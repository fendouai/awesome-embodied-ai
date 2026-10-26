# Copyright (c) 2019 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal


class Distribution(ABC, nn.Module):
    """Abstract base class for distributions."""

    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = None
        self._selected_action = None

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log likelihood
        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    def neg_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the negative log likelihood
        :param x: the taken action
        :return: The negative log likelihood of the distribution
        """
        return - self.log_prob(x)

    def kl_divergence(self, dist) -> torch.Tensor:
        """
        Wrapper for the PyTorch implementation of the full form KL Divergence
        :param self: the p distribution
        :param dist: the q distribution
        :return: KL(dist_true||dist_pred)
        """
        # KL Divergence for different distribution types is out of scope
        assert self.__class__ == dist.__class__, "Error: input distributions should be the same type"
        return torch.distributions.kl_divergence(self.distribution, dist.distribution)

    @abstractmethod
    def entropy(self) -> Optional[torch.Tensor]:
        """
        Returns Shannon's entropy of the probability
        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Returns a sample from the probability distribution
        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution
        :return: the stochastic action
        """

    @property
    def selected_action(self) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        :param deterministic:
        :return:
        """
        if self._selected_action is None:
            self._selected_action = self.sample()
        return self._selected_action


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        """
        Args:
            actions_dim: shape: the dimension of actions
            actions_log_std: will be interpreted as log actions_std
              and can therefore be any real number
        """
        super(DiagGaussianDistribution, self).__init__()
        self.distribution = Normal(mean, log_std.exp())

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.
        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.
    :param action_dim: Dimension of the action space.
     :param epsilon: small value to avoid NaN due to numerical imprecision.
     """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, epsilon: float = 1e-6):
        super(SquashedDiagGaussianDistribution, self).__init__(mean, log_std)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        # It will be clipped to avoid NaN when inversing tanh
        eps = torch.finfo(actions.dtype).eps
        actions = actions.clamp(min=-1.0 + eps, max=1.0 - eps)
        gaussian_actions = 0.5 * (actions.log1p() - (-actions).log1p())
        # Log likelihood for a Gaussian distribution
        log_prob = super(SquashedDiagGaussianDistribution, self).log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions ** 2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        gaussian_actions = super().sample()
        return torch.tanh(gaussian_actions)

    def mode(self) -> torch.Tensor:
        gaussian_actions = super().mode()
        # Squash the output
        return torch.tanh(gaussian_actions)


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.
    The `action_logits` argument will be interpreted as unnormalized log probabilities
    and can therefore be any real number. It will likewise be normalized so that
    the resulting probabilities sum to 1 along the last dimension.
    """

    def __init__(self, logits: torch.Tensor):
        # action_logits: log probabilities (unnormalized): (batch_size, n_actions)
        super(CategoricalDistribution, self).__init__()
        self.distribution = Categorical(logits=logits)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.
    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_logits: torch.Tensor, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.action_dims = action_dims
        self.distribution = [Categorical(logits=split) for split in
                             torch.split(action_logits, tuple(self.action_dims), dim=1)]

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for MultiBinary action spaces.
    :param action_dim: Number of binary actions
    """

    def __init__(self, action_logits: torch.Tensor):
        super(BernoulliDistribution, self).__init__()
        self.distribution = Bernoulli(logits=action_logits)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.round(self.distribution.probs)
