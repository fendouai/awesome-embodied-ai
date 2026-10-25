# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.vq_vae import VQCodebook


class ActorCriticStage3(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_taskpolicy_obs_ego,
        num_taskpolicy_obs_opponent,
        num_critic_obs_ego,
        num_critic_obs_opponent,
        num_prior_obs,
        num_decoder_obs,
        num_actions,
        normalize_z=False,
        taskpolicy_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        prior_hidden_dims=[256, 256, 256],
        decoder_hidden_dims=[256, 256, 256],
        latent_dim=64,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.name = "ActorCriticStage3"
        mlp_input_dim_task_ego = num_taskpolicy_obs_ego
        mlp_input_dim_task_opponent = num_taskpolicy_obs_opponent
        mpl_input_dim_prior = num_prior_obs
        mlp_input_dim_decoder = num_decoder_obs + latent_dim
        mlp_input_dim_criric_ego = num_critic_obs_ego
        mlp_input_dim_criric_opponent = num_critic_obs_opponent
        self.latent_dim = latent_dim
        # Optional VQ bottleneck on the latent fed into the decoder
        self.vq_decoder_latent: VQCodebook | None = None
        self.normalize_z = normalize_z
        # task policy ego
        taskpolicy_ego_layers = []
        taskpolicy_ego_layers.append(nn.Linear(mlp_input_dim_task_ego, taskpolicy_hidden_dims[0]))
        taskpolicy_ego_layers.append(activation)
        for layer_index in range(len(taskpolicy_hidden_dims)):
            if layer_index == len(taskpolicy_hidden_dims) - 1:
                taskpolicy_ego_layers.append(nn.Linear(taskpolicy_hidden_dims[layer_index], latent_dim))
            else:
                taskpolicy_ego_layers.append(
                    nn.Linear(taskpolicy_hidden_dims[layer_index], taskpolicy_hidden_dims[layer_index + 1])
                )
                taskpolicy_ego_layers.append(activation)
        self.taskpolicy_ego = nn.Sequential(*taskpolicy_ego_layers)
        
        # task policy opponent
        taskpolicy_opponent_layers = []
        taskpolicy_opponent_layers.append(nn.Linear(mlp_input_dim_task_opponent, taskpolicy_hidden_dims[0]))
        taskpolicy_opponent_layers.append(activation)
        for layer_index in range(len(taskpolicy_hidden_dims)):
            if layer_index == len(taskpolicy_hidden_dims) - 1:
                taskpolicy_opponent_layers.append(nn.Linear(taskpolicy_hidden_dims[layer_index], latent_dim))
            else:
                taskpolicy_opponent_layers.append(
                    nn.Linear(taskpolicy_hidden_dims[layer_index], taskpolicy_hidden_dims[layer_index + 1])
                )
                taskpolicy_opponent_layers.append(activation)
        self.taskpolicy_opponent = nn.Sequential(*taskpolicy_opponent_layers)
        
        # Value function ego
        critic_ego_layers = []
        critic_ego_layers.append(nn.Linear(mlp_input_dim_criric_ego, critic_hidden_dims[0]))
        critic_ego_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_ego_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_ego_layers.append(
                    nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1])
                )
                critic_ego_layers.append(activation)
        self.critic_ego = nn.Sequential(*critic_ego_layers)
        
        # Value function opponent
        critic_opponent_layers = []
        critic_opponent_layers.append(nn.Linear(mlp_input_dim_criric_opponent, critic_hidden_dims[0]))
        critic_opponent_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_opponent_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_opponent_layers.append(
                    nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1])
                )
                critic_opponent_layers.append(activation)
        self.critic_opponent = nn.Sequential(*critic_opponent_layers)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(mlp_input_dim_decoder, decoder_hidden_dims[0]))
        decoder_layers.append(activation)
        for layer_index in range(len(decoder_hidden_dims)):
            if layer_index == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[layer_index], num_actions))
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[layer_index], decoder_hidden_dims[layer_index + 1]))
                decoder_layers.append(activation)
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Prior
        prior_layers = []
        prior_layers.append(nn.Linear(mpl_input_dim_prior, prior_hidden_dims[0]))
        prior_layers.append(activation)
        for layer_index in range(len(prior_hidden_dims)):
            if layer_index == len(prior_hidden_dims) - 1:
                prior_layers.append(nn.Linear(prior_hidden_dims[layer_index], 2*latent_dim))
            else:
                prior_layers.append(nn.Linear(prior_hidden_dims[layer_index], prior_hidden_dims[layer_index + 1]))
                prior_layers.append(activation)
        self.prior = nn.Sequential(*prior_layers)

        print(f"Task policy ego MLP: {self.taskpolicy_ego}")
        print(f"Task policy opponent MLP: {self.taskpolicy_opponent}")
        print(f"Critic ego MLP: {self.critic_ego}")
        print(f"Critic opponent MLP: {self.critic_opponent}")
        print(f"Decoder MLP: {self.decoder}")
        print(f"Prior MLP: {self.prior}")
        self.task_policy_last_hidden = taskpolicy_hidden_dims[-1]

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std_ego = nn.Parameter(init_noise_std * torch.ones(latent_dim))
            self.std_opponent = nn.Parameter(init_noise_std * torch.ones(latent_dim)) 
        elif self.noise_std_type == "log":
            self.std_ego = nn.Parameter(torch.log(init_noise_std * torch.ones(latent_dim)))
            self.std_opponent = nn.Parameter(torch.log(init_noise_std * torch.ones(latent_dim)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution_ego = None
        self.distribution_opponent = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    
    @property
    def latent_mean_ego(self):
        return self.distribution_ego.mean
    
    @property
    def latent_std_ego(self):
        return self.distribution_ego.stddev

    @property
    def latent_mean_opponent(self):
        return self.distribution_opponent.mean

    @property
    def latent_std_opponent(self):
        return self.distribution_opponent.stddev
    
    @property
    def entropy_ego(self):
        return self.distribution_ego.entropy().sum(dim=-1)

    @property
    def entropy_opponent(self):
        return self.distribution_opponent.entropy().sum(dim=-1)

    def update_distribution_ego(self, observations):
        # compute mean
        mean = self.taskpolicy_ego(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std_ego.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.std_ego).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        if torch.isnan(std).any():
            print("!!! Fatal error: NaN detected in std !!!")
            print("self.std_ego (raw parameter):", self.std_ego)
            print("mean:", mean)
            raise ValueError("std became NaN")
        std = torch.clamp(std, min=1e-4, max=10.0)
        self.distribution_ego = Normal(mean, std)
    
    def update_distribution_opponent(self, observations):
        # compute mean
        mean = self.taskpolicy_opponent(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std_opponent.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.std_opponent).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        std = torch.clamp(std, min=1e-4, max=10.0)
        self.distribution_opponent = Normal(mean, std)
    
    def act_latent_ego(self, observations, **kwargs):
        self.update_distribution_ego(observations)
        return self.distribution_ego.sample()

    def act_latent_opponent(self, observations, **kwargs):
        self.update_distribution_opponent(observations)
        return self.distribution_opponent.sample()
    
    def get_latent_log_prob_ego(self, latent):
        return self.distribution_ego.log_prob(latent).sum(dim=-1)

    def get_latent_log_prob_opponent(self, latent):
        return self.distribution_opponent.log_prob(latent).sum(dim=-1)
    
    def act_inference_latent_ego(self, observations):
        latent_mean = self.taskpolicy_ego(observations)
        return latent_mean

    def act_inference_latent_opponent(self, observations):
        latent_mean = self.taskpolicy_opponent(observations)
        return latent_mean
    
    def evaluate_ego(self, critic_observations, **kwargs):
        value = self.critic_ego(critic_observations)
        return value
    
    def evaluate_opponent(self, critic_observations, **kwargs):
        value = self.critic_opponent(critic_observations)
        return value

    # TODO add decoder and prior inference here.
    def act_inference_decoder(self, observations):
        actions_mean = self.decoder(observations)
        return actions_mean
    
    def act_inference_prior(self, observations):
        prior_latent_mean, prior_latent_logvar = self.prior(observations).chunk(2, dim=-1)
        return prior_latent_mean, prior_latent_logvar

    def apply_decoder_vq(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply VQ-VAE codebook to the latent fed into the decoder, if enabled."""
        if self.vq_decoder_latent is None:
            return latent
        latent_q, _ = self.vq_decoder_latent(latent)
        return latent_q

    def init_vq_decoder_latent(
        self,
        num_codes: int,
        latent_dim: int,
        beta: float = 0.0,
        weight: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the VQ codebook used for decoder latents.

        This is used by VQ-Stage3 to mirror the Stage2 VQ-VAE codebook inside the policy,
        ensuring correct device placement and inclusion in checkpoints.
        """
        if latent_dim != self.latent_dim:
            raise ValueError(
                f"VQ codebook latent_dim ({latent_dim}) does not match policy latent_dim ({self.latent_dim})."
            )
        # Resolve device from policy parameters if not provided explicitly.
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        self.vq_decoder_latent = VQCodebook(num_codes=num_codes, latent_dim=latent_dim, beta=beta).to(device)
        if weight is not None:
            with torch.no_grad():
                if weight.shape != self.vq_decoder_latent.embedding.weight.shape:
                    raise ValueError(
                        f"Provided VQ weight shape {weight.shape} does not match "
                        f"codebook shape {self.vq_decoder_latent.embedding.weight.shape}."
                    )
                self.vq_decoder_latent.embedding.weight.copy_(weight)

    def act_inference(self, taskpolicy_obs_ego, taskpolicy_obs_opponent,
                            prior_obs_ego, prior_obs_opponent,
                            decoder_obs_ego, decoder_obs_opponent):
        latent_ego = self.act_inference_latent_ego(taskpolicy_obs_ego)
        latent_opponent = self.act_inference_latent_opponent(taskpolicy_obs_opponent)
        prior_latent_mean_ego, _ = self.act_inference_prior(prior_obs_ego)
        prior_latent_mean_opponent, _ = self.act_inference_prior(prior_obs_opponent)
        decoder_input_latent_ego = latent_ego + prior_latent_mean_ego
        decoder_input_latent_opponent = latent_opponent + prior_latent_mean_opponent
        if self.normalize_z:
            decoder_input_latent_ego = torch.nn.functional.normalize(decoder_input_latent_ego)
            decoder_input_latent_opponent = torch.nn.functional.normalize(decoder_input_latent_opponent)

        decoder_input_ego = torch.cat([decoder_obs_ego, decoder_input_latent_ego], dim=-1)
        decoder_input_opponent = torch.cat([decoder_obs_opponent, decoder_input_latent_opponent], dim=-1)
        actions_mean_ego = self.decoder(decoder_input_ego)
        actions_mean_opponent = self.decoder(decoder_input_opponent)
        actions = {}
        actions["robot_A"] = actions_mean_ego
        actions["robot_B"] = actions_mean_opponent
        return actions

    def load_state_dict(self, state_dict, strict=True): # TODO
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
