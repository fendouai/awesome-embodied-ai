# Copyright 2024 Tencent. See LICENSE.txt for licensing details.
import torch
import torch.nn as nn
import torch.distributions as dists

import numpy as np
from rl_games.algos_torch import network_builder

from ncp.learning.nn.running_mean_std import RunningMeanStd
from ncp.learning.prior_network_builder import Prior
from ncp.learning.tracking_z_network_builder import LowLevelController
from ncp.learning.distributions import CategoricalDistribution
from ncp.learning.common import pack_sequence_as_structure_like_gym_space


def _norm_initializer(param, std=1.0):
    shape = param.data.shape
    out = np.random.randn(*shape).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=-1, keepdims=True))
    out = torch.from_numpy(out)
    param.data.copy_(out)


class UpperLevelPolicyBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, name, **kwargs):
        net = UpperLevelPolicyBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)

            ob_space = kwargs.get('input_shape')
            prop_dim = ob_space['prop'][0]
            drill_dim = ob_space['drill'][0]
            self.rms_prop = RunningMeanStd(prop_dim, params['rms_momentum'])
            self.rms_drill = RunningMeanStd(drill_dim, params['rms_momentum'])

            action_space = kwargs.get('action_space')

            hidden_dim = params['hidden_dim']
            self.value_size = kwargs.get('value_size', 1)
            self.num_seqs = kwargs.get('num_seqs', 1)
            self.has_rnn = False
            self.use_vq = params['use_vq']
            z_len = params['z_len']
            num_embeddings = params.get("num_embeddings")
            self.value_net = ValueNet(prop_dim, drill_dim, self.value_size, hidden_dim)
            self.pi_net = PiNet(prop_dim, drill_dim, action_space, hidden_dim, z_len, num_embeddings)

        def forward(self, obs_dict):
            is_train = obs_dict.get('is_train', True)
            prev_actions = obs_dict['prev_actions']
            prop = obs_dict['obs']['prop']
            drill = obs_dict['obs']['drill']

            prop_rms, _ = self.rms_prop(prop)
            drill_rms, drill_rms_loss = self.rms_drill(drill)

            prop_rms = prop_rms.clamp(-5.0, 5.0)
            drill_rms = drill_rms.clamp(-5.0, 5.0)

            action_dist = self.pi_net(prop_rms, drill_rms, prev_actions=prev_actions)
            value = self.value_net(prop_rms, drill_rms)
            states = obs_dict.get('rnn_states', None)

            if is_train:
                rms_loss = torch.mean(drill_rms_loss, dim=(0, 1))
                with torch.no_grad():
                    prior_z_dist = self.pi_net.prior(prop_rms)
                kl_div = dists.kl_divergence(action_dist['A_Z'].distribution, prior_z_dist.distribution)
                kl_loss = kl_div.mean()
            else:
                rms_loss = None
                kl_loss = None
            loss_dict = {
                "rms_loss": rms_loss,
                "kl_loss": kl_loss,
            }
            output = (action_dist, value, states) + (loss_dict,)

            return output

        def eval_critic(self, obs):
            prop = obs['prop']
            drill = obs['drill']

            prop_rms, _ = self.rms_prop(prop)
            drill_rms, _ = self.rms_drill(drill)

            prop_rms = prop_rms.clamp(-5.0, 5.0)
            future_rms = drill_rms.clamp(-5.0, 5.0)
            value = self.value_net(prop_rms, future_rms)
            return value


class ValueNet(nn.Module):
    def __init__(self, prop_dim, drill_dim, value_size, hidden_dim):
        super(ValueNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prop_dim + drill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, value_size)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                _norm_initializer(layer.weight)
                _norm_initializer(layer.bias)

    def forward(self, prop, drill):
        embed = torch.hstack((prop, drill))
        return self.mlp(embed)


class HighLevelPolicy(nn.Module):
    def __init__(self, prop_dim, drill_dim, hidden_dim, num_embeddings):
        super(HighLevelPolicy, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prop_dim + drill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_embeddings)
        )

    def forward(self, prop, drill):
        embed = torch.hstack((prop, drill))
        z = self.mlp(embed)
        action_dist = CategoricalDistribution(z)
        return action_dist


class PiNet(nn.Module):
    def __init__(self, prop_dim, drill_dim, action_space, hidden_dim, z_len, num_embeddings):
        super(PiNet, self).__init__()
        self.action_space = action_space
        self.high_level_policy = HighLevelPolicy(prop_dim, drill_dim, hidden_dim, num_embeddings)
        self.prior = Prior(prop_dim, hidden_dim, num_embeddings)
        self.z_len = z_len
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, z_len)
        self.embeddings.weight.data.uniform_(-1. / num_embeddings, 1. / num_embeddings)
        self.num_embeddings = num_embeddings

        self.llc = LowLevelController(prop_dim=prop_dim, z_len=z_len, hidden_dim=hidden_dim,
                                      action_dim=action_space['A_LLC'].shape[0], init_logstd=-2.0)

    def forward(self, prop, drill, prev_actions=None):
        z_dist = self.high_level_policy(prop, drill)
        if prev_actions is None:
            encoding_indices = z_dist.selected_action
        else:
            encoding_indices = prev_actions['A_Z']
        quantized = self.quantize(encoding_indices)
        llc_dist = self.llc(prop, z=quantized)

        action_dist = pack_sequence_as_structure_like_gym_space(self.action_space, [z_dist, llc_dist])
        return action_dist

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)
