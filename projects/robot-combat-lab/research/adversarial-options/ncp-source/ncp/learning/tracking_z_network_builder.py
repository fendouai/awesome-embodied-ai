# Copyright 2024 Tencent. See LICENSE.txt for licensing details.
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from rl_games.algos_torch import network_builder

from ncp.learning.distributions import DiagGaussianDistribution
from ncp.learning.nn.running_mean_std import RunningMeanStd


def _norm_initializer(param, std=1.0):
    shape = param.data.shape
    out = np.random.randn(*shape).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=-1, keepdims=True))
    out = torch.from_numpy(out)
    param.data.copy_(out)


class TrackingZBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, name, **kwargs):
        net = TrackingZBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            ob_space = kwargs.get('input_shape')
            prop_dim = ob_space['prop'][0]
            future_dim = ob_space['future'][0]
            self.rms_prop = RunningMeanStd(prop_dim, params['rms_momentum'])
            self.rms_future = RunningMeanStd(future_dim, params['rms_momentum'])

            action_dim = kwargs.get('action_space').shape[0]

            hidden_dim = params['hidden_dim']
            self.value_size = kwargs.get('value_size', 1)
            self.num_seqs = kwargs.get('num_seqs', 1)
            self.has_rnn = False
            self.use_vq = params['use_vq']
            z_len = params['z_len']
            num_embeddings = params.get("num_embeddings")
            init_logstd = params.get("init_logstd", -2.0)

            self.value_net = ValueNet(prop_dim, future_dim, self.value_size, hidden_dim)
            self.pi_net = PiNet(prop_dim, future_dim, action_dim, hidden_dim, self.use_vq,
                                z_len, num_embeddings, init_logstd)

        def forward(self, obs_dict):
            is_train = obs_dict.get('is_train', True)
            prop = obs_dict['obs']['prop']
            future = obs_dict['obs']['future']

            prop_rms, prop_rms_loss = self.rms_prop(prop)
            future_rms, future_rms_loss = self.rms_future(future)

            prop_rms = prop_rms.clamp(-5.0, 5.0)
            future_rms = future_rms.clamp(-5.0, 5.0)

            if self.use_vq:
                action_dist, quantized, encoder_z, encodings = self.pi_net(prop_rms, future_rms)
                value = self.value_net(prop_rms, future_rms)
                states = obs_dict.get('rnn_states', None)

                if is_train:
                    rms_loss = torch.mean(prop_rms_loss, dim=(0, 1)) + torch.mean(future_rms_loss, dim=(0, 1))
                    e_latent_loss = torch.mean((quantized.detach() - encoder_z) ** 2, dim=(0, 1))
                    q_latent_loss = torch.mean((quantized - encoder_z.detach()) ** 2, dim=(0, 1))
                    avg_probs = torch.mean(encodings.float(), 0)
                    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                else:
                    rms_loss = None
                    e_latent_loss = None
                    q_latent_loss = None
                    perplexity = None
                loss_dict = {
                    "rms_loss": rms_loss,
                    "e_latent_loss": e_latent_loss,
                    "q_latent_loss": q_latent_loss,
                    "perplexity": perplexity,
                }
                output = (action_dist, value, states) + (loss_dict,)

            else:
                action_dist = self.pi_net(prop_rms, future_rms)
                value = self.value_net(prop_rms, future_rms)
                states = obs_dict.get('rnn_states', None)

                if is_train:
                    rms_loss = torch.mean(prop_rms_loss, dim=(0, 1)) + torch.mean(future_rms_loss, dim=(0, 1))
                    e_latent_loss = torch.zeros_like(rms_loss)
                    q_latent_loss = torch.zeros_like(rms_loss)
                else:
                    rms_loss = None
                    e_latent_loss = None
                    q_latent_loss = None
                loss_dict = {
                    "rms_loss": rms_loss,
                    "e_latent_loss": e_latent_loss,
                    "q_latent_loss": q_latent_loss,
                }
                output = (action_dist, value, states) + (loss_dict,)

            return output

        def eval_critic(self, obs):
            prop = obs['prop']
            future = obs['future']

            prop_rms, _ = self.rms_prop(prop)
            future_rms, _ = self.rms_future(future)

            prop_rms = prop_rms.clamp(-5.0, 5.0)
            future_rms = future_rms.clamp(-5.0, 5.0)
            value = self.value_net(prop_rms, future_rms)
            return value


class ValueNet(nn.Module):
    def __init__(self, prop_dim, future_dim, value_size, hidden_dim):
        super(ValueNet, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(prop_dim + future_dim, hidden_dim),
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

    def forward(self, prop, future):
        embed = torch.hstack((prop, future))
        return self.mlp(embed)


class Encoder(nn.Module):
    def __init__(self, prop_dim, future_dim, hidden_dim, z_len):
        super(Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prop_dim + future_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, z_len)
        )

    def forward(self, prop, future):
        embed = torch.hstack((prop, future))
        z = self.mlp(embed)
        return z


class LowLevelController(nn.Module):
    def __init__(self, prop_dim, z_len, hidden_dim, action_dim, init_logstd):
        super(LowLevelController, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prop_dim + z_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        _norm_initializer(self.mlp[-1].weight, 0.01)
        self.log_std = Parameter(torch.ones(action_dim) * torch.tensor(init_logstd), requires_grad=True)

    def forward(self, prop, z):
        z = z.squeeze()
        embed = torch.hstack((prop, z))
        out = self.mlp(embed)
        action_dist = DiagGaussianDistribution(out, self.log_std)
        return action_dist


class PiNet(nn.Module):
    def __init__(self, prop_dim, future_dim, action_dim, hidden_dim, use_vq, z_len, num_embeddings, init_logstd):
        super(PiNet, self).__init__()
        self.encoder = Encoder(prop_dim=prop_dim, future_dim=future_dim, hidden_dim=hidden_dim,
                               z_len=z_len)
        self.use_vq = use_vq
        self.z_len = z_len
        if use_vq:
            self.embeddings = nn.Embedding(num_embeddings, z_len)
            self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
            self.num_embeddings = num_embeddings

        self.llc = LowLevelController(prop_dim=prop_dim, z_len=z_len, hidden_dim=hidden_dim,
                                      action_dim=action_dim, init_logstd=init_logstd)

    def forward(self, prop, future):
        encoder_z = self.encoder(prop, future)
        if self.use_vq:
            flat_z = encoder_z.view(encoder_z.shape[0], self.z_len)
            encoding_indices = self.get_code_indices(flat_z)
            quantized = self.quantize(encoding_indices)
            quantized = quantized.view_as(encoder_z)
            z_curr = encoder_z + (quantized - encoder_z).detach()
            llc_dist = self.llc(prop, z_curr)
            return llc_dist, quantized, encoder_z, torch.nn.functional.one_hot(encoding_indices, self.num_embeddings)
        else:
            llc_dist = self.llc(prop, encoder_z)
            return llc_dist

    def get_code_indices(self, flat_x):
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2. * torch.matmul(flat_x, self.embeddings.weight.t())
                     )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)
