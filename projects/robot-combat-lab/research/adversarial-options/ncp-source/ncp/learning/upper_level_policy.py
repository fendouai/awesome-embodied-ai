# Copyright 2024 Tencent. See LICENSE.txt for licensing details.
from rl_games.algos_torch.models import BaseModel, BaseModelNetwork
import torch

from ncp.learning.common import pack_sequence_as_structure_like_gym_space


class UpperLevelPolicy(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    def build(self, config):
        net = self.network_builder.build('upper_level_policy', **config)
        obs_shape = config['input_shape']
        action_space = config['action_space']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)
        return UpperLevelPolicy.Network(net, action_space=action_space, obs_shape=obs_shape,
                                        normalize_value=normalize_value, normalize_input=normalize_input,
                                        value_size=value_size)

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, action_space, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            self.action_space = action_space
            self.freeze_llc_weights()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            action_dist, value, states, loss_dict = self.a2c_network(input_dict)
            if is_train:
                prev_neglogp = action_dist['A_Z'].neg_log_prob(prev_actions['A_Z'])
                loss_dict.update({
                    "entropy_Z": torch.mean(action_dist['A_Z'].entropy()),
                    "entropy_LLC": torch.mean(action_dist['A_LLC'].entropy()),
                })
                result = {
                    'prev_neglogp': prev_neglogp,
                    'values': value,
                    'entropy': action_dist['A_Z'].entropy(),
                    'rnn_states': states,
                    'loss_dict': loss_dict,
                }
                return result
            else:
                selected_action = pack_sequence_as_structure_like_gym_space(self.action_space,
                                                                            [action_dist[key].selected_action for key in
                                                                             action_dist])
                argmax_actions = pack_sequence_as_structure_like_gym_space(self.action_space,
                                                                           [action_dist[key].mode() for key in
                                                                            action_dist])

                neglogp = action_dist['A_Z'].neg_log_prob(selected_action['A_Z'])
                result = {
                    'neglogpacs': neglogp,
                    'values': value,
                    'actions': selected_action,
                    'rnn_states': states,
                    'argmax_actions': argmax_actions,
                }
                return result

        def load_pretrained_model_state_dict(self, weight):
            model_dict = self.state_dict()
            for key in model_dict:
                if any(reuse_key in key for reuse_key in ['llc', 'rms_prop', 'prior', 'embedding']):
                    model_dict.update({key: weight[key]})
            self.load_state_dict(model_dict)

        def freeze_llc_weights(self):
            for name, param, in self.named_parameters():
                if any(freeze_key in name for freeze_key in ['llc', 'rms_prop', 'prior', 'embedding']):
                    param.requires_grad = False
