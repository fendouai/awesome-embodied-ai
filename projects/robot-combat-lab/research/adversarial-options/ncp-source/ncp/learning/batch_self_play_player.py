# Copyright (c) 2019 Denys88
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright 2024 Tencent. See LICENSE.txt for licensing details.
import time
import gym
import torch

from rl_games.algos_torch.players import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch import model_builder

import numpy as np


class BatchSelfPlayPlayer(BasePlayer):
    def __init__(self, params):
        super(BatchSelfPlayPlayer, self).__init__(params)
        self.network = self.config['network']
        self.opponent_network = self.config['opponent_network']

        self.action_space = self.env_info['action_space']
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        assert isinstance(self.observation_space, gym.spaces.Tuple)

        single_agent_observation_space = self.observation_space[0]
        if isinstance(single_agent_observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in single_agent_observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = single_agent_observation_space.shape

        obs_shape = self.obs_shape
        config = {
            'action_space' : self.action_space,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(config)
        self.opponent_model = self.opponent_network.build(config)
        self.model.to(self.device)
        self.opponent_model.to(self.device)
        self.model.eval()
        self.opponent_model.eval()
        self.is_rnn = self.model.is_rnn()
        self.restore(params['opponent_load_path'], restore_self=False)
        self.states = [None, None]

    def init_rnn(self):
        if self.is_rnn:
            self_rnn_states = self.model.get_default_rnn_state()
            opponent_rnn_states = self.opponent_model.get_default_rnn_state()
            self.states = [
                [torch.zeros((s.size()[0], self.batch_size, s.size()[2]), dtype=torch.float32).to(self.device) for s in
                 rnn_states] for rnn_states in [self_rnn_states, opponent_rnn_states]]

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        # op_agent = getattr(self.env, "create_agent", None)
        # if op_agent:
        #     agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            for _ in range(self.max_steps):
                obses = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                self._post_step(info)

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards / done_count
                        cur_steps_done = cur_steps / done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                        else:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

                done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps
                  / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def _preproc_obs(self, obs_batch):
        return obs_batch

    def env_reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        return [self.obs_to_torch(o) for o in obs]

    def _post_step(self, info):
        return

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def get_masked_action(self, obs, masks, is_deterministic=False):
        raise NotImplementedError

    def get_action(self, obs, is_deterministic=False):
        if not self.has_batch_dimension:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)

        self.model.eval()
        self.opponent_model.eval()
        input_dict = [{
            'is_train': False,
            'prev_actions': None,
            'obs' : obs[i],
            'rnn_states' : state
        } for i, state in enumerate(self.states)]
        with torch.no_grad():
            res_dict = [model(current_dict) for model, current_dict in
                        zip([self.model, self.opponent_model], input_dict)]
        self.states = [current_dict['rnn_states'] for current_dict in res_dict]

        if is_deterministic:
            current_action = [current_dict['argmax_actions'] for current_dict in res_dict]
        else:
            current_action = [current_dict['actions'] for current_dict in res_dict]

        return current_action

    def get_full_state_weights(self):
        state = self.get_weights()
        return state

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)
        self.config['opponent_network'] = builder.load(params)

    def restore(self, fn, restore_self=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        if restore_self:
            self.model.load_state_dict(checkpoint['model'])
            if self.normalize_input and 'running_mean_std' in checkpoint:
                self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        else:
            self.opponent_model.load_state_dict(checkpoint['model'])
            if self.normalize_input and 'running_mean_std' in checkpoint:
                self.opponent_model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()

    def get_batch_size(self, obses, batch_size):
        return super().get_batch_size(obses[0], batch_size)
