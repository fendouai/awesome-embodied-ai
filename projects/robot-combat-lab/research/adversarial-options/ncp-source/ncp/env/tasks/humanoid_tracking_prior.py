# Copyright 2024 Tencent. See LICENSE.txt for licensing details.
import math
from collections import OrderedDict

import torch
from torch import Tensor
import numpy as np
from gym import spaces

from ncp.env.tasks.humanoid import Humanoid, compute_local_feature
from ncp.env.tasks.humanoid import compute_humanoid_reset
from ncp.utils.motion_lib import MotionLib


class HumanoidTrackingPrior(Humanoid):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        motion_file = cfg['env'].get('motion_file', None)
        self._train = True if motion_file else False
        if self._train:
            self._num_splits = cfg['env'].get('numSplit', 1)
            self._load_motion(motion_file)
            assert self.num_envs % self._num_splits == 0
            self._historical_count_steps = cfg['env'].get('historicalCountSteps')
            self._frame_counts = torch.zeros(
                (self._historical_count_steps, self._total_frame_nums), device=self.device, dtype=torch.float)
            self._similarity_threshold = cfg['env'].get('similarityThreshold')

        dict_obs_space = self._build_obs_space()
        self.observation_space = spaces.Dict(dict_obs_space)


    def reset(self, env_ids=None):
        super().reset(env_ids)
        obs = OrderedDict({
            'prop': self.obs_buf.to(self.device),
        })
        return obs

    def step(self, actions):
        super().step(actions)
        obs = OrderedDict({
            'prop': self.obs_buf.to(self.device),
        })
        return obs, self.rew_buf.to(self.device), self.reset_buf.to(self.device), self.extras

    def _load_motion(self, motion_file):
        if motion_file.endswith('yaml'):
            assert (self._dof_offsets[-1] == self.num_dof)
            self._motion_lib = MotionLib(motion_file=motion_file,
                                         dof_body_ids=self._dof_body_ids,
                                         dof_offsets=self._dof_offsets,
                                         key_body_ids=self._key_body_ids.cpu().numpy(),
                                         full_info_body_id=[],
                                         device=self.device,
                                         asset_axis=self._asset_axis)

            self._total_frame_nums = self._motion_lib.get_total_frames()

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, _, key_pos = \
                self._motion_lib.get_all_motion_states()

            _, local_root_vel_kin, local_root_ang_vel_kin, dof_obs_kin, _ = \
                compute_local_feature(root_pos, root_rot, root_vel, root_ang_vel,
                                      dof_pos, key_pos, self._dof_obs_size, self._dof_offsets)

        else:
            tracking_result = np.load(motion_file, allow_pickle=True)
            dof_obs_kin = torch.tensor(tracking_result[()]['joint_pos'], device=self.device)
            local_root_vel_kin = torch.tensor(tracking_result[()]['vel'], device=self.device)
            local_root_ang_vel_kin = torch.tensor(tracking_result[()]['ang_vel'], device=self.device)
            self._total_frame_nums = dof_obs_kin.shape[0]

        self._kinematic_tuple = [local_root_vel_kin, local_root_ang_vel_kin, dof_obs_kin]

    def _reset_actors(self, env_ids):
        dyn_root_states = self._initial_humanoid_root_states[env_ids].clone()

        self._humanoid_root_states[env_ids] = dyn_root_states
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces, self._contact_body_ids,
                                                                           self._rigid_body_pos,
                                                                           self.max_episode_length,
                                                                           self._enable_early_termination,
                                                                           self._termination_heights)

    def _compute_reward(self):
        if self._train:
            self.rew_buf[:] = self._compute_count_reward()
        else:
            self.rew_buf[:] = 1.0

    def _compute_count_reward(self):
        _, local_root_vel_dyn, local_root_ang_vel_dyn, dof_obs_dyn, _ = \
            compute_local_feature(self._rigid_body_pos[:, 0, :],
                                  self._rigid_body_rot[:, 0, :],
                                  self._rigid_body_vel[:, 0, :],
                                  self._rigid_body_ang_vel[:, 0, :],
                                  self._dof_pos,
                                  self._rigid_body_pos[:, self._key_body_ids, :],
                                  self._dof_obs_size,
                                  self._dof_offsets)

        dynamic_feature = (local_root_vel_dyn, local_root_ang_vel_dyn, dof_obs_dyn)
        current_id = self._total_steps % self._historical_count_steps
        self._frame_counts[current_id], frame_match_idx, has_frame_match =\
            self._compute_frame_similarity(dynamic_feature)

        count = torch.sum(self._frame_counts[:, frame_match_idx], dim=0)
        count_reward = torch.sqrt(
            2 * math.log(min(self._total_steps, self._historical_count_steps) * self.num_envs) / count)
        count_reward = torch.where(has_frame_match, count_reward, torch.zeros_like(count_reward))
        return count_reward

    def _compute_frame_similarity(self, dynamic_feature):
        frame_match_idx = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        has_frame_match = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        frame_counts = torch.ones(self._total_frame_nums, dtype=torch.int64, device=self.device)

        expand_kinematic_tuple = [data.unsqueeze(0) for data in self._kinematic_tuple]
        split_num_envs = self.num_envs // self._num_splits
        for i in range(self._num_splits):
            env_ids = torch.arange(i * split_num_envs, (i + 1) * split_num_envs, device=self.device, dtype=torch.long)
            expand_dynamic_tuple = [data[env_ids].unsqueeze(1) for data in dynamic_feature]
            similarity = _compute_frame_similarity(*expand_dynamic_tuple, *expand_kinematic_tuple)

            frame_counts += torch.sum(similarity > self._similarity_threshold, dim=0)
            frame_match_idx[env_ids] = torch.argmax(similarity, dim=1)
            has_frame_match[env_ids] = torch.any(similarity > self._similarity_threshold, dim=1)

        return frame_counts, frame_match_idx, has_frame_match

#####################################################################
#  =========================jit functions=========================  #
#####################################################################


@torch.jit.script
def _compute_frame_similarity(local_root_vel_dyn: Tensor, local_root_ang_vel_dyn: Tensor,
                              dof_obs_dyn: Tensor,

                              local_root_vel_kin: Tensor, local_root_ang_vel_kin: Tensor,
                              dof_obs_kin: Tensor) -> Tensor:
    joint_pos_w = 0.6
    root_vel_w = 0.2

    joint_pos_scale = 2.0
    root_vel_scale = 2.0

    joint_pos_err = torch.linalg.norm((dof_obs_kin - dof_obs_dyn), dim=-1)
    root_vel_err = torch.sum(torch.square(local_root_vel_kin - local_root_vel_dyn), dim=-1)
    root_ang_vel_err = torch.sum(torch.square(local_root_ang_vel_kin - local_root_ang_vel_dyn), dim=-1)
    root_vel_err = root_vel_err + 0.1 * root_ang_vel_err

    joint_pos_similarity = torch.exp(-joint_pos_scale * joint_pos_err)
    root_vel_similarity = torch.exp(-root_vel_scale * root_vel_err)
    similarity = joint_pos_w * joint_pos_similarity + root_vel_w * root_vel_similarity
    return similarity
