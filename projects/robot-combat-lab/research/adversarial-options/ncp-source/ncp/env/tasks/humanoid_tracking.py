# Copyright 2024 Tencent. See LICENSE.txt for licensing details.
import os
from collections import OrderedDict
from enum import Enum
from typing import Tuple, List

import torch
from torch import Tensor
import pandas as pd
import numpy as np
from gym import spaces
from isaacgym import gymapi, gymtorch

from ncp.env.tasks.humanoid import Humanoid, compute_local_feature
from ncp.utils import torch_utils
from ncp.utils.motion_lib import MotionLib


class HumanoidTracking(Humanoid):
    class StateInit(Enum):
        Start = 0  # for testing
        Hybrid = 1  # for training

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = self.StateInit[state_init]
        self._from_start_frame_prob = cfg["env"]["fromStartFrameProb"]
        self._random_state_init_prob = cfg["env"]["randomStateInitProp"]
        self._transition_steps = cfg["env"]["transitionSteps"]
        self._enable_kin = not headless and cfg["env"].get('enable_kin', True)
        cfg["env"]["numHumanoid"] = 1 + self._enable_kin

        self._movable_assets_path = cfg["env"].get("movable_assets_path")
        self._scene_assets_path = cfg["env"].get("scene_assets_path")
        self._future_frame = cfg['env'].get('futureFrame')
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        if self._enable_kin:
            self._kin_root_states = self._root_states.view(self.num_envs, self.get_num_actors_per_env(), 13)[..., 1, :]
            num_humanoid = 2
            dof_per_env = self._dof_state.shape[0] // self.num_envs // num_humanoid
            self._kin_dof_pos = self._dof_state.view(self.num_envs,
                                                     num_humanoid,
                                                     dof_per_env, 2)[:, 1, :self.num_dof, 0]
            self._kin_dof_vel = self._dof_state.view(self.num_envs,
                                                     num_humanoid,
                                                     dof_per_env, 2)[:, 1, :self.num_dof, 1]

        self._time = torch.zeros(self.num_envs, device=self.device)
        self._motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._transition_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._from_start_frame_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.full_info_body_ids = self._build_key_body_ids_tensor(cfg["env"].get('fullInfoBodyIds', []))
        self._load_motion(cfg['env']['motion_file'])

        self._rewards_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._max_steps = self._motion_lib.motion_lengths // self.dt
        self._average_reward = torch.zeros(len(self._motion_lib.motion_lengths), device=self.device, dtype=torch.float)

        self.global_rank = int(os.getenv("RANK", "0"))
        self._prioritized_sampling_factor = cfg['env']['prioritizedSamplingFactor']
        if self._prioritized_sampling_factor is not None:
            assert isinstance(self._prioritized_sampling_factor, int)
            prioritized_sampling_result_path = cfg['env'].get('prioritizedSamplingResultPath', None)
            if prioritized_sampling_result_path is not None:
                self._prioritized_sampling_result_path = os.path.join(prioritized_sampling_result_path,
                                                                      'sampling_result.csv')
                os.makedirs(os.path.dirname(self._prioritized_sampling_result_path), exist_ok=True)
            else:
                self._prioritized_sampling_result_path = None

        self._early_termination_threshold = cfg['env'].get('early_termination_threshold', None)
        self.future_buf = torch.zeros(
            (self.num_envs, self._num_future), device=self.device, dtype=torch.float)
        dict_obs_space = self._build_obs_space()

        self.observation_space = spaces.Dict(dict_obs_space)
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1., np.ones(self._num_actions) * 1.)

    def _build_obs_space(self):
        dict_obs_space = OrderedDict({
            'prop': spaces.Box(0, 0, shape=(self._num_obs,)),
            'future': spaces.Box(0, 0, shape=(self._num_future,)),
        })
        return dict_obs_space

    def _setup_character_props(self):
        super()._setup_character_props()
        key_bodies = self.cfg["env"].get("keyBodies", [])
        self._num_future = \
            (3 + 6 + 3 + 3 + self._dof_obs_size + self._num_actions + len(key_bodies) * 3) * self._future_frame

    def _update_transition_count(self):
        self._transition_counter -= 1
        self._transition_counter = torch.clamp_min(self._transition_counter, 0)

    def pre_physics_step(self, actions):
        self._update_transition_count()

        pd_tar = actions + self._dof_pos
        if self._enable_kin:
            pd_tar = pd_tar.repeat(1, 2)
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        pd_tar_tensor,
                                                        gymtorch.unwrap_tensor(self._humanoid_actor_ids),
                                                        len(self._humanoid_actor_ids))

    def post_physics_step(self):
        self._time += self.dt
        super().post_physics_step()
        if self._enable_kin:
            self._motion_sync()

    def _load_motion(self, motion_file):
        assert (self._dof_offsets[-1] == self.num_dof)
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(),
                                     full_info_body_id=self.full_info_body_ids,
                                     device=self.device,
                                     asset_axis=self._asset_axis)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._scene_assets, self._scene_assets_handles, \
            self._scene_assets_num_bodies, self._scene_assets_num_bodies_shapes = \
            self._load_interacting_assets(self._scene_assets_path, fix_base_link=True)
        self._movable_assets, self._movable_assets_handles, \
            self._movable_assets_num_bodies, self._moving_asset_num_bodies_shapes \
            = self._load_interacting_assets(self._movable_assets_path, fix_base_link=False)
        super()._create_envs(num_envs, spacing, num_per_row)

    def _load_interacting_assets(self, assets_paths, fix_base_link):
        assets, assets_handles, assets_num_bodies, assets_num_bodies_shapes = [], [], [], []
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base_link

        assets_num = len(assets_paths) if assets_paths is not None else 0
        for i in range(assets_num):
            asset_root = self.cfg["env"]["asset"]["assetRoot"]
            asset_file = assets_paths[i]

            curr_asset_path = os.path.join(asset_root, asset_file)
            asset_root = os.path.dirname(curr_asset_path)
            asset_file = os.path.basename(curr_asset_path)
            curr_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

            assets_num_bodies.append(self.gym.get_asset_rigid_body_count(curr_asset))
            assets_num_bodies_shapes.append(self.gym.get_asset_rigid_shape_count(curr_asset))
            assets.append(curr_asset)
            assets_handles.append([])

        return assets, assets_handles, assets_num_bodies, assets_num_bodies_shapes

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        max_agg_bodies = self.num_humanoid_bodies * (1 + self._enable_kin) \
            + sum(self._scene_assets_num_bodies) + sum(self._movable_assets_num_bodies)
        max_agg_shapes = self.num_humanoid_shapes * (1 + self._enable_kin) \
            + sum(self._scene_assets_num_bodies_shapes) + sum(self._moving_asset_num_bodies_shapes)

        self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self._enable_kin:
            self._kin_humanoid_handle = []
            self._build_kin_humanoid(env_id, env_ptr, humanoid_asset)
        self._build_interacting_assets(env_id, env_ptr, self._scene_assets, self._scene_assets_handles, "scene_assets")
        self._build_interacting_assets(env_id, env_ptr, self._movable_assets, self._movable_assets_handles,
                                       "movable_assets")
        self.gym.end_aggregate(env_ptr)

    def _build_kin_humanoid(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id + self.num_envs
        col_filter = self._get_humanoid_collision_filter() + 1
        segmentation_id = 0

        start_pose = gymapi.Transform()
        char_h = 0.89
        start_pose.p = gymapi.Vec3(*torch_utils.get_axis_params(char_h, self.up_axis_idx))
        start_pose.p.x = 1.0
        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose,
                                                "kin", col_group, col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)
        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(0.78, 0.19, 0.19))
        self._kin_humanoid_handle.append(humanoid_handle)

    def _build_interacting_assets(self, env_id, env_ptr, assets, assets_handles, name):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        for i in range(len(assets)):
            handle = self.gym.create_actor(env_ptr, assets[i], default_pose,
                                           name + str(i), col_group, col_filter, segmentation_id)
            assets_handles[i].append(handle)

    def _reset_actors(self, env_ids):
        num_envs = env_ids.shape[0]

        if self._state_init == HumanoidTracking.StateInit.Start:
            self._time[env_ids] = 0.0
            self._motion_ids[env_ids] = self._motion_lib.sample_motions(num_envs)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _, _ \
                = self._motion_lib.get_motion_state(self._motion_ids[env_ids], self._time[env_ids])
            self._set_humanoid_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos,
                                     root_vel=root_vel, root_ang_vel=root_ang_vel, dof_vel=dof_vel)

        elif self._state_init == HumanoidTracking.StateInit.Hybrid:
            self._update_tracking_result(env_ids)
            self._motion_ids[env_ids] = self._motion_lib.sample_motions(num_envs)
            self._transition_counter[env_ids] = 0

            from_start_frame_probs = torch_utils.to_torch([self._from_start_frame_prob] * num_envs, device=self.device)
            from_start_frame_mask = torch.bernoulli(from_start_frame_probs) == 1.0
            from_start_frame_ids = env_ids[from_start_frame_mask]

            if len(from_start_frame_ids) > 0:
                self._time[from_start_frame_ids] = 0.0
                self._from_start_frame_env[from_start_frame_ids] = 1.0

            from_random_frame_mask = torch.logical_not(from_start_frame_mask)
            from_random_frame_ids = env_ids[from_random_frame_mask]

            if len(from_random_frame_ids) > 0:
                self._time[from_random_frame_ids] = \
                    self._motion_lib.sample_time(self._motion_ids[from_random_frame_ids])
                self._from_start_frame_env[from_random_frame_ids] = 0.0

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _, _ \
                = self._motion_lib.get_motion_state(self._motion_ids[env_ids], self._time[env_ids])

            random_init_probs = torch_utils.to_torch([self._random_state_init_prob] * len(env_ids), device=self.device)
            random_init_mask = torch.logical_and(from_random_frame_mask, torch.bernoulli(random_init_probs) == 1.0)
            random_init_ids = env_ids[random_init_mask]

            if len(random_init_ids) > 0:
                self._transition_counter[random_init_ids] = self._transition_steps

                random_init_motion_id = self._motion_lib.sample_motions(len(random_init_ids))
                random_init_time = self._motion_lib.sample_time(random_init_motion_id)

                root_pos_new, root_rot_new, dof_pos[random_init_mask], root_vel[random_init_mask], \
                    root_ang_vel[random_init_mask], dof_vel[random_init_mask], _, _ \
                    = self._motion_lib.get_motion_state(random_init_motion_id, random_init_time)
                root_pos[random_init_mask, 2] = root_pos_new[:, 2]
                root_rot[random_init_mask] = torch_utils.quat_mul(
                    torch_utils.calc_heading_quat(root_rot[random_init_mask]),
                    torch_utils.quat_mul(torch_utils.calc_heading_quat_inv(root_rot_new), root_rot_new))

            self._set_humanoid_state(env_ids=env_ids, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos,
                                     root_vel=root_vel, root_ang_vel=root_ang_vel, dof_vel=dof_vel)

    def _update_tracking_result(self, env_ids):
        if self._prioritized_sampling_factor is not None:
            from_start_frame_env_ids = env_ids[self._from_start_frame_env[env_ids]]
            if len(from_start_frame_env_ids) > 0:
                motion_id = self._motion_ids[from_start_frame_env_ids]
                self._average_reward[motion_id] = \
                    self._rewards_sum[from_start_frame_env_ids] / self._max_steps[motion_id]
                assert (self._average_reward <= 1.0).all()
                self._motion_lib.motion_weights[:] = (1 - self._average_reward) ** self._prioritized_sampling_factor
                self._motion_lib.motion_weights /= self._motion_lib.motion_weights.sum()

            if (self._total_steps + 1) % 500 == 0 and self._prioritized_sampling_result_path is not None:
                result = np.hstack((np.array(self._total_steps + 1), self._average_reward.cpu().numpy()))
                pd.DataFrame(result.reshape(1, -1)).to_csv(self._prioritized_sampling_result_path,
                                                           mode='a', header=False, index=False)
        self._rewards_sum[env_ids] = 0.0

    def _update_tracking_future(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        future_frame = self._future_frame
        future_time = self._time[env_ids].unsqueeze(-1) + (torch.arange(future_frame, device=self.device) + 1) * self.dt
        motion_id = self._motion_ids[env_ids].unsqueeze(-1).repeat((1, future_frame))
        flat_future_time, flat_motion_id = future_time.view(-1), motion_id.view(-1)

        root_pos_kin, root_rot_kin, dof_pos_kin, root_vel_kin, root_ang_vel_kin, dof_vel_kin, key_pos_kin, _ \
            = self._motion_lib.get_motion_state(flat_motion_id, flat_future_time)
        root_pos_dyn, root_rot_dyn = self._rigid_body_pos[env_ids, 0, :], self._rigid_body_rot[env_ids, 0, :]
        future = compute_tracking_future(root_pos_dyn, root_rot_dyn,
                                         root_pos_kin, root_rot_kin, root_vel_kin, root_ang_vel_kin,
                                         dof_pos_kin, dof_vel_kin, key_pos_kin,
                                         self._dof_obs_size, self._dof_offsets, future_frame)
        self.future_buf[env_ids] = future.view(future.shape[0] // future_frame, future_frame * future.shape[1])

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._update_tracking_future(env_ids)
        obs = OrderedDict({
            'prop': self.obs_buf.to(self.device),
            'future': self.future_buf.to(self.device),
        })
        return obs

    def step(self, actions):
        super().step(actions)
        self._update_tracking_future()
        obs = OrderedDict({
            'prop': self.obs_buf.to(self.device),
            'future': self.future_buf.to(self.device),
        })
        return obs, self.rew_buf.to(self.device), self.reset_buf.to(self.device), self.extras

    def _compute_reward(self):
        self.rew_buf[:] = self._compute_humanoid_tracking_reward()
        self._rewards_sum[:] += self.rew_buf

    def _set_humanoid_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids] = torch.hstack((root_pos, root_rot, root_vel, root_ang_vel))
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

    def _motion_sync(self):
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, _, _ \
            = self._motion_lib.get_motion_state(self._motion_ids, self._time)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_kin_humanoid_state(env_ids=env_ids,
                                     root_pos=root_pos,
                                     root_rot=root_rot,
                                     dof_pos=dof_pos,
                                     root_vel=root_vel,
                                     root_ang_vel=root_ang_vel,
                                     dof_vel=dof_vel)

        env_ids_int32 = self._humanoid_actor_ids[env_ids] + 1
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _set_kin_humanoid_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        root_pos[:, 0] += 1.0
        self._kin_root_states[env_ids] = torch.hstack((root_pos, root_rot, root_vel, root_ang_vel))
        self._kin_dof_pos[env_ids] = dof_pos
        self._kin_dof_vel[env_ids] = dof_vel

    def _compute_humanoid_tracking_reward(self):
        root_pos_kin, root_rot_kin, dof_pos_kin, root_vel_kin, root_ang_vel_kin, dof_vel_kin, key_pos_kin, _ \
            = self._motion_lib.get_motion_state(self._motion_ids, self._time)

        _, local_root_vel_kin, local_root_ang_vel_kin, dof_obs_kin, flat_local_key_pos_kin = \
            compute_local_feature(root_pos_kin, root_rot_kin, root_vel_kin, root_ang_vel_kin,
                                  dof_pos_kin, key_pos_kin, self._dof_obs_size, self._dof_offsets)

        _, local_root_vel_dyn, local_root_ang_vel_dyn, dof_obs_dyn, flat_local_key_pos_dyn = \
            compute_local_feature(self._rigid_body_pos[:, 0, :],
                                  self._rigid_body_rot[:, 0, :],
                                  self._rigid_body_vel[:, 0, :],
                                  self._rigid_body_ang_vel[:, 0, :],
                                  self._dof_pos, self._rigid_body_pos[:, self._key_body_ids, :],
                                  self._dof_obs_size, self._dof_offsets)

        kin_global_rotation = self._motion_lib.get_motion_global_rotation(self._motion_ids, self._time)

        self._joint_pos_err, self._joint_vel_err, self._root_pos_err, self._root_rot_angle_err, \
            self._root_vel_err, self._root_ang_vel_err, self._key_body_err, self._key_body_rot_err_angle \
            = compute_humanoid_tracking_difference(self._rigid_body_pos[:, 0, :], self._rigid_body_rot[:, 0, :],
                                                   local_root_vel_dyn, local_root_ang_vel_dyn,
                                                   dof_obs_dyn, self._dof_vel,
                                                   flat_local_key_pos_dyn, self._rigid_body_rot[:, self._key_body_ids],

                                                   root_pos_kin, root_rot_kin,
                                                   local_root_vel_kin, local_root_ang_vel_kin,
                                                   dof_obs_kin, dof_vel_kin,
                                                   flat_local_key_pos_kin, kin_global_rotation[:, self._key_body_ids])

        reward = compute_humanoid_tracking_reward(self._joint_pos_err, self._joint_vel_err,
                                                  self._root_pos_err, self._root_rot_angle_err,
                                                  self._key_body_err, self._root_vel_err,
                                                  self._root_ang_vel_err)

        return reward

    def _compute_reset(self):
        if self._enable_early_termination:
            root_pos_threshold = self._early_termination_threshold['root_pos_threshold']
            root_rot_err_angle_threshold = self._early_termination_threshold['root_rot_err_angle_threshold']
            key_body_pos_threshold = torch.tensor(self._early_termination_threshold['key_body_pos_threshold'],
                                                  device=self.device)
            key_body_rot_threshold = torch.tensor(self._early_termination_threshold['key_body_rot_threshold'],
                                                  device=self.device)

            key_body_err_norm = torch.linalg.norm(self._key_body_err, dim=-1)

            has_big_diff = torch.logical_or(
                torch.logical_or(
                    self._root_pos_err > root_pos_threshold,
                    self._root_rot_angle_err > root_rot_err_angle_threshold
                ),
                torch.logical_or(
                    torch.any(key_body_err_norm > key_body_pos_threshold, dim=1),
                    torch.any(self._key_body_rot_err_angle > key_body_rot_threshold, dim=1))
            )
            has_big_diff = torch.logical_and(has_big_diff, self._transition_counter <= 0)
            has_big_diff *= (self.progress_buf > 1)
            has_big_diff[self._transition_counter > 0] = 0
        else:
            has_big_diff = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        time_end = self._time >= self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf[:] = torch.where(torch.logical_or(has_big_diff, time_end), torch.ones_like(self.reset_buf),
                                        torch.zeros_like(self.reset_buf))
        self._terminate_buf[:] = self.reset_buf

#####################################################################
# =========================jit functions=========================== #
#####################################################################


@torch.jit.script
def compute_humanoid_tracking_difference(root_pos_dyn: Tensor, root_rot_dyn: Tensor, local_root_vel_dyn: Tensor,
                                         local_root_ang_vel_dyn: Tensor, dof_obs_dyn: Tensor, dof_vel_dyn: Tensor,
                                         flat_local_key_pos_dyn: Tensor, key_body_global_rotation_dyn: Tensor,

                                         root_pos_kin: Tensor, root_rot_kin: Tensor, local_root_vel_kin: Tensor,
                                         local_root_ang_vel_kin: Tensor, dof_obs_kin: Tensor, dof_vel_kin: Tensor,
                                         flat_local_key_pos_kin: Tensor, key_body_global_rotation_kin: Tensor
                                         ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    joint_pos_err = torch.abs(dof_obs_kin - dof_obs_dyn)
    joint_vel_err = torch.abs(dof_vel_kin - dof_vel_dyn)

    root_pos_err = torch.linalg.norm(root_pos_kin - root_pos_dyn, dim=1) ** 2
    root_rot_err = torch_utils.quat_mul(root_rot_kin, torch_utils.quat_conjugate(root_rot_dyn))
    root_rot_angle_err, _ = torch_utils.quat_to_angle_axis(root_rot_err)

    root_vel_err = torch.sum(torch.square(local_root_vel_kin - local_root_vel_dyn), dim=1)
    root_ang_vel_err = torch.sum(torch.square(local_root_ang_vel_kin - local_root_ang_vel_dyn), dim=1)

    key_body_err = flat_local_key_pos_kin - flat_local_key_pos_dyn
    key_body_err = torch.where(key_body_err.isnan(), torch.zeros_like(key_body_err), key_body_err)
    key_body_err = key_body_err.view(key_body_err.shape[0], key_body_err.shape[1] // 3, 3)

    key_body_rot_err = torch_utils.quat_mul(key_body_global_rotation_kin,
                                            torch_utils.quat_conjugate(key_body_global_rotation_dyn))
    key_body_rot_err_angle, _ = torch_utils.quat_to_angle_axis(key_body_rot_err)

    return joint_pos_err, joint_vel_err, root_pos_err, root_rot_angle_err, \
        root_vel_err, root_ang_vel_err, key_body_err, key_body_rot_err_angle


@torch.jit.script
def compute_humanoid_tracking_reward(joint_pos_err: Tensor, joint_vel_err: Tensor, root_pos_err: Tensor,
                                     root_rot_angle_err: Tensor, key_body_err: Tensor, root_vel_err: Tensor,
                                     root_ang_vel_err: Tensor) -> Tensor:
    joint_pos_w = 0.30
    joint_vel_w = 0.1
    key_body_w = 0.30
    root_w = 0.2
    root_vel_w = 0.1

    total_w = joint_pos_w + joint_vel_w + key_body_w + root_w + root_vel_w

    joint_pos_w /= total_w
    joint_vel_w /= total_w
    key_body_w /= total_w
    root_w /= total_w
    root_vel_w /= total_w

    joint_pos_scale = 2.0
    joint_vel_scale = 0.1
    key_body_scale = 10
    root_scale = 20.0
    root_vel_scale = 2.0

    root_err = root_pos_err + 0.5 * torch.square(root_rot_angle_err)
    root_vel_err = root_vel_err + 0.1 * root_ang_vel_err
    key_body_err_norm = torch.linalg.norm(key_body_err.view(key_body_err.shape[0], -1), dim=1)

    joint_pos_reward = torch.exp(-joint_pos_scale * torch.linalg.norm(joint_pos_err, dim=1))
    joint_vel_reward = torch.exp(-joint_vel_scale * torch.linalg.norm(joint_vel_err, dim=1))
    key_body_reward = torch.exp(-key_body_scale * key_body_err_norm)
    root_reward = torch.exp(-root_scale * root_err)
    root_vel_reward = torch.exp(-root_vel_scale * root_vel_err)
    reward = joint_pos_w * joint_pos_reward + joint_vel_w * joint_vel_reward + \
        key_body_w * key_body_reward + root_w * root_reward + root_vel_w * root_vel_reward
    return reward


@torch.jit.script
def compute_tracking_future(root_pos_dyn: Tensor, root_rot_dyn: Tensor, root_pos_kin: Tensor, root_rot_kin: Tensor,
                            root_vel_kin: Tensor, root_ang_vel_kin: Tensor, dof_pos_kin: Tensor, dof_vel_kin: Tensor,
                            key_pos_kin: Tensor, dof_obs_size: int, dof_offsets: List[int], future_frame: int
                            ) -> Tensor:

    _, local_root_vel_kin, local_root_ang_vel_kin, dof_obs_kin, flat_local_key_pos_kin = \
        compute_local_feature(root_pos_kin, root_rot_kin, root_vel_kin, root_ang_vel_kin,
                              dof_pos_kin, key_pos_kin, dof_obs_size, dof_offsets)

    pos_dyn_expand = root_pos_dyn.unsqueeze(-2).repeat((1, future_frame, 1))
    flat_pos_dyn = pos_dyn_expand.view(pos_dyn_expand.shape[0] * pos_dyn_expand.shape[1], pos_dyn_expand.shape[2])
    pos_diff = root_pos_kin - flat_pos_dyn

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot_dyn)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, future_frame, 1))
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])

    local_pos_diff = torch_utils.quat_rotate(flat_heading_rot, pos_diff)

    rot_dyn_expand = root_rot_dyn.unsqueeze(-2).repeat((1, future_frame, 1))
    flat_rot_dyn_expand = rot_dyn_expand.view(rot_dyn_expand.shape[0] * rot_dyn_expand.shape[1],
                                              rot_dyn_expand.shape[2])
    root_rot_err = torch_utils.quat_mul(root_rot_kin, torch_utils.quat_conjugate(flat_rot_dyn_expand))
    root_rot_err_obs = torch_utils.quat_to_tan_norm(root_rot_err)

    flat_local_key_pos_kin = torch.where(flat_local_key_pos_kin.isnan(), torch.zeros_like(flat_local_key_pos_kin),
                                         flat_local_key_pos_kin)
    future = torch.cat(
        (local_pos_diff, root_rot_err_obs, local_root_vel_kin, local_root_ang_vel_kin, dof_obs_kin, dof_vel_kin,
         flat_local_key_pos_kin), dim=1)
    return future


@torch.jit.script
def compute_tracking_reset(reset_buf: Tensor, progress_buf: Tensor, max_steps: Tensor, has_big_diff: Tensor
                           ) -> Tuple[Tensor, Tensor]:
    terminated = torch.zeros_like(reset_buf)

    has_big_diff *= (progress_buf > 1)
    terminated = torch.where(has_big_diff, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_steps, torch.ones_like(reset_buf), terminated)

    return reset, terminated
