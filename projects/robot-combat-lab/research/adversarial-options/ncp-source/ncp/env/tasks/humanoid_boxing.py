# Copyright 2024 Tencent. See LICENSE.txt for licensing details.

import os
from collections import OrderedDict
from typing import Tuple, List

import torch
from torch import Tensor
import numpy as np

from gym import spaces
from gym.spaces import Tuple as GymTuple
from isaacgym import gymapi, gymtorch

from ncp.env.tasks.humanoid import Humanoid, dof_to_obs
from ncp.env.tasks.humanoid import compute_humanoid_observations_max
from ncp.utils import torch_utils

OPPONENT_ID = 1
BOXING_RING_ID = 2


class HumanoidBoxing(Humanoid):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.use_boxing_ring = cfg["env"].get('useBoxingRing', False)
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self._build_opponent_tensors()
        self.asset_file = self.cfg["env"]["asset"]["assetFileName"]
        self._num_drill = 289

        self.enable_render = not headless
        self.self_prop_buf = self.obs_buf
        self.oppo_prop_buf = torch.zeros_like(self.obs_buf)
        self.self_drill_buf = torch.zeros(
            (self.num_envs, self._num_drill), device=self.device, dtype=torch.float)
        self.oppo_drill_buf = torch.zeros(
            (self.num_envs, self._num_drill), device=self.device, dtype=torch.float)
        dict_obs_space = self._build_obs_space()

        self.observation_space = GymTuple([spaces.Dict(dict_obs_space), spaces.Dict(dict_obs_space)])
        action_space = spaces.Box(np.ones(self._num_actions) * -1., np.ones(self._num_actions) * 1.)

        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self.action_space = GymTuple([action_space, action_space])
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        foot_name = self.cfg["env"]["footName"]
        self._foot_ids = self._build_contact_body_ids_tensor(foot_name)
        hand_name = self.cfg["env"]["handName"]
        self._hand_ids = self._build_contact_body_ids_tensor(hand_name)

        hit_target_name = self.cfg["env"]["hitTargetName"]
        self._hit_target_ids = self._build_contact_body_ids_tensor(hit_target_name)

        body_name = self.gym.get_actor_rigid_body_names(self.envs[0], self.humanoid_handles[0])
        body_reward_weight_dict = self.cfg["env"]["bodyRewardWeight"]
        body_reward_weight = [body_reward_weight_dict[name] for name in body_name]
        self._body_reward_weight = torch.tensor(body_reward_weight, device=self.device, dtype=torch.float)

        self._oppo_terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._self_contact_force_norm = torch.zeros([self.num_envs, len(body_name)], device=self.device,
                                                    dtype=torch.float)
        self._oppo_contact_force_norm = torch.zeros([self.num_envs, len(body_name)], device=self.device,
                                                    dtype=torch.float)
        self._reward_weight = self.cfg["env"]["rewardWeight"]
        self._total_step_counter = 0
        self._outcome = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._terminated_dist = self.cfg["env"].get('terminated_dist', 1.2)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._opponent_handle = []
        self._load_opponent()
        if self.use_boxing_ring:
            self._boxing_ring_handles = []
            self._load_boxing_ring_asset()
        super()._create_envs(num_envs, spacing, num_per_row)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _build_obs_space(self):
        dict_obs_space = OrderedDict({
            'prop': spaces.Box(0, 0, shape=(self._num_obs,)),
            'drill': spaces.Box(0, 0, shape=(self._num_drill,)),
        })
        return dict_obs_space

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        humanoid_rigid_shape_properties = self.gym.get_asset_rigid_shape_properties(humanoid_asset)
        humanoid_rigid_body_shape_indices = self.gym.get_asset_rigid_body_shape_indices(humanoid_asset)
        for _, index_range in enumerate(humanoid_rigid_body_shape_indices):
            # body_name = self.gym.get_asset_rigid_body_name(humanoid_asset, body_id)
            start, end = index_range.start, index_range.start + index_range.count
            shape_filter = 1
            for rigid_id in range(start, end):
                humanoid_rigid_shape_properties[rigid_id].filter = shape_filter
        self.gym.set_asset_rigid_shape_properties(humanoid_asset, humanoid_rigid_shape_properties)
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_opponent(env_id, env_ptr)
        if self.use_boxing_ring:
            self._build_boxing_ring(env_id, env_ptr)

    def _load_opponent(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._opponent_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        opponent_rigid_shape_properties = self.gym.get_asset_rigid_shape_properties(self._opponent_asset)
        opponent_rigid_body_shape_indices = self.gym.get_asset_rigid_body_shape_indices(self._opponent_asset)
        for _, index_range in enumerate(opponent_rigid_body_shape_indices):
            # body_name = self.gym.get_asset_rigid_body_name(self._opponent_asset, body_id)
            start, end = index_range.start, index_range.start + index_range.count
            shape_filter = 2
            for rigid_id in range(start, end):
                opponent_rigid_shape_properties[rigid_id].filter = shape_filter
        self.gym.set_asset_rigid_shape_properties(self._opponent_asset, opponent_rigid_shape_properties)
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(self._opponent_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(self._opponent_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(self._opponent_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(self._opponent_asset, left_foot_idx, sensor_pose)

    def _load_boxing_ring_asset(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = "mjcf/boxing_ring.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._boxing_ring_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _build_opponent(self, env_id, env_ptr):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*torch_utils.get_axis_params(char_h, self.up_axis_idx))
        start_pose.p.x = 1.2
        start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        humanoid_handle = self.gym.create_actor(env_ptr, self._opponent_asset, start_pose, "opponent", col_group,
                                                col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(0.78, 0.19, 0.19))

        if self._pd_control:
            dof_prop = self.gym.get_asset_dof_properties(self._opponent_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self._opponent_handle.append(humanoid_handle)

    def _build_boxing_ring(self, env_id, env_ptr):
        col_group = env_id
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.z = -1.0 - 0.02
        boxing_ring_handle = self.gym.create_actor(env_ptr, self._boxing_ring_asset, default_pose, "boxing_ring",
                                                   col_group, -1, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, boxing_ring_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0))
        self._boxing_ring_handles.append(boxing_ring_handle)

    def _build_opponent_tensors(self):
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        num_actors = self.get_num_actors_per_env()

        self._opponent_root_states = self._root_states.view(self.num_envs,
                                                            num_actors,
                                                            self._root_states.shape[-1])[..., OPPONENT_ID, :]
        self._initial_opponent_root_states = self._opponent_root_states.clone()
        self._initial_opponent_root_states[:, 7:13] = 0

        self._all_actor_ids = torch.arange(self.num_envs * num_actors,
                                           device=self.device,
                                           dtype=torch.int32).view(self.num_envs, num_actors)
        self._all_humanoid_ids = self._all_actor_ids[:, :OPPONENT_ID + 1].clone()
        num_humanoid = 2
        dofs_per_env = self._dof_state.shape[0] // self.num_envs // num_humanoid
        self._opponent_dof_pos = self._dof_state.view(self.num_envs,
                                                      num_humanoid,
                                                      dofs_per_env, 2)[:, OPPONENT_ID, :self.num_dof, 0]
        self._opponent_dof_vel = self._dof_state.view(self.num_envs,
                                                      num_humanoid,
                                                      dofs_per_env, 2)[:, OPPONENT_ID, :self.num_dof, 1]

        self._initial_opponent_root_states[:, :7] = torch.tensor([0, 2, 0.8739, -0.0202, 0.0282, -0.0027, 0.9994],
                                                                 device=self.device)
        self._initial_opponent_dof_pos = torch.zeros_like(self._opponent_dof_pos, device=self.device, dtype=torch.float)
        self._initial_opponent_dof_vel = torch.zeros_like(self._opponent_dof_pos, device=self.device, dtype=torch.float)

        self._initial_opponent_dof_pos[:] = torch.tensor(
            [0.0962, -0.0388, -0.0436, -0.0695, -0.1147, -0.2051, -0.2319, -0.6831,
             0.1655, -2.0966, 0.2319, -0.6831, -0.1655, -2.0966, 0.1199, -0.2866,
             0.1201, 0.6217, -0.0489, -0.3315, -0.2917, -0.0191, -0.2375, -0.1433,
             0.5284, 0.0246, -0.3087, 0.1618],
            device=self.device)
        self._initial_opponent_dof_vel[:] = 0.0

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._opponent_rigid_body_pos = rigid_body_state_reshaped[..., self.num_bodies: 2 * self.num_bodies, 0:3]
        self._opponent_rigid_body_rot = rigid_body_state_reshaped[..., self.num_bodies:2 * self.num_bodies:, 3:7]
        self._opponent_rigid_body_vel = rigid_body_state_reshaped[..., self.num_bodies:2 * self.num_bodies:, 7:10]
        self._opponent_rigid_body_ang_vel = rigid_body_state_reshaped[..., self.num_bodies:2 * self.num_bodies:, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._opponent_contact_forces = contact_force_tensor.view(self.num_envs,
                                                                  bodies_per_env,
                                                                  3)[..., self.num_bodies:2 * self.num_bodies, :]

    def _reset_actors(self, env_ids):
        num_envs = len(env_ids)
        from_random_probs = torch_utils.to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        from_random_mask = torch.bernoulli(from_random_probs) == 1.0
        from_random_ids = env_ids[from_random_mask]
        if len(from_random_ids) > 0:
            n = len(from_random_ids)
            self_root_states = self._initial_humanoid_root_states[from_random_ids].clone()
            self_root_states[:, 0:2] = torch.randint(-2, 3, (n, 2))

            opponent_root_states = self._initial_opponent_root_states[from_random_ids].clone()
            opponent_root_states[:, 0:2] = torch.randint(-2, 3, (n, 2))

            rand_rot_theta = 2 * np.pi * torch.rand([2 * n], dtype=opponent_root_states.dtype, device=self.device)
            axis = torch.tensor([0.0, 0.0, 1.0], dtype=opponent_root_states.dtype, device=self.device)
            rand_rot = torch_utils.quat_from_angle_axis(rand_rot_theta, axis)

            self_rot = self_root_states[:, 3:7]
            oppo_rot = opponent_root_states[:, 3:7]

            self_root_states[:, 3:7] = \
                torch_utils.quat_mul(rand_rot[:n],
                                     torch_utils.quat_mul(torch_utils.calc_heading_quat_inv(self_rot), self_rot))

            opponent_root_states[:, 3:7] = \
                torch_utils.quat_mul(rand_rot[n:],
                                     torch_utils.quat_mul(torch_utils.calc_heading_quat_inv(oppo_rot), oppo_rot))

            self._humanoid_root_states[from_random_ids] = self_root_states
            self._dof_pos[from_random_ids] = self._initial_dof_pos[from_random_ids]
            self._dof_vel[from_random_ids] = self._initial_dof_vel[from_random_ids]

            self._opponent_root_states[from_random_ids] = opponent_root_states
            self._opponent_dof_pos[from_random_ids] = self._initial_opponent_dof_pos[from_random_ids]
            self._opponent_dof_vel[from_random_ids] = self._initial_opponent_dof_vel[from_random_ids]

        from_init_mask = torch.logical_not(from_random_mask)
        from_init_ids = env_ids[from_init_mask]
        if len(from_init_ids) > 0:
            self_root_states = self._initial_humanoid_root_states[from_init_ids].clone()
            distance = torch.rand(from_init_ids.shape[0], device=self.device) * 0.3 + 0.9
            self_root_states[:, 0] = -distance / 2
            self_root_states[:, 1] = 0.0

            opponent_root_states = self._initial_opponent_root_states[from_init_ids].clone()
            opponent_root_states[:, 0] = distance / 2
            opponent_root_states[:, 1] = 0.0

            self_rot = self_root_states[:, 3:7]
            oppo_rot = opponent_root_states[:, 3:7]
            self_init_rot = torch.zeros_like(self_rot)
            self_init_rot[:, 3] = 1.0

            oppo_init_rot = torch.zeros_like(oppo_rot)
            oppo_init_rot[:, 2] = 1.0

            self_root_states[:, 3:7] = torch_utils.quat_mul(
                self_init_rot, torch_utils.quat_mul(torch_utils.calc_heading_quat_inv(self_rot), self_rot))

            opponent_root_states[:, 3:7] = torch_utils.quat_mul(
                oppo_init_rot, torch_utils.quat_mul(torch_utils.calc_heading_quat_inv(oppo_rot), oppo_rot))

            self._humanoid_root_states[from_init_ids] = self_root_states
            self._dof_pos[from_init_ids] = self._initial_dof_pos[from_init_ids]
            self._dof_vel[from_init_ids] = self._initial_dof_vel[from_init_ids]

            self._opponent_root_states[from_init_ids] = opponent_root_states
            self._opponent_dof_pos[from_init_ids] = self._initial_opponent_dof_pos[from_init_ids]
            self._opponent_dof_vel[from_init_ids] = self._initial_opponent_dof_vel[from_init_ids]

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._all_actor_ids[env_ids].view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        env_id_int32_humanoids = self._all_humanoid_ids[env_ids].view(-1)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_id_int32_humanoids),
                                              len(env_id_int32_humanoids))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._oppo_terminate_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        actions = actions + torch.hstack((self._dof_pos, self._opponent_dof_pos)).view(actions.shape)
        pd_tar = actions
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        pd_tar_tensor,
                                                        gymtorch.unwrap_tensor(self._all_humanoid_ids.view(-1)),
                                                        len(self._all_humanoid_ids.view(-1)))
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]

    def post_physics_step(self):
        super().post_physics_step()
        if torch.any(self.reset_buf):
            outcome = torch.zeros_like(self._outcome) + 1.0 * (self._outcome > 0.0) - 1.0 * (self._outcome < 0.0)
            outcome[self._terminate_buf > 0] = -1.0
            outcome[self._oppo_terminate_buf > 0] = 1.0
            outcome[torch.logical_and(self._terminate_buf > 0, self._oppo_terminate_buf > 0)] = 0.0
            self.extras["outcome"] = outcome

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._update_drill()
        self_obs = OrderedDict({
            'prop': self.self_prop_buf.to(self.device),
            'drill': self.self_drill_buf.to(self.device),
        })
        oppo_obs = OrderedDict({
            'prop': self.oppo_prop_buf.to(self.device),
            'drill': self.oppo_drill_buf.to(self.device),
        })
        obs = (self_obs, oppo_obs)
        self._outcome[env_ids] = 0.0
        return obs

    def step(self, actions):
        actions_concat = torch.hstack(actions).view(self.num_envs, 2, self._num_actions)
        super().step(actions_concat)
        self._update_drill()
        self_obs = OrderedDict({
            'prop': self.self_prop_buf.to(self.device),
            'drill': self.self_drill_buf.to(self.device),
        })
        oppo_obs = OrderedDict({
            'prop': self.oppo_prop_buf.to(self.device),
            'drill': self.oppo_drill_buf.to(self.device),
        })
        obs = (self_obs, oppo_obs)
        self._total_step_counter += 1
        if self.enable_render:
            self._draw_collision_color()
        return obs, self.rew_buf.to(self.device), self.reset_buf.to(self.device), self.extras

    def _draw_collision_color(self):
        self_contact_body = torch.any(torch.abs(self._contact_forces) > 10.0, dim=-1)[0]
        oppo_contact_body = torch.any(torch.abs(self._opponent_contact_forces) > 10.0, dim=-1)[0]

        for i in range(self.num_bodies):
            if self_contact_body[i]:
                self.gym.set_rigid_body_color(self.envs[0], self.humanoid_handles[0], i, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(1, 1, 1))
            else:
                self.gym.set_rigid_body_color(self.envs[0], self.humanoid_handles[0], i, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(0.54, 0.85, 0.2))
            if oppo_contact_body[i]:
                self.gym.set_rigid_body_color(self.envs[0], self._opponent_handle[0], i, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(1, 1, 1))
            else:
                self.gym.set_rigid_body_color(self.envs[0], self._opponent_handle[0], i, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(0.78, 0.19, 0.19))

    def _update_drill(self):
        self.self_drill_buf[:] = compute_fighting_observations(self._rigid_body_pos,
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._dof_obs_size, self._dof_offsets,

                                                               self._opponent_rigid_body_pos,
                                                               self._opponent_rigid_body_rot,
                                                               self._opponent_rigid_body_vel,
                                                               self._opponent_rigid_body_ang_vel,
                                                               self._opponent_dof_pos,
                                                               self._opponent_dof_vel,

                                                               self._self_contact_force_norm,
                                                               self._oppo_contact_force_norm,

                                                               self._hand_ids, self._hit_target_ids
                                                               )

        self.oppo_drill_buf[:] = compute_fighting_observations(self._opponent_rigid_body_pos,
                                                               self._opponent_rigid_body_rot[:, 0, :],
                                                               self._dof_obs_size, self._dof_offsets,

                                                               self._rigid_body_pos,
                                                               self._rigid_body_rot,
                                                               self._rigid_body_vel,
                                                               self._rigid_body_ang_vel,
                                                               self._dof_pos,
                                                               self._dof_vel,

                                                               self._oppo_contact_force_norm,
                                                               self._self_contact_force_norm,

                                                               self._hand_ids, self._hit_target_ids
                                                               )

    def _compute_observations(self, env_ids=None):
        super()._compute_observations(env_ids=env_ids)
        obs = self._compute_opponent_obs(env_ids)
        if env_ids is None:
            self.oppo_prop_buf[:] = obs
        else:
            self.oppo_prop_buf[env_ids] = obs

    def _compute_opponent_obs(self, env_ids=None):
        if env_ids is None:
            body_pos = self._opponent_rigid_body_pos
            body_rot = self._opponent_rigid_body_rot
            body_vel = self._opponent_rigid_body_vel
            body_ang_vel = self._opponent_rigid_body_ang_vel
            dof_pos = self._opponent_dof_pos
            dof_vel = self._opponent_dof_vel
        else:
            body_pos = self._opponent_rigid_body_pos[env_ids]
            body_rot = self._opponent_rigid_body_rot[env_ids]
            body_vel = self._opponent_rigid_body_vel[env_ids]
            body_ang_vel = self._opponent_rigid_body_ang_vel[env_ids]
            dof_pos = self._opponent_dof_pos[env_ids]
            dof_vel = self._opponent_dof_vel[env_ids]
        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, dof_pos, dof_vel,
                                                self._root_height_obs, self._dof_obs_size, self._dof_offsets)
        return obs

    def _compute_reward(self):
        self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                        self._foot_ids, self._rigid_body_pos,
                                                        self._enable_early_termination, self._termination_heights)

        self._oppo_terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                             self._foot_ids, self._opponent_rigid_body_pos,
                                                             self._enable_early_termination, self._termination_heights)

        oppo_pos = self._opponent_root_states[..., 0:3]
        char_root_state = self._humanoid_root_states

        self._self_contact_force_norm = torch.linalg.norm(self._contact_forces, dim=-1)
        self._oppo_contact_force_norm = torch.linalg.norm(self._opponent_contact_forces, dim=-1)

        lower_clipped_force = 200.0
        upper_clipped_force = 1200
        clipped_self_force_norm = self._self_contact_force_norm.clone()
        clipped_self_force_norm[clipped_self_force_norm < lower_clipped_force] = 0.0
        clipped_self_force_norm[clipped_self_force_norm > upper_clipped_force] = upper_clipped_force

        clipped_oppo_force_norm = self._oppo_contact_force_norm.clone()
        clipped_oppo_force_norm[clipped_oppo_force_norm < lower_clipped_force] = 0.0
        clipped_oppo_force_norm[clipped_oppo_force_norm > upper_clipped_force] = upper_clipped_force

        facing_reward_w = self._reward_weight['facing_reward_weight']
        vel_reward_w = self._reward_weight['vel_reward_weight']
        strike_reward_w = self._reward_weight['strike_reward_weight']
        terminate_reward_w = self._reward_weight['terminate_reward_weight']
        strike_reward, vel_reward, facing_reward, terminate_reward = compute_fighting_reward(
            oppo_pos, char_root_state, self._prev_root_pos, self.dt, self._terminate_buf, self._oppo_terminate_buf,
            clipped_self_force_norm, clipped_oppo_force_norm, self._body_reward_weight)
        self.rew_buf[:] = strike_reward_w * strike_reward + vel_reward_w * vel_reward + \
            facing_reward_w * facing_reward + terminate_reward_w * terminate_reward
        clipped_self_force_norm[clipped_self_force_norm < 200] = 0.0
        clipped_self_force_norm[clipped_self_force_norm > 1000] = 1000.0

        clipped_oppo_force_norm[clipped_oppo_force_norm < 200] = 0.0
        clipped_oppo_force_norm[clipped_oppo_force_norm > 1000] = 1000.0

        step_outcome = torch.sum((clipped_oppo_force_norm - clipped_self_force_norm) * self._body_reward_weight, dim=-1)
        self._outcome += step_outcome

    def _compute_reset(self):
        self.reset_buf[:] = torch.where(self.progress_buf > self.max_episode_length - 1,
                                        torch.ones_like(self.reset_buf),
                                        torch.logical_or(self._terminate_buf, self._oppo_terminate_buf) * 1)


#####################################################################
#  =========================jit functions=========================  #
#####################################################################

@torch.jit.script
def compute_fighting_observations(self_body_pos: Tensor, self_root_rot: Tensor, dof_obs_size: int,
                                  dof_offsets: List[int], oppo_body_pos: Tensor, oppo_body_rot: Tensor,
                                  oppo_body_vel: Tensor, oppo_body_ang_vel: Tensor, oppo_dof_pos: Tensor,
                                  oppo_dof_vel: Tensor, self_contact_norm: Tensor, oppo_contact_norm: Tensor,
                                  hand_ids: Tensor, target_ids: Tensor) -> Tensor:
    self_root_pos = self_body_pos[:, 0, :]

    oppo_root_pos = oppo_body_pos[:, 0, :]
    oppo_root_rot = oppo_body_rot[:, 0, :]
    oppo_root_vel = oppo_body_vel[:, 0, :]
    oppo_root_ang_vel = oppo_body_ang_vel[:, 0, :]

    heading_rot = torch_utils.calc_heading_quat_inv(self_root_rot)

    root_pos_diff = oppo_root_pos - self_root_pos
    root_pos_diff[..., -1] = oppo_root_pos[..., -1]
    local_root_pos_diff = torch_utils.quat_rotate(heading_rot, root_pos_diff)

    local_tar_rot = torch_utils.quat_mul(heading_rot, oppo_root_rot)
    local_root_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    local_oppo_vel = torch_utils.quat_rotate(heading_rot, oppo_root_vel)
    local_oppo_ang_vel = torch_utils.quat_rotate(heading_rot, oppo_root_ang_vel)

    oppo_dof_obs = dof_to_obs(oppo_dof_pos, dof_obs_size, dof_offsets)

    other_pos_diff = oppo_body_pos[:, 1:, :] - self_root_pos.unsqueeze(-2).repeat((1,
                                                                                   oppo_body_pos[:, 1:, :].shape[1], 1))
    other_pos_diff[..., -1] = oppo_body_pos[:, 1:, -1]
    flat_other_pos_diff = other_pos_diff.view(other_pos_diff.shape[0] * other_pos_diff.shape[1],
                                              other_pos_diff.shape[2])
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, other_pos_diff.shape[1], 1))

    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_other_pos_diff = torch_utils.quat_rotate(flat_heading_rot, flat_other_pos_diff)
    flat_local_other_pos_diff = local_other_pos_diff.view(other_pos_diff.shape[0],
                                                          other_pos_diff.shape[1] * other_pos_diff.shape[2])
    other_oppo_rot = oppo_body_rot[:, 1:, :].clone()
    flat_other_oppo_rot = other_oppo_rot.view(other_oppo_rot.shape[0] * other_oppo_rot.shape[1],
                                              other_oppo_rot.shape[2])
    local_other_rot = torch_utils.quat_mul(flat_heading_rot, flat_other_oppo_rot)
    local_other_rot_obs = torch_utils.quat_to_tan_norm(local_other_rot)
    flat_local_other_rot = local_other_rot_obs.reshape(other_oppo_rot.shape[0], other_oppo_rot.shape[1] * 6)

    self_hand_pos = self_body_pos[:, hand_ids, :].unsqueeze(-2)
    oppo_target_pos = oppo_body_pos[:, target_ids, :].unsqueeze(-3).repeat((1, self_hand_pos.shape[1], 1, 1))
    global_target_hand_pos_diff = (oppo_target_pos - self_hand_pos).view(-1, 3)
    flat_heading_rot = heading_rot.unsqueeze(-2).\
        repeat((1, global_target_hand_pos_diff.shape[0] // heading_rot.shape[0], 1))
    local_target_hand_pos_diff = torch_utils.quat_rotate(flat_heading_rot.view(-1, 4), global_target_hand_pos_diff)
    local_target_hand_pos_diff = local_target_hand_pos_diff.view(flat_heading_rot.shape[0], -1)

    obs = torch.cat([local_root_pos_diff, local_root_rot_obs, local_oppo_vel, local_oppo_ang_vel,
                     oppo_dof_obs, oppo_dof_vel, flat_local_other_pos_diff, flat_local_other_rot,
                     local_target_hand_pos_diff,

                     self_contact_norm, oppo_contact_norm], dim=-1)
    return obs


@torch.jit.script
def compute_fighting_reward(tar_pos: Tensor, root_state: Tensor, prev_root_pos: Tensor, dt: float,
                            self_terminated: Tensor, oppo_terminated: Tensor,
                            clipped_self_force_norm: Tensor, clipped_oppo_force_norm: Tensor,
                            body_reward_weight: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    tar_speed = 1.0
    vel_err_scale = 4.0
    facing_err_scale = 5.0

    root_pos = root_state[..., 0:3]
    root_rot = root_state[..., 3:7]
    tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    tar_dir_3d = tar_pos[..., 0:3] - root_pos[..., 0:3]
    tar_dir_3d[..., -1] = 0.0
    tar_dir_3d = torch.nn.functional.normalize(tar_dir_3d, dim=-1)
    tar_dir_local = torch_utils.quat_rotate(heading_rot, tar_dir_3d)
    facing_reward = torch.exp(-facing_err_scale * (1 - tar_dir_local[..., 0]))

    strike_reward = torch.sum((clipped_oppo_force_norm - clipped_self_force_norm) * body_reward_weight, dim=-1) / 30

    terminate_reward = oppo_terminated - self_terminated

    return strike_reward, vel_reward, facing_reward, terminate_reward


@torch.jit.script
def compute_humanoid_reset(reset_buf: Tensor, progress_buf: Tensor, foot_ids: Tensor, rigid_body_pos: Tensor,
                           enable_early_termination: bool, termination_heights: Tensor) -> Tensor:
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, foot_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        has_fallen = fall_height
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    return terminated
