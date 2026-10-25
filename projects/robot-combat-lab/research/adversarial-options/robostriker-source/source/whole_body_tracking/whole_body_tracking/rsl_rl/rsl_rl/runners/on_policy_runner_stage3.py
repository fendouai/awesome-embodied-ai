# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from os.path import join as pjoin
import statistics
import time
import torch
from collections import deque
from rsl_rl.algorithms import PPOStage3
import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    EmpiricalNormalization,
    ActorCriticStage3
)
from rsl_rl.utils import store_code_state


class OnPolicyRunnerStage3:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", is_train=True):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.prior_history_length = self.policy_cfg.get("prior_history_length", 1)
        self.stage2_ckpt_path = train_cfg.get("stage2_ckpt_path", None)
        self.normalize_z = self.alg_cfg.get("normalize_z", False)
        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        self.is_train = is_train
        self.training_type = "rl"
        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs_ego = obs["robot_A"].shape[1]
        num_obs_opponent = obs["robot_B"].shape[1]# NOTE this is num task policy obs

        num_privileged_obs_ego = extras["observations"]["robot_A"]["critic"].shape[1]
        num_privileged_obs_opponent = extras["observations"]["robot_B"]["critic"].shape[1]
        num_prior_obs = extras["observations"]["robot_A"]["prior"].shape[-1]
        num_prior_obs = self.prior_history_length * num_prior_obs
        num_decoder_obs = extras["observations"]["robot_A"]["decoder"].shape[1]
        num_actions = self.env.num_actions["robot_A"]
        latent_dim = self.policy_cfg.get("latent_dim", 32)
        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCriticStage3 = policy_class(
            num_obs_ego, num_obs_opponent,
            num_privileged_obs_ego, num_privileged_obs_opponent,
            num_prior_obs, num_decoder_obs,
            num_actions, self.normalize_z, **self.policy_cfg
        ).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # Initialize AMP demo buffer if AMP is enabled.
        self.amp_demo_buffer = None
        if "amp_cfg" in self.alg_cfg and self.alg_cfg["amp_cfg"] is not None:
            amp_cfg = self.alg_cfg["amp_cfg"]
            if amp_cfg.get("enable_amp", False):
                motion_file = amp_cfg.get("motion_file")
                if motion_file:
                    print(f"[INFO] Loading AMP reference motion data from: {motion_file}")
                    self._load_amp_demo_data(motion_file, amp_cfg)
                    # Pass the demo buffer to the algorithm config.
                    amp_cfg["_amp_demo_buffer"] = self.amp_demo_buffer
                else:
                    print("[WARNING] AMP enabled but no motion_file specified!")
                
                # Get environment observations to determine AMP observation dimension.
                temp_obs, temp_extras = self.env.get_observations()
                if "amp" in temp_extras["observations"]["robot_A"]:
                    amp_obs_sample = temp_extras["observations"]["robot_A"]["amp"]  # [num_envs, history_len, obs_dim]
                    # AMP observation is a 2-step history concatenation.
                    num_amp_obs_1_step = amp_obs_sample.shape[-1]  # Single-step dimension.
                    history_len = 2
                    num_amp_obs = num_amp_obs_1_step * history_len  # Total dimension.
                    
                    # Set values in amp_cfg.
                    amp_cfg["num_amp_obs"] = num_amp_obs
                    amp_cfg["history_len"] = history_len
                    
                    # Create AMP normalizer.
                    self.amp_normalizer = EmpiricalNormalization(
                        shape=[num_amp_obs_1_step], 
                        until=1.0e8
                    ).to(self.device)
                    amp_cfg["_amp_normalizer"] = self.amp_normalizer
                    
                    print(f"[INFO] AMP observation dimension: {num_amp_obs} (history_len={history_len}, single_step={num_amp_obs_1_step})")
                else:
                    print("[WARNING] AMP enabled but 'amp' observation not found in environment!")

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPOStage3= alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )
        
        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self.load_run = self.cfg["load_run"]
        self.load_checkpoint = self.cfg["load_checkpoint"]
        if self.empirical_normalization:
            self.taskpolicy_obs_normalizer_ego = EmpiricalNormalization(shape=[num_obs_ego], until=1.0e8).to(self.device)
            self.taskpolicy_obs_normalizer_opponent = EmpiricalNormalization(shape=[num_obs_opponent], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer_ego = EmpiricalNormalization(shape=[num_privileged_obs_ego], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer_opponent = EmpiricalNormalization(shape=[num_privileged_obs_opponent], until=1.0e8).to(self.device)
            self.prior_obs_normalizer = EmpiricalNormalization(shape=[num_prior_obs], until=1.0e8).to(self.device)
            self.decoder_obs_normalizer = EmpiricalNormalization(shape=[num_decoder_obs], until=1.0e8).to(self.device)
        else:
            self.taskpolicy_obs_normalizer_ego = torch.nn.Identity().to(self.device)  # no normalization
            self.taskpolicy_obs_normalizer_opponent = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer_ego = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer_opponent = torch.nn.Identity().to(self.device)  # no normalization
            self.prior_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.decoder_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        if self.is_train:
            self.load_ckpt()
        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs_ego],
            [num_obs_opponent],
            [num_privileged_obs_ego],
            [num_privileged_obs_opponent],
            [num_prior_obs],
            [latent_dim],
        )
        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def load_ckpt(self):
        loaded_dict = torch.load(self.stage2_ckpt_path)
        model_state_dict = loaded_dict["model_state_dict"]

        # --- load decoder (Stage2 student) ---
        decoder_state_dict = {
            k.replace("student.", ""): v
            for k, v in model_state_dict.items()
            if k.startswith("student.")
        }
        if not decoder_state_dict:
            raise ValueError(
                f"Stage2 checkpoint '{self.stage2_ckpt_path}' does not contain any 'student.' parameters. "
                "Please provide a Stage2 distillation checkpoint."
            )
        self.alg.policy.decoder.load_state_dict(decoder_state_dict, strict=True)

        # --- load prior (required for Stage3) ---
        prior_state_dict = {
            k.replace("prior.", ""): v
            for k, v in model_state_dict.items()
            if k.startswith("prior.")
        }
        if not prior_state_dict:
            raise ValueError(
                f"Stage3 currently requires a Stage2 checkpoint with a learnable prior (use_prior=True). "
                f"No 'prior.' parameters were found in '{self.stage2_ckpt_path}'."
            )
        self.alg.policy.prior.load_state_dict(prior_state_dict, strict=True)

        # --- load normalizers ---
        self.decoder_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        self.prior_obs_normalizer.load_state_dict(loaded_dict["prior_obs_norm_state_dict"])
        
    def load_model(self):
        checkpoint_path = pjoin(self.load_run, self.load_checkpoint)
        loaded_dict = torch.load(checkpoint_path)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])

    def _load_amp_demo_data(self, motion_file, amp_cfg):
        """Load AMP reference motion data and store it in the demo buffer.
        
        Args:
            motion_file: Reference motion file path, either a directory or an npz file.
            amp_cfg: AMP configuration dictionary.
        """
        import os
        import numpy as np
        from rsl_rl.storage import CircularBuffer
        
        # Collect all AMP observations.
        all_amp_obs = []
        
        if os.path.isdir(motion_file):
            # If this is a directory, load all npz files.
            import glob
            npz_files = glob.glob(os.path.join(motion_file, "*.npz"))
            print(f"[INFO] Found {len(npz_files)} motion files in {motion_file}")
            
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file)
                    # Build amp_obs from motion data.
                    amp_obs = self._build_amp_obs_from_motion(data)
                    if amp_obs is not None:
                        all_amp_obs.append(amp_obs)
                        print(f"[INFO] Loaded {amp_obs.shape[0]} samples from {os.path.basename(npz_file)}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {npz_file}: {e}")
        
        elif os.path.isfile(motion_file) and motion_file.endswith('.npz'):
            # If this is a single npz file.
            try:
                data = np.load(motion_file)
                amp_obs = self._build_amp_obs_from_motion(data)
                if amp_obs is not None:
                    all_amp_obs.append(amp_obs)
                    print(f"[INFO] Loaded {amp_obs.shape[0]} samples from {motion_file}")
            except Exception as e:
                print(f"[ERROR] Failed to load {motion_file}: {e}")
        else:
            print(f"[ERROR] Motion file not found or unsupported format: {motion_file}")
            # Create an empty buffer.
            buffer_size = amp_cfg.get("amp_demo_buffer_size", 100000)
            self.amp_demo_buffer = CircularBuffer(max_length=buffer_size, device=self.device)
            return
        
        # Merge all data.
        if len(all_amp_obs) > 0:
            all_amp_obs = np.concatenate(all_amp_obs, axis=0)
            print(f"[INFO] Total AMP demo samples: {all_amp_obs.shape[0]}")
            
            # Convert to tensor and create CircularBuffer.
            amp_obs_tensor = torch.from_numpy(all_amp_obs).float().to(self.device)
            buffer_size = amp_cfg.get("amp_demo_buffer_size", len(amp_obs_tensor))
            self.amp_demo_buffer = CircularBuffer(max_length=buffer_size, device=self.device)
            
            # Store data into the buffer.
            self.amp_demo_buffer.store(amp_obs_tensor)
            print(f"[INFO] Initialized AMP demo buffer with {len(self.amp_demo_buffer)} samples")
        else:
            print(f"[WARNING] No AMP demo data loaded, creating empty buffer")
            buffer_size = amp_cfg.get("amp_demo_buffer_size", 100000)
            self.amp_demo_buffer = CircularBuffer(max_length=buffer_size, device=self.device)
    
    def _build_amp_obs_from_motion(self, data):
        """Build amp_obs from motion data, following decoder observation format: base_ang_vel + joint_pos + joint_vel.
        
        Motion data format:
        - body_ang_vel_w: [T, num_bodies, 3]  # world frame angular velocity
        - joint_pos: [T, num_joints]
        - joint_vel: [T, num_joints]
        
        Returns:
        - amp_obs: [T-1, obs_dim * history_len] where obs_dim = 3 (base_ang_vel) + num_joints + num_joints
          Uses a 2-step history, so the final dimension is obs_dim * 2 = 122.
        """
        try:
            import numpy as np
            
            # Extract data.
            body_ang_vel_w = data['body_ang_vel_w']  # [T, num_bodies, 3]
            joint_pos = data['joint_pos']  # [T, num_joints]
            joint_vel = data['joint_vel']  # [T, num_joints]
            
            # Take angular velocity of the root body, body 0.
            base_ang_vel = body_ang_vel_w[:, 0, :]  # [T, 3]
            
            # Concatenate into single-step amp_obs: base_ang_vel + joint_pos + joint_vel.
            amp_obs_single = np.concatenate([
                base_ang_vel,  # [T, 3]
                joint_pos,     # [T, num_joints]
                joint_vel,     # [T, num_joints]
            ], axis=-1)  # [T, 61]
            
            # Build 2-step history by concatenating observations at t-1 and t.
            # amp_obs[i] = [amp_obs_single[i], amp_obs_single[i+1]]
            T = amp_obs_single.shape[0]
            history_len = 2
            
            amp_obs_with_history = []
            for i in range(T - history_len + 1):
                # Concatenate history_len consecutive steps.
                history_obs = amp_obs_single[i:i+history_len].reshape(-1)  # [61*2] = [122]
                amp_obs_with_history.append(history_obs)
            
            amp_obs = np.array(amp_obs_with_history)  # [T-1, 122]
            
            return amp_obs
            
        except Exception as e:
            print(f"[ERROR] Failed to build amp_obs from motion data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        taskpolicy_obs, extras = self.env.get_observations()
        # handle obervations for both agents
        taskpolicy_obs_ego = taskpolicy_obs["robot_A"]
        taskpolicy_obs_opponent = taskpolicy_obs["robot_B"]
        privileged_obs_ego = extras["observations"]["robot_A"]["critic"]
        privileged_obs_opponent = extras["observations"]["robot_B"]["critic"]
        prior_obs_ego = extras["observations"]["robot_A"]["prior"]
        prior_obs_opponent = extras["observations"]["robot_B"]["prior"]
        decoder_obs_ego = extras["observations"]["robot_A"]["decoder"]
        decoder_obs_opponent = extras["observations"]["robot_B"]["decoder"]
        taskpolicy_obs_ego = taskpolicy_obs_ego.to(self.device)
        taskpolicy_obs_opponent = taskpolicy_obs_opponent.to(self.device)
        privileged_obs_ego = privileged_obs_ego.to(self.device)
        privileged_obs_opponent = privileged_obs_opponent.to(self.device)
        prior_obs_ego = self.prior_obs_normalizer(prior_obs_ego.reshape(self.env.num_envs, -1).to(self.device))
        prior_obs_opponent = self.prior_obs_normalizer(prior_obs_opponent.reshape(self.env.num_envs, -1).to(self.device))
        decoder_obs_ego = self.decoder_obs_normalizer(decoder_obs_ego.to(self.device))
        decoder_obs_opponent = self.decoder_obs_normalizer(decoder_obs_opponent.to(self.device))
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        
        ep_infos = []
        rewbuffer_ego = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)

        rewbuffer_opponent = deque(maxlen=100)
        cur_reward_sum_opponent = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        cur_reward_sum_ego = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()

            # === Update reward schedule (AMP) ===
            if self.alg.has_discriminator:
                self.alg.update_reward_schedule(it)
                if it % 100 == 0:  # Print every 100 iterations.
                    print(f"[Epoch {it}] Reward weights: task_w={self.alg.task_reward_w:.3f}, disc_w={self.alg.disc_reward_w:.3f}")

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(taskpolicy_obs_ego, taskpolicy_obs_opponent, 
                                           privileged_obs_ego, privileged_obs_opponent,
                                           prior_obs_ego, prior_obs_opponent,
                                           decoder_obs_ego, decoder_obs_opponent)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions)
                    # Move to device
                    taskpolicy_obs_ego = self.taskpolicy_obs_normalizer_ego(obs["robot_A"].to(self.device))
                    taskpolicy_obs_opponent = self.taskpolicy_obs_normalizer_opponent(obs["robot_B"].to(self.device))
                    privileged_obs_ego = self.privileged_obs_normalizer_ego(infos["observations"]["robot_A"]["critic"].to(self.device))
                    privileged_obs_opponent = self.privileged_obs_normalizer_opponent(infos["observations"]["robot_B"]["critic"].to(self.device))
                    prior_obs_ego = self.prior_obs_normalizer(infos["observations"]["robot_A"]["prior"].reshape(self.env.num_envs, -1).to(self.device))
                    prior_obs_opponent = self.prior_obs_normalizer(infos["observations"]["robot_B"]["prior"].reshape(self.env.num_envs, -1).to(self.device))
                    decoder_obs_ego = self.decoder_obs_normalizer(infos["observations"]["robot_A"]["decoder"].to(self.device))
                    decoder_obs_opponent = self.decoder_obs_normalizer(infos["observations"]["robot_B"]["decoder"].to(self.device))
                    
                    # === Extract AMP observations ===
                    if self.alg.has_discriminator:
                        amp_obs_ego = infos["observations"]["robot_A"]["amp"]
                        amp_obs_opponent = infos["observations"]["robot_B"]["amp"]
                        
                        # Concatenate 2-step history: current step plus previous step.
                        amp_obs_ego_processed = torch.cat([
                            amp_obs_ego[:, -1, :],  # Current step.
                            amp_obs_ego[:, -2, :]   # Previous step.
                        ], dim=-1)
                        
                        amp_obs_opponent_processed = torch.cat([
                            amp_obs_opponent[:, -1, :],
                            amp_obs_opponent[:, -2, :]
                        ], dim=-1)
                        
                        # Normalize AMP observations.
                        # amp_normalizer accepts single-step dimensions, so normalize the two steps separately.
                        with torch.no_grad():
                            num_amp_obs_1_step = amp_obs_ego.shape[-1]  # Single-step dimension.
                            history_len = 2
                            
                            # Normalize AMP observations for the ego agent.
                            amp_obs_ego_norm = amp_obs_ego_processed.clone()
                            for h in range(history_len):
                                start_idx = h * num_amp_obs_1_step
                                end_idx = (h + 1) * num_amp_obs_1_step
                                # Use amp_normalizer without updating statistics.
                                self.amp_normalizer.eval()  # Evaluation mode avoids updating statistics.
                                amp_obs_ego_norm[:, start_idx:end_idx] = self.amp_normalizer(
                                    amp_obs_ego_processed[:, start_idx:end_idx]
                                )
                                
                            # Normalize AMP observations for the opponent agent.
                            amp_obs_opponent_norm = amp_obs_opponent_processed.clone()
                            for h in range(history_len):
                                start_idx = h * num_amp_obs_1_step
                                end_idx = (h + 1) * num_amp_obs_1_step
                                self.amp_normalizer.eval()
                                amp_obs_opponent_norm[:, start_idx:end_idx] = self.amp_normalizer(
                                    amp_obs_opponent_processed[:, start_idx:end_idx]
                                )

                        # Add to infos for algorithm use.
                        infos["amp_obs_ego"] = amp_obs_ego_processed
                        infos["amp_obs_opponent"] = amp_obs_opponent_processed
                        infos["amp_obs_ego_norm"] = amp_obs_ego_norm  
                        infos["amp_obs_opponent_norm"] = amp_obs_opponent_norm
                        
                        # Compute discriminator reward and mix it.
                        disc_reward_ego = self.alg._calc_disc_reward(amp_obs_ego_norm)
                        disc_reward_opponent = self.alg._calc_disc_reward(amp_obs_opponent_norm)
                        
                        # Following the SMPLOlympics style, do not store this to the replay buffer here.
                        # Store it together after training instead.

                    rewards_ego = rewards["robot_A"].to(self.device)
                    rewards_opponent = rewards["robot_B"].to(self.device)
                    dones = dones.to(self.device)
                    # process the step

                    # === Mix task reward and discriminator reward if AMP is enabled ===
                    if self.alg.has_discriminator:
                        task_reward_ego = rewards_ego.clone()
                        task_reward_opponent = rewards_opponent.clone()
                        
                        # Mix rewards: combined = task_w * task_r + disc_w * disc_r.
                        rewards_ego = (self.alg.task_reward_w * task_reward_ego + 
                                      self.alg.disc_reward_w * disc_reward_ego.squeeze(-1))
                        rewards_opponent = (self.alg.task_reward_w * task_reward_opponent + 
                                           self.alg.disc_reward_w * disc_reward_opponent.squeeze(-1))
                        
                    self.alg.process_env_step(rewards_ego, rewards_opponent, dones, infos)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum_ego += rewards_ego
                            cur_reward_sum_opponent += rewards_opponent
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer_ego.extend(cur_reward_sum_ego[new_ids][:, 0].cpu().numpy().tolist())
                        rewbuffer_opponent.extend(cur_reward_sum_opponent[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum_ego[new_ids] = 0
                        cur_reward_sum_opponent[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs_ego, privileged_obs_opponent)

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for robot in ["robot_A", "robot_B"]:
                
                for key in locs["ep_infos"][0][robot]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs["ep_infos"]:
                        # handle scalar and zero dimensional tensor infos
                        if key not in ep_info[robot]:
                            continue
                        if not isinstance(ep_info[robot][key], torch.Tensor):
                            ep_info[robot][key] = torch.Tensor([ep_info[robot][key]])
                        if len(ep_info[robot][key].shape) == 0:
                            ep_info[robot][key] = ep_info[robot][key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[robot][key].to(self.device)))
                    value = torch.mean(infotensor)
                    # log to logger and terminal
                    if "/" in key:
                        self.writer.add_scalar(robot +"_"+ key, value, locs["it"])
                        ep_string += f"""{f'{robot} {key}:':>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/"+ robot + key, value, locs["it"])
                        ep_string += f"""{f'Mean episode {robot} {key}:':>{pad}} {value:.4f}\n"""

        mean_std_ego = self.alg.policy.latent_std_ego.mean()
        mean_std_opponent = self.alg.policy.latent_std_opponent.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate_ego", self.alg.learning_rate_ego, locs["it"])
        self.writer.add_scalar("Loss/learning_rate_opponent", self.alg.learning_rate_opponent, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std_ego", mean_std_ego.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std_opponent", mean_std_opponent.item(), locs["it"])
        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer_ego"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # AMP logging
            if self.alg.has_discriminator:
                self.writer.add_scalar("AMP/task_reward_w", self.alg.task_reward_w, locs["it"])
                self.writer.add_scalar("AMP/disc_reward_w", self.alg.disc_reward_w, locs["it"])
                if "disc_loss" in locs["loss_dict"]:
                    self.writer.add_scalar("AMP/disc_loss", locs["loss_dict"]["disc_loss"], locs["it"])
                    self.writer.add_scalar("AMP/disc_agent_acc", locs["loss_dict"]["disc_agent_acc"], locs["it"])
                    self.writer.add_scalar("AMP/disc_demo_acc", locs["loss_dict"]["disc_demo_acc"], locs["it"])
                    self.writer.add_scalar("AMP/disc_grad_penalty", locs["loss_dict"]["disc_grad_penalty"], locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward_ego", statistics.mean(locs["rewbuffer_ego"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward_opponent", statistics.mean(locs["rewbuffer_opponent"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward_ego/time", statistics.mean(locs["rewbuffer_ego"]), self.tot_time)
                self.writer.add_scalar("Train/mean_reward_opponent/time", statistics.mean(locs["rewbuffer_opponent"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer_ego"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std ego:':>{pad}} {mean_std_ego.item():.2f}\n"""
                f"""{'Mean action noise std opponent:':>{pad}} {mean_std_opponent.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward ego:':>{pad}} {statistics.mean(locs['rewbuffer_ego']):.2f}\n"""
            log_string += f"""{'Mean reward opponent:':>{pad}} {statistics.mean(locs['rewbuffer_opponent']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std ego:':>{pad}} {mean_std_ego.item():.2f}\n"""
                f"""{'Mean action noise std opponent:':>{pad}} {mean_std_opponent.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_ego_state_dict": self.alg.optimizer_ego.state_dict(),
            "optimizer_opponent_state_dict": self.alg.optimizer_opponent.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["taskpolicy_obs_normalizer_ego_state_dict"] = self.taskpolicy_obs_normalizer_ego.state_dict()
            saved_dict["taskpolicy_obs_normalizer_opponent_state_dict"] = self.taskpolicy_obs_normalizer_opponent.state_dict()
            
            saved_dict["privileged_obs_normalizer_ego_state_dict"] = self.privileged_obs_normalizer_ego.state_dict()
            saved_dict["privileged_obs_normalizer_opponent_state_dict"] = self.privileged_obs_normalizer_opponent.state_dict()
            saved_dict["prior_obs_normalizer_state_dict"] = self.prior_obs_normalizer.state_dict()
            saved_dict["decoder_obs_normalizer_state_dict"] = self.decoder_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        # Extract taskpolicy_ego weights and load them into both taskpolicy_ego and taskpolicy_opponent.
        model_state_dict = loaded_dict["model_state_dict"]
        
        resumed_training = self.alg.policy.load_state_dict(model_state_dict)
        # # Create a new state_dict and copy taskpolicy_ego weights to taskpolicy_opponent.
        # modified_state_dict = model_state_dict.copy()
        
        # # Extract all taskpolicy_ego weights.
        # taskpolicy_ego_keys = [k for k in model_state_dict.keys() if k.startswith("taskpolicy_ego.")]
        
        # # Copy taskpolicy_ego weights to taskpolicy_opponent.
        # for ego_key in taskpolicy_ego_keys:
        #     opponent_key = ego_key.replace("taskpolicy_ego.", "taskpolicy_opponent.", 1)
        #     if opponent_key in modified_state_dict:
        #         modified_state_dict[opponent_key] = model_state_dict[ego_key].clone()
        #         print(f"[INFO] Loaded taskpolicy_ego weights to taskpolicy_opponent: {ego_key} -> {opponent_key}")
        #     else:
        #         print(f"[WARNING] Key {opponent_key} not found in model state dict, skipping...")
        
        # # Load the model using the modified state_dict.
        # resumed_training = self.alg.policy.load_state_dict(modified_state_dict)
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.taskpolicy_obs_normalizer_ego.load_state_dict(loaded_dict["taskpolicy_obs_normalizer_ego_state_dict"])
                self.taskpolicy_obs_normalizer_opponent.load_state_dict(loaded_dict["taskpolicy_obs_normalizer_opponent_state_dict"])
                self.privileged_obs_normalizer_ego.load_state_dict(loaded_dict["privileged_obs_normalizer_ego_state_dict"])
                self.privileged_obs_normalizer_opponent.load_state_dict(loaded_dict["privileged_obs_normalizer_opponent_state_dict"])
                self.prior_obs_normalizer.load_state_dict(loaded_dict["prior_obs_normalizer_state_dict"])
                self.decoder_obs_normalizer.load_state_dict(loaded_dict["decoder_obs_normalizer_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                pass # Identity
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.taskpolicy_obs_normalizer_ego.to(device)
                self.taskpolicy_obs_normalizer_opponent.to(device)
                self.privileged_obs_normalizer_ego.to(device)
                self.privileged_obs_normalizer_opponent.to(device)
                self.prior_obs_normalizer.to(device)
                self.decoder_obs_normalizer.to(device)
            policy = lambda extras: self.alg.policy.act_inference(self.taskpolicy_obs_normalizer_ego(extras["observations"]["robot_A"]["task_policy"]),
                                                                  self.taskpolicy_obs_normalizer_opponent(extras["observations"]["robot_B"]["task_policy"]),
                                                                  self.prior_obs_normalizer(extras["observations"]["robot_A"]["prior"].reshape(self.env.num_envs, -1)),
                                                                  self.prior_obs_normalizer(extras["observations"]["robot_B"]["prior"].reshape(self.env.num_envs, -1)),
                                                                  self.decoder_obs_normalizer(extras["observations"]["robot_A"]["decoder"]),
                                                                  self.decoder_obs_normalizer(extras["observations"]["robot_B"]["decoder"]))  # noqa: E731
        return policy

    def train_mode(self):
        self.freeze_network(self.alg.policy.decoder)
        self.freeze_network(self.alg.policy.prior)
        self.activate_network(self.alg.policy.taskpolicy_ego)
        self.activate_network(self.alg.policy.taskpolicy_opponent)
        self.activate_network(self.alg.policy.critic_ego)
        self.activate_network(self.alg.policy.critic_opponent)

        if self.empirical_normalization:
            self.privileged_obs_normalizer_ego.train()
            self.privileged_obs_normalizer_opponent.train()
            self.taskpolicy_obs_normalizer_ego.train()
            self.taskpolicy_obs_normalizer_opponent.train()
            self.prior_obs_normalizer.eval()
            self.decoder_obs_normalizer.eval()
            self.prior_obs_normalizer.training = False
            self.decoder_obs_normalizer.training = False

    
    def freeze_network(self, net):
        for param in net.parameters():
            param.requires_grad = False
    
    def activate_network(self, net):
        for param in net.parameters():
            param.requires_grad = True
            
    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.taskpolicy_obs_normalizer_ego.eval()
            self.taskpolicy_obs_normalizer_opponent.eval()
            self.privileged_obs_normalizer_ego.eval()
            self.privileged_obs_normalizer_opponent.eval()
            self.prior_obs_normalizer.eval()
            self.decoder_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
