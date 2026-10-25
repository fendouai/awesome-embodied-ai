# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCriticStage3
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorageStage3
from rsl_rl.utils import string_to_callable

try:
    from rsl_rl.storage import CircularBuffer
except ImportError:
    CircularBuffer = None  # Will be handled in init_storage


class PPOStage3:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCriticStage3
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        kl_constraint=False,
        kl_constraint_coef=0.001,
        normalize_z=False,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Get the AMP config        
        amp_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract learning rate and remove it from the original dict
            learning_rate = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer_ego = optim.Adam(
            chain(self.policy.taskpolicy_ego.parameters(),
                self.policy.critic_ego.parameters(), [self.policy.std_ego]),
            lr=learning_rate
        )
        self.optimizer_opponent = optim.Adam(
            chain(self.policy.taskpolicy_opponent.parameters(),
                self.policy.critic_opponent.parameters(), [self.policy.std_opponent]),
            lr=learning_rate
        )
        # Create rollout storage
        self.storage: RolloutStorageStage3 = None  # type: ignore
        self.transition = RolloutStorageStage3.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate_ego = learning_rate
        self.learning_rate_opponent = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.kl_constraint = kl_constraint
        self.kl_constraint_coef = kl_constraint_coef
        self.normalize_z = normalize_z

        ## AMP components
        self.amp_cfg = amp_cfg
        self.has_discriminator = amp_cfg is not None and amp_cfg.get("enable_amp", True)

        if self.has_discriminator:
            # Type assertion: ensure amp_cfg is not None and silence Pylance warnings.
            assert amp_cfg is not None, "amp_cfg should not be None when has_discriminator is True"
            
            # Initialize discriminator.
            self.discriminator = self._build_discriminator(amp_cfg)
            
            # Discriminator optimizer (all parameters with weight decay)
            self.disc_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr=amp_cfg["disc_learning_rate"],
                weight_decay=amp_cfg.get("disc_weight_decay", 0.0001)
            )
            self.disc_max_grad_norm = amp_cfg.get("disc_max_grad_norm", 50.0)
            
            # AMP replay buffer (import later in init_storage)
            self.amp_replay_buffer = None
            
            # AMP demo buffer (from runner)
            self.amp_demo_buffer = amp_cfg.get("_amp_demo_buffer", None)
            
            # Discriminator loss scales
            self.disc_coef = amp_cfg.get("disc_coef", 5.0)
            self.disc_logit_reg = amp_cfg.get("disc_logit_reg", 0.01)
            self.disc_grad_penalty = amp_cfg.get("disc_grad_penalty", 5.0)
            
            # Reward configuration: initial and target values.
            self.task_reward_w_initial = amp_cfg.get("task_reward_w_initial", 0.0)
            self.disc_reward_w_initial = amp_cfg.get("disc_reward_w_initial", 1.0)
            self.task_reward_w_final = amp_cfg.get("task_reward_w_final", 0.5)
            self.disc_reward_w_final = amp_cfg.get("disc_reward_w_final", 0.5)
            self.reward_schedule_epoch = amp_cfg.get("reward_schedule_epoch", 400)
            
            # Current reward weights.
            self.task_reward_w = self.task_reward_w_initial
            self.disc_reward_w = self.disc_reward_w_initial
            self.disc_reward_scale = amp_cfg.get("disc_reward_scale", 2)
            
            # AMP minibatch size
            self.amp_batch_size = amp_cfg.get("amp_batch_size", 512)
            
            print(f"[INFO] AMP enabled with reward scheduling:")
            print(f"  Initial: task_reward_w={self.task_reward_w_initial}, disc_reward_w={self.disc_reward_w_initial}")
            print(f"  Final:   task_reward_w={self.task_reward_w_final}, disc_reward_w={self.disc_reward_w_final}")
            print(f"  Schedule epoch: {self.reward_schedule_epoch}")
            print(f"  Discriminator coef={self.disc_coef}, grad_penalty={self.disc_grad_penalty}")
        else:
            self.discriminator = None
            self.disc_optimizer = None
            self.amp_replay_buffer = None
            self.task_reward_w = 1.0
            self.disc_reward_w = 0.0
            print("[INFO] AMP disabled, using only task rewards")
    
    # Build discriminator.
    def _build_discriminator(self, amp_cfg):
        """Build the AMP discriminator.
        
        Discriminator input: 2-step history concatenation of decoder_obs.
        """
        from rsl_rl.modules.amp_discriminator import AMPDiscriminator
        
        num_amp_obs = amp_cfg["num_amp_obs"]  
        
        # Save normalizer and observation dimension.
        self.amp_normalizer = amp_cfg["_amp_normalizer"]
        self.num_amp_obs = num_amp_obs

        print(f"[INFO] Building AMP discriminator with input dim: {num_amp_obs}")
        
        discriminator = AMPDiscriminator(
            input_dim=num_amp_obs,
            hidden_dims=amp_cfg.get("discriminator_hidden_dims", [1024, 512]),
            activation=amp_cfg.get("discriminator_activation", "relu"),
        ).to(self.device)
        
        return discriminator

    def update_reward_schedule(self, current_epoch):
        """
        Update reward weights by epoch.
        Before reward_schedule_epoch: only use disc_reward to imitate reference motions.
        After reward_schedule_epoch: use a mixture of task_reward and disc_reward.
        """
        if not self.has_discriminator:
            return
        
        if current_epoch < self.reward_schedule_epoch:
            self.task_reward_w = self.task_reward_w_initial
            self.disc_reward_w = self.disc_reward_w_initial
        else:
            self.task_reward_w = self.task_reward_w_final
            self.disc_reward_w = self.disc_reward_w_final
    
    def _calc_disc_reward(self, amp_obs_norm):
        """Compute discriminator reward."""
        with torch.no_grad():
            disc_logits = self.discriminator(amp_obs_norm)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self.disc_reward_scale
        return disc_r
    
    def _disc_loss(self, disc_agent_logit, disc_agent_replay_logit, disc_demo_logit, amp_obs_demo):
        """
        Compute discriminator loss, following the SMPLOlympics BCE style.
        disc_agent_logit: discriminator output for AMP observations generated by the current policy.
        disc_agent_replay_logit: discriminator output for AMP observations from the replay buffer.
        disc_demo_logit: discriminator output for AMP observations from reference demo data.
        amp_obs_demo: AMP observations from reference data, used for gradient penalty.
        """
        # BCE loss: make the discriminator distinguish policy fake data from demo real data.
        # agent (fake) should be classified as 0, demo (real) should be 1
        bce = torch.nn.BCEWithLogitsLoss()
        
        disc_loss_agent = bce(disc_agent_logit, torch.zeros_like(disc_agent_logit))
        disc_loss_agent_replay = bce(disc_agent_replay_logit, torch.zeros_like(disc_agent_replay_logit))
        disc_loss_demo = bce(disc_demo_logit, torch.ones_like(disc_demo_logit))
        
        # Combine losses: average of agent losses + demo loss
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_agent_replay) + disc_loss_demo
        
        # Logit regularization: prevent logits from becoming too large using output-layer L2 regularization.
        # Get output layer weights (last linear layer)
        output_layer = None
        for module in self.discriminator.modules():
            if isinstance(module, torch.nn.Linear):
                output_layer = module  # Keep updating to get the last one
        
        if output_layer is not None:
            logit_weights = torch.flatten(output_layer.weight)
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss += self.disc_logit_reg * disc_logit_loss
        else:
            disc_logit_loss = torch.tensor(0.0, device=self.device)
        
        # Gradient penalty on demo data, SMPLOlympics style.
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit, 
            amp_obs_demo, 
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self.disc_grad_penalty * disc_grad_penalty
        
        # Compute accuracies (logit < 0 means classified as fake, logit > 0 means real)
        disc_agent_acc = torch.mean((disc_agent_logit < 0.0).float())
        disc_agent_replay_acc = torch.mean((disc_agent_replay_logit < 0.0).float())
        disc_demo_acc = torch.mean((disc_demo_logit > 0.0).float())
        
        disc_info = {
            "disc_loss": disc_loss,
            "disc_agent_acc": disc_agent_acc,
            "disc_agent_replay_acc": disc_agent_replay_acc,
            "disc_demo_acc": disc_demo_acc,
            "disc_agent_logit": disc_agent_logit.mean(),
            "disc_agent_replay_logit": disc_agent_replay_logit.mean(),
            "disc_demo_logit": disc_demo_logit.mean(),
            "disc_grad_penalty": disc_grad_penalty,
            "disc_logit_loss": disc_logit_loss,
        }

        return disc_info

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape_ego, actor_obs_shape_opponent,
        critic_obs_shape_ego, critic_obs_shape_opponent, prior_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorageStage3(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape_ego,
            actor_obs_shape_opponent,
            critic_obs_shape_ego,
            critic_obs_shape_opponent,
            prior_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

        # Initialize AMP replay buffer.
        if self.has_discriminator:
            from rsl_rl.storage import CircularBuffer
            num_amp_obs = self.amp_cfg["num_amp_obs"]
            replay_buffer_size = self.amp_cfg.get("amp_replay_buffer_size", 100000)
            self.amp_replay_buffer = CircularBuffer(
                max_length=replay_buffer_size,
                device=self.device
            )
            print(f"[INFO] Initialized AMP replay buffer with size {replay_buffer_size}")
    
    def act(self, taskpolicy_obs_ego, taskpolicy_obs_opponent, privileged_obs_ego, privileged_obs_opponent,
            prior_obs_ego, prior_obs_opponent, decoder_obs_ego, decoder_obs_opponent):
        # compute the actions and values
        # NOTE Below is ppo_ego info
        self.transition.actions_ego = self.policy.act_latent_ego(taskpolicy_obs_ego).detach()
        self.transition.values_ego = self.policy.evaluate_ego(privileged_obs_ego).detach()
        self.transition.actions_log_prob_ego = self.policy.get_latent_log_prob_ego(self.transition.actions_ego).detach()
        self.transition.action_mean_ego = self.policy.latent_mean_ego.detach()
        self.transition.action_sigma_ego = self.policy.latent_std_ego.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations_ego = taskpolicy_obs_ego
        self.transition.privileged_observations_ego = privileged_obs_ego
        # NOTE Below is ppo_opponent info
        self.transition.actions_opponent = self.policy.act_latent_opponent(taskpolicy_obs_opponent).detach()
        self.transition.values_opponent = self.policy.evaluate_opponent(privileged_obs_opponent).detach()
        self.transition.actions_log_prob_opponent = self.policy.get_latent_log_prob_opponent(self.transition.actions_opponent).detach()
        self.transition.action_mean_opponent = self.policy.latent_mean_opponent.detach()
        self.transition.action_sigma_opponent = self.policy.latent_std_opponent.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations_opponent = taskpolicy_obs_opponent
        self.transition.privileged_observations_opponent = privileged_obs_opponent
        
        self.transition.prior_observations_ego = prior_obs_ego
        self.transition.prior_observations_opponent = prior_obs_opponent
        # NOTE Compute actions step in the env
        prior_z_ego, _ = self.policy.act_inference_prior(prior_obs_ego)
        input_z_ego = prior_z_ego.detach() + self.transition.actions_ego
        if self.normalize_z:
            input_z_ego = torch.nn.functional.normalize(input_z_ego, dim=-1)

        deocder_obs_cat_ego = torch.cat([decoder_obs_ego, input_z_ego], dim=-1)
        robot_ego_actions = self.policy.act_inference_decoder(deocder_obs_cat_ego).detach()

        prior_z_opponent, _ = self.policy.act_inference_prior(prior_obs_opponent)
        input_z_opponent = prior_z_opponent.detach() + self.transition.actions_opponent

        if self.normalize_z:
            input_z_opponent = torch.nn.functional.normalize(input_z_opponent, dim=-1)
        deocder_obs_cat_opponent = torch.cat([decoder_obs_opponent, input_z_opponent], dim=-1)
        robot_opponent_actions = self.policy.act_inference_decoder(deocder_obs_cat_opponent).detach()
        actions = {}
        actions["robot_A"] = robot_ego_actions.to(self.device)
        actions["robot_B"] = robot_opponent_actions.to(self.device)
        return actions
        

    def process_env_step(self, rewards_ego, rewards_opponent, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards_ego = rewards_ego.clone()
        self.transition.dones_ego = dones
        self.transition.rewards_opponent = rewards_opponent.clone()
        self.transition.dones_opponent = dones

        # Record AMP observations if discriminator is enabled
        if self.has_discriminator:
            if "amp_obs_ego" in infos:
                self.transition.amp_obs_ego = infos["amp_obs_ego"].clone()
            if "amp_obs_opponent" in infos:
                self.transition.amp_obs_opponent = infos["amp_obs_opponent"].clone()

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()
        # Bootstrapping on time outs
        if "time_outs" in infos["robot_A"]["dones"]:
            self.transition.rewards_ego += self.gamma * torch.squeeze(
                self.transition.values_ego * infos["robot_A"]["dones"]["time_outs"].unsqueeze(1).to(self.device), 1
            )
        if "time_outs" in infos["robot_B"]["dones"]:
            self.transition.rewards_opponent += self.gamma * torch.squeeze(
                self.transition.values_opponent * infos["robot_B"]["dones"]["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs_ego, last_critic_obs_opponent):
        # compute value for the last step
        last_values_ego = self.policy.evaluate_ego(last_critic_obs_ego).detach()
        last_values_opponent = self.policy.evaluate_opponent(last_critic_obs_opponent).detach()
        self.storage.compute_returns_ego(
            last_values_ego, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )
        self.storage.compute_returns_opponent(
            last_values_opponent, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss_ego = 0
        mean_value_loss_opponent = 0
        mean_surrogate_loss_ego = 0
        mean_surrogate_loss_opponent = 0
        mean_entropy_ego = 0
        mean_entropy_opponent = 0
        mean_kl_constraint_loss_ego = 0
        mean_kl_constraint_loss_opponent = 0
        
        # -- RND loss``
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # Collect AMP observations from current rollout for discriminator training
        current_amp_obs_ego = None
        current_amp_obs_opponent = None
        if self.has_discriminator:
            if self.storage.amp_obs_ego is not None:
                # Shape: [num_transitions_per_env, num_envs, amp_obs_dim]
                # Flatten to [num_transitions_per_env * num_envs, amp_obs_dim]
                current_amp_obs_ego = self.storage.amp_obs_ego.view(-1, self.num_amp_obs)
            if self.storage.amp_obs_opponent is not None:
                current_amp_obs_opponent = self.storage.amp_obs_opponent.view(-1, self.num_amp_obs)

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch_ego, 
            obs_batch_opponent,
            critic_obs_batch_ego, 
            critic_obs_batch_opponent,
            prior_obs_batch_ego,
            prior_obs_batch_opponent,
            actions_batch_ego, 
            actions_batch_opponent,
            target_values_batch_ego, 
            target_values_batch_opponent,
            advantages_batch_ego, 
            advantages_batch_opponent,
            returns_batch_ego, 
            returns_batch_opponent,
            old_actions_log_prob_batch_ego, 
            old_actions_log_prob_batch_opponent,
            old_mu_batch_ego, 
            old_mu_batch_opponent,
            old_sigma_batch_ego, 
            old_sigma_batch_opponent,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch_ego.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch_ego = (advantages_batch_ego - advantages_batch_ego.mean()) / (advantages_batch_ego.std() + 1e-8)
                    advantages_batch_opponent = (advantages_batch_opponent - advantages_batch_opponent.mean()) / (advantages_batch_opponent.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            if torch.isnan(obs_batch_ego).any():
                print("!!! Fatal error: NaN detected in ego obs !!!")
                print("obs_batch_ego (raw parameter):", obs_batch_ego)
                raise ValueError("ego obs became NaN")
            self.policy.act_latent_ego(obs_batch_ego, masks=masks_batch, hidden_states=hid_states_batch[0])
            self.policy.act_latent_opponent(obs_batch_opponent, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch_ego = self.policy.get_latent_log_prob_ego(actions_batch_ego)
            actions_log_prob_batch_opponent = self.policy.get_latent_log_prob_opponent(actions_batch_opponent)
            # -- critic
            value_batch_ego = self.policy.evaluate_ego(critic_obs_batch_ego, masks=masks_batch, hidden_states=hid_states_batch[1])
            value_batch_opponent = self.policy.evaluate_opponent(critic_obs_batch_opponent, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch_ego = self.policy.latent_mean_ego[:original_batch_size]
            mu_batch_opponent = self.policy.latent_mean_opponent[:original_batch_size]
            sigma_batch_ego = self.policy.latent_std_ego[:original_batch_size]
            sigma_batch_opponent = self.policy.latent_std_opponent[:original_batch_size]
            entropy_batch_ego = self.policy.entropy_ego[:original_batch_size]
            entropy_batch_opponent = self.policy.entropy_opponent[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_ego = torch.sum(
                        torch.log(sigma_batch_ego / old_sigma_batch_ego + 1.0e-5)
                        + (torch.square(old_sigma_batch_ego) + torch.square(old_mu_batch_ego - mu_batch_ego))
                        / (2.0 * torch.square(sigma_batch_ego))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean_ego = torch.mean(kl_ego)
                    
                    kl_opponent = torch.sum(
                        torch.log(sigma_batch_opponent / old_sigma_batch_opponent + 1.0e-5)
                        + (torch.square(old_sigma_batch_opponent) + torch.square(old_mu_batch_opponent - mu_batch_opponent))
                        / (2.0 * torch.square(sigma_batch_opponent))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean_opponent = torch.mean(kl_opponent)
                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean_ego, op=torch.distributed.ReduceOp.SUM)
                        kl_mean_ego /= self.gpu_world_size
                        torch.distributed.all_reduce(kl_mean_opponent, op=torch.distributed.ReduceOp.SUM)
                        kl_mean_opponent /= self.gpu_world_size
                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0: # TODO seperate ego and opponent kl
                        if kl_mean_ego > self.desired_kl * 2.0:
                            self.learning_rate_ego = max(1e-5, self.learning_rate_ego / 1.5)
                        elif kl_mean_ego < self.desired_kl / 2.0 and kl_mean_ego > 0.0:
                            self.learning_rate_ego = min(1e-2, self.learning_rate_ego * 1.5)
                        if kl_mean_opponent > self.desired_kl * 2.0:
                            self.learning_rate_opponent = max(1e-5, self.learning_rate_opponent / 1.5)
                        elif kl_mean_opponent < self.desired_kl / 2.0 and kl_mean_opponent > 0.0:
                            self.learning_rate_opponent = min(1e-2, self.learning_rate_opponent * 1.5)
                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer_ego.param_groups:
                        param_group["lr"] = self.learning_rate_ego
                    for param_group in self.optimizer_opponent.param_groups:
                        param_group["lr"] = self.learning_rate_opponent

            # Surrogate loss
            ratio_ego = torch.exp(actions_log_prob_batch_ego - torch.squeeze(old_actions_log_prob_batch_ego))
            surrogate_ego = -torch.squeeze(advantages_batch_ego) * ratio_ego
            surrogate_clipped_ego = -torch.squeeze(advantages_batch_ego) * torch.clamp(
                ratio_ego, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss_ego = torch.max(surrogate_ego, surrogate_clipped_ego).mean()

            ratio_opponent = torch.exp(actions_log_prob_batch_opponent - torch.squeeze(old_actions_log_prob_batch_opponent))
            surrogate_opponent = -torch.squeeze(advantages_batch_opponent) * ratio_opponent
            surrogate_clipped_opponent = -torch.squeeze(advantages_batch_opponent) * torch.clamp(
                ratio_opponent, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss_opponent = torch.max(surrogate_opponent, surrogate_clipped_opponent).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped_ego = target_values_batch_ego + (value_batch_ego - target_values_batch_ego).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses_ego = (value_batch_ego - returns_batch_ego).pow(2)
                value_losses_clipped_ego = (value_clipped_ego - returns_batch_ego).pow(2)
                value_loss_ego = torch.max(value_losses_ego, value_losses_clipped_ego).mean()
                

                value_clipped_opponent = target_values_batch_opponent + (value_batch_opponent - target_values_batch_opponent).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses_opponent = (value_batch_opponent - returns_batch_opponent).pow(2)
                value_losses_clipped_opponent = (value_clipped_opponent - returns_batch_opponent).pow(2)
                value_loss_opponent = torch.max(value_losses_opponent, value_losses_clipped_opponent).mean()
            else:
                value_loss_ego = (returns_batch_ego - value_batch_ego).pow(2).mean()
                value_loss_opponent = (returns_batch_opponent - value_batch_opponent).pow(2).mean()

            if self.kl_constraint:
                prior_mu_ego, prior_logvar_ego = self.policy.act_inference_prior(prior_obs_batch_ego)
                prior_mu_opponent, prior_logvar_opponent = self.policy.act_inference_prior(prior_obs_batch_opponent)
                kl_constraint_loss_ego = self.compute_kl_loss(mu_batch_ego, 2*torch.log(self.policy.std_ego+1e-6), 
                                                              prior_mu_ego, prior_logvar_ego)
                kl_constraint_loss_opponent = self.compute_kl_loss(mu_batch_opponent, 2*torch.log(self.policy.std_opponent+1e-6)
                                                                   , prior_mu_opponent, prior_logvar_opponent)
            else:
                kl_constraint_loss_ego = 0.0
                kl_constraint_loss_opponent = 0.0

            loss_ego = surrogate_loss_ego + self.value_loss_coef * value_loss_ego - self.entropy_coef * entropy_batch_ego.mean() + self.kl_constraint_coef * kl_constraint_loss_ego
            loss_opponent = surrogate_loss_opponent + self.value_loss_coef * value_loss_opponent - self.entropy_coef * entropy_batch_opponent.mean() + self.kl_constraint_coef * kl_constraint_loss_opponent
            loss = loss_ego + loss_opponent

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer_ego.zero_grad()
            self.optimizer_opponent.zero_grad()
            if loss.requires_grad:
                loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            ego_params = (
                list(self.policy.taskpolicy_ego.parameters())
                + list(self.policy.critic_ego.parameters())
                + [self.policy.std_ego]
            )
            ego_params_with_grad = [p for p in ego_params if p.requires_grad and p.grad is not None]
            if len(ego_params_with_grad) > 0:
                nn.utils.clip_grad_norm_(ego_params_with_grad, self.max_grad_norm)
                self.optimizer_ego.step()

            opponent_params = (
                list(self.policy.taskpolicy_opponent.parameters())
                + list(self.policy.critic_opponent.parameters())
                + [self.policy.std_opponent]
            )
            opponent_params_with_grad = [p for p in opponent_params if p.requires_grad and p.grad is not None]
            if len(opponent_params_with_grad) > 0:
                nn.utils.clip_grad_norm_(opponent_params_with_grad, self.max_grad_norm)
                self.optimizer_opponent.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss_ego += value_loss_ego.item()
            mean_value_loss_opponent += value_loss_opponent.item()
            mean_surrogate_loss_ego += surrogate_loss_ego.item()
            mean_surrogate_loss_opponent += surrogate_loss_opponent.item()
            mean_entropy_ego += entropy_batch_ego.mean().item()
            mean_entropy_opponent += entropy_batch_opponent.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # ===== Train Discriminator =====
        # Following SMPLOlympics pattern: use current rollout + replay buffer + demo
        if self.has_discriminator and current_amp_obs_ego is not None:
            mean_disc_loss = 0
            mean_disc_agent_acc = 0
            mean_disc_agent_replay_acc = 0
            mean_disc_demo_acc = 0
            mean_disc_agent_logit = 0
            mean_disc_agent_replay_logit = 0
            mean_disc_demo_logit = 0
            mean_disc_grad_penalty = 0
            mean_disc_logit_loss = 0
            
            # Number of discriminator mini-batch updates.
            disc_num_updates = self.num_learning_epochs * self.num_mini_batches
            
            for _ in range(disc_num_updates):
                # Get AMP observations from 3 sources:
                # 1. Current rollout (recent agent data): current rollout data collected from storage.
                # 2. Replay buffer (older agent data): historical data sampled from the replay buffer.
                # 3. Demo buffer (reference motion data): reference data sampled from the demo buffer.
                
                # Sample from current rollout - combine ego and opponent
                num_current = current_amp_obs_ego.shape[0]
                current_indices = torch.randperm(num_current, device=self.device)[:self.amp_batch_size // 2]
                amp_obs_current_ego = current_amp_obs_ego[current_indices]
                
                num_current_opp = current_amp_obs_opponent.shape[0]
                current_indices_opp = torch.randperm(num_current_opp, device=self.device)[:self.amp_batch_size // 2]
                amp_obs_current_opponent = current_amp_obs_opponent[current_indices_opp]
                
                # Combine ego and opponent current observations
                amp_obs_current_batch = torch.cat([amp_obs_current_ego, amp_obs_current_opponent], dim=0)
                
                # Sample from replay buffer (if not empty) - otherwise use current
                if len(self.amp_replay_buffer) > 0:
                    amp_obs_replay_batch = self.amp_replay_buffer.sample(self.amp_batch_size)
                else:
                    # If the replay buffer is empty in the first epoch, use current data.
                    amp_obs_replay_batch = amp_obs_current_batch.clone()
                
                # Sample demo data.
                amp_obs_demo_batch = self.amp_demo_buffer.sample(self.amp_batch_size)
                
                # Normalize AMP observations.
                # num_amp_obs includes history, so the single-step observation dimension is num_amp_obs / history_len.
                history_len = self.amp_cfg.get("history_len", 2)
                num_amp_obs_1_step = self.num_amp_obs // history_len
                
                # Debug: print dimension information once.
                if not hasattr(self, '_amp_debug_printed'):
                    print(f"[DEBUG] AMP normalization:")
                    print(f"  num_amp_obs: {self.num_amp_obs}")
                    print(f"  history_len: {history_len}")
                    print(f"  num_amp_obs_1_step: {num_amp_obs_1_step}")
                    print(f"  amp_obs_current_batch shape: {amp_obs_current_batch.shape}")
                    print(f"  amp_obs_replay_batch shape: {amp_obs_replay_batch.shape}")
                    print(f"  amp_obs_demo_batch shape: {amp_obs_demo_batch.shape}")
                    print(f"  amp_normalizer._mean shape: {self.amp_normalizer._mean.shape}")
                    self._amp_debug_printed = True
                
                # Normalize each step separately
                amp_obs_current_batch_norm = amp_obs_current_batch.clone()
                amp_obs_replay_batch_norm = amp_obs_replay_batch.clone()
                amp_obs_demo_batch_norm = amp_obs_demo_batch.clone()
                
                for h in range(history_len):
                    start = h * num_amp_obs_1_step
                    end = (h + 1) * num_amp_obs_1_step
                    
                    # Debug: print slice information.
                    if not hasattr(self, '_amp_debug_printed2'):
                        print(f"[DEBUG] Loop h={h}: start={start}, end={end}")
                        print(f"  amp_obs_current_batch[:, start:end] shape: {amp_obs_current_batch[:, start:end].shape}")
                        print(f"  amp_obs_demo_batch[:, start:end] shape: {amp_obs_demo_batch[:, start:end].shape}")
                        if h == history_len - 1:
                            self._amp_debug_printed2 = True
                    
                    amp_obs_current_batch_norm[:, start:end] = self.amp_normalizer(amp_obs_current_batch[:, start:end])
                    amp_obs_replay_batch_norm[:, start:end] = self.amp_normalizer(amp_obs_replay_batch[:, start:end])
                    amp_obs_demo_batch_norm[:, start:end] = self.amp_normalizer(amp_obs_demo_batch[:, start:end])
                
                # Need gradients for gradient penalty.
                amp_obs_demo_batch_norm.requires_grad_(True)
                
                # Forward pass through discriminator
                disc_agent_logit = self.discriminator(amp_obs_current_batch_norm)
                disc_agent_replay_logit = self.discriminator(amp_obs_replay_batch_norm)
                disc_demo_logit = self.discriminator(amp_obs_demo_batch_norm)
                
                # Compute discriminator loss with 3 sources
                disc_info = self._disc_loss(
                    disc_agent_logit,
                    disc_agent_replay_logit,
                    disc_demo_logit, 
                    amp_obs_demo_batch_norm
                )
                
                disc_loss = disc_info["disc_loss"] * self.disc_coef
                
                # Backward and optimize
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                
                if self.is_multi_gpu:
                    # Reduce discriminator gradients
                    disc_grads = [param.grad.view(-1) for param in self.discriminator.parameters() if param.grad is not None]
                    all_disc_grads = torch.cat(disc_grads)
                    torch.distributed.all_reduce(all_disc_grads, op=torch.distributed.ReduceOp.SUM)
                    all_disc_grads /= self.gpu_world_size
                    
                    offset = 0
                    for param in self.discriminator.parameters():
                        if param.grad is not None:
                            numel = param.numel()
                            param.grad.data.copy_(all_disc_grads[offset : offset + numel].view_as(param.grad.data))
                            offset += numel
                
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.disc_max_grad_norm)
                self.disc_optimizer.step()
                
                # Accumulate losses
                mean_disc_loss += disc_info["disc_loss"].item()
                mean_disc_agent_acc += disc_info["disc_agent_acc"].item()
                mean_disc_agent_replay_acc += disc_info["disc_agent_replay_acc"].item()
                mean_disc_demo_acc += disc_info["disc_demo_acc"].item()
                mean_disc_agent_logit += disc_info["disc_agent_logit"].item()
                mean_disc_agent_replay_logit += disc_info["disc_agent_replay_logit"].item()
                mean_disc_demo_logit += disc_info["disc_demo_logit"].item()
                mean_disc_grad_penalty += disc_info["disc_grad_penalty"].item()
                mean_disc_logit_loss += disc_info["disc_logit_loss"].item()
            
            # Average discriminator losses
            mean_disc_loss /= disc_num_updates
            mean_disc_agent_acc /= disc_num_updates
            mean_disc_agent_replay_acc /= disc_num_updates
            mean_disc_demo_acc /= disc_num_updates
            mean_disc_agent_logit /= disc_num_updates
            mean_disc_agent_replay_logit /= disc_num_updates
            mean_disc_demo_logit /= disc_num_updates
            mean_disc_grad_penalty /= disc_num_updates
            mean_disc_logit_loss /= disc_num_updates
            
            # Store current rollout amp_obs to replay buffer for next epoch
            # Following SMPLOlympics pattern: store after discriminator training
            if self.amp_replay_buffer is not None:
                if current_amp_obs_ego is not None:
                    self.amp_replay_buffer.store(current_amp_obs_ego)
                if current_amp_obs_opponent is not None:
                    self.amp_replay_buffer.store(current_amp_obs_opponent)
        else:
            # If discriminator is not trained, set losses to None
            mean_disc_loss = None
            mean_disc_agent_acc = None
            mean_disc_agent_replay_acc = None
            mean_disc_demo_acc = None
            mean_disc_agent_logit = None
            mean_disc_agent_replay_logit = None
            mean_disc_demo_logit = None
            mean_disc_grad_penalty = None
            mean_disc_logit_loss = None
            
        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss_ego /= num_updates
        mean_value_loss_opponent /= num_updates
        mean_surrogate_loss_ego /= num_updates
        mean_surrogate_loss_opponent /= num_updates
        mean_entropy_ego /= num_updates
        mean_entropy_opponent /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function_ego": mean_value_loss_ego,
            "value_function_opponent": mean_value_loss_opponent,
            "surrogate_ego": mean_surrogate_loss_ego,
            "surrogate_opponent": mean_surrogate_loss_opponent,
            "entropy_ego": mean_entropy_ego,
            "entropy_opponent": mean_entropy_opponent,
            "kl_constraint_ego": mean_kl_constraint_loss_ego,
            "kl_constraint_opponent": mean_kl_constraint_loss_opponent
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        
        # Add AMP discriminator metrics to loss_dict
        if self.has_discriminator:
            if mean_disc_loss is not None:
                loss_dict["disc_loss"] = mean_disc_loss
            if mean_disc_agent_acc is not None:
                loss_dict["disc_agent_acc"] = mean_disc_agent_acc
            if mean_disc_agent_replay_acc is not None:
                loss_dict["disc_agent_replay_acc"] = mean_disc_agent_replay_acc
            if mean_disc_demo_acc is not None:
                loss_dict["disc_demo_acc"] = mean_disc_demo_acc
            if mean_disc_agent_logit is not None:
                loss_dict["disc_agent_logit"] = mean_disc_agent_logit
            if mean_disc_agent_replay_logit is not None:
                loss_dict["disc_agent_replay_logit"] = mean_disc_agent_replay_logit
            if mean_disc_demo_logit is not None:
                loss_dict["disc_demo_logit"] = mean_disc_demo_logit
            if mean_disc_grad_penalty is not None:
                loss_dict["disc_grad_penalty"] = mean_disc_grad_penalty
            if mean_disc_logit_loss is not None:
                loss_dict["disc_logit_loss"] = mean_disc_logit_loss

        return loss_dict

    """
    Helper functions
    """

    def compute_kl_loss(self, mu1, logvar1, mu2, logvar2):
        """
        Compute KL( N(mu1+mu2, var1+var2) || N(mu2, var2) ).
        Inputs:
            mu1, logvar1: mean and log variance of z1.
            mu2, logvar2: mean and log variance of z2.
        Returns:
            KL Divergence (scalar)
        """
        logvar_sum = torch.logaddexp(logvar1, logvar2)
        term1 = 0.5 * (logvar2 - logvar_sum)
        numerator = torch.exp(logvar_sum) + mu1.pow(2)
        denominator = 2 * torch.exp(logvar2)
        term2 = numerator / denominator
        kl_loss = term1 + term2 - 0.5
        
        return kl_loss.mean()
    
    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
