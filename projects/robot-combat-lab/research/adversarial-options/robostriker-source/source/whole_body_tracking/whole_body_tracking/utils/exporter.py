# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx
from typing import Tuple, Optional

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand
import copy
import torch.nn as nn

class OnnxStage3PolicyExporter(nn.Module):
    def __init__(
        self,
        taskpolicy,
        prior,
        decoder,
        latent_dim,
        normalize_z,
        task_obs_normalizer=None,
        prior_obs_normalizer=None,
        decoder_obs_normalizer=None,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.taskpolicy = copy.deepcopy(taskpolicy).eval()
        self.prior = copy.deepcopy(prior).eval()
        self.decoder = copy.deepcopy(decoder).eval()
        self.latent_dim = latent_dim
        self.normalize_z = normalize_z
        
        self.task_obs_normalizer = (
            copy.deepcopy(task_obs_normalizer) if task_obs_normalizer is not None else nn.Identity()
        )
        self.prior_obs_normalizer = (
            copy.deepcopy(prior_obs_normalizer) if prior_obs_normalizer is not None else nn.Identity()
        )
        self.decoder_obs_normalizer = (
            copy.deepcopy(decoder_obs_normalizer) if decoder_obs_normalizer is not None else nn.Identity()
        )

        # Infer dimensions
        self.task_obs_dim = self.taskpolicy[0].in_features
        self.prior_obs_dim = self.prior[0].in_features
        self.decoder_obs_dim = self.decoder[0].in_features - self.latent_dim

    def forward(self, task_obs, prior_obs, decoder_obs):
        # Normalize observations
        task_obs = self.task_obs_normalizer(task_obs)
        prior_obs = self.prior_obs_normalizer(prior_obs)
        decoder_obs = self.decoder_obs_normalizer(decoder_obs)

        # Forward pass
        latent = self.taskpolicy(task_obs)
        prior_latent_mean, _ = self.prior(prior_obs).chunk(2, dim=-1)
        
        combined_latent = latent + prior_latent_mean
        if self.normalize_z:
            combined_latent = torch.nn.functional.normalize(combined_latent, dim=-1)
            
        decoder_input = torch.cat([decoder_obs, combined_latent], dim=-1)
        actions = self.decoder(decoder_input)
        
        return actions

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        self.eval()
        
        # Dummy inputs
        task_obs = torch.zeros(1, self.task_obs_dim)
        prior_obs = torch.zeros(1, self.prior_obs_dim)
        decoder_obs = torch.zeros(1, self.decoder_obs_dim)
        
        torch.onnx.export(
            self,
            (task_obs, prior_obs, decoder_obs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["task_obs", "prior_obs", "decoder_obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def export_stage3_policy_as_onnx(
    policy,
    path,
    runner,
    onnx_prefix: str = "",
    verbose=False,
):
    # Export Robot A
    exporter_A = OnnxStage3PolicyExporter(
        taskpolicy=policy.taskpolicy_ego,
        prior=policy.prior,
        decoder=policy.decoder,
        latent_dim=policy.latent_dim,
        normalize_z=policy.normalize_z,
        task_obs_normalizer=runner.taskpolicy_obs_normalizer_ego,
        prior_obs_normalizer=runner.prior_obs_normalizer,
        decoder_obs_normalizer=runner.decoder_obs_normalizer,
        verbose=verbose,
    )
    exporter_A.export(path, f"{onnx_prefix}policy_robot_A.onnx")
    
    # Export Robot B
    exporter_B = OnnxStage3PolicyExporter(
        taskpolicy=policy.taskpolicy_opponent,
        prior=policy.prior,
        decoder=policy.decoder,
        latent_dim=policy.latent_dim,
        normalize_z=policy.normalize_z,
        task_obs_normalizer=runner.taskpolicy_obs_normalizer_opponent,
        prior_obs_normalizer=runner.prior_obs_normalizer,
        decoder_obs_normalizer=runner.decoder_obs_normalizer,
        verbose=verbose,
    )
    exporter_B.export(path, f"{onnx_prefix}policy_robot_B.onnx")



def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
    data_slice: Optional[Tuple[int, int]] = None,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if data_slice is not None:
        filename = f"policy_{data_slice[0]}_{data_slice[1]}.onnx"
    if actor_critic.name == "ActorCriticMoe":
        policy_exporter = _OnnxMotionMoePolicyExporter(env, actor_critic, normalizer, verbose, data_slice)
    elif actor_critic.name == "ActorCriticLowlevelLatent":
        policy_exporter = _OnnxMotionPolicyLowlevelLatentExporter(env, actor_critic, normalizer, verbose, data_slice)
    else:
        policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose, data_slice)
    if data_slice is not None:
        filename = f"{filename.split('.onnx')[0]}_{data_slice[0]}_{data_slice[1]}.onnx"
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False, data_slice: Optional[Tuple[int, int]] = None):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion").motion_set.gather_all()
        if data_slice is None:
            data_slice = (0, cmd.joint_pos.shape[0])
        self.joint_pos = cmd.joint_pos[data_slice[0]:data_slice[1]].to("cpu")
        self.joint_vel = cmd.joint_vel[data_slice[0]:data_slice[1]].to("cpu")
        self.body_pos_w = cmd.body_pos_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_quat_w = cmd.body_quat_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_lin_vel_w = cmd.body_lin_vel_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_ang_vel_w = cmd.body_ang_vel_w[data_slice[0]:data_slice[1]].to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


class _OnnxMotionPolicyLowlevelLatentExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False, data_slice: Optional[Tuple[int, int]] = None):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion").motion_set.gather_all()
        if data_slice is None:
            data_slice = (0, cmd.joint_pos.shape[0])
        self.joint_pos = cmd.joint_pos[data_slice[0]:data_slice[1]].to("cpu")
        self.joint_vel = cmd.joint_vel[data_slice[0]:data_slice[1]].to("cpu")
        self.body_pos_w = cmd.body_pos_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_quat_w = cmd.body_quat_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_lin_vel_w = cmd.body_lin_vel_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_ang_vel_w = cmd.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

        # Infer dimensions for actor / encoder so we don't hard-code latent size
        # Actor takes [obs, latent] as input.
        self._actor_input_dim = self.actor[0].in_features
        # Encoder maps encoder_obs -> latent
        self._encoder_input_dim = self.policy.encoder[0].in_features
        self._latent_dim = self.policy.encoder[-1].out_features
        self._obs_dim = self._actor_input_dim - self._latent_dim

    def forward(self, obs, encoder_obs, time_step):
        """Forward pass used for ONNX export.

        Inputs:
            obs:          low-level policy observations (without latent).
            encoder_obs:  observations for the encoder network.
            time_step:    scalar index for motion library.
        """
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)

        # 1) Encode latent from encoder observations
        latent = self.policy.encode(encoder_obs)
        # 2) Normalize policy observations if a normalizer is provided
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        # 3) Concatenate obs and latent, then decode to actions
        actor_input = torch.cat([obs, latent], dim=-1)
        actions = self.actor(actor_input)

        return (
            actions,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        # Dummy tensors to trace the full encoder-decoder graph.
        obs = torch.zeros(1, self._obs_dim)
        encoder_obs = torch.zeros(1, self._encoder_input_dim)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, encoder_obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "encoder_obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


# export moe to onnx
class _OnnxMotionMoePolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False, data_slice: Optional[Tuple[int, int]] = None):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion").motion_set.gather_all()

        if data_slice is None:
            data_slice = (0, cmd.joint_pos.shape[0])

        self.joint_pos = cmd.joint_pos[data_slice[0]:data_slice[1]].to("cpu")
        self.joint_vel = cmd.joint_vel[data_slice[0]:data_slice[1]].to("cpu")
        self.body_pos_w = cmd.body_pos_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_quat_w = cmd.body_quat_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_lin_vel_w = cmd.body_lin_vel_w[data_slice[0]:data_slice[1]].to("cpu")
        self.body_ang_vel_w = cmd.body_ang_vel_w[data_slice[0]:data_slice[1]].to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        
        mean, std, gate_probs = self.actor(self.normalizer(x), deterministic=True)

        return (
            mean,  # Export only mean as the action.
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        # Use self.actor.input_dim instead of self.actor[0].in_features.
        obs = torch.zeros(1, self.actor.input_dim)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )

def export_motion_moe_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
    data_slice: Optional[Tuple[int, int]] = None,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionMoePolicyExporter(env, actor_critic, normalizer, verbose, data_slice)
    if data_slice is not None:
        filename = f"{filename.split('.onnx')[0]}_{data_slice[0]}_{data_slice[1]}.onnx"
    policy_exporter.export(path, filename)



def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)
    motion_ends = env.command_manager.get_term("motion").motion_set.get_motion_ends()
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
        "motion_ends": motion_ends,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
