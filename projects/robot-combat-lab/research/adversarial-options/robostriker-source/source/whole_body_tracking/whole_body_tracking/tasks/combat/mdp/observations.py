from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply_inverse
from isaaclab.managers import SceneEntityCfg, CommandManager

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


anchor_body_name = "torso_link"
body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

def robot_anchor_ori_w(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    mat = matrix_from_quat(robot.data.body_quat_w[:, robot.body_names.index(anchor_body_name)])
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_lin_vel_w[:, robot.body_names.index(anchor_body_name)].view(env.num_envs, -1)



def robot_anchor_ang_vel_w(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_ang_vel_w[:, robot.body_names.index(anchor_body_name)].view(env.num_envs, -1)

def robot_body_pos_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    num_bodies = len(body_names)
    robot_anchor_pos_w = robot.data.body_pos_w[:, robot.body_names.index(anchor_body_name)]
    robot_anchor_quat_w = robot.data.body_quat_w[:, robot.body_names.index(anchor_body_name)]
    body_index, _ = robot.find_bodies(body_names, preserve_order=True)
    body_index = torch.tensor(body_index, dtype=torch.long, device=env.device)
    robot_body_pos_w = robot.data.body_pos_w[:, body_index]
    robot_body_quat_w = robot.data.body_quat_w[:, body_index]

    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        robot_body_pos_w,
        robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)

def robot_body_ori_b(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    num_bodies = len(body_names)
    robot_anchor_pos_w = robot.data.body_pos_w[:, robot.body_names.index(anchor_body_name)]
    robot_anchor_quat_w = robot.data.body_quat_w[:, robot.body_names.index(anchor_body_name)]
    body_index, _ = robot.find_bodies(body_names, preserve_order=True)
    body_index = torch.tensor(body_index, dtype=torch.long, device=env.device)
    robot_body_pos_w = robot.data.body_pos_w[:, body_index]
    robot_body_quat_w = robot.data.body_quat_w[:, body_index]

    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        robot_body_pos_w,
        robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

def custom_generated_commands(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command_manager: CommandManager = env.command_managers[asset_cfg.name] if hasattr(env, "command_managers") else env.command_manager
    return command_manager.get_command(command_name)


def custom_last_action(env, action_name: str | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """

    action_manager = env.action_managers[asset_cfg.name] if hasattr(env, "action_managers") else env.action_manager
    if action_name is None:
        return action_manager.action
    else:
        return action_manager.get_term(action_name).raw_actions


def task_obs_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    ego_agent: Articulation = env.scene[asset_cfg.name]
    opponent_agent: Articulation = env.get_opponent(asset_cfg.name)
    torso_index = ego_agent.body_names.index("torso_link")
    left_wrist_yaw_index = ego_agent.body_names.index("left_wrist_yaw_link")
    right_wrist_yaw_index = ego_agent.body_names.index("right_wrist_yaw_link")
    ego_root_quat = ego_agent.data.root_quat_w
    N = ego_root_quat.shape[0]
    relative_pos_selftorso_to_opponentleftwrist = (ego_agent.data.body_pos_w[:, torso_index:torso_index+1] - opponent_agent.data.body_pos_w[:, left_wrist_yaw_index:left_wrist_yaw_index+1])
    relative_pos_selftorso_to_opponentrightwrist = (ego_agent.data.body_pos_w[:, torso_index:torso_index+1] - opponent_agent.data.body_pos_w[:, right_wrist_yaw_index:right_wrist_yaw_index+1])
    relative_pos_opponenttorso_to_selfleftwrist = (ego_agent.data.body_pos_w[:, left_wrist_yaw_index:left_wrist_yaw_index+1] - opponent_agent.data.body_pos_w[:, torso_index:torso_index+1])
    relative_pos_opponenttorso_to_selfrightwrist = (ego_agent.data.body_pos_w[:, right_wrist_yaw_index:right_wrist_yaw_index+1] - opponent_agent.data.body_pos_w[:, torso_index:torso_index+1])
    # Wentao: Remove the abs.
    task_obs_pos = torch.cat((
        relative_pos_selftorso_to_opponentleftwrist,
        relative_pos_selftorso_to_opponentrightwrist,
        relative_pos_opponenttorso_to_selfleftwrist,
        relative_pos_opponenttorso_to_selfrightwrist,
    ), dim=1)  # N x 4 x 3
    return quat_apply_inverse(ego_root_quat.unsqueeze(1).repeat(1,4,1), task_obs_pos).view(N, -1) # Rotate to ego's local frame.

def body_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    ego_agent: Articulation = env.scene[asset_cfg.name]
    ego_pos = ego_agent.data.body_pos_w
    # Rotate according to ego's root quat
    ego_root_quat = ego_agent.data.root_quat_w
    ego_root_pos = ego_agent.data.root_pos_w
    ego_pos = ego_pos - ego_root_pos.unsqueeze(1).repeat(1,ego_pos.shape[1],1)
    N = ego_root_quat.shape[0]
    ego_pos = quat_apply_inverse(ego_root_quat.unsqueeze(1).repeat(1,ego_pos.shape[1],1), ego_pos).view(N, -1)
    return ego_pos.reshape(env.num_envs, -1)

def body_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    ego_agent: Articulation = env.scene[asset_cfg.name]
    ego_vel = ego_agent.data.body_lin_vel_w
    # Rotate according to ego's root quat
    ego_root_quat = ego_agent.data.root_quat_w
    root_vel = ego_agent.data.root_lin_vel_w
    link_offset = ego_agent.data.body_pos_w - ego_agent.data.root_pos_w.unsqueeze(1)
    root_ang_vel = ego_agent.data.root_ang_vel_w
    ego_vel = ego_vel - root_vel.unsqueeze(1) - torch.cross(root_ang_vel.unsqueeze(1), link_offset, dim=-1)
    N = ego_root_quat.shape[0]
    ego_vel = quat_apply_inverse(ego_root_quat.unsqueeze(1).repeat(1,ego_vel.shape[1],1), ego_vel).view(N, -1)
    return ego_vel.reshape(env.num_envs, -1)

def opponent_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    opponent_agent: Articulation = env.get_opponent(asset_cfg.name)
    opponent_pos = opponent_agent.data.body_pos_w
    # Rotate according to ego's root quat
    ego_agent: Articulation = env.scene[asset_cfg.name]
    ego_root_quat = ego_agent.data.root_quat_w
    ego_root_pos = ego_agent.data.root_pos_w
    opponent_pos = opponent_pos - ego_root_pos.unsqueeze(1).repeat(1,opponent_pos.shape[1],1)
    N = ego_root_quat.shape[0]
    opponent_pos = quat_apply_inverse(ego_root_quat.unsqueeze(1).repeat(1,opponent_pos.shape[1],1), opponent_pos).view(N, -1)
    return opponent_pos.reshape(env.num_envs, -1)

def opponent_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    ego: Articulation = env.scene[asset_cfg.name]
    opp: Articulation = env.get_opponent(asset_cfg.name)

    root_pos = ego.data.root_pos_w            # ego frame origin
    root_lin = ego.data.root_lin_vel_w
    root_ang = ego.data.root_ang_vel_w
    root_quat = ego.data.root_quat_w

    opp_pos_w = opp.data.body_pos_w
    opp_vel_w = opp.data.body_lin_vel_w
    rel_pos_w = opp_pos_w - root_pos.unsqueeze(1)
    rel_vel_w = opp_vel_w - root_lin.unsqueeze(1) - torch.cross(root_ang.unsqueeze(1), rel_pos_w, dim=-1)

    rel_vel_b = quat_apply_inverse(root_quat.unsqueeze(1).repeat(1, opp_vel_w.shape[1], 1), rel_vel_w)
    return rel_vel_b.reshape(env.num_envs, -1)

def punch_target(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    ego_agent: Articulation = env.scene[asset_cfg.name]
    opponent_agent: Articulation = env.get_opponent(asset_cfg.name)
    torso_index = ego_agent.body_names.index("torso_link")
    left_wrist_yaw_index = ego_agent.body_names.index("left_wrist_yaw_link")
    right_wrist_yaw_index = ego_agent.body_names.index("right_wrist_yaw_link")

    ego_root_quat = ego_agent.data.root_quat_w
    N = ego_root_quat.shape[0]
    relative_pos_egofist_to_oppotorso_left = (ego_agent.data.body_pos_w[:, left_wrist_yaw_index:left_wrist_yaw_index+1] - opponent_agent.data.body_pos_w[:, torso_index:torso_index+1])
    relative_pos_egofist_to_oppotorso_right = (ego_agent.data.body_pos_w[:, right_wrist_yaw_index:right_wrist_yaw_index+1] - opponent_agent.data.body_pos_w[:, torso_index:torso_index+1])
    task_obs_pos = torch.cat((
        relative_pos_egofist_to_oppotorso_left,
        relative_pos_egofist_to_oppotorso_right,
    ), dim=1)  # N x 2 x 3
    # return task_obs_pos.reshape(env.num_envs, -1)
    return quat_apply_inverse(ego_root_quat.unsqueeze(1).repeat(1,2,1), task_obs_pos).view(N, -1) # Rotate to ego's local frame.

def opponent_punch_target(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Calculate opponent's two fists to ego's torso distance, in ego's local frame.
    
    Args:
        env: The environment instance
        asset_cfg: Scene entity configuration for the ego agent
        
    Returns:
        Tensor of shape (N, 6) containing relative positions of opponent's 
        left and right fists to ego's torso, rotated to ego's local frame.
    """
    ego_agent: Articulation = env.scene[asset_cfg.name]
    opponent_agent: Articulation = env.get_opponent(asset_cfg.name)
    
    # Get ego's torso index
    ego_torso_index = ego_agent.body_names.index("torso_link")
    
    # Get opponent's left and right wrist indices
    opponent_left_wrist_yaw_index = opponent_agent.body_names.index("left_wrist_yaw_link")
    opponent_right_wrist_yaw_index = opponent_agent.body_names.index("right_wrist_yaw_link")

    ego_root_quat = ego_agent.data.root_quat_w
    N = ego_root_quat.shape[0]
    
    # Calculate relative positions: opponent's fists to ego's torso
    relative_pos_oppofist_to_egotorso_left = (opponent_agent.data.body_pos_w[:, opponent_left_wrist_yaw_index:opponent_left_wrist_yaw_index+1] - ego_agent.data.body_pos_w[:, ego_torso_index:ego_torso_index+1])
    relative_pos_oppofist_to_egotorso_right = (opponent_agent.data.body_pos_w[:, opponent_right_wrist_yaw_index:opponent_right_wrist_yaw_index+1] - ego_agent.data.body_pos_w[:, ego_torso_index:ego_torso_index+1])
    
    task_obs_pos = torch.cat((
        relative_pos_oppofist_to_egotorso_left,
        relative_pos_oppofist_to_egotorso_right,
    ), dim=1)  # N x 2 x 3
    
    # Rotate to ego's local frame
    return quat_apply_inverse(ego_root_quat.unsqueeze(1).repeat(1,2,1), task_obs_pos).view(N, -1)
    
def task_obs_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    ego: Articulation = env.scene[asset_cfg.name]
    opp: Articulation = env.get_opponent(asset_cfg.name)

    torso_idx = ego.body_names.index("torso_link")
    l_wrist_idx = ego.body_names.index("left_wrist_yaw_link")
    r_wrist_idx = ego.body_names.index("right_wrist_yaw_link")

    root_pos = ego.data.root_pos_w
    root_lin = ego.data.root_lin_vel_w
    root_ang = ego.data.root_ang_vel_w
    root_quat = ego.data.root_quat_w

    # Ego wrists
    ego_pos_w = ego.data.body_pos_w[:, [l_wrist_idx, r_wrist_idx]]
    ego_vel_w = ego.data.body_lin_vel_w[:, [l_wrist_idx, r_wrist_idx]]
    ego_rel_pos_w = ego_pos_w - root_pos.unsqueeze(1)
    ego_rel_vel_w = ego_vel_w - root_lin.unsqueeze(1) - torch.cross(root_ang.unsqueeze(1), ego_rel_pos_w, dim=-1)

    # Opponent wrists (same indices assumed)
    opp_pos_w = opp.data.body_pos_w[:, [l_wrist_idx, r_wrist_idx]]
    opp_vel_w = opp.data.body_lin_vel_w[:, [l_wrist_idx, r_wrist_idx]]
    opp_rel_pos_w = opp_pos_w - root_pos.unsqueeze(1)
    opp_rel_vel_w = opp_vel_w - root_lin.unsqueeze(1) - torch.cross(root_ang.unsqueeze(1), opp_rel_pos_w, dim=-1)

    stacked_rel_vel_w = torch.cat([ego_rel_vel_w, opp_rel_vel_w], dim=1)  # [B, 4, 3]
    stacked_rel_vel_b = quat_apply_inverse(root_quat.unsqueeze(1).repeat(1, stacked_rel_vel_w.shape[1], 1), stacked_rel_vel_w)
    return stacked_rel_vel_b.reshape(env.num_envs, -1)

def motion_anchor_pos_b(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    command_manager: MotionCommand = env.command_managers[asset_cfg.name] if hasattr(env, "command_managers") else env.command_manager
    command: MotionCommand = command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    command_manager: CommandManager = env.command_managers[asset_cfg.name] if hasattr(env, "command_managers") else env.command_manager
    command: MotionCommand = command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)