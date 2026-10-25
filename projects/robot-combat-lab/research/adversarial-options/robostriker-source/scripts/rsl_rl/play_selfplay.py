"""Script to play/inference with self-play trained models."""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Play/inference with self-play trained RL agents.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during inference.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Combating-G1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--episode_length_s", type=float, default=100.0, help="Episode length in seconds.")
parser.add_argument("--global_view", action="store_true", default=False, help="Use global view (world origin) instead of following robot.")
parser.add_argument("--export_only", action="store_true", default=False)

# Cinematic camera controls (viewport camera)
parser.add_argument("--cinematic", action="store_true", default=False, help="Enable cinematic camera (orbit + hit zoom). Recommended with --video.")
parser.add_argument("--cinematic_orbit_radius", type=float, default=6.0, help="Orbit camera radius (meters).")
parser.add_argument("--cinematic_orbit_height", type=float, default=2.2, help="Orbit camera height above look-at (meters).")
parser.add_argument("--cinematic_orbit_period_s", type=float, default=10.0, help="Seconds per full 360-degree rotation.")
parser.add_argument("--cinematic_lookat_z", type=float, default=1.1, help="Look-at Z offset (meters).")
parser.add_argument("--cinematic_hit_reward_threshold", type=float, default=0.5, help="Reward threshold to trigger hit zoom (robot_A reward).")
parser.add_argument("--cinematic_hit_cooldown_steps", type=int, default=60, help="Cooldown steps between zoom triggers.")
parser.add_argument("--cinematic_zoom_radius", type=float, default=2.4, help="Zoom-in radius (meters).")
parser.add_argument("--cinematic_zoom_height", type=float, default=1.6, help="Zoom-in height above look-at (meters).")
parser.add_argument("--cinematic_zoom_hold_steps", type=int, default=40, help="How many steps to hold the zoomed-in view.")
parser.add_argument("--cinematic_zoom_ease_steps", type=int, default=20, help="How many steps to ease in/out for zoom transition.")
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from loguru import logger

from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from rsl_rl.env import RslVecMAEnvWrapper  # isort: skip
from whole_body_tracking.envs.mix_marl_env import MixMarlEnvCfg
from whole_body_tracking.rsl_rl.rsl_rl.runners.on_policy_runner_selfplay import OnPolicyRunnerSelfPlay
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyVQRunnerPPOStage3 as OnPolicyRunnerStage3VQ
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunnerPPOStage3 as OnPolicyRunnerStage3
from whole_body_tracking.utils.exporter import export_stage3_policy_as_onnx

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: MixMarlEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play/inference with self-play trained models."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.episode_length_s = args_cli.episode_length_s

    # set the environment seed
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # View configuration
    # NOTE: cinematic mode needs global (world) view so we can freely orbit around the scene center.
    if args_cli.cinematic:
        args_cli.global_view = True
        if not args_cli.video:
            logger.warning("Cinematic camera is enabled but --video is not set. You will see the camera in GUI, but no mp4 will be saved.")

    if args_cli.global_view:
        print("[INFO] Using global view (world origin)")
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (5.0, 5.0, 5.0)  # Higher and further for overview
        env_cfg.viewer.lookat = (1.0, 0.0, 0.76)  # Look at center between robots
    else:
        print("[INFO] Using local view (following robot_A)")
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot_A"
        env_cfg.viewer.eye = (2.0, 2.0, 2.0)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_marl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    logger.info(f"Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    logger.info(f"Loading model checkpoint from: {resume_path}")
    
    log_dir = os.path.dirname(resume_path)
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_selfplay"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        logger.info("Recording videos during inference.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslVecMAEnvWrapper(env)
    device = env.device
    
    # Create runner - use OnPolicyRunnerStage3 for inference (same policy structure as self-play)
    # Note: OnPolicyRunnerSelfPlay inherits from OnPolicyRunnerStage3, so the model structure is compatible
    if "VQ" in args_cli.task:
        runner = OnPolicyRunnerStage3VQ(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        # Use MotionOnPolicyRunnerPPOStage3 (same as play_marl_env.py) for consistency
        runner = OnPolicyRunnerStage3(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, is_train=False
        )
    
    # Load the trained model
    runner.load(resume_path)
    # Export policy to ONNX
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_stage3_policy_as_onnx(
        policy=runner.alg.policy,
        path=export_model_dir,
        runner=runner,
        onnx_prefix=f"{args_cli.checkpoint.split('.')[0]}_",
    )
    if args_cli.export_only:
        return
    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Reset environment
    obs, extras = env.get_observations()
    
    logger.info("=" * 60)
    logger.info("Starting self-play inference...")
    logger.info(f"Task: {args_cli.task}")
    logger.info(f"Episode length: {env_cfg.episode_length_s}s")
    logger.info(f"Number of environments: {env_cfg.scene.num_envs}")
    logger.info(f"Model checkpoint: {resume_path}")
    logger.info(f"Video recording: {'Enabled' if args_cli.video else 'Disabled'}")
    if args_cli.video:
        logger.info(f"Video length: {args_cli.video_length} steps")
    logger.info("=" * 60)
    
    timestep = 0
    episode_count = 0
    dt = env.unwrapped.step_dt

    # Cinematic camera state
    cinematic_t = 0.0
    last_hit_step = -10**9
    zoom_state = "orbit"  # orbit | ease_in | hold | ease_out
    zoom_counter = 0

    def _smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _get_lookat_xy() -> torch.Tensor:
        """Returns [3] look-at position (world frame) for env 0."""
        base_env = env.unwrapped
        # Prefer midpoint between robot_A and robot_B if available.
        try:
            pos_a = base_env.scene["robot_A"].data.root_pos_w[0]
            pos_b = base_env.scene["robot_B"].data.root_pos_w[0]
            lookat = 0.5 * (pos_a + pos_b)
        except Exception:
            # Fallback to robot_A root.
            lookat = base_env.scene["robot_A"].data.root_pos_w[0]
        lookat = lookat.clone()
        lookat[2] = lookat[2] + float(args_cli.cinematic_lookat_z)
        return lookat

    def _set_orbit_camera(lookat: torch.Tensor, radius: float, height: float, t_sec: float):
        """Sets viewport camera to orbit around lookat point."""
        # angle in XY plane
        period = max(1e-3, float(args_cli.cinematic_orbit_period_s))
        theta = 2.0 * 3.141592653589793 * (t_sec / period)
        eye = lookat.clone()
        eye[0] = eye[0] + float(radius) * float(torch.cos(torch.tensor(theta)))
        eye[1] = eye[1] + float(radius) * float(torch.sin(torch.tensor(theta)))
        eye[2] = eye[2] + float(height)
        # set camera
        env.unwrapped.sim.set_camera_view(eye=eye.detach().cpu().numpy(), target=lookat.detach().cpu().numpy())
    
    # Statistics tracking
    total_rewards_ego = torch.zeros(env_cfg.scene.num_envs, device=device)
    total_rewards_opponent = torch.zeros(env_cfg.scene.num_envs, device=device)
    episode_lengths = torch.zeros(env_cfg.scene.num_envs, device=device, dtype=torch.long)
    
    # Inference loop
    while simulation_app.is_running():
        start_time = time.time()
        
        # Run inference
        with torch.inference_mode():
            # Get actions from policy
            actions = policy(extras)
            
            # Step environment
            obs, rewards, dones, extras = env.step(actions)

            # Cinematic camera update (only meaningful when GUI is enabled; still safe to call)
            if args_cli.cinematic:
                cinematic_t += float(dt)
                lookat = _get_lookat_xy()

                # Detect "hit" using offensive_hit_reward_punch function directly
                hit_now = False
                try:
                    base_env = env.unwrapped
                    # Import the reward function
                    import whole_body_tracking.tasks.combat.mdp.rewards as mdp_rewards
                    from isaaclab.managers import SceneEntityCfg
                    
                    # Get reward configuration from environment
                    reward_cfg = base_env.cfg.rewards.get("robot_A")
                    if reward_cfg is not None and hasattr(reward_cfg, "offensive_hit_reward_punch"):
                        hit_term_cfg = reward_cfg.offensive_hit_reward_punch
                        # Call the function directly with current environment state
                        # This gives us the raw hit detection value (0 or 1 typically)
                        hit_value = hit_term_cfg.func(
                            base_env,
                            hit_term_cfg.params["sensor_cfg1"],
                            hit_term_cfg.params["sensor_cfg2"],
                            hit_term_cfg.params["asset_cfg"],
                            hit_term_cfg.params.get("threshold", 1.0),
                            hit_term_cfg.params.get("vel_threshold", 1.0),
                            hit_term_cfg.params.get("relative_by_ego", True),
                        )
                        # Check if environment 0 has a hit
                        hit_now = bool(hit_value[0].item() > 0.5)
                except Exception as e:
                    # Fallback to reward spike detection if direct function call fails
                    try:
                        r_a = float(rewards["robot_A"][0].item())
                        hit_now = (r_a >= float(args_cli.cinematic_hit_reward_threshold))
                    except Exception:
                        hit_now = False

                if hit_now and (timestep - last_hit_step) >= int(args_cli.cinematic_hit_cooldown_steps):
                    last_hit_step = timestep
                    zoom_state = "ease_in"
                    zoom_counter = 0

                # Camera radius/height scheduling
                r_orbit = float(args_cli.cinematic_orbit_radius)
                h_orbit = float(args_cli.cinematic_orbit_height)
                r_zoom = float(args_cli.cinematic_zoom_radius)
                h_zoom = float(args_cli.cinematic_zoom_height)
                ease_steps = max(1, int(args_cli.cinematic_zoom_ease_steps))
                hold_steps = max(0, int(args_cli.cinematic_zoom_hold_steps))

                if zoom_state == "orbit":
                    radius, height = r_orbit, h_orbit
                elif zoom_state == "ease_in":
                    alpha = _smoothstep(zoom_counter / ease_steps)
                    radius = (1 - alpha) * r_orbit + alpha * r_zoom
                    height = (1 - alpha) * h_orbit + alpha * h_zoom
                    zoom_counter += 1
                    if zoom_counter >= ease_steps:
                        zoom_state = "hold"
                        zoom_counter = 0
                elif zoom_state == "hold":
                    radius, height = r_zoom, h_zoom
                    zoom_counter += 1
                    if zoom_counter >= hold_steps:
                        zoom_state = "ease_out"
                        zoom_counter = 0
                else:  # ease_out
                    alpha = _smoothstep(zoom_counter / ease_steps)
                    radius = (1 - alpha) * r_zoom + alpha * r_orbit
                    height = (1 - alpha) * h_zoom + alpha * h_orbit
                    zoom_counter += 1
                    if zoom_counter >= ease_steps:
                        zoom_state = "orbit"
                        zoom_counter = 0

                _set_orbit_camera(lookat=lookat, radius=radius, height=height, t_sec=cinematic_t)
            
            # Update statistics (only for non-done environments)
            # Note: rewards are accumulated before checking dones, which is correct
            # because the reward is for the current step before termination
            total_rewards_ego += rewards["robot_A"]
            total_rewards_opponent += rewards["robot_B"]
            episode_lengths += 1
            
            # Check for episode completion
            if torch.any(dones):
                done_ids = torch.where(dones)[0].cpu().tolist()
                for env_id in done_ids:
                    episode_count += 1
                    logger.info(
                        f"Episode {episode_count} completed (env {env_id}): "
                        f"ego_reward={total_rewards_ego[env_id].item():.2f}, "
                        f"opponent_reward={total_rewards_opponent[env_id].item():.2f}, "
                        f"length={episode_lengths[env_id].item()} steps"
                    )
                    # Reset statistics for completed episodes
                    total_rewards_ego[env_id] = 0
                    total_rewards_opponent[env_id] = 0
                    episode_lengths[env_id] = 0
        
        timestep += 1
        
        # Exit after recording video if specified
        if args_cli.video:
            if timestep >= args_cli.video_length:
                logger.info(f"Reached video length ({args_cli.video_length} steps), exiting.")
                break
        
        # Real-time simulation delay (optional)
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Close the simulator
    logger.info("Closing environment...")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

