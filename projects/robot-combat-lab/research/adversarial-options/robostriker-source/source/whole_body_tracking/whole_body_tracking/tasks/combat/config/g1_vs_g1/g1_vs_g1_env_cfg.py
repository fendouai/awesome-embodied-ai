from typing import Dict

from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs.common import AgentID

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.combat import combat_env_cfg
from whole_body_tracking.tasks.tracking import tracking_env_cfg


@configclass
class G1vsG1EnvCfg(combat_env_cfg.EnvCfg):
    #NOTE Entry
    def set_robots(self): # TODO CHECK
        self.scene.robot_A = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotA",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(0., 0., .76), # TODO init rot here, face to face
                rot=(1, 0, 0, 0)
            )
        )
        self.scene.robot_B = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotB",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(2.0, 0, .76),
                rot=(0, 0, 0, -1)
            )
        )

        # add contact sensors for robotA
        self.scene.contact_sensor_robot_A = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotA/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            # filter_prim_paths_expr=["/World/defaultGroundPlane"] # filter ground contact, only detect collisions with other robots
        )
        
        self.scene.contact_sensor_robot_B = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotB/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            # filter_prim_paths_expr=["/World/defaultGroundPlane"],
        )

        self.scene.contact_sensor_robot_A_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotA/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotB/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotB/right_wrist_yaw_link"],
        )
        self.scene.contact_sensor_robot_B_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotB/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotA/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotA/right_wrist_yaw_link"],
        )



    def customize_cfg(self):
        """Customize robots, contact sensors and space definition
        """
        self.observations: Dict[AgentID, combat_env_cfg.ObservationsCfg] = {
            agent: combat_env_cfg.ObservationsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.actions: Dict[AgentID, combat_env_cfg.ActionsCfg] = {
            agent: combat_env_cfg.ActionsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.terminations: Dict[AgentID, combat_env_cfg.TerminationsCfg] = {
            agent: combat_env_cfg.TerminationsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.rewards: Dict[AgentID, combat_env_cfg.RewardsCfg] = {
            agent: combat_env_cfg.RewardsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.event = {agent: combat_env_cfg.EventsCfg(ego_agent=agent) for agent in self.possible_agents}
        self.commands = None  # remove commands to disable command manager
        self.actions["robot_A"].joint_pos.scale = G1_ACTION_SCALE
        self.actions["robot_B"].joint_pos.scale = G1_ACTION_SCALE


@configclass
class G1vsSandbagEnvCfg(combat_env_cfg.EnvCfg):
    """G1 vs 'sandbag' environment.

    robot_A: full combat rewards (attacker).
    robot_B: passive sandbag, only regularization rewards (balance / joint_limit / action_rate_l2).
    """

    def set_robots(self):
        # Reuse the same robot placement as G1vsG1EnvCfg.
        self.scene.robot_A = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotA",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.76),
                rot=(1, 0, 0, 0),
            ),
        )
        self.scene.robot_B = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotB",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(1.2, 0.0, 0.76),
                rot=(0, 0, 0, -1),
            ),
        )

        self.scene.contact_sensor_robot_A = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotA/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_B = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotB/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_A_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotA/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotB/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotB/right_wrist_yaw_link"],
        )
        self.scene.contact_sensor_robot_B_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotB/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotA/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotA/right_wrist_yaw_link"],
        )


    def customize_cfg(self):
        """Customize robots, contact sensors and spaces for attacker vs sandbag."""
        self.observations = {
            agent: combat_env_cfg.ObservationsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.actions = {
            agent: combat_env_cfg.ActionsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.terminations = {
            agent: combat_env_cfg.TerminationsCfg(ego_agent=agent) for agent in self.possible_agents
        }

        # Rewards: robot_A uses full combat rewards, robot_B acts as sandbag.
        self.rewards = {
            "robot_A": combat_env_cfg.RewardsCfg(ego_agent="robot_A"),
            "robot_B": combat_env_cfg.SandbagRewardsCfg(ego_agent="robot_B"),
        }

        self.event = {agent: combat_env_cfg.EventsCfg(ego_agent=agent) for agent in self.possible_agents}
        self.commands = None  # disable command manager

        # Action scales
        self.actions["robot_A"].joint_pos.scale = G1_ACTION_SCALE
        self.actions["robot_B"].joint_pos.scale = G1_ACTION_SCALE


@configclass
class G1vsSandbagEnvCfg_SMPLREW(combat_env_cfg.EnvCfg):
    """G1 vs 'sandbag' environment.

    robot_A: full combat rewards (attacker).
    robot_B: passive sandbag, only regularization rewards (balance / joint_limit / action_rate_l2).
    """

    def set_robots(self):
        # Reuse the same robot placement as G1vsG1EnvCfg.
        self.scene.robot_A = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotA",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.76),
                rot=(1, 0, 0, 0),
            ),
        )
        self.scene.robot_B = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotB",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(1.2, 0.0, 0.76),
                rot=(0, 0, 0, -1),
            ),
        )

        self.scene.contact_sensor_robot_A = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotA/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_B = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotB/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_A_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotA/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotB/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotB/right_wrist_yaw_link"],
        )
        self.scene.contact_sensor_robot_B_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotB/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotA/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotA/right_wrist_yaw_link"],
        )


    def customize_cfg(self):
        """Customize robots, contact sensors and spaces for attacker vs sandbag."""
        self.observations = {
            agent: combat_env_cfg.ObservationsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.actions = {
            agent: combat_env_cfg.ActionsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.terminations = {
            agent: combat_env_cfg.TerminationsCfg(ego_agent=agent) for agent in self.possible_agents
        }

        # Rewards: robot_A uses full combat rewards, robot_B acts as sandbag.
        self.rewards = {
            "robot_A": combat_env_cfg.SMPLOLYMPICRewardsCfg(ego_agent="robot_A"),
            "robot_B": combat_env_cfg.SandbagRewardsCfg(ego_agent="robot_B"),
        }

        self.event = {agent: combat_env_cfg.EventsCfg(ego_agent=agent) for agent in self.possible_agents}
        self.commands = None  # disable command manager

        # Action scales
        self.actions["robot_A"].joint_pos.scale = G1_ACTION_SCALE
        self.actions["robot_B"].joint_pos.scale = G1_ACTION_SCALE

@configclass
class G1vsG1EnvCfg_SMPLREW(combat_env_cfg.EnvCfg):
    """G1 vs G1 environment.

    robot_A: full combat rewards (attacker).
    robot_B: full combat rewards (attacker).
    """

    def set_robots(self):
        # Reuse the same robot placement as G1vsG1EnvCfg.
        self.scene.robot_A = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotA",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.76),
                rot=(1, 0, 0, 0),
            ),
        )
        self.scene.robot_B = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotB",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(2, 0.0, 0.76),
                rot=(0, 0, 0, -1),
            ),
        )

        self.scene.contact_sensor_robot_A = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotA/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_B = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotB/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_A_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotA/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotB/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotB/right_wrist_yaw_link"],
        )
        self.scene.contact_sensor_robot_B_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotB/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotA/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotA/right_wrist_yaw_link"],
        )


    def customize_cfg(self):
        """Customize robots, contact sensors and spaces for attacker vs sandbag."""
        self.observations = {
            agent: combat_env_cfg.ObservationsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.actions = {
            agent: combat_env_cfg.ActionsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.terminations = {
            agent: combat_env_cfg.TerminationsCfg(ego_agent=agent) for agent in self.possible_agents
        }

        # Rewards: robot_A uses full combat rewards, robot_B acts as sandbag.
        self.rewards = {
            "robot_A": combat_env_cfg.SMPLOLYMPICRewardsCfg(ego_agent="robot_A"),
            "robot_B": combat_env_cfg.SMPLOLYMPICRewardsCfg(ego_agent="robot_B"),
        }

        self.event = {agent: combat_env_cfg.EventsCfg(ego_agent=agent) for agent in self.possible_agents}
        self.commands = None  # disable command manager

        # Action scales
        self.actions["robot_A"].joint_pos.scale = G1_ACTION_SCALE
        self.actions["robot_B"].joint_pos.scale = G1_ACTION_SCALE

@configclass
class G1vsSandbagEnvCfg_KO(combat_env_cfg.EnvCfg):
    """G1 vs 'sandbag' environment.

    robot_A: full combat rewards (attacker).
    robot_B: passive sandbag, only regularization rewards (balance / joint_limit / action_rate_l2).
    """

    def set_robots(self):
        # Reuse the same robot placement as G1vsG1EnvCfg.
        self.scene.robot_A = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotA",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.76),
                rot=(1, 0, 0, 0),
            ),
        )
        self.scene.robot_B = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotB",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(1.2, 0.0, 0.76),
                rot=(0, 0, 0, -1),
            ),
        )

        self.scene.contact_sensor_robot_A = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotA/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_B = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotB/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_A_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotA/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotB/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotB/right_wrist_yaw_link"],
        )
        self.scene.contact_sensor_robot_B_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotB/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotA/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotA/right_wrist_yaw_link"],
        )


    def customize_cfg(self):
        """Customize robots, contact sensors and spaces for attacker vs sandbag."""
        self.observations = {
            agent: combat_env_cfg.ObservationsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.actions = {
            agent: combat_env_cfg.ActionsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.terminations = {
            agent: combat_env_cfg.TerminationsCfg(ego_agent=agent) for agent in self.possible_agents
        }

        # Rewards: robot_A uses full combat rewards, robot_B acts as sandbag.
        self.rewards = {
            "robot_A": combat_env_cfg.KORewardsCfg(ego_agent="robot_A"),
            "robot_B": combat_env_cfg.SandbagRewardsCfg(ego_agent="robot_B"),
        }

        self.event = {agent: combat_env_cfg.EventsCfg(ego_agent=agent) for agent in self.possible_agents}
        self.commands = None  # disable command manager

        # Action scales
        self.actions["robot_A"].joint_pos.scale = G1_ACTION_SCALE
        self.actions["robot_B"].joint_pos.scale = G1_ACTION_SCALE

@configclass
class G1vsG1EnvCfg_SMPLREW_BoxingRing(combat_env_cfg.EnvCfg):
    """G1 vs G1 environment with boxing ring.
    
    Features:
    - Both robots use SMPL Olympic rewards
    - Boxing ring with physical walls (5m x 5m arena)
    - Robots start facing each other at opposite ends
    """
    
    # Override scene config to increase env_spacing for boxing ring (5m x 5m)
    # Need at least 5.5m spacing to avoid overlap, use 6m for safety margin
    scene: combat_env_cfg.SceneCfg = combat_env_cfg.SceneCfg(num_envs=4096, env_spacing=6.0)

    def set_robots(self):
        from whole_body_tracking.assets.boxing_ring import BOXING_RING_CFG
        
        # Add boxing ring to the scene
        self.scene.boxing_ring = BOXING_RING_CFG
        
        # Robot positions: face to face, 2m apart
        # robot_A at x=-1, facing +x (towards robot_B)
        # robot_B at x=+1, facing -x (towards robot_A)
        # Distance: 1 - (-1) = 2m
        self.scene.robot_A = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotA",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(-1.0, 0.0, 0.76),
                rot=(1, 0, 0, 0),  # Default orientation (facing +x)
            ),
        )
        
        self.scene.robot_B = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotB",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(1.0, 0.0, 0.76),
                rot=(0, 0, 0, -1),  # Rotate 180 degrees around z-axis (facing -x)
            ),
        )

        self.scene.contact_sensor_robot_A = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotA/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_B = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotB/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_A_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotA/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotB/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotB/right_wrist_yaw_link"],
        )
        self.scene.contact_sensor_robot_B_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotB/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotA/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotA/right_wrist_yaw_link"],
        )


    def customize_cfg(self):
        """Customize robots, contact sensors and reward configurations."""
        self.observations = {
            agent: combat_env_cfg.ObservationsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.actions = {
            agent: combat_env_cfg.ActionsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.terminations = {
            agent: combat_env_cfg.TerminationsCfg(ego_agent=agent) for agent in self.possible_agents
        }

        # Both robots use SMPL Olympic rewards
        self.rewards = {
            "robot_A": combat_env_cfg.SMPLOLYMPICRewardsCfg(ego_agent="robot_A"),
            "robot_B": combat_env_cfg.SMPLOLYMPICRewardsCfg(ego_agent="robot_B"),
        }

        self.event = {agent: combat_env_cfg.EventsCfg(ego_agent=agent) for agent in self.possible_agents}
        self.commands = None  # disable command manager

        # Action scales
        self.actions["robot_A"].joint_pos.scale = G1_ACTION_SCALE
        self.actions["robot_B"].joint_pos.scale = G1_ACTION_SCALE


@configclass
class G1vsSandbagEnvCfg_SMPLREW_BoxingRing(combat_env_cfg.EnvCfg):
    """G1 vs Sandbag environment with boxing ring.
    
    Features:
    - Robot A uses SMPL Olympic rewards (attacker)
    - Robot B is sandbag (defender with limited rewards)
    - Boxing ring with physical walls (5m x 5m arena)
    """
    
    # Override scene config to increase env_spacing for boxing ring (5m x 5m)
    # Need at least 5.5m spacing to avoid overlap, use 6m for safety margin
    scene: combat_env_cfg.SceneCfg = combat_env_cfg.SceneCfg(num_envs=4096, env_spacing=6.0)

    def set_robots(self):
        from whole_body_tracking.assets.boxing_ring import BOXING_RING_CFG
        
        # Add boxing ring to the scene
        self.scene.boxing_ring = BOXING_RING_CFG
        
        # Robot positions: face to face, 2m apart
        # robot_A at x=-1, facing +x (towards robot_B)
        # robot_B at x=+1, facing -x (towards robot_A)
        # Distance: 1 - (-1) = 2m
        self.scene.robot_A = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotA",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(-1.0, 0.0, 0.76),
                rot=(1, 0, 0, 0),  # Default orientation (facing +x)
            ),
        )
        
        # Robot B (sandbag) at opposite position
        self.scene.robot_B = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/robotB",
            init_state=G1_CYLINDER_CFG.init_state.replace(
                pos=(1.0, 0.0, 0.76),
                rot=(0, 0, 0, -1),  # Rotate 180 degrees around z-axis (facing -x)
            ),
        )

        self.scene.contact_sensor_robot_A = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotA/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_B = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robotB/.*",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
        )

        self.scene.contact_sensor_robot_A_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotA/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotB/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotB/right_wrist_yaw_link"],
        )
        self.scene.contact_sensor_robot_B_torso = ContactSensorCfg(
            prim_path = "{ENV_REGEX_NS}/robotB/.*torso_link",
            history_length=6,
            track_air_time=True,
            debug_vis=True,
            force_threshold=10.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/robotA/left_wrist_yaw_link", "{ENV_REGEX_NS}/robotA/right_wrist_yaw_link"],
        )


    def customize_cfg(self):
        """Customize for attacker vs sandbag in boxing ring."""
        self.observations = {
            agent: combat_env_cfg.ObservationsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.actions = {
            agent: combat_env_cfg.ActionsCfg(ego_agent=agent) for agent in self.possible_agents
        }
        self.terminations = {
            agent: combat_env_cfg.TerminationsCfg(ego_agent=agent) for agent in self.possible_agents
        }

        # Robot A attacks, Robot B is sandbag
        self.rewards = {
            "robot_A": combat_env_cfg.SMPLOLYMPICRewardsCfg(ego_agent="robot_A"),
            "robot_B": combat_env_cfg.SandbagRewardsCfg(ego_agent="robot_B"),
        }

        self.event = {agent: combat_env_cfg.EventsCfg(ego_agent=agent) for agent in self.possible_agents}
        self.commands = None  # disable command manager

        # Action scales
        self.actions["robot_A"].joint_pos.scale = G1_ACTION_SCALE
        self.actions["robot_B"].joint_pos.scale = G1_ACTION_SCALE