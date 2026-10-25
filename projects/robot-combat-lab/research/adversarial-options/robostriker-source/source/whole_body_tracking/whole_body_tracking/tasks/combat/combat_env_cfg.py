from typing import List

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.configclass import MISSING
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

import whole_body_tracking.tasks.combat.mdp as mdp


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

@configclass
class SceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    #ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    robot_A: ArticulationCfg = MISSING

    robot_B: ArticulationCfg = MISSING

    contact_sensor_robot_A: ContactSensorCfg = MISSING
    """Contact sensors for robot_A"""
    
    contact_sensor_robot_B: ContactSensorCfg = MISSING
    """Contact sensors for robot_B"""
    
    contact_sensor_robot_A_torso: ContactSensorCfg = MISSING
    contact_sensor_robot_B_torso: ContactSensorCfg = MISSING
    """Only Torso Sensor: One Body, Can Use Filter For 'Many-to-one' """

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


from isaaclab.managers import ObservationTermCfg as ObsTerm

from whole_body_tracking.tasks.tracking import tracking_env_cfg


@configclass
class ObservationsCfg:

    ego_agent: str = MISSING
    @configclass
    class DecoderCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.custom_last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class PriorCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.custom_last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.history_length = 10
            self.flatten_history_dim = False
    
    @configclass
    class CriticCfg(ObsGroup):
        body_pos = ObsTerm(func=mdp.robot_body_pos_b)
        body_ori = ObsTerm(func=mdp.robot_body_ori_b)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.custom_last_action)
        # opponent_body_pos_w = ObsTerm(func=mdp.opponent_pos, noise=Unoise(n_min=-0.2, n_max=0.2))
        # opponent_body_vel_w = ObsTerm(func=mdp.opponent_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        punch_target = ObsTerm(func=mdp.punch_target, noise=Unoise(n_min=-0.2, n_max=0.2))

        # TODO add task obs
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
        
    @configclass
    class TaskPolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.custom_last_action)

        # base_ang_vel_opponent = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # joint_pos_opponent = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel_opponent = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        # actions_opponent = ObsTerm(func=mdp.custom_last_action)
        
        # # TODO add task obs
        # task_obs_pos = ObsTerm(func=mdp.task_obs_pos, noise=Unoise(n_min=-0.2, n_max=0.2))
        # task_obs_vel = ObsTerm(func=mdp.task_obs_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        body_pos_w = ObsTerm(func=mdp.body_pos, noise=Unoise(n_min=-0.2, n_max=0.2))
        # body_vel_w = ObsTerm(func=mdp.body_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # opponent_body_pos_w = ObsTerm(func=mdp.opponent_pos, noise=Unoise(n_min=-0.2, n_max=0.2))
        # opponent_body_vel_w = ObsTerm(func=mdp.opponent_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        punch_target = ObsTerm(func=mdp.punch_target, noise=Unoise(n_min=-0.2, n_max=0.2))
        opponent_punch_target = ObsTerm(func=mdp.opponent_punch_target, noise=Unoise(n_min=-0.2, n_max=0.2))
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class AMPCfg(ObsGroup):
        """AMP (Adversarial Motion Priors) observations for discriminator.
        Same as decoder observations but with history for temporal information.
        """
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False  # No noise for AMP
            self.history_length = 2  # 2-step history for temporal context
            self.flatten_history_dim = False  # Keep history dimension separate
            self.concatenate_terms = True
    
    decoder: DecoderCfg = DecoderCfg()
    prior: PriorCfg = PriorCfg()
    critic: CriticCfg = CriticCfg()
    task_policy: TaskPolicyCfg = TaskPolicyCfg()
    amp: AMPCfg = AMPCfg()

    def __post_init__(self):
        if self.ego_agent == "robot_A": #FIXME HardCode now
            opponent_agent = "robot_B"
        else:
            opponent_agent = "robot_A"
        for v in self.decoder.__dict__.values():
            if isinstance(v, ObsTerm):
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
        for v in self.prior.__dict__.values():
            if isinstance(v, ObsTerm):
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
        for v in self.amp.__dict__.values():
            if isinstance(v, ObsTerm):
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
        for k, v in self.critic.__dict__.items():
            if isinstance(v, ObsTerm):
                if k.endswith("_opponent"):
                    v.params['asset_cfg'] = SceneEntityCfg(opponent_agent)
                else:
                    v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
        for k, v in self.task_policy.__dict__.items():
            if isinstance(v, ObsTerm):
                if k.endswith("_opponent"):
                    v.params['asset_cfg'] = SceneEntityCfg(opponent_agent)
                else:
                    v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
        del self.ego_agent


@configclass
class ActionsCfg(tracking_env_cfg.ActionsCfg):# TODO Check
    """Action configuration"""

    ego_agent: str = MISSING

    def __post_init__(self):
        self.joint_pos.asset_name = self.ego_agent
        del self.ego_agent


from isaaclab.managers import RewardTermCfg

@configclass
class RewardsCfg:

    ego_agent: str = MISSING
    
    """ID for ego agent"""

    # action_rate_l2 = RewTerm(func=mdp.custom_action_rate_l2, 
    #                          weight=-1e-1,
    #                          params={"asset_cfg": SceneEntityCfg("robot")}
    # )# TODO custom, assetcfg
    
    # joint_limit = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-10.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    # )
    
    # undesired_contacts = RewTerm( # TODO remove it
    #     func=mdp.undesired_contacts,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[
    #                 r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    #             ],
    #         ),
    #         "threshold": 1.0,
    #     },
    # )
    
    offensive_hit_reward_punch = RewTerm( # TODO 
        func=mdp.offensive_hit_reward_punch,
        weight=5.0,
        params={
            "sensor_cfg1": SceneEntityCfg(
                "contact_sensor_robot_A", 
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "sensor_cfg2": SceneEntityCfg(
                "contact_sensor_robot_B", 
                body_names=[
                    r"^(?!right_hip_roll_link$)(?!left_hip_roll_link$)(?!torso_link$).+$"
                ],
            ), 
            "threshold": 1.0,
            "vel_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "relative_by_ego": True,
        },
    )
    offensive_hit_reward_punch_2d = RewTerm( # TODO 
        func=mdp.offensive_hit_reward_punch_2d,
        weight=5.0,
        params={
            "sensor_cfg1": SceneEntityCfg(
                "contact_sensor_robot_A", 
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "sensor_cfg2": SceneEntityCfg(
                "contact_sensor_robot_B", 
                body_names=[
                    r"^(?!right_hip_roll_link$)(?!left_hip_roll_link$)(?!torso_link$).+$"
                ],
            ), 
            "threshold": 1.0,
            "vel_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "relative_by_ego": True,
        },
    )
    # offensive_hit_reward_kick = RewTerm(
    #     func=mdp.offensive_hit_reward_kick,
    #     weight=10.0,
    #     params={
    #         "sensor_cfg1": SceneEntityCfg(
    #             "contact_sensor_robot_A", 
    #             body_names=[
    #                 r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    #             ],
    #         ),
    #         "sensor_cfg2": SceneEntityCfg(
    #             "contact_sensor_robot_B", 
    #             body_names=[
    #                 r"^(?!right_hip_roll_link$)(?!left_hip_roll_link$)(?!torso_link$).+$"
    #             ],
    #         ), 
    #         "threshold": 10.0,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
    
    # defensive_hit_penalty = RewTerm( # TODO 
    #     func=mdp.defensive_hit_penalty,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg1": SceneEntityCfg(
    #             "contact_sensor_robot_A", 
    #             body_names=[
    #                 r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    #             ],
    #         ),
    #         "sensor_cfg2": SceneEntityCfg(
    #             "contact_sensor_robot_B", 
    #             body_names=[
    #                 r"^(?!right_hip_roll_link$)(?!left_hip_roll_link$)(?!torso_link$).+$"
    #             ],
    #         ), 
    #         "threshold": 1.0,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
    
    # offensive_initiative = RewTerm(
    #     func=mdp.offensive_initiative,
    #     weight=1.0,
    #     params={
    #         "std": 0.2,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     }
    # )
    
    # target_dist = RewTerm( # TODO
    #     func=mdp.target_dist,
    #     weight=1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "std": 0.3,
    #         "min_dist": 0.5,
    #     }
    # )
    target_dist = RewTerm( # NOTE fist to torso instead of torso to torso
        func=mdp.punch_proximity_reward,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_body_names": ["torso_link"],
            "scale": 10,
        }
    )

    # target_dist = RewTerm(
    #     func=mdp.distance_penalty,
    #     weight=0.25,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "max_dist": 0.75,
    #         "min_dist": 0.25,
    #         "scale": 0.1,
    #     }
    # )
    
    glove_distance = RewTerm(
        func=mdp.glove_distance_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.8,
        }
    )
    
    # dodge_success = RewTerm(
    #     func=mdp.dodge_success_reward,
    #     weight=0.0,
    #     params={
    #         "sensor_cfg1": SceneEntityCfg(
    #             "contact_sensor_robot_A", 
    #             # body_names=[
    #             #     r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    #             # ],
    #             body_names=["torso_link"],
    #         ),

    #         "sensor_cfg2": SceneEntityCfg(
    #             "contact_sensor_robot_B", 
    #             body_names=[
    #                 r".*(wrist_yaw).*_link$"
    #             ],
    #         ), 
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "dist_threshold": 0.35,
    #         "vel_threshold": 1.0,
    #         "force_threshold": 10.0,
    #     }
    # )
    # punch_speed = RewTerm(
    #     func=mdp.punch_speed,
    #     weight=0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     }
    # )
    # dodge_success = RewTerm(
    #     func=mdp.dodge_success_reward,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg1": SceneEntityCfg(
    #             "contact_sensor_robot_A", 
    #             body_names=[
    #                 r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    #             ],
    #         ),
    #         "sensor_cfg2": SceneEntityCfg(
    #             "contact_sensor_robot_B", 
    #             body_names=[
    #                 r"^(?!right_hip_roll_link$)(?!left_hip_roll_link$)(?!torso_link$).+$"
    #             ],
    #         ), 
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "dist_threshold": 0.35,
    #         "vel_threshold": 1.0
    #     }
    # )
    face_opponent = RewTerm(
        func=mdp.face_opponent_reward,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "temperature": 3,
            "sigma": 0.3
        }
    )

    balance = RewTerm(
        func=mdp.balance_reward,
        weight=0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "upright_threshold": 0.7,
            "with_angular": False,
        }
    )
    velocity_towards_opponent = RewTerm(
        func=mdp.velocity_towards_opponent_reward,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    
    dense_punch = RewTerm(
        func=mdp.dense_punch_guide_reward,
        weight=0.00,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "vel_threshold": 10.0,
            "distance_scale": 1.5,
        }
    )
    # terminate = RewTerm(
    #     func=mdp.is_terminated,
    #     weight = -500,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )
    # is_alive = RewTerm(
    #     func=mdp.is_alive,
    #     weight= 0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )
    energy_saving_penalty = RewTerm(
        func=mdp.energy_saving_penalty,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "distance_threshold": 0.75,
                "std": 0.5,
                "joint_names": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",]
                }
    )
    # opp_is_terminated = RewTerm(
    #     func=mdp.opp_is_terminated,
    #     weight= 400.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # ) # TODO: THIS HAVE BUG.
    def __post_init__(self):
        if self.ego_agent == "robot_A": #FIXME HardCode now
            opponent_agent = "robot_B"
        else:
            opponent_agent = "robot_A"

        for k, v in self.__dict__.items():
            if not isinstance(v, RewardTermCfg):
                continue
            # TODO
            if k == 'offensive_hit_reward_punch':
                v.params['sensor_cfg1'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg1'].body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link"
                ]
                v.params['sensor_cfg2'].name = f"contact_sensor_{opponent_agent}"
                v.params['sensor_cfg2'].body_names=[
                    "right_hip_roll_link",
                    "left_hip_roll_link",
                    "torso_link"
                    "", 
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            
            elif k == 'offensive_hit_reward_kick':
                v.params['sensor_cfg1'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg1'].body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ]
                v.params['sensor_cfg2'].name = f"contact_sensor_{opponent_agent}"
                v.params['sensor_cfg2'].body_names=[
                    "torso_link"
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
                
            elif k == 'defensive_hit_penalty':
                v.params['sensor_cfg1'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg1'].body_names=[
                    "right_hip_roll_link",
                    "left_hip_roll_link",
                    "torso_link"
                ]
                v.params['sensor_cfg2'].name = f"contact_sensor_{opponent_agent}"
                v.params['sensor_cfg2'].body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link"
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            elif k == 'dodge_success_reward':
                v.params['sensor_cfg1'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg1'].body_names=["torso_link"]
                v.params['sensor_cfg2'].name = f"contact_sensor_{opponent_agent}"
                v.params['sensor_cfg2'].body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            else:
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)

        # self.joint_limit.params['asset_cfg'].joint_names = [".*"]
        del self.ego_agent

@configclass
class SandbagRewardsCfg:
    """Rewards configuration for a passive 'sandbag' agent.

    Only keeps basic physical regularization terms (smooth actions, joint limits, balance).
    """

    ego_agent: str = MISSING

    # action_rate_l2 = RewTerm(
    #     func=mdp.custom_action_rate_l2,
    #     weight=-1e-1,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # joint_limit = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-10.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    # )

    stand_still = RewTerm(
        func=mdp.stand_still_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    balance = RewTerm(
        func=mdp.balance_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "upright_threshold": 0.7,
            "with_angular": True,
        },
    )

    def __post_init__(self):
        # Bind all reward terms to the correct ego agent.
        for _, v in self.__dict__.items():
            if not isinstance(v, RewardTermCfg):
                continue
            v.params["asset_cfg"] = SceneEntityCfg(self.ego_agent)

        # Ensure joint limits apply to all joints.
        # self.joint_limit.params["asset_cfg"].joint_names = [".*"]
        del self.ego_agent

from isaaclab.managers import TerminationTermCfg

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    ego_agent: str = MISSING
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    falldone = DoneTerm(func=mdp.root_height_below_minimum, 
                        params={"asset_cfg": SceneEntityCfg("robot"),
                                "minimum_height":0.3}) # TODO more termination conditions?
    contact_time_out = DoneTerm(
        func=mdp.time_termination,
        params={"threshold": 5.0,
                "asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if k != 'time_out' and isinstance(v, TerminationTermCfg):
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
        del self.ego_agent

from isaaclab.managers import EventTermCfg

@configclass
class EventsCfg:
    """Configuration for events."""
    ego_agent: str = MISSING
    
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )
    maintain_contact = EventTerm(
        func=mdp.maintain_contact,
        mode="interval",
        interval_range_s=(0.5, 0.5),
        params={
            "sensor_cfg1": SceneEntityCfg(
                "contact_sensor_robot_A", 
                body_names=[r".*(wrist_yaw).*_link$"],
            ),
            "sensor_cfg2": SceneEntityCfg(
                "contact_sensor_robot_B", 
                body_names=[
                    "torso_link"
                ],
            ), 
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "check_time_interval": 0.5,
        },
    )
    reset_contact_time_buffer = EventTerm(
        func=mdp.reset_contact_time_buffer,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_punch_counts = EventTerm(
        func=mdp.reset_punch_counts,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        opponent_agent = "robot_B" if self.ego_agent == "robot_A" else "robot_A"
        for k, v in self.__dict__.items():
            if k == 'maintain_contact':
                v.params['sensor_cfg1'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg1'].body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link"
                ]
                v.params['sensor_cfg2'].name = f"contact_sensor_{opponent_agent}"
                v.params['sensor_cfg2'].body_names=[
                    "torso_link"
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            if isinstance(v, EventTermCfg):
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
        del self.ego_agent
    

from whole_body_tracking.envs.mix_marl_env import MixMarlEnvCfg

@configclass
class EnvCfg(MixMarlEnvCfg):

    """Initialize two robots for combating"""

    # Sceneraio configuration
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)

    # ============= env =============
    decimation: int = 4
    episode_length_s: float = 50.

    possible_agents: List[str] = []
    """A list of all possible agents the environment could generate.

    The contents of the list cannot be modified during the entire training process.
    """

    # ============ simulation ====================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    def set_robots(self):
        raise NotImplementedError

    def customize_cfg(self):
        """Customize agents/robots, key_links and contact bodies here"""
        raise NotImplementedError

    def __post_init__(self):
        # unset observation / state / action spaces
        del self.observation_spaces
        del self.action_spaces
        del self.state_space

        self.set_robots()

        possible_agents = []
        for k, v in self.scene.__dict__.items():
            if isinstance(v, ArticulationCfg):
                possible_agents.append(k)
        self.possible_agents = possible_agents

        assert len(possible_agents), f"No available robots registered"

        self.customize_cfg()

@configclass
class SMPLOLYMPICRewardsCfg:
    '''
        SMPL OLYMPIC's style Rewards configuration
        
    '''

    ego_agent: str = MISSING

    action_rate_l2 = RewTerm(func=mdp.custom_action_rate_l2, 
                             weight=-0.05,
                             params={"asset_cfg": SceneEntityCfg("robot")}
    )# TODO custom, assetcfg
    
    foot_slip_penalty = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_sensor_robot_A"),
            "threshold": 0.1,
        },
    )
    
    foot_lift_reward = RewTerm(
        func=mdp.foot_lift_reward,
        weight=0.,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_sensor_robot_A"),
            "contact_threshold": 0.5,
            "min_height": 0.1,
            "max_height": 0.3,
        },
    )
    
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    balance = RewTerm(
        func=mdp.balance_reward,
        weight=0.,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "upright_threshold": 0.7,
            "with_angular": False,
        },
    )

    facing_reward = RewTerm(
        func=mdp.facing_reward,
        weight=1.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_body_name": "torso_link",
            "sigma": 0.5,
        }
    )
    vel_reward = RewTerm(
        func = mdp.vel_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_body_name": "torso_link",
            "sigma": 0.25,
            "tar_speed": 1.0,
        }
    )
    strike_reward = RewTerm(
        func=mdp.strike_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg_ego": SceneEntityCfg(
                "contact_sensor_robot_A", 
                body_names=[
                    "torso_link",
                ],
            ), 
            "sensor_cfg_opponent": SceneEntityCfg(
                "contact_sensor_robot_B", 
                body_names=[
                    "torso_link",
                ],
            ), 
            "target_body_weights": [1.0], 
        }
    )
    # NOTE: In self-play, a pure distance-based hit shaping becomes "free reward"
    # when the opponent actively walks into you. Use an *active* shaping that is
    # gated by your own punching speed towards the opponent.
    hit_reward = RewTerm(
        func=mdp.hit_reward,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "hit_body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            "target_body_names": ["torso_link"],
            "sigma": 1.0,
        },
    )

    active_hit_shaping_reward = RewTerm(
        func=mdp.active_hit_shaping_reward,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "hit_body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            "target_body_names": ["torso_link"],
            "sigma": 1.0,
            "vel_threshold": 0.8,
            "gate_temperature": 10.0,
        },
    )
    terminate_reward = RewTerm(
        func=mdp.terminate_reward,
        weight = 0.3,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "minimum_height":0.4}
    )

    energy_saving_penalty = RewTerm(
        func=mdp.energy_saving_penalty,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "distance_threshold": 0.75,
                "std": 0.5,
                "joint_names": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",]
                }
    )
    
    offensive_hit_reward_punch = RewTerm( # TODO 
        func=mdp.offensive_hit_reward_punch,
        weight=50.0,
        params={
            "sensor_cfg1": SceneEntityCfg(
                "contact_sensor_robot_A", 
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "sensor_cfg2": SceneEntityCfg(
                "contact_sensor_robot_B", 
                body_names=[
                    r"^(?!right_hip_roll_link$)(?!left_hip_roll_link$)(?!torso_link$).+$"
                ],
            ), 
            "threshold": 1.0,
            "vel_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "relative_by_ego": True,
        },
    )

    complete_punch_classify_reward = RewTerm(
        func=mdp.complete_punch_classify_reward,
        weight=0.,
        params={
            "sensor_cfg_opp": SceneEntityCfg(
                "contact_sensor_robot_B_torso", 
            ),
            "sensor_cfg_ego": SceneEntityCfg(
                "contact_sensor_robot_A",
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "head_height_threshold": 0.4,
            "upward_punch_threshold": 0.3,
            "force_threshold": 1.0,
            "contact_dt": 0.1,
            "reward_weight_decay_coeff": 0.5,
            "head_weight": 2,
            "body_weight": 1,
        }
    )

    defensive_hit_penalty_punch = RewTerm(
        func=mdp.defensive_hit_penalty_punch,
        weight=-8.0,
        params={
            "sensor_cfg1": SceneEntityCfg(
                "contact_sensor_robot_A", 
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "sensor_cfg2": SceneEntityCfg(
                "contact_sensor_robot_B", 
                body_names=[
                    r"^(?!right_hip_roll_link$)(?!left_hip_roll_link$)(?!torso_link$).+$"
                ],
            ), 
            "threshold": 1.0,
            "vel_threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "relative_by_ego": True,
        },
    )

    complete_punch_reward = RewTerm(
        func=mdp.complete_punch_reward,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_steps": 40,
            "v_start_thresh": 0.75,
            "retract_ratio": 0.3,
            "dist_rescale": True,
            "max_dist": 1.0,
            "min_dist": 0.5,
            "scale": 0.05,
            "penalty_shift": 1.0,
        }
    )


    def __post_init__(self):
        if self.ego_agent == "robot_A": #FIXME HardCode now
            opponent_agent = "robot_B"
        else:
            opponent_agent = "robot_A"

        for k, v in self.__dict__.items():
            if not isinstance(v, RewardTermCfg):
                continue
            if k == 'offensive_hit_reward_punch':
                v.params['sensor_cfg1'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg1'].body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link"
                ]
                v.params['sensor_cfg2'].name = f"contact_sensor_{opponent_agent}"
                v.params['sensor_cfg2'].body_names=[
                    # "right_hip_roll_link",
                    # "left_hip_roll_link",
                    "torso_link",
                    # "pelvis",
                    # "right_shoulder_yaw_link",
                    # "left_shoulder_yaw_link"
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            
            elif k == "complete_punch_classify_reward":
                v.params['sensor_cfg_opp'].name = f"contact_sensor_{opponent_agent}_torso"
                v.params['sensor_cfg_ego'].name = f"contact_sensor_{self.ego_agent}"
                v.params["asset_cfg"] = SceneEntityCfg(self.ego_agent)

            elif k == 'offensive_hit_reward_punch_left' or k == 'offensive_hit_reward_punch_right':
                v.params['sensor_cfg1'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg1'].body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link"
                ]
                v.params['sensor_cfg2'].name = f"contact_sensor_{opponent_agent}"
                v.params['sensor_cfg2'].body_names=[
                    "right_hip_roll_link",
                    "left_hip_roll_link",
                    "torso_link"
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            elif k == 'defensive_hit_penalty_punch':
                v.params['sensor_cfg1'].name = f"contact_sensor_{opponent_agent}"
                v.params['sensor_cfg1'].body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link"
                ]
                v.params['sensor_cfg2'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg2'].body_names=[
                    "right_hip_roll_link",
                    "left_hip_roll_link",
                    "torso_link"
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            elif k == 'foot_slip_penalty':
                v.params['sensor_cfg'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg'].body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link"
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            elif k == 'foot_lift_reward':
                v.params['sensor_cfg'].name = f"contact_sensor_{self.ego_agent}"
                v.params['sensor_cfg'].body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link"
                ]
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            else:
                if 'asset_cfg' in v.params.keys():
                    v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
                if 'sensor_cfg_ego' in v.params.keys():
                    v.params['sensor_cfg_ego'].name = f"contact_sensor_{self.ego_agent}"
                if 'sensor_cfg_opponent' in v.params.keys():
                    v.params['sensor_cfg_opponent'].name = f"contact_sensor_{opponent_agent}"
            
        # self.joint_limit.params['asset_cfg'].joint_names = [".*"]
        del self.ego_agent

@configclass
class KORewardsCfg:
    '''
        KO Rewards configuration
        
    '''

    ego_agent: str = MISSING

    facing_reward = RewTerm(
        func=mdp.facing_reward,
        weight=0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_body_name": "torso_link",
            "sigma": 0.5,
        }
    )
    vel_reward = RewTerm(
        func = mdp.vel_reward,
        weight = 0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_body_name": "torso_link",
            "sigma": 0.25,
            "tar_speed": 1.0,
        }
    )
    terminate_reward = RewTerm(
        func=mdp.terminate_reward,
        weight = 1,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "minimum_height":0.3}
    )
    balance = RewTerm(
        func=mdp.balance_reward,
        weight=0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "upright_threshold": 0.7,
            "with_angular": True,
            "is_ego": True,
        },
    )

    balance = RewTerm(
        func=mdp.balance_reward,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "upright_threshold": 0.7,
            "with_angular": True,
            "is_ego": False,
        },
    )

    def __post_init__(self):
        if self.ego_agent == "robot_A": #FIXME HardCode now
            opponent_agent = "robot_B"
        else:
            opponent_agent = "robot_A"

        for k, v in self.__dict__.items():
            if not isinstance(v, RewardTermCfg):
                continue
            # TODO
            if 'asset_cfg' in v.params.keys():
                v.params['asset_cfg'] = SceneEntityCfg(self.ego_agent)
            if 'sensor_cfg_ego' in v.params.keys():
                v.params['sensor_cfg_ego'].name = f"contact_sensor_{self.ego_agent}"
            if 'sensor_cfg_opponent' in v.params.keys():
                v.params['sensor_cfg_opponent'].name = f"contact_sensor_{opponent_agent}"
            
        # self.joint_limit.params['asset_cfg'].joint_names = [".*"]
        del self.ego_agent