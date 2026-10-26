from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from dataclasses import MISSING
from typing import Literal
from isaaclab_rl.rsl_rl.rnd_cfg import RslRlRndCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
from isaaclab_rl.rsl_rl.distillation_cfg import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg

VQ_STAGE2_PATH_64 = "logs/rsl_rl/g1_flat_stage2_latent_vq_noprior/2025-12-09_09-20-50_Stage2_noprior_vq_64/model_6000.pt"
VQ_STAGE2_PATH_128 = "logs/rsl_rl/g1_flat_stage2_latent_vq_noprior/2025-12-09_09-20-05_Stage2_noprior_vq_128/model_6000.pt"

@configclass
class RslRlPpoActorCriticStage3Cfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""
    
    taskpolicy_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the task policy network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""
    
    prior_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the prior network."""
    
    decoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the decoder network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
    latent_dim: int = MISSING
    """The dimension of the latent variable."""
    prior_history_length: int = MISSING
    """The history length for the prior input."""
    
@configclass
class RslRlPpoAlgorithmStage3Cfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""
    
    kl_constraint: bool = MISSING
    """Whether to use KL constraint."""
    
    kl_constraint_coef: float = MISSING
    """The coefficient for the KL constraint."""

    normalize_z: bool = MISSING
    """Whether to normalize the latent variable z."""

    normalize_advantage_per_mini_batch: bool = MISSING
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the entire collected trajectories.
    Otherwise, the advantage is normalized over the mini-batches only.
    """

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the entire collected trajectories.
    Otherwise, the advantage is normalized over the mini-batches only.
    """

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: RslRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """
    amp_cfg: dict | None = None
    """The configuration for AMP (Adversarial Motion Priors). Default is None,
    in which case AMP is not used.
    """
    
@configclass
class RslRlOnPolicyRunnerStage3Cfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticStage3Cfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmStage3Cfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
    
@configclass
class G1FlatPPORunnerStage3Cfg(RslRlOnPolicyRunnerStage3Cfg):
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 500
    experiment_name = "g1_flat_rsl_rl_ppo_stage3"
    empirical_normalization = True
    # stage2_ckpt_path = "logs/rsl_rl/g1_flat_stage2_latent/stage2_20251227_03/model_18000.pt"
    stage2_ckpt_path = "logs/rsl_marl/g1_flat_rsl_rl_ppo_stage3/stage3_sandbag_20260108_02/model_18000.pt"
    policy = RslRlPpoActorCriticStage3Cfg(
        class_name="ActorCriticStage3",
        init_noise_std=1.0,
        taskpolicy_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        prior_hidden_dims=[512, 256, 128],
        decoder_hidden_dims=[512, 256, 128],
        noise_std_type="log",
        activation="elu",
        latent_dim=32,
        prior_history_length=10,
    )
    
    algorithm = RslRlPpoAlgorithmStage3Cfg(
        class_name="PPOStage3",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        kl_constraint=False,
        kl_constraint_coef=0.001,
        normalize_z=True,
        # AMP configuration.
        amp_cfg={
            "enable_amp": True,
            "disc_learning_rate": 5e-5,
            "disc_weight_decay": 0.0001,
            "disc_max_grad_norm": 50.0,
            "disc_coef": 5.0,
            "disc_logit_reg": 0.01,
            "disc_grad_penalty": 5.0,
            "disc_reward_scale": 0.1, 
            "amp_batch_size": 512,
            "amp_replay_buffer_size": 100000,
            "discriminator_hidden_dims": [1024, 1024, 512],
            "discriminator_activation": "relu",
            # Reward scheduling configuration.
            "task_reward_w_initial": 0.0,  # Early phase: only use disc_reward.
            "disc_reward_w_initial": 1.0,
            "task_reward_w_final": 0.8,    # Late phase: 0.7 task + 0.3 disc.
            "disc_reward_w_final": 0.2,
            "reward_schedule_epoch": 0, # Switch to mixed reward after 500 epochs.
            # Motion file path.
            "motion_file": "source/whole_body_tracking/whole_body_tracking/motions/boxing_20251226",
        }
    ) 

@configclass
class G1FlatPPORunnerStage3FSQCfg(G1FlatPPORunnerStage3Cfg):
    """Stage3 runner that loads a Stage2 FSQ-with-prior checkpoint."""
    experiment_name = "g1_flat_rsl_rl_ppo_stage3_fsq"
    # NOTE: Please point this to the checkpoint produced by
    # G1FlatDistillationStage2LatentFSQRunnerCfg (with use_prior=True).
    stage2_ckpt_path = "logs/rsl_rl/g1_flat_stage2_latent_fsq/model_32000.pt"


@configclass
class G1FlatPPORunnerStage3VQCfg(RslRlOnPolicyRunnerStage3Cfg):
    """Stage3 runner that loads a Stage2 VQ-VAE (no-prior) checkpoint and uses VQStage3."""

    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 100
    experiment_name = "g1_flat_rsl_rl_ppo_stage3_vq"
    empirical_normalization = True
    # NOTE: Please point this to the checkpoint produced by
    # G1FlatDistillationStage2LatentVQNoPriorRunnerCfg.
    stage2_ckpt_path = VQ_STAGE2_PATH_128

    policy = RslRlPpoActorCriticStage3Cfg(
        class_name="ActorCriticStage3VQ",
        init_noise_std=1.0,
        taskpolicy_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        prior_hidden_dims=[512, 256, 128],
        decoder_hidden_dims=[512, 256, 128],
        noise_std_type="log",
        activation="elu",
        latent_dim=32,
        prior_history_length=10,
    )

    algorithm = RslRlPpoAlgorithmStage3Cfg(
        class_name="PPOVQStage3",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        kl_constraint=False,
        kl_constraint_coef=0.001,
    )

@configclass
class G1FlatPPORunnerStage3VQIndexCfg(RslRlOnPolicyRunnerStage3Cfg):
    """Stage3 runner that loads a Stage2 VQ-VAE (no-prior) checkpoint and uses VQStage3."""

    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 100
    experiment_name = "g1_flat_rsl_rl_ppo_stage3_vqindex"
    empirical_normalization = True
    # NOTE: Please point this to the checkpoint produced by
    # G1FlatDistillationStage2LatentVQNoPriorRunnerCfg.
    stage2_ckpt_path = VQ_STAGE2_PATH_128

    policy = RslRlPpoActorCriticStage3Cfg(
        class_name="ActorCriticStage3VQIndex",
        init_noise_std=1.0,
        taskpolicy_hidden_dims=[512, 256, 512, 256],
        critic_hidden_dims=[512, 256, 128],
        prior_hidden_dims=[512, 256, 128],
        decoder_hidden_dims=[512, 256, 128],
        noise_std_type="log",
        activation="elu",
        latent_dim=32,
        prior_history_length=10,
    )

    algorithm = RslRlPpoAlgorithmStage3Cfg(
        class_name="PPOVQIndexStage3",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        kl_constraint=False,
        kl_constraint_coef=0.001,
    )
