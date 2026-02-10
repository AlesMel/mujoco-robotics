"""Config-first environment schema for manager-based reach."""
from __future__ import annotations

from dataclasses import dataclass, field

from .mdp import ReachMDPCfg, ReachRewardCfg


@dataclass
class SceneCfg:
    """Scene-level settings."""

    robot: str = "ur3e"
    render_mode: str | None = None
    render_size: tuple[int, int] = (960, 720)


@dataclass
class EpisodeCfg:
    """Episode and reproducibility settings."""

    time_limit: int | None = None
    seed: int | None = None


@dataclass
class ActionCfg:
    """Control-space settings."""

    control_variant: str = "joint_pos"
    ee_step: float = 0.06
    ori_step: float = 0.5
    ori_abs_max: float = 3.141592653589793
    joint_action_scale: float = 0.5
    # Optional smoothing for relative joint targets (0=no update, 1=no smoothing).
    joint_target_ema_alpha: float = 1.0


@dataclass
class CommandCfg:
    """Goal command sampling settings."""

    goal_resample_time_range_s: tuple[float, float] | None = None


@dataclass
class SuccessCfg:
    """Goal thresholds and hold/resampling policy."""

    reach_threshold: float = 0.001
    ori_threshold: float = 0.01
    terminate_on_success: bool = False
    terminate_on_collision: bool = False
    success_hold_steps: int = 10
    success_bonus: float = 0.25
    stay_reward_weight: float = 0.05
    resample_on_success: bool = False


@dataclass
class RandomizationCfg:
    """Initial state randomization settings."""

    randomize_init: bool = True
    init_q_range: tuple[float, float] = (0.75, 1.25)


@dataclass
class PhysicsCfg:
    """Physics and actuator tuning."""

    actuator_kp: float = 250.0
    min_joint_damping: float = 20.0
    min_joint_frictionloss: float = 1.0
    obs_noise: float = 0.01
    action_rate_weight: float = 0.0001
    joint_vel_weight: float = 0.0001


@dataclass
class ManagerCfg:
    """Manager overrides."""

    mdp_cfg: ReachMDPCfg | None = None
    reward_cfg: ReachRewardCfg | None = None


@dataclass
class ReachEnvCfg:
    """Top-level config for manager-based reach environment."""

    scene: SceneCfg = field(default_factory=SceneCfg)
    episode: EpisodeCfg = field(default_factory=EpisodeCfg)
    actions: ActionCfg = field(default_factory=ActionCfg)
    commands: CommandCfg = field(default_factory=CommandCfg)
    success: SuccessCfg = field(default_factory=SuccessCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    managers: ManagerCfg = field(default_factory=ManagerCfg)

    def to_legacy_kwargs(self) -> dict:
        """Convert cfg to kwargs accepted by current reach env constructors."""
        kwargs: dict = {
            "ee_step": self.actions.ee_step,
            "ori_step": self.actions.ori_step,
            "ori_abs_max": self.actions.ori_abs_max,
            "joint_action_scale": self.actions.joint_action_scale,
            "reach_threshold": self.success.reach_threshold,
            "ori_threshold": self.success.ori_threshold,
            "terminate_on_success": self.success.terminate_on_success,
            "terminate_on_collision": self.success.terminate_on_collision,
            "success_hold_steps": self.success.success_hold_steps,
            "success_bonus": self.success.success_bonus,
            "stay_reward_weight": self.success.stay_reward_weight,
            "resample_on_success": self.success.resample_on_success,
            "obs_noise": self.physics.obs_noise,
            "action_rate_weight": self.physics.action_rate_weight,
            "joint_vel_weight": self.physics.joint_vel_weight,
            "randomize_init": self.randomization.randomize_init,
            "init_q_range": self.randomization.init_q_range,
            "actuator_kp": self.physics.actuator_kp,
            "min_joint_damping": self.physics.min_joint_damping,
            "min_joint_frictionloss": self.physics.min_joint_frictionloss,
            "render_size": self.scene.render_size,
        }

        if self.episode.time_limit is not None:
            kwargs["time_limit"] = self.episode.time_limit
        if self.episode.seed is not None:
            kwargs["seed"] = self.episode.seed

        if self.commands.goal_resample_time_range_s is not None:
            kwargs["goal_resample_time_range_s"] = self.commands.goal_resample_time_range_s

        if self.managers.mdp_cfg is not None:
            kwargs["mdp_cfg"] = self.managers.mdp_cfg
        if self.managers.reward_cfg is not None:
            kwargs["reward_cfg"] = self.managers.reward_cfg

        return kwargs
