"""Joint-position reach variant with Isaac Lab reward terms.

This environment keeps the same action mapping as :class:`ReachJointPosEnv`
and only changes reward computation to match Isaac Lab's Franka joint-position
reach task reward terms exactly:

1. ``position_command_error_tanh`` with weight ``+0.1`` and std ``0.1``
2. ``orientation_command_error`` with weight ``-0.1``
3. ``action_rate_l2`` with weight ``-1e-4``
4. ``joint_vel_l2`` with weight ``-1e-4``

As in Isaac Lab's reward manager, the weighted sum is scaled by the control
step duration ``step_dt``.
"""
from __future__ import annotations

from mujoco_robot.envs.reach.mdp import ReachMDPCfg, RewardTermCfg, rewards
from mujoco_robot.envs.reach.reach_env_base import ReachGymnasiumBase
from mujoco_robot.envs.reach.reach_env_joint_pos import ReachJointPosEnv


ISAAC_CONTROL_DT_S = 0.005 * 4
ISAAC_EPISODE_SECONDS = 3.0
ISAAC_EPISODE_STEPS = int(round(ISAAC_EPISODE_SECONDS / ISAAC_CONTROL_DT_S))
ISAAC_GOAL_RESAMPLE_TIME_RANGE_S = (4.0, 4.0)
ISAAC_JOINT_ACTION_SCALE = 0.25
ISAAC_ACTION_DEADZONE = 0.0
ISAAC_EMA_ALPHA = 1.0


class ReachJointPosIsaacRewardEnv(ReachJointPosEnv):
    """Reach task with joint-position control and Isaac Lab reward.

    Defaults are configured to mirror common Isaac Lab reach settings:
    - episode length ~= 3.0 s
    - command resampling window = [4.0, 4.0] s
    - direct relative joint-position targets (no EMA smoothing)
    - no action deadzone
    - no task-based termination by default (timeout only)
    """

    DEFAULT_TIME_LIMIT = ISAAC_EPISODE_STEPS
    _ema_alpha = ISAAC_EMA_ALPHA

    def __init__(self, robot: str = "ur5e", **kwargs) -> None:
        kwargs.setdefault("time_limit", self.DEFAULT_TIME_LIMIT)
        kwargs.setdefault("goal_resample_time_range_s", ISAAC_GOAL_RESAMPLE_TIME_RANGE_S)
        kwargs.setdefault("joint_action_scale", ISAAC_JOINT_ACTION_SCALE)
        kwargs.setdefault("terminate_on_success", False)
        kwargs.setdefault("terminate_on_collision", False)
        kwargs.setdefault("hold_seconds", 0.0)
        super().__init__(robot=robot, **kwargs)
        # Isaac action term has no deadzone around zero.
        self.hold_eps = ISAAC_ACTION_DEADZONE

    def _build_default_mdp_cfg(self) -> ReachMDPCfg:
        cfg = super()._build_default_mdp_cfg()
        cfg.reward_terms = (
            RewardTermCfg(
                name="position_command_error",
                fn=rewards.position_error_l2,
                weight=rewards.scaled_by_step_dt(-0.2),
            ),
            RewardTermCfg(
                name="position_command_error_tanh",
                fn=rewards.position_error_tanh_std_01,
                weight=rewards.scaled_by_step_dt(0.1),
            ),
            RewardTermCfg(
                name="orientation_command_error",
                fn=rewards.orientation_error,
                weight=rewards.scaled_by_step_dt(-0.1),
            ),
            RewardTermCfg(
                name="action_rate_l2",
                fn=rewards.action_rate_l2,
                weight=rewards.scaled_by_step_dt(-1.0e-4),
            ),
            RewardTermCfg(
                name="joint_vel_l2",
                fn=rewards.joint_vel_l2,
                weight=rewards.scaled_by_step_dt(-1.0e-4),
            ),
        )
        cfg.include_reward_terms_in_info = True
        return cfg


class ReachJointPosIsaacRewardGymnasium(ReachGymnasiumBase):
    """Gymnasium wrapper for :class:`ReachJointPosIsaacRewardEnv`."""

    _env_cls = ReachJointPosIsaacRewardEnv

    def __init__(
        self,
        robot: str = "ur5e",
        seed: int | None = None,
        render_mode: str | None = None,
        time_limit: int = ISAAC_EPISODE_STEPS,
        **env_kwargs,
    ):
        super().__init__(
            robot=robot,
            seed=seed,
            render_mode=render_mode,
            time_limit=time_limit,
            **env_kwargs,
        )
