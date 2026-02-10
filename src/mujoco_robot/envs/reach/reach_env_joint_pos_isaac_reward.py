"""Joint-position reach variant with IsaacLab-aligned defaults."""
from __future__ import annotations

from mujoco_robot.envs.reach.reach_env_base import ReachGymnasiumBase
from mujoco_robot.envs.reach.reach_env_joint_pos import ReachJointPosEnv


ISAAC_CONTROL_DT_S = (1.0 / 60.0) * 2.0
ISAAC_EPISODE_SECONDS = 12.0
ISAAC_EPISODE_STEPS = int(round(ISAAC_EPISODE_SECONDS / ISAAC_CONTROL_DT_S))
ISAAC_JOINT_ACTION_SCALE = 0.5
ISAAC_ACTION_DEADZONE = 0.0
ISAAC_EMA_ALPHA = 1.0


class ReachJointPosIsaacRewardEnv(ReachJointPosEnv):
    """Reach task with joint-position control and IsaacLab defaults.

    Defaults are configured to mirror IsaacLab reach settings:
    - episode length ~= 12.0 s
    - default command resampling disabled within episode (window > episode)
    - direct relative joint-position targets (no EMA smoothing)
    - action scale = 0.5
    - no task-based termination by default (timeout only)
    """

    DEFAULT_TIME_LIMIT = ISAAC_EPISODE_STEPS
    _ema_alpha = ISAAC_EMA_ALPHA

    def __init__(self, robot: str = "ur5e", **kwargs) -> None:
        kwargs.setdefault("time_limit", self.DEFAULT_TIME_LIMIT)
        kwargs.setdefault("joint_action_scale", ISAAC_JOINT_ACTION_SCALE)
        kwargs.setdefault("terminate_on_success", False)
        kwargs.setdefault("terminate_on_collision", False)
        kwargs.setdefault("hold_seconds", 0.0)
        super().__init__(robot=robot, **kwargs)
        # Isaac action term has no deadzone around zero.
        self.hold_eps = ISAAC_ACTION_DEADZONE


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
