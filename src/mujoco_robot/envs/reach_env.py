"""Backward-compatible shim — delegates to modular reach variants.

Existing code such as::

    from mujoco_robot.envs.reach_env import URReachEnv, ReachGymnasium

continues to work.  :class:`URReachEnv` is a **factory** that returns
the appropriate variant subclass based on ``control_variant``.

For new code, prefer importing directly from the modular packages::

    from mujoco_robot.envs.reach import ReachIKRelEnv, ReachIKRelGymnasium
"""
from __future__ import annotations

from typing import Optional

from mujoco_robot.envs.reach.reach_env_base import StepResult, URReachEnvBase
from mujoco_robot.envs.reach import (
    REACH_VARIANTS,
    ReachGymnasiumBase,
    ReachIKAbsEnv,
    ReachIKAbsGymnasium,
    ReachIKRelEnv,
    ReachIKRelGymnasium,
    ReachJointPosEnv,
    ReachJointPosGymnasium,
    ReachJointPosIsaacRewardGymnasium,
)


# ---------------------------------------------------------------------------
# Factory function (looks like the old class constructor)
# ---------------------------------------------------------------------------
def URReachEnv(
    robot: str = "ur5e",
    control_variant: Optional[str] = None,
    action_mode: str = "cartesian",
    **kwargs,
) -> URReachEnvBase:
    """Factory that returns the correct :class:`URReachEnvBase` subclass.

    Parameters
    ----------
    robot : str
        Robot model name (``"ur5e"`` or ``"ur3e"``).
    control_variant : str | None
        Variant key from :data:`mujoco_robot.envs.reach.REACH_VARIANTS`.
    action_mode : str
        Backward-compat alias: ``"cartesian"`` → ``"ik_rel"``,
        ``"joint"`` → ``"joint_pos"``.
    **kwargs
        Forwarded to the variant constructor.

    Returns
    -------
    URReachEnvBase
        A concrete variant instance.
    """
    if control_variant is None:
        if action_mode == "cartesian":
            control_variant = "ik_rel"
        elif action_mode == "joint":
            control_variant = "joint_pos"
        else:
            raise ValueError(
                "action_mode must be 'cartesian' or 'joint' when "
                f"control_variant is None, got '{action_mode}'"
            )

    if control_variant not in REACH_VARIANTS:
        raise ValueError(
            f"Unknown control_variant '{control_variant}'. "
            f"Available: {list(REACH_VARIANTS.keys())}"
        )

    env_cls, _ = REACH_VARIANTS[control_variant]
    return env_cls(robot=robot, **kwargs)


# ---------------------------------------------------------------------------
# Gymnasium shim — delegates to variant via control_variant kwarg
# ---------------------------------------------------------------------------
class ReachGymnasium(ReachGymnasiumBase):
    """Backward-compatible Gymnasium wrapper.

    Routes to the correct variant env class based on ``control_variant``.

    Usage::

        env = ReachGymnasium(robot="ur3e", control_variant="ik_rel")
    """

    def __init__(
        self,
        robot: str = "ur5e",
        seed: int | None = None,
        render_mode: str | None = None,
        time_limit: int | None = None,
        control_variant: str | None = None,
        action_mode: str = "cartesian",
        **env_kwargs,
    ):
        # Resolve variant
        if control_variant is None:
            if action_mode == "cartesian":
                control_variant = "ik_rel"
            elif action_mode == "joint":
                control_variant = "joint_pos"
            else:
                raise ValueError(
                    "action_mode must be 'cartesian' or 'joint' when "
                    f"control_variant is None, got '{action_mode}'"
                )

        if control_variant not in REACH_VARIANTS:
            raise ValueError(
                f"Unknown control_variant '{control_variant}'. "
                f"Available: {list(REACH_VARIANTS.keys())}"
            )

        env_cls, _ = REACH_VARIANTS[control_variant]
        resolved_time_limit = (
            time_limit
            if time_limit is not None
            else int(getattr(env_cls, "DEFAULT_TIME_LIMIT", 375))
        )
        super().__init__(
            robot=robot,
            seed=seed,
            render_mode=render_mode,
            time_limit=resolved_time_limit,
            env_cls=env_cls,
            **env_kwargs,
        )


# ---------------------------------------------------------------------------
# Convenience aliases (for old-style variant-specific imports)
# ---------------------------------------------------------------------------
ReachIKRelGymnasium = ReachIKRelGymnasium
ReachIKAbsGymnasium = ReachIKAbsGymnasium
ReachJointPosGymnasium = ReachJointPosGymnasium
ReachJointPosIsaacRewardGymnasium = ReachJointPosIsaacRewardGymnasium

__all__ = [
    "URReachEnv",
    "ReachGymnasium",
    "ReachIKRelGymnasium",
    "ReachIKAbsGymnasium",
    "ReachJointPosGymnasium",
    "ReachJointPosIsaacRewardGymnasium",
    "StepResult",
]
