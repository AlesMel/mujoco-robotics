"""Backward-compatible shim for lift-suction env imports."""
from __future__ import annotations

import warnings

warnings.warn(
    "mujoco_robot.envs.lift_suction_env is deprecated and will be removed in "
    "v0.4.0; import from mujoco_robot.envs.lift_suction instead.",
    DeprecationWarning,
    stacklevel=2,
)

from mujoco_robot.envs.lift_suction.lift_suction_env import (
    LiftSuctionGymnasium,
    StepResult,
    URLiftSuctionEnv,
)

__all__ = ["URLiftSuctionEnv", "LiftSuctionGymnasium", "StepResult"]
