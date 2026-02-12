"""Lift-suction task entrypoints and config."""

from mujoco_robot.tasks.lift_suction.config import LiftSuctionTaskConfig
from mujoco_robot.tasks.lift_suction.factory import (
    make_lift_suction_env,
    make_lift_suction_gymnasium,
)

__all__ = [
    "LiftSuctionTaskConfig",
    "make_lift_suction_env",
    "make_lift_suction_gymnasium",
]

