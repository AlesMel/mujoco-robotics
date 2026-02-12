"""Backward-compatible shim for slot-sorter task modules.

For new code, prefer:

    from mujoco_robot.envs.slot_sorter import URSlotSorterEnv, SlotSorterGymnasium
"""
from __future__ import annotations

import warnings

warnings.warn(
    "mujoco_robot.envs.slot_sorter_env is deprecated and will be removed in "
    "v0.4.0; import from mujoco_robot.envs.slot_sorter instead.",
    DeprecationWarning,
    stacklevel=2,
)

from mujoco_robot.envs.slot_sorter.slot_sorter_env import (
    StepResult,
    SlotSorterGymnasium,
    URSlotSorterEnv,
)

__all__ = ["StepResult", "URSlotSorterEnv", "SlotSorterGymnasium"]
