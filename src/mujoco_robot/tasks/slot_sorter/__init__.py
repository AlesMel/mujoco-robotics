"""Slot-sorter task entrypoints and config."""

from mujoco_robot.tasks.slot_sorter.config import SlotSorterTaskConfig
from mujoco_robot.tasks.slot_sorter.factory import (
    make_slot_sorter_env,
    make_slot_sorter_gymnasium,
)
from mujoco_robot.tasks.slot_sorter.slot_sorter_env import URSlotSorterEnv

__all__ = [
    "SlotSorterTaskConfig",
    "URSlotSorterEnv",
    "make_slot_sorter_env",
    "make_slot_sorter_gymnasium",
]
