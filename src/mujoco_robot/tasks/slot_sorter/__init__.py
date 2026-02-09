"""Slot-sorter task entrypoints and config."""

from mujoco_robot.tasks.slot_sorter.config import SlotSorterTaskConfig
from mujoco_robot.tasks.slot_sorter.factory import (
    make_slot_sorter_env,
    make_slot_sorter_gymnasium,
)

__all__ = [
    "SlotSorterTaskConfig",
    "make_slot_sorter_env",
    "make_slot_sorter_gymnasium",
]
