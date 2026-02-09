"""IsaacLab-style task layer: configs, factories, registry."""

from mujoco_robot.tasks.reach import (
    ReachTaskConfig,
    make_reach_env,
    make_reach_gymnasium,
)
from mujoco_robot.tasks.slot_sorter import (
    SlotSorterTaskConfig,
    make_slot_sorter_env,
    make_slot_sorter_gymnasium,
)
from mujoco_robot.tasks.registry import (
    TaskSpec,
    TASK_REGISTRY,
    get_task_spec,
    list_tasks,
    make_task,
)

__all__ = [
    "ReachTaskConfig",
    "make_reach_env",
    "make_reach_gymnasium",
    "SlotSorterTaskConfig",
    "make_slot_sorter_env",
    "make_slot_sorter_gymnasium",
    "TaskSpec",
    "TASK_REGISTRY",
    "get_task_spec",
    "list_tasks",
    "make_task",
]
