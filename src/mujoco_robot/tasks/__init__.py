"""IsaacLab-style task layer: configs, factories, registry."""

from mujoco_robot.tasks.manager_based.manipulation.reach import (
    ReachEnvCfg,
    make_reach_manager_based_env,
    make_reach_manager_based_gymnasium,
    list_reach_cfgs,
)
from mujoco_robot.tasks.lift_suction import (
    LiftSuctionTaskConfig,
    make_lift_suction_env,
    make_lift_suction_gymnasium,
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


def make_reach_env(cfg: ReachEnvCfg | None = None):
    """Create a raw reach environment from manager-based config."""
    return make_reach_manager_based_env(cfg)


def make_reach_gymnasium(cfg: ReachEnvCfg | None = None):
    """Create a Gymnasium reach environment from manager-based config."""
    return make_reach_manager_based_gymnasium(cfg)


__all__ = [
    "ReachEnvCfg",
    "make_reach_env",
    "make_reach_gymnasium",
    "make_reach_manager_based_env",
    "make_reach_manager_based_gymnasium",
    "list_reach_cfgs",
    "LiftSuctionTaskConfig",
    "make_lift_suction_env",
    "make_lift_suction_gymnasium",
    "SlotSorterTaskConfig",
    "make_slot_sorter_env",
    "make_slot_sorter_gymnasium",
    "TaskSpec",
    "TASK_REGISTRY",
    "get_task_spec",
    "list_tasks",
    "make_task",
]
