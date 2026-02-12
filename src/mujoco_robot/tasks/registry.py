"""Task registry and high-level task constructors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type

from mujoco_robot.tasks.reach import (
    ReachEnvCfg,
    make_reach_manager_based_env,
    make_reach_manager_based_gymnasium,
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


@dataclass(frozen=True)
class TaskSpec:
    """Describes one registered task and its factories."""

    name: str
    description: str
    config_type: Type[Any]
    make_raw: Callable[[Any | None], Any]
    make_gymnasium: Callable[[Any | None], Any]


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "reach": TaskSpec(
        name="reach",
        description="Config-first manager-based 3D end-effector reach task.",
        config_type=ReachEnvCfg,
        make_raw=make_reach_manager_based_env,
        make_gymnasium=make_reach_manager_based_gymnasium,
    ),
    "slot_sorter": TaskSpec(
        name="slot_sorter",
        description="Pick-and-place task with color/slot matching.",
        config_type=SlotSorterTaskConfig,
        make_raw=make_slot_sorter_env,
        make_gymnasium=make_slot_sorter_gymnasium,
    ),
    "lift_suction": TaskSpec(
        name="lift_suction",
        description="UR3e single-object lifting with a suction end-effector.",
        config_type=LiftSuctionTaskConfig,
        make_raw=make_lift_suction_env,
        make_gymnasium=make_lift_suction_gymnasium,
    ),
}


def get_task_spec(name: str) -> TaskSpec:
    """Return the task spec for ``name``."""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]


def list_tasks() -> Tuple[str, ...]:
    """List registered task names."""
    return tuple(TASK_REGISTRY.keys())


def make_task(
    name: str,
    *,
    gymnasium: bool = False,
    config: Any | None = None,
) -> Any:
    """Create a task environment from the registry.

    Parameters
    ----------
    name : str
        Task key (for example ``reach`` or ``slot_sorter``).
    gymnasium : bool
        If ``True``, create the Gymnasium wrapper. Otherwise create raw env.
    config : Any | None
        Task config object matching the spec's ``config_type``.
    """
    spec = get_task_spec(name)
    if config is not None and not isinstance(config, spec.config_type):
        raise TypeError(
            f"Task '{name}' expects config type {spec.config_type.__name__}, "
            f"got {type(config).__name__}."
        )
    if gymnasium:
        return spec.make_gymnasium(config)
    return spec.make_raw(config)


__all__ = ["TaskSpec", "TASK_REGISTRY", "get_task_spec", "list_tasks", "make_task"]
