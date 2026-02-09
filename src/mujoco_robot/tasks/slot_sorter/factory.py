"""Factories for creating slot-sorter task environments."""
from __future__ import annotations

from mujoco_robot.envs.slot_sorter import SlotSorterGymnasium, URSlotSorterEnv
from mujoco_robot.tasks.slot_sorter.config import SlotSorterTaskConfig


def make_slot_sorter_env(config: SlotSorterTaskConfig | None = None) -> URSlotSorterEnv:
    """Create a raw slot-sorter environment from ``SlotSorterTaskConfig``."""
    cfg = config or SlotSorterTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    kwargs.setdefault("time_limit", cfg.time_limit)
    kwargs.setdefault("seed", cfg.seed)
    kwargs.setdefault("actuator_profile", cfg.actuator_profile)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return URSlotSorterEnv(**kwargs)


def make_slot_sorter_gymnasium(
    config: SlotSorterTaskConfig | None = None,
) -> SlotSorterGymnasium:
    """Create a Gymnasium slot-sorter environment from ``SlotSorterTaskConfig``."""
    cfg = config or SlotSorterTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return SlotSorterGymnasium(
        seed=cfg.seed,
        render=cfg.render,
        time_limit=cfg.time_limit,
        actuator_profile=cfg.actuator_profile,
        **kwargs,
    )
