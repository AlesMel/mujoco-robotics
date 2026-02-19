"""Factories for creating cable-routing task environments."""
from __future__ import annotations

from mujoco_robot.tasks.cable_routing.cable_routing_env import (
    CableRoutingGymnasium,
    URCableRoutingEnv,
)
from mujoco_robot.tasks.cable_routing.config import CableRoutingTaskConfig


def make_cable_routing_env(
    config: CableRoutingTaskConfig | None = None,
) -> URCableRoutingEnv:
    """Create a raw cable-routing environment from ``CableRoutingTaskConfig``."""
    cfg = config or CableRoutingTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    kwargs.setdefault("time_limit", cfg.time_limit)
    kwargs.setdefault("seed", cfg.seed)
    kwargs.setdefault("actuator_profile", cfg.actuator_profile)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return URCableRoutingEnv(**kwargs)


def make_cable_routing_gymnasium(
    config: CableRoutingTaskConfig | None = None,
) -> CableRoutingGymnasium:
    """Create a Gymnasium cable-routing environment from task config."""
    cfg = config or CableRoutingTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return CableRoutingGymnasium(
        seed=cfg.seed,
        render=cfg.render,
        render_mode=cfg.render_mode,
        time_limit=cfg.time_limit,
        actuator_profile=cfg.actuator_profile,
        **kwargs,
    )
