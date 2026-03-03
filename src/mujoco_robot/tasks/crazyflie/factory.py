"""Factories for creating Crazyflie task environments (hover + reach)."""
from __future__ import annotations

from mujoco_robot.tasks.crazyflie.crazyflie_env import (
    CrazyflieHoverEnv,
    CrazyflieHoverGymnasium,
)
from mujoco_robot.tasks.crazyflie.crazyflie_reach_env import (
    CrazyflieReachEnv,
    CrazyflieReachGymnasium,
)
from mujoco_robot.tasks.crazyflie.crazyflie_obstacle_env import (
    CrazyflieObstacleEnv,
    CrazyflieObstacleGymnasium,
)
from mujoco_robot.tasks.crazyflie.config import CrazyflieTaskConfig


def make_crazyflie_hover_env(
    config: CrazyflieTaskConfig | None = None,
) -> CrazyflieHoverEnv:
    """Create a raw Crazyflie hover environment from ``CrazyflieTaskConfig``."""
    cfg = config or CrazyflieTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    kwargs.setdefault("time_limit", cfg.time_limit)
    kwargs.setdefault("seed", cfg.seed)
    kwargs.setdefault("actuator_profile", cfg.actuator_profile)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return CrazyflieHoverEnv(**kwargs)


def make_crazyflie_hover_gymnasium(
    config: CrazyflieTaskConfig | None = None,
) -> CrazyflieHoverGymnasium:
    """Create a Gymnasium Crazyflie environment from task config."""
    cfg = config or CrazyflieTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return CrazyflieHoverGymnasium(
        seed=cfg.seed,
        render=cfg.render,
        render_mode=cfg.render_mode,
        time_limit=cfg.time_limit,
        actuator_profile=cfg.actuator_profile,
        **kwargs,
    )


def make_crazyflie_reach_env(
    config: CrazyflieTaskConfig | None = None,
) -> CrazyflieReachEnv:
    """Create a raw Crazyflie reach environment from ``CrazyflieTaskConfig``."""
    cfg = config or CrazyflieTaskConfig(time_limit=800)
    kwargs = dict(cfg.env_kwargs)
    kwargs.setdefault("time_limit", cfg.time_limit)
    kwargs.setdefault("seed", cfg.seed)
    kwargs.setdefault("actuator_profile", cfg.actuator_profile)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return CrazyflieReachEnv(**kwargs)


def make_crazyflie_reach_gymnasium(
    config: CrazyflieTaskConfig | None = None,
) -> CrazyflieReachGymnasium:
    """Create a Gymnasium Crazyflie reach environment from task config."""
    cfg = config or CrazyflieTaskConfig(time_limit=800)
    kwargs = dict(cfg.env_kwargs)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return CrazyflieReachGymnasium(
        seed=cfg.seed,
        render=cfg.render,
        render_mode=cfg.render_mode,
        time_limit=cfg.time_limit,
        actuator_profile=cfg.actuator_profile,
        **kwargs,
    )


def make_crazyflie_obstacle_env(
    config: CrazyflieTaskConfig | None = None,
) -> CrazyflieObstacleEnv:
    """Create a raw Crazyflie obstacle-avoidance environment."""
    cfg = config or CrazyflieTaskConfig(time_limit=800)
    kwargs = dict(cfg.env_kwargs)
    kwargs.setdefault("time_limit", cfg.time_limit)
    kwargs.setdefault("seed", cfg.seed)
    kwargs.setdefault("actuator_profile", cfg.actuator_profile)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return CrazyflieObstacleEnv(**kwargs)


def make_crazyflie_obstacle_gymnasium(
    config: CrazyflieTaskConfig | None = None,
) -> CrazyflieObstacleGymnasium:
    """Create a Gymnasium Crazyflie obstacle-avoidance environment."""
    cfg = config or CrazyflieTaskConfig(time_limit=800)
    kwargs = dict(cfg.env_kwargs)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return CrazyflieObstacleGymnasium(
        seed=cfg.seed,
        render=cfg.render,
        render_mode=cfg.render_mode,
        time_limit=cfg.time_limit,
        actuator_profile=cfg.actuator_profile,
        **kwargs,
    )
