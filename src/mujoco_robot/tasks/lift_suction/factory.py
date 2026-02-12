"""Factories for creating lift-suction task environments."""
from __future__ import annotations

from mujoco_robot.envs.lift_suction import LiftSuctionGymnasium, URLiftSuctionEnv
from mujoco_robot.tasks.lift_suction.config import LiftSuctionTaskConfig


def make_lift_suction_env(
    config: LiftSuctionTaskConfig | None = None,
) -> URLiftSuctionEnv:
    """Create a raw lift-suction environment from ``LiftSuctionTaskConfig``."""
    cfg = config or LiftSuctionTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    kwargs.setdefault("time_limit", cfg.time_limit)
    kwargs.setdefault("seed", cfg.seed)
    kwargs.setdefault("actuator_profile", cfg.actuator_profile)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return URLiftSuctionEnv(**kwargs)


def make_lift_suction_gymnasium(
    config: LiftSuctionTaskConfig | None = None,
) -> LiftSuctionGymnasium:
    """Create a Gymnasium lift-suction environment from task config."""
    cfg = config or LiftSuctionTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    if cfg.model_path is not None:
        kwargs.setdefault("model_path", cfg.model_path)
    return LiftSuctionGymnasium(
        seed=cfg.seed,
        render=cfg.render,
        time_limit=cfg.time_limit,
        actuator_profile=cfg.actuator_profile,
        **kwargs,
    )

