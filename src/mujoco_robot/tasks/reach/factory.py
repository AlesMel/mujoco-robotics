"""Factories for creating reach task environments."""
from __future__ import annotations

from mujoco_robot.envs.reach_env import ReachGymnasium, URReachEnv
from mujoco_robot.tasks.reach.config import ReachTaskConfig


def make_reach_env(config: ReachTaskConfig | None = None):
    """Create a raw reach environment from ``ReachTaskConfig``."""
    cfg = config or ReachTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    if cfg.time_limit is not None:
        kwargs.setdefault("time_limit", cfg.time_limit)
    if cfg.seed is not None:
        kwargs.setdefault("seed", cfg.seed)
    if cfg.mdp_cfg is not None:
        kwargs.setdefault("mdp_cfg", cfg.mdp_cfg)
    return URReachEnv(
        robot=cfg.robot,
        control_variant=cfg.control_variant,
        **kwargs,
    )


def make_reach_gymnasium(config: ReachTaskConfig | None = None) -> ReachGymnasium:
    """Create a Gymnasium reach environment from ``ReachTaskConfig``."""
    cfg = config or ReachTaskConfig()
    kwargs = dict(cfg.env_kwargs)
    if cfg.time_limit is not None:
        kwargs.setdefault("time_limit", cfg.time_limit)
    if cfg.mdp_cfg is not None:
        kwargs.setdefault("mdp_cfg", cfg.mdp_cfg)
    return ReachGymnasium(
        robot=cfg.robot,
        control_variant=cfg.control_variant,
        seed=cfg.seed,
        render_mode=cfg.render_mode,
        **kwargs,
    )
