"""Manager-based reach task package."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .reach_env_cfg import (
    ActionCfg,
    CommandCfg,
    EpisodeCfg,
    ManagerCfg,
    PhysicsCfg,
    RandomizationCfg,
    ReachEnvCfg,
    SceneCfg,
    SuccessCfg,
)
from .config import get_reach_cfg, list_reach_cfgs

if TYPE_CHECKING:
    from .reach_env import ReachManagerBasedEnv, ReachManagerBasedRLEnv


def resolve_reach_env_cfg(cfg: ReachEnvCfg | str | None):
    from .reach_env import resolve_reach_env_cfg as _impl

    return _impl(cfg)


def make_reach_manager_based_env(cfg: ReachEnvCfg | str | None = None):
    from .reach_env import make_reach_manager_based_env as _impl

    return _impl(cfg)


def make_reach_manager_based_gymnasium(cfg: ReachEnvCfg | str | None = None):
    from .reach_env import make_reach_manager_based_gymnasium as _impl

    return _impl(cfg)


def __getattr__(name: str):
    if name in {"ReachManagerBasedEnv", "ReachManagerBasedRLEnv"}:
        from .reach_env import ReachManagerBasedEnv, ReachManagerBasedRLEnv

        return {
            "ReachManagerBasedEnv": ReachManagerBasedEnv,
            "ReachManagerBasedRLEnv": ReachManagerBasedRLEnv,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SceneCfg",
    "EpisodeCfg",
    "ActionCfg",
    "CommandCfg",
    "SuccessCfg",
    "RandomizationCfg",
    "PhysicsCfg",
    "ManagerCfg",
    "ReachEnvCfg",
    "resolve_reach_env_cfg",
    "make_reach_manager_based_env",
    "make_reach_manager_based_gymnasium",
    "ReachManagerBasedEnv",
    "ReachManagerBasedRLEnv",
    "get_reach_cfg",
    "list_reach_cfgs",
]

