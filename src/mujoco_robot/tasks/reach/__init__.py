"""Reach task — modular action-space variants (Isaac Lab style).

Each control variant lives in its own module:

* :mod:`.reach_env_joint_pos` — Joint-Position (default)
* :mod:`.reach_env_ik_rel` — IK-Relative
* :mod:`.reach_env_ik_abs` — IK-Absolute
* :mod:`.reach_env_joint_pos_isaac_reward` — Joint-Position + Isaac reward

Manager-based environment:

* :class:`ReachManagerBasedEnv` — raw env backed by manager runtime
* :class:`ReachManagerBasedRLEnv` — Gymnasium wrapper

Config-first API::

    from mujoco_robot.tasks.reach import ReachEnvCfg, get_reach_cfg
    cfg = get_reach_cfg("ur3e_joint_pos_dense_stable")
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .reach_env_base import (
    URReachEnvBase,
    ReachGymnasiumBase,
)
from mujoco_robot.envs.step_result import StepResult
from .success_tracker import SuccessTracker, SuccessResult
from .reach_env_ik_rel import ReachIKRelEnv, ReachIKRelGymnasium
from .reach_env_ik_abs import ReachIKAbsEnv, ReachIKAbsGymnasium
from .reach_env_joint_pos import ReachJointPosEnv, ReachJointPosGymnasium
from .reach_env_joint_pos_isaac_reward import (
    ReachJointPosIsaacRewardEnv,
    ReachJointPosIsaacRewardGymnasium,
)
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
from .mdp import (
    ActionTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    ReachMDPCfg,
    make_default_reach_mdp_cfg,
)

if TYPE_CHECKING:
    from .reach_env import ReachManagerBasedEnv, ReachManagerBasedRLEnv


# Variant registry — maps string keys to (env_cls, gymnasium_cls) pairs
REACH_VARIANTS = {
    "ik_rel": (ReachIKRelEnv, ReachIKRelGymnasium),
    "ik_abs": (ReachIKAbsEnv, ReachIKAbsGymnasium),
    "joint_pos": (ReachJointPosEnv, ReachJointPosGymnasium),
    "joint_pos_isaac_reward": (
        ReachJointPosIsaacRewardEnv,
        ReachJointPosIsaacRewardGymnasium,
    ),
}


def resolve_reach_env_cfg(cfg: ReachEnvCfg | str | None):
    from .reach_env import resolve_reach_env_cfg as _impl
    return _impl(cfg)


def make_reach_manager_based_env(cfg: ReachEnvCfg | str | None = None, **kwargs):
    from .reach_env import make_reach_manager_based_env as _impl
    return _impl(cfg, **kwargs)


def make_reach_manager_based_gymnasium(cfg: ReachEnvCfg | str | None = None, **kwargs):
    from .reach_env import make_reach_manager_based_gymnasium as _impl
    return _impl(cfg, **kwargs)


def __getattr__(name: str):
    if name in {"ReachManagerBasedEnv", "ReachManagerBasedRLEnv"}:
        from .reach_env import ReachManagerBasedEnv, ReachManagerBasedRLEnv
        return {
            "ReachManagerBasedEnv": ReachManagerBasedEnv,
            "ReachManagerBasedRLEnv": ReachManagerBasedRLEnv,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base
    "URReachEnvBase",
    "ReachGymnasiumBase",
    "StepResult",
    "SuccessTracker",
    "SuccessResult",
    # Modular variants
    "ReachIKRelEnv",
    "ReachIKRelGymnasium",
    "ReachIKAbsEnv",
    "ReachIKAbsGymnasium",
    "ReachJointPosEnv",
    "ReachJointPosGymnasium",
    "ReachJointPosIsaacRewardEnv",
    "ReachJointPosIsaacRewardGymnasium",
    "REACH_VARIANTS",
    # Cfg
    "SceneCfg", "EpisodeCfg", "ActionCfg", "CommandCfg", "SuccessCfg",
    "RandomizationCfg", "PhysicsCfg", "ManagerCfg", "ReachEnvCfg",
    "get_reach_cfg", "list_reach_cfgs",
    # MDP
    "ActionTermCfg", "ObservationTermCfg", "RewardTermCfg",
    "TerminationTermCfg", "ReachMDPCfg", "make_default_reach_mdp_cfg",
    # Manager-based env (lazy)
    "resolve_reach_env_cfg",
    "make_reach_manager_based_env",
    "make_reach_manager_based_gymnasium",
    "ReachManagerBasedEnv", "ReachManagerBasedRLEnv",
]
