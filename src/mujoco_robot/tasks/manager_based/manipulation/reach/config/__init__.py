"""Profile registry for manager-based reach configs."""
from __future__ import annotations

from typing import Callable

from ..reach_env_cfg import ReachEnvCfg
from .ur3e import (
    make_ur3e_ik_abs_cfg,
    make_ur3e_ik_rel_cfg,
    make_ur3e_joint_pos_cfg,
)
from .ur5e import (
    make_ur5e_ik_abs_cfg,
    make_ur5e_ik_rel_cfg,
    make_ur5e_joint_pos_cfg,
)


_CFG_FACTORIES: dict[str, Callable[[], ReachEnvCfg]] = {
    "ur3e_joint_pos": make_ur3e_joint_pos_cfg,
    "ur3e_ik_rel": make_ur3e_ik_rel_cfg,
    "ur3e_ik_abs": make_ur3e_ik_abs_cfg,
    "ur5e_joint_pos": make_ur5e_joint_pos_cfg,
    "ur5e_ik_rel": make_ur5e_ik_rel_cfg,
    "ur5e_ik_abs": make_ur5e_ik_abs_cfg,
    # Alias names that encode intended profile intent.
    "ur3e_joint_pos_dense_stable": make_ur3e_joint_pos_cfg,
    "ur5e_joint_pos_dense_stable": make_ur5e_joint_pos_cfg,
}


def get_reach_cfg(name: str) -> ReachEnvCfg:
    """Build one named reach config profile."""
    if name not in _CFG_FACTORIES:
        raise ValueError(
            f"Unknown reach cfg '{name}'. Available: {sorted(_CFG_FACTORIES)}"
        )
    return _CFG_FACTORIES[name]()


def list_reach_cfgs() -> tuple[str, ...]:
    """List available reach config profile names."""
    return tuple(sorted(_CFG_FACTORIES.keys()))


__all__ = ["get_reach_cfg", "list_reach_cfgs"]

