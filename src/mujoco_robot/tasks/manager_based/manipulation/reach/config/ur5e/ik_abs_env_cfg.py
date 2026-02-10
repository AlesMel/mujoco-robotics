"""UR5e IK-absolute reach config profile."""
from __future__ import annotations

from ...reach_env_cfg import ActionCfg, ReachEnvCfg, SceneCfg


def make_ur5e_ik_abs_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur5e"),
        actions=ActionCfg(control_variant="ik_abs"),
    )

