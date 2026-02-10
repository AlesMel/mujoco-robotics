"""UR3e IK-relative reach config profile."""
from __future__ import annotations

from ...reach_env_cfg import ActionCfg, ReachEnvCfg, SceneCfg


def make_ur3e_ik_rel_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur3e"),
        actions=ActionCfg(control_variant="ik_rel"),
    )

