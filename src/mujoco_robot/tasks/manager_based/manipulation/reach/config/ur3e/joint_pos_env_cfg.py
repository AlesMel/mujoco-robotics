"""UR3e joint-position reach config profile."""
from __future__ import annotations

from ...reach_env_cfg import ActionCfg, PhysicsCfg, ReachEnvCfg, SceneCfg


def make_ur3e_joint_pos_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur3e"),
        actions=ActionCfg(control_variant="joint_pos", joint_target_ema_alpha=0.35),
        physics=PhysicsCfg(
            actuator_kp=120.0,
            min_joint_damping=24.0,
            min_joint_frictionloss=1.2,
        ),
    )
