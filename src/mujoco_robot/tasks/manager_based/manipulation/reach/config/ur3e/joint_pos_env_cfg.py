"""UR3e joint-position reach config profile."""
from __future__ import annotations

import math

from ...reach_env_cfg import ActionCfg, CommandCfg, PhysicsCfg, ReachEnvCfg, SceneCfg


def make_ur3e_joint_pos_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur3e"),
        actions=ActionCfg(control_variant="joint_pos", joint_target_ema_alpha=1.0),
        commands=CommandCfg(
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(0.0, 0.0),
            goal_yaw_range=(-0.5, 0.5),
        ),
        physics=PhysicsCfg(
            actuator_kp=300.0,
            min_joint_damping=30.0,
            min_joint_frictionloss=1.2,
        ),
    )
