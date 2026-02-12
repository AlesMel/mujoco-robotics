"""UR5e IK-absolute reach config profile."""
from __future__ import annotations

import math

from ...reach_env_cfg import ActionCfg, CommandCfg, ReachEnvCfg, SceneCfg


def make_ur5e_ik_abs_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur5e"),
        actions=ActionCfg(control_variant="ik_abs"),
        commands=CommandCfg(
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(math.pi / 2.0, math.pi / 2.0),
            goal_yaw_range=(-math.pi, math.pi),
        ),
    )
