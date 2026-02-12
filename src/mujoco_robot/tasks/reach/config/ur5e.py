"""UR5e reach config profiles (joint-pos, IK-rel, IK-abs)."""
from __future__ import annotations

import math

from ..reach_env_cfg import ActionCfg, CommandCfg, PhysicsCfg, ReachEnvCfg, SceneCfg


def make_ur5e_joint_pos_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur5e"),
        actions=ActionCfg(control_variant="joint_pos", joint_target_ema_alpha=0.35),
        commands=CommandCfg(
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(math.pi / 2.0, math.pi / 2.0),
            goal_yaw_range=(-math.pi, math.pi),
        ),
        physics=PhysicsCfg(
            actuator_kp=120.0,
            min_joint_damping=24.0,
            min_joint_frictionloss=1.2,
        ),
    )


def make_ur5e_ik_rel_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur5e"),
        actions=ActionCfg(control_variant="ik_rel"),
        commands=CommandCfg(
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(math.pi / 2.0, math.pi / 2.0),
            goal_yaw_range=(-math.pi, math.pi),
        ),
    )


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


__all__ = ["make_ur5e_joint_pos_cfg", "make_ur5e_ik_rel_cfg", "make_ur5e_ik_abs_cfg"]
