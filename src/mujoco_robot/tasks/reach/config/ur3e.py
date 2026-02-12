"""UR3e reach config profiles (joint-pos, IK-rel, IK-abs)."""
from __future__ import annotations

import math

from ..mdp import ReachRewardCfg
from ..reach_env_cfg import ActionCfg, CommandCfg, ManagerCfg, PhysicsCfg, ReachEnvCfg, SceneCfg


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
            actuator_kp=100.0,
            min_joint_damping=60.0,
            min_joint_frictionloss=1.2,
        ),
        managers=ManagerCfg(
            reward_cfg=ReachRewardCfg(
                orientation_error_weight=-0.2,
                orientation_tanh_weight=0.05,
                orientation_tanh_std=0.2,
            ),
        ),
    )


def make_ur3e_ik_rel_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur3e"),
        actions=ActionCfg(control_variant="ik_rel"),
        commands=CommandCfg(
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(math.pi / 2.0, math.pi / 2.0),
            goal_yaw_range=(-math.pi, math.pi),
        ),
    )


def make_ur3e_ik_abs_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur3e"),
        actions=ActionCfg(control_variant="ik_abs"),
        commands=CommandCfg(
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(math.pi / 2.0, math.pi / 2.0),
            goal_yaw_range=(-math.pi, math.pi),
        ),
    )


__all__ = ["make_ur3e_joint_pos_cfg", "make_ur3e_ik_rel_cfg", "make_ur3e_ik_abs_cfg"]
