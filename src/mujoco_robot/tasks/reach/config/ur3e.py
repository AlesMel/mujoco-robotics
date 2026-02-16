"""UR3e reach config profiles (joint-pos, IK-rel, IK-abs)."""
from __future__ import annotations

import math

from ..mdp import ReachRewardCfg
from ..reach_env_cfg import ActionCfg, CommandCfg, ManagerCfg, PhysicsCfg, ReachEnvCfg, SceneCfg


def make_ur3e_joint_pos_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur3e"),
        actions=ActionCfg(control_variant="joint_pos", joint_target_ema_alpha=0.35),
        commands=CommandCfg(
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(0.0, 0.0),
            goal_yaw_range=(-math.pi, math.pi),
        ),
        physics=PhysicsCfg(
            actuator_kp=500.0,
            min_joint_damping=20.0,
            min_joint_frictionloss=1.2,
        ),
        managers=ManagerCfg(
            reward_cfg=ReachRewardCfg(
                reward_mode="dense_bounded",
                dense_position_std=0.05,
                dense_orientation_std=0.2,
                dense_position_weight=0.7,
                dense_orientation_weight=0.3,
                clip_to_unit_interval=True,
                include_action_rate=True,
                action_rate_weight=-0.0015,
                include_joint_vel=True,
                joint_vel_weight=-0.0002,
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
