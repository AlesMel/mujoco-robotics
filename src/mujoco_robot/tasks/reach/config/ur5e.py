"""UR5e reach config profiles (joint-pos, IK-rel, IK-abs)."""
from __future__ import annotations

import math

from ..reach_env_cfg import ActionCfg, CommandCfg, PhysicsCfg, ReachEnvCfg, SceneCfg, SuccessCfg


def make_ur5e_joint_pos_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur5e"),
        actions=ActionCfg(
            control_variant="joint_pos",
            joint_target_ema_alpha=0.5,
            # Per-joint action scale: [shoulder_pan, shoulder_lift, elbow,
            #                          wrist1, wrist2, wrist3]
            # Conservative for shoulder/elbow, large for wrist joints.
            joint_action_scale=(1.5, 0.5, 0.5, 1.5, 1.5, 3.14),
        ),
        commands=CommandCfg(
            # IsaacLab-style: resample goal every 4-8 s within the episode.
            goal_resample_time_range_s=(4.0, 8.0),
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(math.pi / 2.0, math.pi / 2.0),
            goal_yaw_range=(-math.pi / 2, math.pi / 2),
        ),
        success=SuccessCfg(
            resample_on_success=True,
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
