"""UR3e reach config profiles (joint-pos, IK-rel, IK-abs)."""
from __future__ import annotations

import math

from ..mdp import ReachRewardCfg
from ..reach_env_cfg import ActionCfg, CommandCfg, ManagerCfg, PhysicsCfg, ReachEnvCfg, SceneCfg, SuccessCfg


def make_ur3e_joint_pos_cfg() -> ReachEnvCfg:
    return ReachEnvCfg(
        scene=SceneCfg(robot="ur3e"),
        actions=ActionCfg(
            control_variant="joint_pos",
            joint_target_ema_alpha=0.5,
            # Per-joint action scale: [shoulder_pan, shoulder_lift, elbow,
            #                          wrist1, wrist2, wrist3]
            # shoulder_pan (j0) is the primary world-z rotator — needs
            # enough range to cover the goal yaw span.  Wrist joints
            # handle fine orientation adjustments.
            joint_action_scale=(1.5, 0.5, 0.5, 1.5, 1.5, 3.14),
        ),
        commands=CommandCfg(
            # IsaacLab-style: resample goal every 4-5 s within the episode.
            goal_resample_time_range_s=(4.0, 5.0),
            goal_roll_range=(0.0, 0.0),
            goal_pitch_range=(0.0, 0.0),
            goal_yaw_range=(-math.pi / 2, math.pi / 2),
        ),
        success=SuccessCfg(
            # Resample immediately when the robot reaches the current goal
            # (no idle waiting for the timer).
            resample_on_success=True,
        ),
        physics=PhysicsCfg(
            actuator_kp=500.0,
            min_joint_damping=20.0,
            min_joint_frictionloss=1.2,
        ),
        managers=ManagerCfg(
            reward_cfg=ReachRewardCfg(
                reward_mode="dense_bounded",
                dense_position_std=0.22,
                dense_position_fine_std=0.025,
                dense_position_fine_weight=0.5,
                dense_orientation_std=0.3,
                dense_orientation_fine_std=0.15,
                dense_orientation_fine_weight=0.4,
                dense_orientation_linear_weight=0.2,
                dense_position_weight=0.5,
                dense_orientation_weight=0.5,
                clip_to_unit_interval=False,
                include_action_rate=True,
                action_rate_weight=-0.0005,
                include_joint_vel=True,
                joint_vel_weight=-0.00005,
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
