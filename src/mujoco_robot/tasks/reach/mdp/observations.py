"""Observation terms for the reach task."""
from __future__ import annotations

import numpy as np


def joint_pos_rel(env) -> np.ndarray:
    # Vectorized fancy indexing — avoids per-joint Python loop.
    return (env.data.qpos[env._qpos_idx] - env._init_q_f32).astype(np.float32)


def joint_vel_rel(env) -> np.ndarray:
    # Vectorized fancy indexing — avoids per-joint Python loop.
    return env.data.qvel[env._dof_idx].astype(np.float32)


def generated_commands_ee_pose(env) -> np.ndarray:
    # During ObservationManager dim inference, command manager may not yet exist.
    try:
        return env._manager("command")._pose_command.copy()
    except KeyError:
        goal_pos_base = env.goal_pos - env._BASE_POS
        return np.concatenate(
            [goal_pos_base.astype(np.float32), env.goal_quat.astype(np.float32)]
        )


# Pre-allocated buffer for ee_pose_base to avoid per-step allocation.
_ee_pose_buf = np.empty(7, dtype=np.float32)


def ee_pose_base(env) -> np.ndarray:
    """Current EE pose in base frame: [pos_xyz(3), quat_wxyz(4)] → 7-D.

    Providing the EE orientation explicitly makes it much easier for the
    policy to learn orientation tracking (vs. inferring FK from joint angles).
    """
    _ee_pose_buf[:3] = env.data.site_xpos[env.ee_site] - env._BASE_POS
    _ee_pose_buf[3:] = env._cached_ee_quat()
    return _ee_pose_buf.copy()


def last_action(env) -> np.ndarray:
    return env._last_action.copy()


# Backward-compatible aliases for existing imports.
joint_pos_relative = joint_pos_rel
joint_vel = joint_vel_rel
pose_command_base = generated_commands_ee_pose
