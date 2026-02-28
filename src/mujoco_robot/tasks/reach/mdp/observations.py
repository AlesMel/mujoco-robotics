"""Observation terms for the reach task."""
from __future__ import annotations

import numpy as np

from mujoco_robot.core.ik_controller import quat_to_rot6d, rot_mat_to_rot6d


def joint_pos_rel(env) -> np.ndarray:
    out = np.empty(len(env._robot_qpos_ids), dtype=np.float32)
    for i, qpos_adr in enumerate(env._robot_qpos_ids):
        out[i] = float(env.data.qpos[qpos_adr] - env.init_q[i])
    return out


def joint_vel_rel(env) -> np.ndarray:
    out = np.empty(len(env.robot_dofs), dtype=np.float32)
    for i, dof_adr in enumerate(env.robot_dofs):
        out[i] = float(env.data.qvel[dof_adr])
    return out


def goal_pos_base(env) -> np.ndarray:
    """Goal position in base frame (3,)."""
    try:
        pose_command = env._manager("command").pose_command.astype(np.float32)
        return (pose_command[:3] - env._BASE_POS.astype(np.float32))
    except KeyError:
        return (env.goal_pos - env._BASE_POS).astype(np.float32)


def goal_rot6d(env) -> np.ndarray:
    """Goal orientation as 6D continuous rotation representation (6,).

    Uses the first two columns of the rotation matrix (Zhou et al. CVPR 2019).
    Unlike quaternions, this representation has no discontinuities.
    """
    try:
        quat = env._manager("command").pose_command[3:7]
    except KeyError:
        quat = env.goal_quat
    return quat_to_rot6d(quat).astype(np.float32)


def ee_rot6d(env) -> np.ndarray:
    """Current EE orientation as 6D continuous rotation representation (6,)."""
    mat = env.data.site_xmat[env.ee_site].reshape(3, 3)
    return rot_mat_to_rot6d(mat).astype(np.float32)


def generated_commands_ee_pose(env) -> np.ndarray:
    """Goal pose as [pos_base(3), rot6d(6)] = (9,) — continuous representation.

    Replaces the old [pos(3), quat(4)] = (7,) layout to avoid quaternion
    discontinuities that make orientation hard to learn for neural networks.
    """
    pos = goal_pos_base(env)
    rot = goal_rot6d(env)
    return np.concatenate([pos, rot])


def orientation_error_vec(env) -> np.ndarray:
    """Axis-angle orientation error vector (EE -> goal), shape (3,)."""
    return env._orientation_error().astype(np.float32)


def last_action(env) -> np.ndarray:
    return env._last_action.astype(np.float32).copy()


# Backward-compatible aliases for existing imports.
joint_pos_relative = joint_pos_rel
joint_vel = joint_vel_rel
pose_command_base = generated_commands_ee_pose
