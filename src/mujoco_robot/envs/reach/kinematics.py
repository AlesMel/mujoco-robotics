"""Kinematics and pose helper functions for reach environments."""
from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_robot.core.ik_controller import (
    orientation_error_axis_angle,
    quat_error_magnitude,
    quat_multiply,
    quat_unique,
)


def ee_quaternion(env: Any) -> np.ndarray:
    """Return current end-effector quaternion (w, x, y, z)."""
    return env._ik.ee_quat()


def desired_ee_relative_pose(
    env: Any,
    delta_xyz: np.ndarray,
    delta_ori: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute desired EE pose from relative Cartesian deltas."""
    pos = env.data.site_xpos[env.ee_site].copy() + delta_xyz
    for i in range(3):
        pos[i] = float(np.clip(pos[i], env.ee_bounds[i, 0], env.ee_bounds[i, 1]))

    current_quat = ee_quaternion(env)
    angle = float(np.linalg.norm(delta_ori))
    if angle > 1e-8:
        axis = delta_ori / angle
        half = angle / 2.0
        dq = np.array(
            [
                np.cos(half),
                axis[0] * np.sin(half),
                axis[1] * np.sin(half),
                axis[2] * np.sin(half),
            ]
        )
        target_quat = quat_unique(quat_multiply(dq, current_quat))
    else:
        target_quat = current_quat

    return pos, target_quat.copy()


def desired_ee_absolute_pose(
    env: Any,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute desired EE pose from absolute Cartesian action."""
    lo = env.ee_bounds[:, 0]
    hi = env.ee_bounds[:, 1]
    pos = lo + 0.5 * (action[:3] + 1.0) * (hi - lo)

    aa = action[3:6] * env.ori_abs_max
    angle = float(np.linalg.norm(aa))
    if angle > env.ori_abs_max and angle > 1e-8:
        aa = aa * (env.ori_abs_max / angle)
        angle = float(env.ori_abs_max)

    if angle > 1e-8:
        axis = aa / angle
        half = angle / 2.0
        dq = np.array(
            [
                np.cos(half),
                axis[0] * np.sin(half),
                axis[1] * np.sin(half),
                axis[2] * np.sin(half),
            ]
        )
        target_quat = quat_unique(quat_multiply(dq, env._home_quat))
    else:
        target_quat = env._home_quat.copy()

    return pos.astype(float), target_quat


def ik_cartesian_joint_targets(
    env: Any,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> np.ndarray:
    """Solve Cartesian target pose into joint targets via IK."""
    return env._ik.solve(target_pos, target_quat)


def clamp_joint_targets(env: Any, targets: np.ndarray) -> np.ndarray:
    """Clamp joint targets to model joint limits."""
    out = np.asarray(targets, dtype=float).copy()
    for k, jid in enumerate(env._robot_joint_ids):
        lo, hi = env.model.jnt_range[jid]
        if lo < hi:
            out[k] = float(np.clip(out[k], lo, hi))
    return out


def ee_goal_distance(env: Any) -> float:
    """Compute end-effector to goal position distance."""
    return float(np.linalg.norm(env.data.site_xpos[env.ee_site] - env.goal_pos))


def orientation_error_vector(env: Any) -> np.ndarray:
    """Compute axis-angle orientation error vector (EE -> goal)."""
    return orientation_error_axis_angle(ee_quaternion(env), env.goal_quat)


def orientation_error_magnitude(env: Any) -> float:
    """Compute scalar quaternion orientation error in radians."""
    return quat_error_magnitude(ee_quaternion(env), env.goal_quat)
