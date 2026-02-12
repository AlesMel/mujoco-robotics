"""Goal sampling and command-resample helpers for reach environments."""
from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_robot.core.ik_controller import quat_multiply, quat_unique
from mujoco_robot.robots.configs import get_robot_config


def goal_sampling_bounds(
    env: Any,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return goal XYZ sampling bounds clipped to table footprint when available."""
    x_lo, x_hi = float(env.goal_bounds[0, 0]), float(env.goal_bounds[0, 1])
    y_lo, y_hi = float(env.goal_bounds[1, 0]), float(env.goal_bounds[1, 1])
    z_lo, z_hi = float(env.goal_bounds[2, 0]), float(env.goal_bounds[2, 1])

    if env._table_xy_bounds is not None:
        x_lo = max(x_lo, float(env._table_xy_bounds[0, 0]) + env._table_spawn_margin_xy)
        x_hi = min(x_hi, float(env._table_xy_bounds[0, 1]) - env._table_spawn_margin_xy)
        y_lo = max(y_lo, float(env._table_xy_bounds[1, 0]) + env._table_spawn_margin_xy)
        y_hi = min(y_hi, float(env._table_xy_bounds[1, 1]) - env._table_spawn_margin_xy)
    if env._table_top_z is not None:
        z_lo = max(z_lo, env._table_top_z + env._table_goal_z_margin)

    # Keep ranges valid even for unusual custom robot/table setups.
    if x_lo >= x_hi:
        if env._table_xy_bounds is not None:
            x_lo = float(env._table_xy_bounds[0, 0]) + 0.01
            x_hi = float(env._table_xy_bounds[0, 1]) - 0.01
        if x_lo >= x_hi:
            x_lo, x_hi = float(env.goal_bounds[0, 0]), float(env.goal_bounds[0, 1])
    if y_lo >= y_hi:
        if env._table_xy_bounds is not None:
            y_lo = float(env._table_xy_bounds[1, 0]) + 0.01
            y_hi = float(env._table_xy_bounds[1, 1]) - 0.01
        if y_lo >= y_hi:
            y_lo, y_hi = float(env.goal_bounds[1, 0]), float(env.goal_bounds[1, 1])
    if z_lo >= z_hi:
        z_lo, z_hi = float(env.goal_bounds[2, 0]), float(env.goal_bounds[2, 1])
        if z_lo >= z_hi:
            z_mid = 0.5 * (z_lo + z_hi)
            z_lo, z_hi = z_mid - 0.01, z_mid + 0.01

    return (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)


def sample_goal_position(env: Any) -> np.ndarray:
    """Sample a random reachable goal position."""
    cfg = get_robot_config(env.robot)
    ee_pos = env.data.site_xpos[env.ee_site].copy()
    min_base, max_base = cfg.goal_distance
    min_height = cfg.goal_min_height
    min_ee = cfg.goal_min_ee_dist
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = goal_sampling_bounds(env)
    if env._table_top_z is not None:
        min_height = max(min_height, env._table_top_z + env._table_goal_z_margin)

    for _ in range(500):
        goal = np.array([
            env._rng.uniform(x_lo, x_hi),
            env._rng.uniform(y_lo, y_hi),
            env._rng.uniform(z_lo, z_hi),
        ])
        if goal[2] < min_height:
            continue
        dist_from_base = np.linalg.norm(goal - env._BASE_POS)
        if dist_from_base < min_base or dist_from_base > max_base:
            continue
        if np.linalg.norm(goal - ee_pos) < min_ee:
            continue
        return goal
    fallback = env._BASE_POS + np.array([0.25, 0.0, 0.30])
    fallback[0] = float(np.clip(fallback[0], x_lo, x_hi))
    fallback[1] = float(np.clip(fallback[1], y_lo, y_hi))
    fallback[2] = float(np.clip(fallback[2], z_lo, z_hi))
    return fallback


def sample_goal_quaternion(env: Any) -> np.ndarray:
    """Sample a goal orientation as a perturbation from the home EE pose.

    Roll/pitch/yaw ranges define *deviations* from the canonical home EE
    orientation (``_home_quat``).  A range of ``(0, 0)`` means "keep the
    home orientation on that axis".  The perturbation quaternion is left-
    multiplied onto ``_home_quat`` and then expressed in world frame.
    """
    roll = float(env._rng.uniform(*env.goal_roll_range))
    pitch = float(env._rng.uniform(*env.goal_pitch_range))
    yaw = float(env._rng.uniform(*env.goal_yaw_range))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    # Perturbation quaternion from Euler angles (intrinsic ZYX convention)
    delta_q = np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dtype=float,
    )

    # Compose perturbation with home EE orientation:
    #   goal_quat = delta_q ⊗ home_quat
    # This makes roll/pitch/yaw act as deviations FROM the home EE pose,
    # not from the identity quaternion.
    home_quat = getattr(env, "_home_quat", np.array([1.0, 0.0, 0.0, 0.0]))
    quat_w = quat_multiply(delta_q, home_quat)
    return quat_unique(quat_w)


def resample_goal_from_command(env: Any) -> None:
    """Apply the currently sampled command pose as the active goal."""
    pose_command = env._manager("command").pose_command
    env.goal_pos = pose_command[:3].astype(float)
    env.goal_quat = quat_unique(pose_command[3:7].astype(float))
    env._place_goal_marker(env.goal_pos, env.goal_quat)
    env._success_streak_steps = 0
    env._goal_hold_completed = False


def maybe_resample_goal_by_timer(env: Any) -> tuple[bool, str]:
    """Advance command timer and resample when the sampled duration elapses."""
    step_dt = env.model.opt.timestep * env.n_substeps
    if not env._manager("command").step(step_dt):
        return False, "none"

    env._goals_resampled += 1
    resample_goal_from_command(env)
    return True, "timer"


def resample_goal_after_success(env: Any) -> tuple[bool, str]:
    """Resample command pose immediately after hold-success."""
    env._manager("command").reset()
    env._goals_resampled += 1
    resample_goal_from_command(env)
    return True, "success"
