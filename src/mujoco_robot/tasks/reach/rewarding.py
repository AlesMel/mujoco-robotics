"""Reward and termination helpers for reach environments."""
from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_robot.tasks.reach.goals import (
    maybe_resample_goal_by_timer,
    resample_goal_after_success,
)
from mujoco_robot.tasks.reach.info import build_step_info


def compute_done_flags(
    env: Any,
    dist: float,
    ori_err_mag: float,
) -> tuple[bool, bool, bool, bool, bool]:
    """Compute done flags via the configured termination manager."""
    ctx = {"dist": float(dist), "ori_err": float(ori_err_mag)}
    flags = env._manager("termination").compute(ctx)
    return (
        bool(flags["success"]),
        bool(flags["failure"]),
        bool(flags["terminated"]),
        bool(flags["time_out"]),
        bool(flags["done"]),
    )


def compute_step_reward(env: Any) -> tuple[float, bool, dict]:
    """Compute reward and done flags from configured manager terms."""
    goal_pos_step = env.goal_pos.copy()
    goal_quat_step = env.goal_quat.copy()

    dist = env._ee_goal_dist()
    ori_err_mag = env._orientation_error_magnitude()
    action_delta = env._last_action - env._prev_action
    action_rate_l2 = float(np.dot(action_delta, action_delta))
    joint_vel = env.data.qvel[env.robot_dofs]
    joint_vel_l2 = float(np.dot(joint_vel, joint_vel))
    ctx = {
        "dist": dist,
        "ori_err": ori_err_mag,
        "action_rate_l2": action_rate_l2,
        "joint_vel_l2": joint_vel_l2,
    }

    reward, raw_reward_terms, _weighted_terms = env._manager("reward").compute(ctx)

    success, failure, terminated, time_up, done = compute_done_flags(env, dist, ori_err_mag)

    # --- Success tracking via SuccessTracker ---
    step_dt = float(env.model.opt.timestep * env.n_substeps)
    sr = env._success_tracker.update(success, step_dt)
    reward = float(reward + sr.total_reward_add)
    if bool(getattr(env._mdp_cfg, "reward_clip_to_unit_interval", False)):
        reward = float(np.clip(reward, 0.0, 1.0))

    # --- Goal resampling ---
    goal_resampled = False
    goal_resample_reason = "none"
    if not done:
        if env.resample_on_success and sr.first_hold:
            goal_resampled, goal_resample_reason = resample_goal_after_success(env)
        else:
            goal_resampled, goal_resample_reason = maybe_resample_goal_by_timer(env)

    # --- Info dict ---
    info = build_step_info(
        env,
        dist=dist,
        ori_err_mag=ori_err_mag,
        success_result=sr,
        failure=failure,
        terminated=terminated,
        time_out=time_up,
        goal_resampled=goal_resampled,
        goal_resample_reason=goal_resample_reason,
        goal_pos_step=goal_pos_step,
        goal_quat_step=goal_quat_step,
        raw_reward_terms=raw_reward_terms,
        step_dt=step_dt,
    )
    return float(reward), done, info
