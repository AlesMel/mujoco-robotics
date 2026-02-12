"""Reward and termination helpers for reach environments."""
from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_robot.envs.reach.goals import (
    maybe_resample_goal_by_timer,
    resample_goal_after_success,
)


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
    # Keep goal snapshots so returned metrics stay consistent even if we
    # resample the command pose at the end of this step.
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
    if success and not env._prev_success:
        env._goals_reached += 1
    env._prev_success = success
    if success:
        env._success_streak_steps += 1
    else:
        env._success_streak_steps = 0

    hold_success = bool(success and env._success_streak_steps >= env.success_hold_steps)
    first_hold = bool(hold_success and not env._goal_hold_completed)
    if first_hold:
        env._goal_hold_completed = True
        env._goals_held += 1

    step_dt = float(env.model.opt.timestep * env.n_substeps)
    stay_reward = float(env.stay_reward_weight * step_dt if hold_success else 0.0)
    success_bonus = float(env.success_bonus if first_hold else 0.0)
    reward = float(reward + stay_reward + success_bonus)
    success_streak_steps = int(env._success_streak_steps)

    goal_resampled = False
    goal_resample_reason = "none"
    if not done:
        if env.resample_on_success and first_hold:
            goal_resampled, goal_resample_reason = resample_goal_after_success(env)
        else:
            goal_resampled, goal_resample_reason = maybe_resample_goal_by_timer(env)
    command_manager = env._manager("command")

    info = {
        "dist": dist,
        "ori_err": ori_err_mag,
        "reach_threshold": float(env.reach_threshold),
        "ori_threshold": float(env.ori_threshold),
        "within_position_threshold": bool(dist < env.reach_threshold),
        "within_orientation_threshold": bool(ori_err_mag < env.ori_threshold),
        "success": success,
        "hold_success": hold_success,
        "first_hold_success": first_hold,
        "success_streak_steps": success_streak_steps,
        "success_hold_steps": env.success_hold_steps,
        "success_bonus": success_bonus,
        "stay_reward": stay_reward,
        "failure": failure,
        "terminated": terminated,
        "time_out": time_up,
        "goal_resample_elapsed_s": command_manager.elapsed_s,
        "goal_resample_target_s": command_manager.target_s,
        "goal_resampled": goal_resampled,
        "goal_resample_reason": goal_resample_reason,
        "goals_reached": env._goals_reached,
        "goals_held": env._goals_held,
        "goals_resampled": env._goals_resampled,
        "self_collisions": env._self_collision_count,
        "ee_pos": env.data.site_xpos[env.ee_site].copy(),
        "ee_quat": env._ee_quat(),
        # Goal used to compute this step's dist/orientation metrics.
        "goal_pos": goal_pos_step,
        "goal_quat": goal_quat_step,
        # Active command after potential end-of-step resample.
        "active_goal_pos": env.goal_pos.copy(),
        "active_goal_quat": env.goal_quat.copy(),
    }
    if env._mdp_cfg.include_reward_terms_in_info:
        reward_terms = dict(raw_reward_terms)
        reward_terms["success_bonus"] = success_bonus
        reward_terms["stay_reward"] = stay_reward
        reward_terms["step_dt"] = step_dt
        info["reward_terms"] = reward_terms
    return float(reward), done, info
