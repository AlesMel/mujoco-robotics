"""Step-info dict builder for reach environments.

Extracts the 30-field info dict construction from ``rewarding.py``
so the reward orchestrator stays short and readable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_robot.tasks.reach.success_tracker import SuccessResult


def build_step_info(
    env: Any,
    *,
    dist: float,
    ori_err_mag: float,
    success_result: SuccessResult,
    failure: bool,
    terminated: bool,
    time_out: bool,
    goal_resampled: bool,
    goal_resample_reason: str,
    goal_pos_step: np.ndarray,
    goal_quat_step: np.ndarray,
    raw_reward_terms: dict[str, float] | None,
    step_dt: float,
) -> dict:
    """Assemble the canonical info dict returned by ``env.step()``.

    All heavy state reads happen here so ``compute_step_reward``
    can stay focused on reward logic.
    """
    command_manager = env._manager("command")
    tracker = env._success_tracker

    info: dict[str, Any] = {
        "dist": dist,
        "ori_err": ori_err_mag,
        "reach_threshold": float(env.reach_threshold),
        "ori_threshold": float(env.ori_threshold),
        "within_position_threshold": bool(dist < env.reach_threshold),
        "within_orientation_threshold": bool(ori_err_mag < env.ori_threshold),
        "success": success_result.success,
        "hold_success": success_result.hold_success,
        "first_hold_success": success_result.first_hold,
        "success_streak_steps": success_result.streak_steps,
        "success_hold_steps": tracker.hold_steps,
        "success_bonus": success_result.success_bonus,
        "stay_reward": success_result.stay_reward,
        "failure": failure,
        "terminated": terminated,
        "time_out": time_out,
        "goal_resample_elapsed_s": command_manager.elapsed_s,
        "goal_resample_target_s": command_manager.target_s,
        "goal_resampled": goal_resampled,
        "goal_resample_reason": goal_resample_reason,
        "goals_reached": tracker.goals_reached,
        "goals_held": tracker.goals_held,
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

    if env._mdp_cfg.include_reward_terms_in_info and raw_reward_terms is not None:
        reward_terms = dict(raw_reward_terms)
        reward_terms["success_bonus"] = success_result.success_bonus
        reward_terms["stay_reward"] = success_result.stay_reward
        reward_terms["step_dt"] = step_dt
        info["reward_terms"] = reward_terms

    return info
