"""Reset/bootstrap helpers for reach environments."""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def reset_episode_state(env: Any, seed: int | None) -> None:
    """Reset MuJoCo data and per-episode bookkeeping."""
    if seed is not None:
        env._rng = np.random.default_rng(seed)

    mujoco.mj_resetData(env.model, env.data)
    env.step_id = 0
    env._ee_quat_step_id = -1  # Invalidate EE cache
    env._success_tracker.reset()
    env._goals_resampled = 0
    env._self_collision_count = 0
    env._last_targets[:] = env.init_q
    env._last_action[:] = 0.0
    env._prev_action[:] = 0.0
    env._last_reward = 0.0
    env._last_step_info = {}
    env._last_obs = None


def initialize_robot_state(env: Any) -> None:
    """Set robot to randomized home pose and refresh derived state."""
    env._total_episodes += 1

    # --- 1. Set joints to canonical init_q and compute home EE orientation ---
    # Vectorized writes instead of per-joint Python loop.
    env.data.qpos[env._qpos_idx] = env.init_q
    env.data.qvel[env._dof_idx] = 0.0
    env.data.ctrl[env._act_idx] = env.init_q
    mujoco.mj_forward(env.model, env.data)
    env._ee_quat_step_id = -1
    env._home_quat = env._ee_quat()

    # --- 2. Optionally randomize from init_q ---
    if env.randomize_init:
        scales = env._rng.uniform(*env.init_q_range, size=len(env.init_q))
        q_rand = env.init_q * scales
        q_rand = np.clip(q_rand, env._joint_lo, env._joint_hi)
        env.data.qpos[env._qpos_idx] = q_rand
        env.data.qvel[env._dof_idx] = 0.0
        env.data.ctrl[env._act_idx] = q_rand

    env._last_targets[:] = env.data.qpos[env._qpos_idx]
    mujoco.mj_forward(env.model, env.data)


def initialize_goal_and_settle(env: Any) -> None:
    """Sample command goal, settle physics, and cache initial errors."""
    env._manager("command").reset()
    env._resample_goal()

    for _ in range(env.settle_steps):
        mujoco.mj_step(env.model, env.data)

    env._init_dist = env._ee_goal_dist()
    env._init_ori_err = env._orientation_error_magnitude()
