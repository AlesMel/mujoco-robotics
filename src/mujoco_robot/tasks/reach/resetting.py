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
    env._success_tracker.reset()
    env._goals_resampled = 0
    env._self_collision_count = 0
    env._last_targets = env.init_q.copy()
    env._last_action = np.zeros(env.action_dim, dtype=np.float32)
    env._prev_action = np.zeros(env.action_dim, dtype=np.float32)
    env._last_reward = 0.0
    env._last_step_info = {}
    env._last_obs = None


def initialize_robot_state(env: Any) -> None:
    """Set robot to randomized home pose and refresh derived state."""
    env._total_episodes += 1

    # --- 1. Set joints to canonical init_q and compute home EE orientation ---
    for qi, (qpos_adr, dof_adr, act_id) in enumerate(
        zip(env._robot_qpos_ids, env.robot_dofs, env.robot_actuators)
    ):
        env.data.qpos[qpos_adr] = env.init_q[qi]
        env.data.qvel[dof_adr] = 0.0
        env.data.ctrl[act_id] = env.init_q[qi]
    mujoco.mj_forward(env.model, env.data)
    env._home_quat = env._ee_quat()

    # --- 2. Optionally randomize from init_q ---
    for qi, (jid, qpos_adr, dof_adr, act_id) in enumerate(
        zip(
            env._robot_joint_ids,
            env._robot_qpos_ids,
            env.robot_dofs,
            env.robot_actuators,
        )
    ):
        q_home = env.init_q[qi]
        if env.randomize_init:
            scale = float(env._rng.uniform(*env.init_q_range))
            q_home = q_home * scale
            lo, hi = env.model.jnt_range[jid]
            if lo < hi:
                q_home = float(np.clip(q_home, lo, hi))
        env.data.qpos[qpos_adr] = q_home
        env.data.qvel[dof_adr] = 0.0
        env.data.ctrl[act_id] = q_home

    for qi, qpos_adr in enumerate(env._robot_qpos_ids):
        env._last_targets[qi] = env.data.qpos[qpos_adr]

    mujoco.mj_forward(env.model, env.data)


def initialize_goal_and_settle(env: Any) -> None:
    """Sample command goal, settle physics, and cache initial errors."""
    env._manager("command").reset()
    env._resample_goal()

    for _ in range(env.settle_steps):
        mujoco.mj_step(env.model, env.data)

    env._init_dist = env._ee_goal_dist()
    env._init_ori_err = env._orientation_error_magnitude()
