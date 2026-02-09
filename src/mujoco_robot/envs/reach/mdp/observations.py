"""Observation terms for the reach task."""
from __future__ import annotations

import numpy as np


def joint_pos_relative(env) -> np.ndarray:
    out = np.empty(len(env._robot_qpos_ids), dtype=np.float32)
    for i, qpos_adr in enumerate(env._robot_qpos_ids):
        out[i] = float(env.data.qpos[qpos_adr] - env.init_q[i])
    return out


def joint_vel(env) -> np.ndarray:
    out = np.empty(len(env.robot_dofs), dtype=np.float32)
    for i, dof_adr in enumerate(env.robot_dofs):
        out[i] = float(env.data.qvel[dof_adr])
    return out


def pose_command_base(env) -> np.ndarray:
    goal_pos_base = env.goal_pos - env._BASE_POS
    return np.concatenate(
        [goal_pos_base.astype(np.float32), env.goal_quat.astype(np.float32)]
    )


def last_action(env) -> np.ndarray:
    return env._last_action.astype(np.float32).copy()
