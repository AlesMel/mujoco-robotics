"""Observation terms for the reach task."""
from __future__ import annotations

import numpy as np


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


def generated_commands_ee_pose(env) -> np.ndarray:
    # During ObservationManager dim inference, command manager may not yet exist.
    try:
        return env._manager("command").pose_command.astype(np.float32)
    except KeyError:
        goal_pos_base = env.goal_pos - env._BASE_POS
        return np.concatenate(
            [goal_pos_base.astype(np.float32), env.goal_quat.astype(np.float32)]
        )


def last_action(env) -> np.ndarray:
    return env._last_action.astype(np.float32).copy()


# Backward-compatible aliases for existing imports.
joint_pos_relative = joint_pos_rel
joint_vel = joint_vel_rel
pose_command_base = generated_commands_ee_pose
