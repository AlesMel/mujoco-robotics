"""Action terms for reach task variants."""
from __future__ import annotations

import numpy as np


def call_variant_apply_action(env, action: np.ndarray) -> np.ndarray:
    """Backward-compatible action dispatch via variant's ``_apply_action``."""
    return env._apply_action(action)


def ik_relative_joint_targets(env, action: np.ndarray) -> np.ndarray:
    """IK-relative action: delta pose -> joint targets."""
    delta_pos = action[:3] * env.ee_step
    delta_ori = action[3:6] * env.ori_step
    target_pos, target_quat = env._desired_ee_relative(delta_pos, delta_ori)

    qvel_cmd = env._ik_cartesian(target_pos, target_quat)
    qvel_cmd = np.clip(qvel_cmd, -env.max_joint_vel, env.max_joint_vel)

    if np.linalg.norm(action) < env.hold_eps:
        return env._last_targets.copy()
    ik_gain = 0.35
    return env._last_targets + qvel_cmd * ik_gain


def ik_absolute_joint_targets(env, action: np.ndarray) -> np.ndarray:
    """IK-absolute action: absolute pose command -> joint targets."""
    target_pos, target_quat = env._desired_ee_absolute(action)

    qvel_cmd = env._ik_cartesian(target_pos, target_quat)
    qvel_cmd = np.clip(qvel_cmd, -env.max_joint_vel, env.max_joint_vel)

    ik_gain = 0.35
    return env._last_targets + qvel_cmd * ik_gain


def joint_relative_targets(env, action: np.ndarray) -> np.ndarray:
    """IsaacLab-style joint-position target: default_q + action * scale.

    This mirrors ``use_default_offset=True`` semantics in IsaacLab's
    ``JointPositionAction``: policy outputs are interpreted as offsets around
    the robot's default joint posture (``env.init_q``), not around the current
    measured joints. The optional EMA then smooths target updates in command
    space.
    """
    raw_target = env.init_q + action * env.joint_action_scale
    return env._ema_alpha * raw_target + (1.0 - env._ema_alpha) * env._last_targets
