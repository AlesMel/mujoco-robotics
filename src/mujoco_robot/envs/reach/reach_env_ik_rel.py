"""ReachIKRelEnv — IK-Relative control variant for the reach task.

Action (6-D): ``[dx, dy, dz, dwx, dwy, dwz]`` in ``[-1, 1]``.
First 3 components are scaled by ``ee_step`` and applied as a position
delta in world frame.  Last 3 are scaled by ``ori_step`` as an
axis-angle orientation increment.

This is the default and most Isaac-Lab-aligned variant.  The action is
Markov — the delta is applied relative to the **current** EE pose, not
an accumulated target.  Zero action ≈ hold in place.

Usage::

    from mujoco_robot.envs.reach import ReachIKRelEnv, ReachIKRelGymnasium

    env = ReachIKRelEnv(robot="ur3e")
    obs = env.reset()
    result = env.step(env.sample_action())

    gym_env = ReachIKRelGymnasium(robot="ur3e")
"""
from __future__ import annotations

import numpy as np

from mujoco_robot.envs.reach.reach_env_base import (
    ReachGymnasiumBase,
    URReachEnvBase,
)
from mujoco_robot.envs.reach.mdp import actions


class ReachIKRelEnv(URReachEnvBase):
    """Reach task with **relative Cartesian IK** actions (Isaac Lab IK-Rel).

    Action layout (all in [-1, 1]):
        ========  ========================================
        Index     Description
        ========  ========================================
        0–2       Position delta (×ee_step) in world frame
        3–5       Orientation delta (×ori_step) axis-angle
        ========  ========================================
    """

    _action_dim = 6

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        return actions.ik_relative_joint_targets(self, action)


class ReachIKRelGymnasium(ReachGymnasiumBase):
    """Gymnasium wrapper for :class:`ReachIKRelEnv`."""

    _env_cls = ReachIKRelEnv
