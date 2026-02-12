"""Smoke tests for phase-3 manager-based reach runtime."""

from __future__ import annotations

import numpy as np

from mujoco_robot.core.ik_controller import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_multiply,
)
from mujoco_robot.tasks import ReachEnvCfg, list_tasks, make_task


def _make_smoke_cfg() -> ReachEnvCfg:
    cfg = ReachEnvCfg()
    cfg.scene.robot = "ur3e"
    cfg.actions.control_variant = "joint_pos"
    cfg.episode.time_limit = 2
    cfg.episode.seed = 0
    cfg.randomization.randomize_init = False
    return cfg


def test_manager_based_reach_task_is_registered() -> None:
    names = list_tasks()
    assert "reach" in names


def test_make_reach_task_raw_smoke() -> None:
    env = make_task("reach", config=_make_smoke_cfg())
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)

    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step.obs.shape == (env.observation_dim,)
    assert env._manager_runtime.has("action")
    assert env._manager_runtime.has("command")
    assert env._manager_runtime.has("observation")
    assert env._manager_runtime.has("reward")
    assert env._manager_runtime.has("termination")
    env.close()


def test_make_reach_task_gym_smoke() -> None:
    env = make_task("reach", gymnasium=True, config=_make_smoke_cfg())
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, _reward, _terminated, _truncated, _info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert obs2.shape == env.observation_space.shape
    assert env.base._manager_runtime.has("action")
    assert env.base._manager_runtime.has("command")
    assert env.base._manager_runtime.has("observation")
    assert env.base._manager_runtime.has("reward")
    assert env.base._manager_runtime.has("termination")
    env.close()


def test_joint_pos_action_uses_default_offset_not_current_qpos() -> None:
    """Joint-pos action should be based on init_q offset (IsaacLab-style)."""
    cfg = _make_smoke_cfg()
    cfg.physics.obs_noise = 0.0
    env = make_task("reach", config=cfg)
    env.reset(seed=0)

    action = np.full((env.action_dim,), 0.25, dtype=np.float32)
    target_ref = env._manager("action").compute_joint_targets(action)

    # Perturb MuJoCo qpos heavily; default-offset targets should not change.
    env.data.qpos[list(env._robot_qpos_ids)] += 0.5
    target_after_perturb = env._manager("action").compute_joint_targets(action)

    np.testing.assert_allclose(target_after_perturb, target_ref, atol=1e-8, rtol=0.0)
    env.close()


def test_goal_orientation_sampling_is_structured_yaw_not_random_axis() -> None:
    """Default goal orientation perturbation should be yaw-only relative to home."""
    cfg = _make_smoke_cfg()
    cfg.physics.obs_noise = 0.0
    env = make_task("reach", config=cfg)
    env.reset(seed=0)

    for _ in range(64):
        goal_q = env._sample_goal_quat()
        # Extract the perturbation relative to home: delta = goal ⊗ home⁻¹
        home_q = env._home_quat.copy()
        delta_q = quat_multiply(goal_q, quat_conjugate(home_q))
        aa = axis_angle_from_quat(delta_q)
        # No roll/pitch by default: perturbation axis-angle x/y components stay near zero.
        assert abs(float(aa[0])) < 1e-6
        assert abs(float(aa[1])) < 1e-6

    env.close()
