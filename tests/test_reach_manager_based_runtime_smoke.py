"""Smoke tests for phase-3 manager-based reach runtime."""

from __future__ import annotations

import numpy as np

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
