"""Smoke/integration tests for the Crazyflie reach-random-goals task."""

from __future__ import annotations

import gymnasium
import numpy as np

import mujoco_robot  # noqa: F401  – ensures Gym IDs are registered
from mujoco_robot.tasks import (
    CrazyflieTaskConfig,
    get_crazyflie_cfg,
    list_crazyflie_cfgs,
    list_tasks,
    make_task,
)
from mujoco_robot.tasks.crazyflie import (
    CrazyflieReachEnv,
    make_crazyflie_reach_env,
    make_crazyflie_reach_gymnasium,
)


# ------------------------------------------------------------------ Imports

def test_crazyflie_reach_env_importable() -> None:
    assert CrazyflieReachEnv is not None
    assert callable(CrazyflieReachEnv)


def test_crazyflie_reach_factory_importable() -> None:
    assert callable(make_crazyflie_reach_env)
    assert callable(make_crazyflie_reach_gymnasium)


# ------------------------------------------------------------------ Config

def test_crazyflie_reach_cfg_profiles() -> None:
    names = list_crazyflie_cfgs()
    assert "crazyflie_reach" in names
    assert "crazyflie_reach_dense_stable" in names

    cfg = get_crazyflie_cfg("crazyflie_reach")
    assert isinstance(cfg, CrazyflieTaskConfig)
    assert cfg.time_limit == 800


# ------------------------------------------------------------------ Raw env

def test_reach_raw_env_step_smoke() -> None:
    env = make_crazyflie_reach_env(
        CrazyflieTaskConfig(time_limit=5, seed=0),
    )
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)

    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step.obs.shape == (env.observation_dim,)
    assert isinstance(step.reward, float)
    env.close()


def test_reach_raw_env_goal_changes_on_success() -> None:
    """When the drone holds near the goal, a new one should be sampled."""
    env = CrazyflieReachEnv(
        time_limit=500,
        seed=42,
        reach_threshold=999.0,   # always "near goal"
        reach_hold_steps=2,
        actuator_noise_std=0.0,
        disturbance_sigma=0.0,
    )
    env.reset(seed=42)

    first_goal = env._goal_pos.copy()
    # Step a few times to trigger goal_just_reached
    for _ in range(5):
        env.step(np.zeros(4))

    assert env._goals_reached >= 1
    # Goal should have changed
    assert not np.allclose(env._goal_pos, first_goal)
    env.close()


# ------------------------------------------------------------------ Gym wrapper

def test_reach_gym_step_smoke() -> None:
    env = make_crazyflie_reach_gymnasium(
        CrazyflieTaskConfig(time_limit=5, seed=0),
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert obs2.shape == env.observation_space.shape
    assert "goals_reached" in info
    env.close()


# ------------------------------------------------------------------ Registry

def test_reach_task_registered() -> None:
    assert "crazyflie_reach" in list_tasks()


def test_reach_make_task_raw() -> None:
    env = make_task(
        "crazyflie_reach",
        config=CrazyflieTaskConfig(time_limit=3, seed=0),
    )
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)
    env.close()


def test_reach_make_task_gym() -> None:
    env = make_task(
        "crazyflie_reach",
        gymnasium=True,
        config=CrazyflieTaskConfig(time_limit=3, seed=0),
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    env.close()


# ------------------------------------------------------------------ Gym registration

def test_crazyflie_reach_gym_id() -> None:
    env = gymnasium.make("MuJoCoRobot/Crazyflie-Reach-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, _r, _t, _tr, info = env.step(env.action_space.sample())
    assert obs2.shape == env.observation_space.shape
    assert "goals_reached" in info
    env.close()


# ------------------------------------------------------------------ Physics

def test_motor_lag_reach() -> None:
    env = CrazyflieReachEnv(time_limit=2, seed=0)
    env.reset(seed=0)
    env.step(np.ones(env.action_dim, dtype=np.float32))
    assert float(np.max(env.motor_omega)) < env.max_motor_omega
    env.close()


def test_obs_dimension_matches_property() -> None:
    for mode in ("state_estimate", "flowdeck", "privileged"):
        env = CrazyflieReachEnv(
            time_limit=2, seed=0, observation_mode=mode,
        )
        obs = env.reset(seed=0)
        assert obs.shape == (env.observation_dim,), f"mismatch for mode={mode}"
        env.close()
