"""Smoke/integration tests for the Crazyflie hover task."""

from __future__ import annotations

import gymnasium
import numpy as np

import mujoco_robot  # noqa: F401  # ensures Gym IDs are registered
from mujoco_robot.tasks import (
    CrazyflieTaskConfig,
    get_crazyflie_cfg,
    list_crazyflie_cfgs,
    list_tasks,
    make_task,
)
from mujoco_robot.tasks.crazyflie import (
    CrazyflieHoverEnv,
    make_crazyflie_hover_env,
    make_crazyflie_hover_gymnasium,
)


def test_crazyflie_env_class_importable() -> None:
    """Crazyflie env class should be importable from tasks.crazyflie."""
    assert CrazyflieHoverEnv is not None
    assert callable(CrazyflieHoverEnv)


def test_crazyflie_factory_functions_importable() -> None:
    """Crazyflie factories should be importable from tasks.crazyflie."""
    assert callable(make_crazyflie_hover_env)
    assert callable(make_crazyflie_hover_gymnasium)


def test_crazyflie_cfg_registry_exposes_profiles() -> None:
    """Crazyflie config registry should expose all planned profiles."""
    names = list_crazyflie_cfgs()
    assert "crazyflie_hover" in names
    assert "crazyflie_hover_dense_stable" in names
    assert "crazyflie_hover_flowdeck" in names

    cfg = get_crazyflie_cfg("crazyflie_hover")
    assert isinstance(cfg, CrazyflieTaskConfig)
    assert cfg.actuator_profile == "crazyflie"


def test_make_crazyflie_task_raw_smoke() -> None:
    """Raw task factory should build and step Crazyflie hover env."""
    env = make_task(
        "crazyflie_hover",
        config=CrazyflieTaskConfig(time_limit=2, seed=0),
    )
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)

    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step.obs.shape == (env.observation_dim,)
    env.close()


def test_make_crazyflie_task_gym_smoke() -> None:
    """Gym task factory should build and step Crazyflie hover env."""
    env = make_task(
        "crazyflie_hover",
        gymnasium=True,
        config=CrazyflieTaskConfig(time_limit=2, seed=0),
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, _reward, _terminated, _truncated, _info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert obs2.shape == env.observation_space.shape
    env.close()


def test_crazyflie_task_registered() -> None:
    """Task registry should include the Crazyflie key."""
    assert "crazyflie_hover" in list_tasks()


def test_crazyflie_gym_registration_smoke() -> None:
    """Public Gymnasium ID should construct and step."""
    env = gymnasium.make("MuJoCoRobot/Crazyflie-Hover-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, _reward, _terminated, _truncated, _info = env.step(
        env.action_space.sample()
    )
    assert obs2.shape == env.observation_space.shape
    env.close()


def test_motor_lag_prevents_instant_max_speed() -> None:
    """First-order motor lag should prevent instant jump to max omega."""
    env = CrazyflieHoverEnv(time_limit=2, seed=0)
    env.reset(seed=0)

    env.step(np.ones(env.action_dim, dtype=np.float32))
    assert float(np.max(env.motor_omega)) < env.max_motor_omega
    env.close()


def test_ground_effect_monotonicity() -> None:
    """Ground-effect multiplier should be larger near the ground."""
    env = CrazyflieHoverEnv(time_limit=2, seed=0)
    # cos_tilt=1.0 (upright), horiz_speed=0.0 (no lateral motion)
    low = env._ground_effect_multiplier(0.01, cos_tilt=1.0, horiz_speed=0.0)
    high = env._ground_effect_multiplier(0.30, cos_tilt=1.0, horiz_speed=0.0)
    assert low > high
    assert high >= 1.0
    # Tilted drone should get less ground effect
    low_tilted = env._ground_effect_multiplier(0.01, cos_tilt=0.5, horiz_speed=0.0)
    assert low_tilted < low
    # Moving laterally should reduce ground effect
    low_moving = env._ground_effect_multiplier(0.01, cos_tilt=1.0, horiz_speed=1.0)
    assert low_moving < low
    env.close()
