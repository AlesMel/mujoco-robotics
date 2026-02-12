"""Tests for the IsaacLab-style task layer and registry."""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium

import mujoco_robot  # noqa: F401  # ensures Gym IDs are registered
from mujoco_robot.tasks.lift_suction import URLiftSuctionEnv
from mujoco_robot.tasks.slot_sorter import URSlotSorterEnv
from mujoco_robot.tasks import (
    LiftSuctionTaskConfig,
    ReachEnvCfg,
    SlotSorterTaskConfig,
    get_task_spec,
    list_tasks,
    make_task,
)


def test_slot_sorter_env_class_importable() -> None:
    """URSlotSorterEnv should be importable from tasks.slot_sorter."""
    assert URSlotSorterEnv is not None
    assert callable(URSlotSorterEnv)


def test_lift_suction_env_class_importable() -> None:
    """URLiftSuctionEnv should be importable from tasks.lift_suction."""
    assert URLiftSuctionEnv is not None
    assert callable(URLiftSuctionEnv)


def test_task_registry_lists_builtin_tasks() -> None:
    """Registry should expose all core tasks."""
    names = list_tasks()
    assert "reach" in names
    assert "slot_sorter" in names
    assert "lift_suction" in names


def test_get_task_spec_unknown_raises() -> None:
    """Unknown task names should raise a clear error."""
    with pytest.raises(ValueError):
        get_task_spec("does-not-exist")


def test_make_task_rejects_wrong_config_type() -> None:
    """Task factory should validate config type."""
    with pytest.raises(TypeError):
        make_task("reach", config=SlotSorterTaskConfig())


def test_make_reach_task_raw_smoke() -> None:
    """Reach task factory should build a working raw environment."""
    cfg = ReachEnvCfg()
    cfg.scene.robot = "ur3e"
    cfg.actions.control_variant = "ik_rel"
    cfg.episode.seed = 0
    cfg.episode.time_limit = 0
    cfg.randomization.randomize_init = False
    env = make_task(
        "reach",
        config=cfg,
    )
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)

    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step.obs.shape == (env.observation_dim,)
    env.close()


def test_make_reach_task_gym_smoke() -> None:
    """Reach task factory should build a working Gymnasium wrapper."""
    cfg = ReachEnvCfg()
    cfg.scene.robot = "ur3e"
    cfg.actions.control_variant = "joint_pos"
    cfg.episode.time_limit = 2
    cfg.episode.seed = 0
    cfg.randomization.randomize_init = False
    env = make_task(
        "reach",
        gymnasium=True,
        config=cfg,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, _reward, _terminated, _truncated, _info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert obs2.shape == env.observation_space.shape
    env.close()


def test_make_slot_sorter_task_raw_smoke() -> None:
    """Slot-sorter task factory should build a working raw environment."""
    env = make_task(
        "slot_sorter",
        config=SlotSorterTaskConfig(time_limit=1, seed=0),
    )
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)

    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step.obs.shape == (env.observation_dim,)
    env.close()


def test_slot_sorter_gym_registration_smoke() -> None:
    """Registered slot-sorter Gym ID should construct and step."""
    env = gymnasium.make("MuJoCoRobot/Slot-Sorter-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, _reward, _terminated, _truncated, _info = env.step(
        env.action_space.sample()
    )
    assert obs2.shape == env.observation_space.shape
    env.close()


def test_make_lift_suction_task_raw_smoke() -> None:
    """Lift-suction task factory should build a working raw environment."""
    env = make_task(
        "lift_suction",
        config=LiftSuctionTaskConfig(time_limit=2, seed=0),
    )
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)

    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step.obs.shape == (env.observation_dim,)
    env.close()


def test_lift_suction_gym_registration_smoke() -> None:
    """Registered lift-suction Gym ID should construct and step."""
    env = gymnasium.make("MuJoCoRobot/Lift-Suction-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, _reward, _terminated, _truncated, _info = env.step(
        env.action_space.sample()
    )
    assert obs2.shape == env.observation_space.shape
    env.close()
