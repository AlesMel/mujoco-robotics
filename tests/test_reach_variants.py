"""Smoke tests for reach control variants and Gym registrations."""

from __future__ import annotations

import gymnasium
import mujoco
import numpy as np
import pytest

import mujoco_robot  # noqa: F401  # ensures Gym IDs are registered
from mujoco_robot.tasks.reach import REACH_VARIANTS


def _make_variant_env(variant: str, **kwargs):
    env_cls, _ = REACH_VARIANTS[variant]
    return env_cls(**kwargs)


def _make_variant_gym(variant: str, **kwargs):
    _, gym_cls = REACH_VARIANTS[variant]
    return gym_cls(**kwargs)


@pytest.mark.parametrize(
    "variant", ["ik_rel", "ik_abs", "joint_pos", "joint_pos_isaac_reward"]
)
def test_urreach_control_variants_smoke(variant: str) -> None:
    """Each control variant should reset and step without errors."""
    env = _make_variant_env(
        variant,
        robot="ur3e",
        time_limit=0,
        randomize_init=False,
        seed=0,
    )
    obs = env.reset(seed=0)
    assert obs.shape == (env.observation_dim,)

    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step.obs.shape == (env.observation_dim,)
    env.close()


@pytest.mark.parametrize(
    "env_id",
    [
        "MuJoCoRobot/Reach-v0",
        "MuJoCoRobot/Reach-IK-Rel-v0",
        "MuJoCoRobot/Reach-IK-Abs-v0",
        "MuJoCoRobot/Reach-Joint-Pos-v0",
        "MuJoCoRobot/Reach-Joint-Pos-Isaac-Reward-v0",
    ],
)
def test_reach_gym_registration_variants(env_id: str) -> None:
    """All public reach IDs should construct and step once."""
    env = gymnasium.make(env_id, robot="ur3e")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    obs2, _reward, _terminated, _truncated, _info = env.step(env.action_space.sample())
    assert obs2.shape == env.observation_space.shape
    env.close()


@pytest.mark.parametrize(
    "variant", ["ik_rel", "ik_abs", "joint_pos", "joint_pos_isaac_reward"]
)
def test_reach_collision_count_resets_on_reset(variant: str) -> None:
    """Reset should clear stale collision state in internals."""
    env = _make_variant_env(
        variant,
        robot="ur3e",
        time_limit=0,
        randomize_init=False,
        obs_noise=0.0,
        seed=0,
    )
    env.reset(seed=0)

    # Simulate stale state from previous episode and ensure reset clears it.
    env._self_collision_count = 1
    env.reset(seed=1)

    assert env._self_collision_count == 0
    env.close()


@pytest.mark.parametrize(
    "variant", ["ik_rel", "ik_abs", "joint_pos", "joint_pos_isaac_reward"]
)
def test_reach_time_limit_done_boundary_raw_env(variant: str) -> None:
    """Raw env done should flip exactly on the time_limit-th step."""
    env = _make_variant_env(
        variant,
        robot="ur3e",
        time_limit=3,
        randomize_init=False,
        seed=0,
    )
    env.reset(seed=0)

    dones = []
    for _ in range(3):
        step = env.step(np.zeros(env.action_dim, dtype=np.float32))
        dones.append(step.done)

    assert dones == [False, False, True]
    env.close()


@pytest.mark.parametrize(
    "variant", ["ik_rel", "ik_abs", "joint_pos", "joint_pos_isaac_reward"]
)
def test_reach_time_limit_boundary_gym_wrapper(variant: str) -> None:
    """Gym wrapper truncation should align with the same step boundary."""
    env = _make_variant_gym(
        variant,
        robot="ur3e",
        time_limit=3,
    )
    env.reset(seed=0)

    truncated = []
    terminated = []
    for _ in range(3):
        _, _, term, trunc, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        terminated.append(term)
        truncated.append(trunc)

    assert terminated == [False, False, False]
    assert truncated == [False, False, True]
    env.close()


def test_joint_pos_isaac_reward_resample_period_is_time_based() -> None:
    """Goal resampling should follow elapsed seconds, not a hard-coded step count."""
    env = _make_variant_env(
        "joint_pos_isaac_reward",
        robot="ur3e",
        time_limit=0,
        randomize_init=False,
        goal_resample_time_range_s=(4.0, 4.0),
        seed=0,
    )
    env.reset(seed=0)
    zero = np.zeros(env.action_dim, dtype=np.float32)
    step_dt = env.model.opt.timestep * env.n_substeps
    expected_steps = int(np.floor(4.0 / step_dt))

    for _ in range(max(0, expected_steps - 1)):
        step = env.step(zero)
        assert not step.info["goal_resampled"]

    # Allow a tiny floating-point boundary tolerance around the expected step.
    resampled = False
    for _ in range(3):
        step = env.step(zero)
        if step.info["goal_resampled"]:
            resampled = True
            break

    assert resampled
    env.close()


def test_joint_pos_isaac_reward_success_termination_flag_gym() -> None:
    """With task termination enabled, success should map to Gym terminated=True."""
    env = _make_variant_gym(
        "joint_pos_isaac_reward",
        robot="ur3e",
        time_limit=0,
        terminate_on_success=True,
        # Force immediate success independent of the sampled command.
        reach_threshold=10.0,
        ori_threshold=float(np.pi),
    )
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    env.close()


def test_joint_pos_isaac_reward_defaults_do_not_resample_within_episode() -> None:
    """Built-in defaults should keep one goal for the full 12s episode."""
    env = _make_variant_gym(
        "joint_pos_isaac_reward",
        robot="ur3e",
    )
    env.reset(seed=0)

    step_dt = env.base.model.opt.timestep * env.base.n_substeps
    assert env.base.time_limit == int(round(12.0 / step_dt))

    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    any_resample = False
    terminated = False
    truncated = False
    for _ in range(env.base.time_limit):
        _, _, terminated, truncated, info = env.step(zero)
        any_resample = any_resample or bool(info.get("goal_resampled", False))

    assert terminated is False
    assert truncated is True
    assert any_resample is False
    env.close()


def test_joint_pos_isaac_reward_control_defaults_match_isaac_style() -> None:
    """Isaac-style variant should use direct, no-deadzone relative joint targets."""
    env = _make_variant_env(
        "joint_pos_isaac_reward",
        robot="ur3e",
        time_limit=0,
        randomize_init=False,
        seed=0,
    )
    env.reset(seed=0)

    assert env.joint_action_scale == pytest.approx(0.5)
    assert env.hold_eps == pytest.approx(0.0)
    assert env._ema_alpha == pytest.approx(1.0)

    env.close()


def test_reach_goal_sampling_respects_table_boundaries() -> None:
    """Sampled goals should stay within the table footprint and z clearance."""
    env = _make_variant_env(
        "ik_rel",
        robot="ur3e",
        time_limit=0,
        randomize_init=False,
        seed=0,
    )

    table_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    assert table_gid >= 0
    table_pos = env.model.geom_pos[table_gid].copy()
    table_size = env.model.geom_size[table_gid].copy()
    x_lo = table_pos[0] - table_size[0] + env._table_spawn_margin_xy
    x_hi = table_pos[0] + table_size[0] - env._table_spawn_margin_xy
    y_lo = table_pos[1] - table_size[1] + env._table_spawn_margin_xy
    y_hi = table_pos[1] + table_size[1] - env._table_spawn_margin_xy
    z_min = table_pos[2] + table_size[2] + env._table_goal_z_margin

    for seed in range(8):
        env.reset(seed=seed)
        gx, gy, gz = env.goal_pos
        assert x_lo <= gx <= x_hi
        assert y_lo <= gy <= y_hi
        assert gz >= z_min

    env.close()
