"""Tests for manager-based reach MDP modular overrides."""

from __future__ import annotations

import numpy as np

from mujoco_robot.envs.reach_env import URReachEnv
from mujoco_robot.envs.reach.mdp import (
    ActionTermCfg,
    ObservationTermCfg,
    ReachRewardCfg,
    RewardTermCfg,
    TerminationTermCfg,
    make_default_reach_mdp_cfg,
)


def _constant_reward(_env, _ctx: dict[str, float]) -> float:
    return 1.0


def _two_dim_obs(env) -> np.ndarray:
    return env.goal_pos[:2].astype(np.float32).copy()


def _hold_action(env, _action: np.ndarray) -> np.ndarray:
    return env._last_targets.copy()


def _always_true(_env, _ctx: dict[str, float]) -> bool:
    return True


def test_reach_mdp_reward_override_constant() -> None:
    cfg = make_default_reach_mdp_cfg()
    cfg.reward_terms = (RewardTermCfg("constant", _constant_reward, weight=2.5),)

    env = URReachEnv(
        robot="ur3e",
        control_variant="ik_rel",
        randomize_init=False,
        time_limit=0,
        mdp_cfg=cfg,
        seed=0,
    )
    env.reset(seed=0)
    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    step_dt = env.model.opt.timestep * env.n_substeps
    assert np.isclose(step.reward, 2.5 * step_dt)
    env.close()


def test_reach_mdp_observation_override_dimension() -> None:
    cfg = make_default_reach_mdp_cfg()
    cfg.observation_terms = (ObservationTermCfg("goal_xy", _two_dim_obs),)

    env = URReachEnv(
        robot="ur3e",
        control_variant="ik_rel",
        randomize_init=False,
        time_limit=0,
        mdp_cfg=cfg,
        seed=0,
    )
    obs = env.reset(seed=0)
    assert env.observation_dim == 2
    assert obs.shape == (2,)
    env.close()


def test_reach_mdp_action_override_hold_targets() -> None:
    cfg = make_default_reach_mdp_cfg()
    cfg.action_term = ActionTermCfg("hold", _hold_action)

    env = URReachEnv(
        robot="ur3e",
        control_variant="ik_rel",
        randomize_init=False,
        time_limit=0,
        mdp_cfg=cfg,
        seed=0,
    )
    env.reset(seed=0)
    before = env._last_targets.copy()
    env.step(np.ones(env.action_dim, dtype=np.float32))
    after = env._last_targets.copy()
    np.testing.assert_allclose(before, after)
    env.close()


def test_reach_mdp_termination_override_success() -> None:
    cfg = make_default_reach_mdp_cfg()
    cfg.success_term = TerminationTermCfg("always_success", _always_true)

    env = URReachEnv(
        robot="ur3e",
        control_variant="ik_rel",
        randomize_init=False,
        time_limit=0,
        terminate_on_success=True,
        mdp_cfg=cfg,
        seed=0,
    )
    env.reset(seed=0)
    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert step.done is True
    assert bool(step.info["terminated"]) is True
    env.close()


def test_reach_reward_cfg_disables_all_default_terms() -> None:
    reward_cfg = ReachRewardCfg(
        position_error_weight=0.0,
        position_tanh_weight=0.0,
        orientation_error_weight=0.0,
        include_action_rate=False,
        include_joint_vel=False,
    )

    env = URReachEnv(
        robot="ur3e",
        control_variant="ik_rel",
        randomize_init=False,
        time_limit=0,
        reward_cfg=reward_cfg,
        success_bonus=0.0,
        stay_reward_weight=0.0,
        reach_threshold=0.0,
        ori_threshold=0.0,
        seed=0,
    )
    env.reset(seed=0)
    step = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert np.isclose(step.reward, 0.0)
    env.close()
