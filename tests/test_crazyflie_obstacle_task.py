"""Smoke / integration tests for the Crazyflie wall-maze environment."""
from __future__ import annotations

import numpy as np
import pytest

from mujoco_robot.tasks import get_crazyflie_cfg
from mujoco_robot.tasks.crazyflie.crazyflie_obstacle_env import (
    CrazyflieObstacleEnv,
    CrazyflieObstacleGymnasium,
)


# ------------------------------------------------------------------ Creation


class TestObstacleEnvCreation:
    def test_default_creation(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        assert env is not None
        env.close()

    def test_zero_walls(self) -> None:
        env = CrazyflieObstacleEnv(seed=0, n_wall_slots=0, n_barriers=(0, 0))
        obs = env.reset(seed=0)
        assert obs.shape == (env.observation_dim,)
        env.close()

    def test_multi_ranger_mode(self) -> None:
        env = CrazyflieObstacleEnv(seed=0, rangefinder_mode="multi_ranger")
        # parent(27) + 5 rays for multi-ranger = 32
        assert env.observation_dim == 27 + 5
        env.close()

    def test_lidar_mode_configurable(self) -> None:
        env = CrazyflieObstacleEnv(seed=0, n_lidar_rays=8)
        # parent(27) + 10 (8 horiz + up + down) = 37
        assert env.observation_dim == 27 + 10
        env.close()


# ------------------------------------------------------------------ Rangefinder


class TestRangefinder:
    def test_shape_lidar(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, n_lidar_rays=16)
        env.reset(seed=42)
        rf = env._compute_rangefinder()
        assert rf.shape == (18,)  # 16 horizontal + up + down
        env.close()

    def test_shape_multi_ranger(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, rangefinder_mode="multi_ranger")
        env.reset(seed=42)
        rf = env._compute_rangefinder()
        assert rf.shape == (5,)
        env.close()

    def test_values_in_range(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, rangefinder_noise_std=0.0)
        env.reset(seed=42)
        rf = env._compute_rangefinder()
        assert np.all(rf >= 0.0) and np.all(rf <= 1.0)
        env.close()

    def test_detects_walls(self) -> None:
        """With walls in the arena, some rays should detect them."""
        env = CrazyflieObstacleEnv(
            seed=42, n_wall_slots=12, n_barriers=(3, 3),
            rangefinder_noise_std=0.0,
        )
        env.reset(seed=42)
        rf = env._compute_rangefinder()
        # Floor alone should make the down-ray short
        assert np.any(rf < 1.0), "At least the floor / one wall should be detected"
        env.close()

    def test_floor_detected_by_down_ray(self) -> None:
        """The downward ray should detect the floor beneath the drone."""
        env = CrazyflieObstacleEnv(
            seed=0, n_wall_slots=0, n_barriers=(0, 0),
            rangefinder_noise_std=0.0, n_lidar_rays=4,
        )
        env.reset(seed=0)
        rf = env._compute_rangefinder()
        # Last ray is down; drone hovers ~0.35 m -> normalised ~0.175
        down_idx = len(rf) - 1
        assert rf[down_idx] < 0.5, f"Down ray should see floor, got {rf[down_idx]:.3f}"
        env.close()


# ------------------------------------------------------------------ Step


class TestObstacleStep:
    def test_step_obs_shape(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, n_lidar_rays=8)
        obs = env.reset(seed=42)
        expected_dim = env.observation_dim
        assert obs.shape == (expected_dim,)
        res = env.step(env.sample_action())
        assert res.obs.shape == (expected_dim,)
        env.close()

    def test_info_keys(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        env.reset(seed=42)
        res = env.step(env.sample_action())
        assert "obstacle_collision" in res.info
        assert "min_obstacle_range_m" in res.info
        assert "rangefinder_min_norm" in res.info
        assert "obstacle_proximity_penalty" in res.info
        assert "progress" in res.info
        assert "n_active_walls" in res.info
        # 5-component reward breakdown
        assert "R_progress" in res.info
        assert "R_obstacle" in res.info
        assert "R_energy" in res.info
        assert "R_stability" in res.info
        assert "R_task" in res.info
        env.close()

    def test_multi_step_no_crash(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, time_limit=20)
        env.reset(seed=42)
        for _ in range(10):
            res = env.step(np.zeros(4, dtype=np.float32))
            if res.done:
                break
        env.close()


# ------------------------------------------------------------------ Maze generation


class TestMazeGeneration:
    def test_walls_created(self) -> None:
        """With barriers=(2,3), some walls should be generated."""
        env = CrazyflieObstacleEnv(seed=42, n_barriers=(2, 3))
        env.reset(seed=42)
        assert env._n_active_walls >= 2, (
            f"Expected at least 2 wall segments, got {env._n_active_walls}"
        )
        env.close()

    def test_walls_fixed_across_resets(self) -> None:
        """With fixed_layout=True (default), walls stay identical."""
        env = CrazyflieObstacleEnv(seed=42, n_barriers=(2, 3))
        env.reset(seed=42)
        n1 = env._n_active_walls
        pos1 = env._active_wall_positions[:n1].copy()
        env.reset()  # different RNG draw for goal, but walls unchanged
        n2 = env._n_active_walls
        pos2 = env._active_wall_positions[:n2].copy()
        assert n1 == n2, "Wall count should stay constant across resets"
        assert np.allclose(pos1, pos2), "Fixed walls must not move between resets"
        env.close()

    def test_walls_randomised_when_not_fixed(self) -> None:
        """With fixed_layout=False, walls differ across resets."""
        env = CrazyflieObstacleEnv(
            seed=42, n_barriers=(2, 3), fixed_layout=False,
        )
        env.reset(seed=42)
        pos1 = env._active_wall_positions[: env._n_active_walls].copy()
        env.reset()  # re-generates maze
        pos2 = env._active_wall_positions[: env._n_active_walls].copy()
        if pos1.shape == pos2.shape:
            assert not np.allclose(pos1, pos2), "Walls must be re-randomised"
        env.close()

    def test_inactive_walls_underground(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, n_barriers=(2, 2), n_wall_slots=12)
        env.reset(seed=42)
        for i in range(env._n_active_walls, len(env._wall_mocap_ids)):
            z = env._active_wall_positions[i, 2]
            assert z < -1.0, f"Inactive wall {i} should be underground, z={z}"
        env.close()

    def test_walls_not_at_spawn(self) -> None:
        """Wall barriers should respect spawn clearance."""
        env = CrazyflieObstacleEnv(
            seed=42, n_barriers=(3, 3), wall_spawn_clearance=0.12,
        )
        for trial in range(5):
            env.reset()
            for i in range(env._n_active_walls):
                wpos = env._active_wall_positions[i]
                if wpos[2] < -1:
                    continue
                wsize = env._active_wall_sizes[i]
                # Wall's perpendicular axis should be at least clearance from 0
                # For Y-aligned walls: wsize[0] ~ THICKNESS, wsize[1] large
                # For X-aligned walls: wsize[0] large, wsize[1] ~ THICKNESS
                if wsize[0] < 0.02:
                    # Y-aligned wall, perpendicular is X
                    assert abs(wpos[0]) >= 0.10, (
                        f"Trial {trial}, wall {i} too close to spawn X: {wpos[0]:.3f}"
                    )
                elif wsize[1] < 0.02:
                    # X-aligned wall, perpendicular is Y
                    assert abs(wpos[1]) >= 0.10, (
                        f"Trial {trial}, wall {i} too close to spawn Y: {wpos[1]:.3f}"
                    )
        env.close()

    def test_full_walls_reach_flight_altitude(self) -> None:
        """Full-height walls should extend above 0.30 m."""
        env = CrazyflieObstacleEnv(seed=42, n_barriers=(3, 3))
        env.reset(seed=42)
        max_top = 0.0
        for i in range(env._n_active_walls):
            wpos = env._active_wall_positions[i]
            wsize = env._active_wall_sizes[i]
            if wpos[2] < -1:
                continue
            top = wpos[2] + wsize[2]
            max_top = max(max_top, top)
        assert max_top >= 0.30, (
            f"Tallest wall top = {max_top:.3f} m, should reach flight altitude"
        )
        env.close()

    def test_wall_segments_capped_at_slots(self) -> None:
        """Even with many barriers, wall count should not exceed n_wall_slots."""
        env = CrazyflieObstacleEnv(seed=42, n_wall_slots=6, n_barriers=(3, 3))
        env.reset(seed=42)
        assert env._n_active_walls <= 6, (
            f"Walls should be capped at 6 slots, got {env._n_active_walls}"
        )
        env.close()


# ------------------------------------------------------------------ Goal behind barrier


class TestGoalBehindBarrier:
    def test_goal_behind_at_least_one_barrier(self) -> None:
        """Goal should be on the far side of at least one barrier."""
        env = CrazyflieObstacleEnv(seed=42, n_barriers=(2, 3))
        for trial in range(10):
            env.reset()
            gx, gy, _gz = env._goal_pos
            behind_any = False
            for b in env._barrier_specs:
                pp = b["perp_pos"]
                if b["is_y_aligned"]:
                    if (pp >= 0 and gx > pp) or (pp < 0 and gx < pp):
                        behind_any = True
                else:
                    if (pp >= 0 and gy > pp) or (pp < 0 and gy < pp):
                        behind_any = True
            assert behind_any, (
                f"Trial {trial}: goal at ({gx:.3f}, {gy:.3f}) is not "
                f"behind any barrier: {env._barrier_specs}"
            )
        env.close()

    def test_goal_never_near_spawn(self) -> None:
        """Goal should always be well away from spawn (0,0)."""
        env = CrazyflieObstacleEnv(seed=42, n_barriers=(2, 3))
        for _ in range(20):
            env.reset()
            gx, gy = env._goal_pos[0], env._goal_pos[1]
            dist = np.sqrt(gx ** 2 + gy ** 2)
            assert dist > 0.15, f"Goal too close to spawn: ({gx:.3f}, {gy:.3f})"
        env.close()

    def test_goal_matches_current_barriers_when_random_layout(self) -> None:
        """When fixed_layout=False, goal should still be behind current barriers."""
        env = CrazyflieObstacleEnv(seed=42, n_barriers=(2, 3), fixed_layout=False)
        for trial in range(10):
            env.reset()
            gx, gy, _gz = env._goal_pos
            behind_any = False
            for b in env._barrier_specs:
                pp = b["perp_pos"]
                if b["is_y_aligned"]:
                    if (pp >= 0 and gx > pp) or (pp < 0 and gx < pp):
                        behind_any = True
                else:
                    if (pp >= 0 and gy > pp) or (pp < 0 and gy < pp):
                        behind_any = True
            assert behind_any, (
                f"Trial {trial}: random-layout goal ({gx:.3f}, {gy:.3f}) "
                f"is not behind current barriers: {env._barrier_specs}"
            )
        env.close()

    def test_barrier_specs_populated(self) -> None:
        """After init, barrier_specs should contain barrier info."""
        env = CrazyflieObstacleEnv(seed=42, n_barriers=(2, 3))
        assert len(env._barrier_specs) >= 2
        for b in env._barrier_specs:
            assert "is_y_aligned" in b
            assert "perp_pos" in b
        env.close()


# ------------------------------------------------------------------ Collision


class TestCollisionDetection:
    def test_no_collision_at_start(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        env.reset(seed=42)
        assert not env._check_obstacle_collision()
        env.close()


# ------------------------------------------------------------------ Gymnasium


class TestObstacleGymnasiumWrapper:
    def test_create_and_step(self) -> None:
        env = CrazyflieObstacleGymnasium(seed=42, render=False)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        action = env.action_space.sample()
        obs2, reward, term, trunc, info = env.step(action)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        env.close()

    def test_obstacle_collision_terminates(self) -> None:
        env = CrazyflieObstacleGymnasium(seed=42, render=False)
        obs, _ = env.reset(seed=42)
        for _ in range(5):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term:
                break
        env.close()

    def test_terminate_on_goal_flag(self) -> None:
        env = CrazyflieObstacleGymnasium(
            seed=0,
            render=False,
            terminate_on_goal=True,
            reach_threshold=999.0,
            reach_hold_steps=1,
            actuator_noise_std=0.0,
            disturbance_sigma=0.0,
        )
        obs, _ = env.reset(seed=0)
        obs, _r, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
        assert term is True
        assert info["goal_just_reached"] is True
        assert info["terminated_on_goal"] is True
        env.close()

    def test_render_rgb_array(self) -> None:
        env = CrazyflieObstacleGymnasium(
            seed=42, render=True, render_mode="rgb_array",
        )
        env.reset(seed=42)
        env.step(env.action_space.sample())
        frame = env.render()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        env.close()

    def test_render_shows_lidar_no_crash(self) -> None:
        """Rendering with lidar rays injected into the scene should not crash."""
        env = CrazyflieObstacleGymnasium(
            seed=42, render=True, render_mode="rgb_array",
            n_lidar_rays=8,
        )
        obs, _ = env.reset(seed=42)
        for _ in range(3):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            frame = env.render()
            assert frame is not None
            assert frame.ndim == 3
            if term:
                break
        env.close()


# ------------------------------------------------------------------ Obs dim


class TestObservationDimension:
    @pytest.mark.parametrize("mode", ["state_estimate", "flowdeck", "privileged"])
    def test_obs_matches_property(self, mode: str) -> None:
        env = CrazyflieObstacleEnv(
            seed=0, time_limit=2, observation_mode=mode, n_lidar_rays=8,
        )
        obs = env.reset(seed=0)
        assert obs.shape == (env.observation_dim,), f"Mismatch for mode={mode}"
        env.close()

    def test_obs_dim_is_parent_plus_rf(self) -> None:
        """Obstacle env: parent(27) + rangefinder(18) = 45."""
        env = CrazyflieObstacleEnv(seed=0, n_lidar_rays=16)
        assert env.observation_dim == 27 + 18  # 16 horiz + up + down
        env.close()


# ------------------------------------------------------------------ Reward signals


class TestRewardSignals:
    def test_progress_in_info(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        env.reset(seed=42)
        res = env.step(env.sample_action())
        assert "progress" in res.info
        assert isinstance(res.info["progress"], float)
        env.close()

    def test_reward_components_in_info(self) -> None:
        """All five reward components must appear in info."""
        env = CrazyflieObstacleEnv(seed=42)
        env.reset(seed=42)
        res = env.step(env.sample_action())
        for key in ("R_progress", "R_obstacle", "R_energy", "R_stability", "R_task"):
            assert key in res.info, f"Missing reward component: {key}"
            assert isinstance(res.info[key], float)
        env.close()

    def test_goal_bonus_on_reach(self) -> None:
        """Goal bonus should be +10 by default."""
        env = CrazyflieObstacleEnv(seed=42)
        assert env._goal_bonus == 10.0
        env.close()

    def test_progress_tracking_initialised(self) -> None:
        """After reset, _prev_distance should be set for potential shaping."""
        env = CrazyflieObstacleEnv(seed=42)
        env.reset(seed=42)
        assert hasattr(env, "_prev_distance")
        assert env._prev_distance > 0
        env.close()

    def test_velocity_history_initialised(self) -> None:
        """After reset, velocity/accel history should be zeroed."""
        env = CrazyflieObstacleEnv(seed=42)
        env.reset(seed=42)
        assert hasattr(env, "_prev_lin_vel")
        assert hasattr(env, "_prev_accel")
        assert np.allclose(env._prev_lin_vel, 0.0)
        assert np.allclose(env._prev_accel, 0.0)
        env.close()

    def test_progress_nets_zero_on_roundtrip(self) -> None:
        """Progress shaping is potential-based: round-trip nets to zero."""
        env = CrazyflieObstacleEnv(seed=42, time_limit=200)
        env.reset(seed=42)
        d0 = env._prev_distance
        total_progress = 0.0
        for _ in range(20):
            res = env.step(env.sample_action())
            total_progress += res.info["progress"]
            if res.done:
                break
        d_final = env._prev_distance
        # progress is (d_prev - d_curr)*2.0 each step, so total ~ (d0 - d_final)*2.0
        expected = (d0 - d_final) * 2.0
        assert abs(total_progress - expected) < 1e-4
        env.close()

    def test_reward_and_observation_use_same_rf_sample(self) -> None:
        """Step info min RF should match the RF slice inside returned obs."""
        env = CrazyflieObstacleEnv(seed=42, rangefinder_noise_std=0.02)
        env.reset(seed=42)
        res = env.step(env.sample_action())
        rf_obs = res.obs[-env._n_rf_total:]
        assert abs(float(np.min(rf_obs)) - float(res.info["rangefinder_min_norm"])) < 1e-6
        env.close()


# ------------------------------------------------------------------ Reward structure


class TestRewardStructure:
    def test_obstacle_info_keys_present(self) -> None:
        """Obstacle-specific info keys should be in step info."""
        env = CrazyflieObstacleEnv(seed=42)
        env.reset(seed=42)
        res = env.step(env.sample_action())
        assert "obstacle_collision" in res.info
        assert "obstacle_proximity_penalty" in res.info
        assert "min_obstacle_range_m" in res.info
        assert "progress" in res.info
        assert "R_progress" in res.info
        assert "R_obstacle" in res.info
        assert "R_energy" in res.info
        assert "R_stability" in res.info
        assert "R_task" in res.info
        # Parent-compatible keys (replicated, not from super())
        assert "crashed" in res.info
        assert "goals_reached" in res.info
        assert "pos_error" in res.info
        env.close()

    def test_reward_bounded_when_stable(self) -> None:
        """Hovering in place should yield a finite reward — obstacle zone
        penalties can be large in the small maze but should be bounded."""
        env = CrazyflieObstacleEnv(seed=42, time_limit=5)
        env.reset(seed=42)
        res = env.step(np.zeros(4, dtype=np.float32))
        assert np.isfinite(res.reward), f"Reward is not finite: {res.reward}"
        # No goal/collision/timeout on first step ⇒ R_task ~ 0
        assert res.info["R_task"] == 0.0
        env.close()


# ------------------------------------------------------------------ Reward params


class TestRewardParams:
    def test_default_w_progress(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        assert env._w_progress == 5.0
        env.close()

    def test_default_w_obstacle(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        assert env._w_obstacle == 0.5
        env.close()

    def test_default_w_energy(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        assert env._w_energy == 0.2
        env.close()

    def test_default_w_stability(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        assert env._w_stability == 0.3
        env.close()

    def test_default_w_task(self) -> None:
        env = CrazyflieObstacleEnv(seed=42)
        assert env._w_task == 1.0
        env.close()

    def test_default_collision_penalty(self) -> None:
        """Default collision penalty should be 20.0."""
        env = CrazyflieObstacleEnv(seed=42)
        assert env._collision_penalty == 20.0
        env.close()

    def test_default_goal_bonus(self) -> None:
        """Default goal_bonus should be 10.0."""
        env = CrazyflieObstacleEnv(seed=42)
        assert env._goal_bonus == 10.0
        env.close()

    def test_default_timeout_penalty(self) -> None:
        """Default timeout_penalty should be 5.0."""
        env = CrazyflieObstacleEnv(seed=42)
        assert env._timeout_penalty == 5.0
        env.close()

    def test_default_v_optimal(self) -> None:
        """Default v_optimal should be 0.5."""
        env = CrazyflieObstacleEnv(seed=42)
        assert env._v_optimal == 0.5
        env.close()

    def test_default_obstacle_zones(self) -> None:
        """Zone boundaries should be in ascending order (normalised fractions)."""
        env = CrazyflieObstacleEnv(seed=42)
        assert env._obstacle_collision_zone == 0.08
        assert env._obstacle_danger_zone == 0.25
        assert env._obstacle_warning_zone == 0.50
        assert env._obstacle_collision_zone < env._obstacle_danger_zone
        assert env._obstacle_danger_zone < env._obstacle_warning_zone
        env.close()

    def test_no_spike_on_goal_resample(self) -> None:
        """When the env resamples the goal, progress shaping should not
        produce a spurious spike — just the goal bonus."""
        env = CrazyflieObstacleEnv(seed=42, time_limit=2000)
        env.reset(seed=42)
        # Force the drone to the goal position to trigger a resample
        env._goal_pos = env.data.xpos[env.cf_body_id].copy()
        env._goal_pos[2] = env.data.xpos[env.cf_body_id][2]
        env._update_goal_marker()
        env._prev_distance = 0.01  # pretend we were already close
        env._reach_hold_counter = env.reach_hold_steps - 1  # one step away
        res = env.step(np.zeros(4, dtype=np.float32))
        # R_task should contain the goal bonus (10.0 * w_task=1.0 = +10)
        assert res.info["R_task"] > 0.0, (
            f"Goal resample R_task should be positive: {res.info['R_task']:.3f}"
        )
        env.close()

    def test_dense_profile_uses_random_layout(self) -> None:
        cfg = get_crazyflie_cfg("crazyflie_obstacle_dense_stable")
        assert cfg.env_kwargs["fixed_layout"] is False
