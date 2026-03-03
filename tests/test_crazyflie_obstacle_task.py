"""Smoke / integration tests for the Crazyflie obstacle-avoidance environment."""
from __future__ import annotations

import numpy as np
import pytest

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

    def test_zero_obstacles(self) -> None:
        env = CrazyflieObstacleEnv(seed=0, n_obstacle_slots=0, n_obstacles_range=(0, 0))
        obs = env.reset(seed=0)
        assert obs.shape == (env.observation_dim,)
        env.close()

    def test_multi_ranger_mode(self) -> None:
        env = CrazyflieObstacleEnv(seed=0, rangefinder_mode="multi_ranger")
        # 5 rays for multi-ranger
        base_dim = 27  # same as reach base (state_estimate)
        assert env.observation_dim == base_dim + 5
        env.close()

    def test_lidar_mode_configurable(self) -> None:
        env = CrazyflieObstacleEnv(seed=0, n_lidar_rays=8)
        base_dim = 27
        assert env.observation_dim == base_dim + 10  # 8 horiz + up + down
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

    def test_detects_obstacles(self) -> None:
        """With many obstacles in a small arena, some rays should detect them."""
        env = CrazyflieObstacleEnv(
            seed=42, n_obstacle_slots=12, n_obstacles_range=(10, 12),
            rangefinder_noise_std=0.0,
        )
        env.reset(seed=42)
        rf = env._compute_rangefinder()
        # Floor alone should make the down-ray short
        assert np.any(rf < 1.0), "At least the floor / one obstacle should be detected"
        env.close()

    def test_floor_detected_by_down_ray(self) -> None:
        """The downward ray should detect the floor beneath the drone."""
        env = CrazyflieObstacleEnv(
            seed=0, n_obstacle_slots=0, n_obstacles_range=(0, 0),
            rangefinder_noise_std=0.0, n_lidar_rays=4,
        )
        env.reset(seed=0)
        rf = env._compute_rangefinder()
        # Last ray is down; drone hovers ~0.35 m → normalised ≈ 0.175
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
        env.close()

    def test_multi_step_no_crash(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, time_limit=20)
        env.reset(seed=42)
        for _ in range(10):
            res = env.step(np.zeros(4, dtype=np.float32))
            if res.done:
                break
        env.close()


# ------------------------------------------------------------------ Reset


class TestObstacleReset:
    def test_obstacle_count_within_range(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, n_obstacles_range=(4, 6))
        env.reset(seed=42)
        assert 4 <= env._n_active_obstacles <= 6
        env.close()

    def test_obstacles_randomised_across_resets(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, n_obstacles_range=(5, 5))
        env.reset(seed=42)
        pos1 = env._active_obstacle_positions[:5].copy()
        env.reset()  # different RNG state
        pos2 = env._active_obstacle_positions[:5].copy()
        assert not np.allclose(pos1, pos2), "Obstacles must be re-randomised"
        env.close()

    def test_inactive_underground(self) -> None:
        env = CrazyflieObstacleEnv(seed=42, n_obstacles_range=(2, 2))
        env.reset(seed=42)
        for i in range(2, len(env._obstacle_mocap_ids)):
            z = env._active_obstacle_positions[i, 2]
            assert z < -1.0, f"Inactive obstacle {i} should be underground, z={z}"
        env.close()

    def test_obstacles_not_overlapping_spawn(self) -> None:
        env = CrazyflieObstacleEnv(
            seed=42, n_obstacles_range=(8, 8), obstacle_spawn_clearance=0.20,
        )
        for trial in range(5):
            env.reset()
            for i in range(env._n_active_obstacles):
                opos = env._active_obstacle_positions[i]
                if opos[2] < -1:
                    continue
                d = np.linalg.norm(opos[:2] - env.spawn_pos[:2])
                assert d >= 0.18, (
                    f"Trial {trial}, obs {i} too close to spawn: d={d:.3f}"
                )
        env.close()

    def test_blocker_obstacles_on_corridor(self) -> None:
        """Blocker obstacles should be placed between spawn and goal."""
        env = CrazyflieObstacleEnv(
            seed=42, n_obstacles_range=(6, 6), n_blocker_obstacles=2,
        )
        for trial in range(10):
            env.reset()
            spawn_xy = env.spawn_pos[:2]
            goal_xy = env._goal_pos[:2]
            corridor_len = float(np.linalg.norm(goal_xy - spawn_xy))
            if corridor_len < 0.05:
                continue  # spawn ≈ goal, skip
            direction = (goal_xy - spawn_xy) / corridor_len

            for bi in range(2):
                opos = env._active_obstacle_positions[bi]
                if opos[2] < -1:
                    continue
                # Project obstacle onto spawn→goal line
                vec = opos[:2] - spawn_xy
                proj = float(np.dot(vec, direction))
                # Should be in the 0.20–0.80 fraction of the corridor
                frac = proj / corridor_len
                assert 0.15 <= frac <= 0.85, (
                    f"Trial {trial}, blocker {bi}: frac={frac:.3f} outside corridor"
                )
        env.close()

    def test_tall_obstacles_reach_flight_altitude(self) -> None:
        """At least some obstacles should have tops above 0.30 m."""
        env = CrazyflieObstacleEnv(
            seed=42, n_obstacles_range=(10, 10),
        )
        env.reset(seed=42)
        max_top = 0.0
        for i in range(env._n_active_obstacles):
            opos = env._active_obstacle_positions[i]
            if opos[2] < -1:
                continue
            hh = env._obstacle_half_heights[i] if i < len(env._obstacle_half_heights) else 0.1
            top = opos[2] + hh
            max_top = max(max_top, top)
        assert max_top >= 0.30, (
            f"Tallest obstacle top = {max_top:.3f} m, should reach flight altitude"
        )
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
        # Step with zero action for a while; not testing specific collision,
        # just that the terminated flag can include obstacle_collision
        for _ in range(5):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term:
                # Obstacle collision *or* another termination — both fine
                break
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
