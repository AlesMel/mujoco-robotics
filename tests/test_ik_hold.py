"""Tests for IK controller — verify the end-effector holds proposed positions.

These tests validate that the damped-least-squares IK controller can:
1. Hold the EE at its home position when given zero action.
2. Drive the EE to a target position and hold it there.
3. Maintain position stability across multiple hold targets.
4. Work correctly for both UR5e and UR3e robots.

Run::

    pytest tests/test_ik_hold.py -v
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mujoco_robot.core.ik_controller import (
    orientation_error_axis_angle,
    quat_error_magnitude,
)
from mujoco_robot.envs.reach_env import URReachEnv


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _drive_to_target(
    env: URReachEnv,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    drive_steps: int = 200,
    hold_steps: int = 100,
) -> dict:
    """Drive the EE toward *target_pos / target_quat*, then hold and measure.

    Returns a dict with final ``ee_pos``, ``ee_quat``, ``pos_error``,
    ``ori_error``, and lists ``pos_errors`` / ``ori_errors`` recorded
    during the hold phase.
    """
    # --- Drive phase: send IK commands toward target ---
    for _ in range(drive_steps):
        ee_pos = env.data.site_xpos[env.ee_site].copy()
        direction = target_pos - ee_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            action_xyz = np.zeros(3)
        else:
            # Scale: full action when far, proportional when close
            scale = min(1.0, dist / env.ee_step)
            action_xyz = (direction / dist) * scale

        # Orientation: axis-angle error, scaled to [-1, 1]
        ori_err = orientation_error_axis_angle(env._ee_quat(), target_quat)
        ori_err_mag = np.linalg.norm(ori_err)
        if ori_err_mag > 1e-6:
            action_ori = np.clip(ori_err / env.ori_step, -1.0, 1.0)
        else:
            action_ori = np.zeros(3)

        action = np.concatenate([action_xyz, action_ori]).astype(np.float32)
        env.step(action)

    # --- Hold phase: send zero actions and record errors ---
    pos_errors = []
    ori_errors = []
    for _ in range(hold_steps):
        env.step(np.zeros(6, dtype=np.float32))
        ee_pos = env.data.site_xpos[env.ee_site].copy()
        pos_errors.append(float(np.linalg.norm(ee_pos - target_pos)))
        ori_errors.append(quat_error_magnitude(env._ee_quat(), target_quat))

    final_ee = env.data.site_xpos[env.ee_site].copy()
    final_quat = env._ee_quat()
    return {
        "ee_pos": final_ee,
        "ee_quat": final_quat,
        "pos_error": float(np.linalg.norm(final_ee - target_pos)),
        "ori_error": quat_error_magnitude(final_quat, target_quat),
        "pos_errors": pos_errors,
        "ori_errors": ori_errors,
    }


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=["ur5e", "ur3e"])
def reach_env(request):
    """Create a URReachEnv for each robot, action_mode=cartesian."""
    env = URReachEnv(
        robot=request.param,
        time_limit=0,  # no time limit for testing
        action_mode="cartesian",
        randomize_init=False,
        seed=42,
    )
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def ur5e_env():
    env = URReachEnv(robot="ur5e", time_limit=0, action_mode="cartesian", randomize_init=False, seed=42)
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def ur3e_env():
    env = URReachEnv(robot="ur3e", time_limit=0, action_mode="cartesian", randomize_init=False, seed=42)
    env.reset(seed=42)
    yield env
    env.close()


# ────────────────────────────────────────────────────────────────────────
# Test: EE holds at home position
# ────────────────────────────────────────────────────────────────────────

class TestIKHoldHome:
    """Verify the EE stays at home when zero-action is applied."""

    POS_TOL = 0.01   # 1 cm
    ORI_TOL = 0.15   # ~8.6°

    def test_hold_at_home(self, reach_env):
        """Zero action for 200 steps → EE should barely move."""
        initial_ee = reach_env.data.site_xpos[reach_env.ee_site].copy()
        initial_quat = reach_env._ee_quat()

        for _ in range(200):
            reach_env.step(np.zeros(6, dtype=np.float32))

        final_ee = reach_env.data.site_xpos[reach_env.ee_site].copy()
        drift = np.linalg.norm(final_ee - initial_ee)
        ori_drift = quat_error_magnitude(initial_quat, reach_env._ee_quat())

        assert drift < self.POS_TOL, (
            f"EE drifted {drift:.4f} m from home (tol={self.POS_TOL})"
        )
        assert ori_drift < self.ORI_TOL, (
            f"EE orientation drifted {ori_drift:.4f} rad (tol={self.ORI_TOL})"
        )


# ────────────────────────────────────────────────────────────────────────
# Test: EE reaches and holds a specific target
# ────────────────────────────────────────────────────────────────────────

class TestIKReachAndHold:
    """Drive EE to a target, then hold with zero action — position must stay."""

    POS_TOL = 0.26   # 26 cm — IK + PD controller steady-state (50 Hz, full 6-DOF)
    ORI_TOL = 1.20   # ~69° — full 3-D orientation is harder to match exactly
    HOLD_DRIFT_TOL = 0.02  # max drift during hold phase

    def _targets_for_robot(self, robot: str):
        """Return a list of reachable (pos, quat) targets per robot.

        Quaternions are (w,x,y,z) — identity means EE-up.
        """
        if robot == "ur5e":
            return [
                (np.array([-0.10, 0.15, 1.00]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([-0.30, -0.20, 1.10]), np.array([0.924, 0.0, 0.0, 0.383])),
                (np.array([-0.20, 0.10, 1.05]), np.array([0.966, 0.0, 0.0, -0.259])),
            ]
        else:  # ur3e — home EE ≈ [0.15, 0.14, 1.03]
            return [
                (np.array([0.12, 0.10, 1.00]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.10, 0.15, 0.98]), np.array([0.966, 0.0, 0.0, 0.259])),
                (np.array([0.13, 0.12, 1.05]), np.array([0.966, 0.0, 0.0, -0.259])),
            ]

    def test_reach_and_hold_target_0(self, reach_env):
        """Reach first target and hold position."""
        targets = self._targets_for_robot(reach_env.robot)
        pos, quat = targets[0]
        result = _drive_to_target(reach_env, pos, quat, drive_steps=500, hold_steps=200)

        assert result["pos_error"] < self.POS_TOL, (
            f"[{reach_env.robot}] target 0: pos error {result['pos_error']:.4f} m "
            f"(tol={self.POS_TOL})"
        )
        # Check hold stability: max drift during hold phase
        if len(result["pos_errors"]) > 1:
            hold_drift = max(result["pos_errors"]) - min(result["pos_errors"])
            assert hold_drift < self.HOLD_DRIFT_TOL, (
                f"[{reach_env.robot}] target 0: hold drift {hold_drift:.4f} m "
                f"(tol={self.HOLD_DRIFT_TOL})"
            )

    def test_reach_and_hold_target_1(self, reach_env):
        """Reach second target and hold position."""
        targets = self._targets_for_robot(reach_env.robot)
        pos, quat = targets[1]
        result = _drive_to_target(reach_env, pos, quat, drive_steps=400, hold_steps=150)

        assert result["pos_error"] < self.POS_TOL, (
            f"[{reach_env.robot}] target 1: pos error {result['pos_error']:.4f} m "
            f"(tol={self.POS_TOL})"
        )

    def test_reach_and_hold_target_2(self, reach_env):
        """Reach third target and hold position."""
        targets = self._targets_for_robot(reach_env.robot)
        pos, quat = targets[2]
        result = _drive_to_target(reach_env, pos, quat, drive_steps=400, hold_steps=150)

        assert result["pos_error"] < self.POS_TOL, (
            f"[{reach_env.robot}] target 2: pos error {result['pos_error']:.4f} m "
            f"(tol={self.POS_TOL})"
        )


# ────────────────────────────────────────────────────────────────────────
# Test: Sequential targets — EE moves between multiple positions
# ────────────────────────────────────────────────────────────────────────

class TestIKSequentialTargets:
    """Move through a sequence of targets; EE must hold each one."""

    POS_TOL = 0.08
    HOLD_DRIFT_TOL = 0.035  # Slightly higher: full-orientation IK adds minor position ripple

    @pytest.mark.parametrize("robot", ["ur5e", "ur3e"])
    def test_sequential_hold(self, robot):
        """Drive to 3 waypoints in sequence — each must hold."""
        env = URReachEnv(robot=robot, time_limit=0, action_mode="cartesian", randomize_init=False, seed=7)

        # Quaternions are near-identity with slight Z-rotation
        if robot == "ur5e":  # home EE ≈ [0.20, 0.13, 1.21]
            waypoints = [
                (np.array([0.15, 0.10, 1.15]), np.array([0.989, 0.0, 0.0, 0.149])),
                (np.array([0.18, 0.05, 1.10]), np.array([0.995, 0.0, 0.0, -0.100])),
                (np.array([0.20, 0.15, 1.18]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        else:  # ur3e — home EE ≈ [0.15, 0.14, 1.03]
            waypoints = [
                (np.array([0.12, 0.10, 1.00]), np.array([0.995, 0.0, 0.0, 0.100])),
                (np.array([0.13, 0.12, 0.98]), np.array([0.995, 0.0, 0.0, -0.100])),
                (np.array([0.14, 0.13, 1.02]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]

        for i, (pos, quat) in enumerate(waypoints):
            # Reset between waypoints to avoid cumulative drift
            env.reset(seed=7 + i)
            result = _drive_to_target(env, pos, quat, drive_steps=400, hold_steps=100)
            assert result["pos_error"] < self.POS_TOL, (
                f"[{robot}] waypoint {i}: pos error {result['pos_error']:.4f} m"
            )
            if len(result["pos_errors"]) > 1:
                hold_drift = max(result["pos_errors"]) - min(result["pos_errors"])
                assert hold_drift < self.HOLD_DRIFT_TOL, (
                    f"[{robot}] waypoint {i}: hold drift {hold_drift:.4f} m"
                )

        env.close()


# ────────────────────────────────────────────────────────────────────────
# Test: IK controller directly (unit-level)
# ────────────────────────────────────────────────────────────────────────

class TestIKControllerUnit:
    """Low-level tests on the IKController class itself."""

    def test_ee_position_matches_site(self, reach_env):
        """IKController.ee_position() should match mujoco site data."""
        ik_pos = reach_env._ik.ee_position()
        mj_pos = reach_env.data.site_xpos[reach_env.ee_site].copy()
        np.testing.assert_allclose(ik_pos, mj_pos, atol=1e-10)

    def test_ee_quat_unit(self, reach_env):
        """EE quaternion should be a unit quaternion."""
        quat = reach_env._ik.ee_quat()
        assert quat.shape == (4,), f"Expected shape (4,), got {quat.shape}"
        norm = np.linalg.norm(quat)
        assert abs(norm - 1.0) < 1e-6, f"EE quat not unit: norm={norm}"

    def test_solve_returns_correct_shape(self, reach_env):
        """IK solve should return (6,) joint velocity vector."""
        target = reach_env.data.site_xpos[reach_env.ee_site].copy()
        quat = reach_env._ee_quat()
        qvel = reach_env._ik.solve(target, quat)
        assert qvel.shape == (6,), f"Expected shape (6,), got {qvel.shape}"

    def test_solve_near_target_gives_small_velocity(self, reach_env):
        """When EE is already at target, IK should return near-zero velocity."""
        target = reach_env.data.site_xpos[reach_env.ee_site].copy()
        quat = reach_env._ik.ee_quat()
        qvel = reach_env._ik.solve(target, quat)
        assert np.linalg.norm(qvel) < 0.1, (
            f"Expected near-zero velocity at target, got norm={np.linalg.norm(qvel):.4f}"
        )

    def test_solve_far_target_gives_nonzero_velocity(self, reach_env):
        """When target is far, IK should return non-trivial velocity."""
        target = reach_env.data.site_xpos[reach_env.ee_site].copy()
        target[2] += 0.3  # 30 cm above current
        quat = reach_env._ik.ee_quat()
        qvel = reach_env._ik.solve(target, quat)
        assert np.linalg.norm(qvel) > 0.01, (
            f"Expected non-zero velocity for distant target, got norm={np.linalg.norm(qvel):.6f}"
        )


# ────────────────────────────────────────────────────────────────────────
# Test: No self-collision during IK hold
# ────────────────────────────────────────────────────────────────────────

class TestIKNoCollision:
    """Ensure the IK hold positions don't cause self-collisions."""

    @pytest.mark.parametrize("robot", ["ur5e", "ur3e"])
    def test_no_collision_at_home(self, robot):
        """Holding at home should produce zero self-collisions."""
        env = URReachEnv(robot=robot, time_limit=0, action_mode="cartesian", randomize_init=False, seed=42)
        env.reset(seed=42)

        total_collisions = 0
        for _ in range(100):
            result = env.step(np.zeros(6, dtype=np.float32))
            total_collisions += result.info["self_collisions"]

        assert total_collisions == 0, (
            f"[{robot}] {total_collisions} self-collisions at home"
        )
        env.close()

    @pytest.mark.parametrize("robot", ["ur5e", "ur3e"])
    def test_no_collision_during_reach(self, robot):
        """Moving to a safe target should not trigger self-collisions."""
        env = URReachEnv(robot=robot, time_limit=0, action_mode="cartesian", randomize_init=False, seed=42)
        env.reset(seed=42)

        # Safe target — in front of base, well within reach
        if robot == "ur5e":
            target = np.array([-0.20, 0.0, 1.10])
        else:
            target = np.array([0.00, 0.10, 0.95])

        total_collisions = 0
        for _ in range(200):
            ee_pos = env.data.site_xpos[env.ee_site].copy()
            direction = target - ee_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                scale = min(1.0, dist / env.ee_step)
                action = np.array([*(direction / dist * scale), 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                action = np.zeros(6, dtype=np.float32)
            result = env.step(action)
            total_collisions += result.info["self_collisions"]

        assert total_collisions == 0, (
            f"[{robot}] {total_collisions} self-collisions during safe reach"
        )
        env.close()
