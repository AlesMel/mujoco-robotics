"""Comprehensive diagnostic tests for all reach variants.

Converted from ``scripts/diag_all_variants.py`` into a proper pytest module
so these checks run as part of CI and ``pytest tests/``.

Checks per variant (ik_rel, ik_abs, joint_pos):
1. Quaternion math consistency (shared, run once)
2. Observation vector shape and layout
3. Observation consistency with MuJoCo ground truth
4. Zero-action stability
5. Proportional controller convergence
6. Orientation error direction consistency
7. Reward sanity
8. IK controller produces sensible velocities (IK variants only)
9. ``last_action`` tracking in observation
10. Goal quaternion consistency after stepping
11. Cross-variant initial-obs consistency
"""
from __future__ import annotations

import numpy as np
import pytest

from mujoco_robot.core.ik_controller import (
    axis_angle_from_quat,
    orientation_error_axis_angle,
    quat_conjugate,
    quat_error_magnitude,
    quat_multiply,
    quat_unique,
)
from mujoco_robot.tasks.reach import ReachIKAbsEnv, ReachIKRelEnv, ReachJointPosEnv

ROBOT = "ur3e"
SEED = 42
VARIANT_CLS = {
    "ik_rel": ReachIKRelEnv,
    "ik_abs": ReachIKAbsEnv,
    "joint_pos": ReachJointPosEnv,
}


def _make(variant: str, **kw):
    return VARIANT_CLS[variant](
        robot=ROBOT,
        time_limit=0,
        randomize_init=False,
        obs_noise=0.0,
        seed=SEED,
        **kw,
    )


# ───────────────────────────────────────────────────────────────────
# 0. Quaternion math unit checks (shared — not per-variant)
# ───────────────────────────────────────────────────────────────────
class TestQuaternionMath:
    """Validate quaternion helpers used throughout the library."""

    def test_identity_multiply(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(quat_multiply(q_id, q_id), q_id)

    def test_q_times_conjugate_is_identity(self):
        q = quat_unique(np.array([0.5, 0.5, 0.5, 0.5]))
        prod = quat_multiply(q, quat_conjugate(q))
        np.testing.assert_allclose(prod, [1, 0, 0, 0], atol=1e-10)

    def test_axis_angle_90_deg_z(self):
        angle_90 = np.pi / 2
        half = angle_90 / 2
        q_z90 = np.array([np.cos(half), 0, 0, np.sin(half)])
        aa = axis_angle_from_quat(q_z90)
        assert abs(np.linalg.norm(aa) - angle_90) < 1e-8
        assert abs(aa[2]) > abs(aa[0]) and abs(aa[2]) > abs(aa[1])

    def test_error_magnitude_identical_is_zero(self):
        q = quat_unique(np.array([0.5, 0.5, 0.5, 0.5]))
        assert quat_error_magnitude(q, q) < 1e-8

    def test_error_magnitude_opposed_is_large(self):
        half = np.pi / 4
        q_z90 = np.array([np.cos(half), 0, 0, np.sin(half)])
        err = quat_error_magnitude(q_z90, quat_conjugate(q_z90))
        assert err > 0.5

    def test_orientation_error_direction_positive_z(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        half = np.pi / 4
        q_z90 = np.array([np.cos(half), 0, 0, np.sin(half)])
        err = orientation_error_axis_angle(q_id, q_z90)
        assert err[2] > 0, f"Expected +Z direction, got {err}"

    def test_orientation_error_direction_negative_z(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        half = np.pi / 4
        q_z90 = np.array([np.cos(half), 0, 0, np.sin(half)])
        err = orientation_error_axis_angle(q_z90, q_id)
        assert err[2] < 0, f"Expected -Z direction, got {err}"


# ───────────────────────────────────────────────────────────────────
# Per-variant parametrised tests
# ───────────────────────────────────────────────────────────────────
@pytest.fixture(params=["ik_rel", "ik_abs", "joint_pos"])
def variant(request):
    return request.param


@pytest.fixture
def env(variant):
    e = _make(variant)
    e.reset(seed=SEED)
    yield e
    e.close()


class TestObservationShape:
    """Obs vector shape and layout (36-D for 6-DOF with rot6d)."""

    def test_shape(self, env, variant):
        obs = env.reset(seed=SEED)
        expected_dim = 30 + env.action_dim
        assert obs.shape == (expected_dim,), f"{variant}: obs shape {obs.shape}"
        assert env.observation_dim == expected_dim
        assert env.action_dim == 6

    def test_layout_consumes_all_dims(self, env, variant):
        obs = env.reset(seed=SEED)
        expected_dim = 30 + env.action_dim
        # Decompose: joint_pos(6) + joint_vel(6) + ee_rot6d(6) + goal_pos(3) + goal_rot6d(6) + ori_err_vec(3) + last_action(6)
        idx = 6 + 6 + 6 + 3 + 6 + 3 + 6
        assert idx == expected_dim, f"Observation layout mismatch: {idx} vs {expected_dim}"


class TestObservationGroundTruth:
    """Observation values should match MuJoCo state."""

    def test_goal_pos_in_base_frame(self, env):
        obs = env.reset(seed=SEED)
        goal_pos_obs = obs[18:21]
        goal_pos_base = env.goal_pos - env._BASE_POS
        np.testing.assert_allclose(goal_pos_obs, goal_pos_base, atol=1e-5)

    def test_goal_rot6d_is_unit_columns(self, env):
        obs = env.reset(seed=SEED)
        goal_rot6d_obs = obs[21:27]
        # rot6d is two columns of rotation matrix; each should be unit length
        col1 = goal_rot6d_obs[:3]
        col2 = goal_rot6d_obs[3:6]
        assert abs(np.linalg.norm(col1) - 1.0) < 1e-4
        assert abs(np.linalg.norm(col2) - 1.0) < 1e-4

    def test_last_action_zeros_at_reset(self, env):
        obs = env.reset(seed=SEED)
        last_action = obs[30:36]
        np.testing.assert_allclose(last_action, 0.0, atol=1e-8)

    def test_orientation_error_vector_matches_env(self, env):
        obs = env.reset(seed=SEED)
        ori_err_obs = obs[27:30]
        np.testing.assert_allclose(ori_err_obs, env._orientation_error(), atol=1e-5)

    def test_joint_pos_relative_to_home(self, env):
        import mujoco

        obs = env.reset(seed=SEED)
        for qi, j in enumerate(env.robot_joints):
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            qpos = env.data.qpos[env.model.jnt_qposadr[jid]]
            expected = qpos - env.init_q[qi]
            assert abs(obs[qi] - expected) < 1e-5, (
                f"joint_pos[{qi}] obs={obs[qi]:.5f} vs expected={expected:.5f}"
            )


class TestZeroActionStability:
    """Physics must not explode under zero action."""

    def test_no_nans(self, env, variant):
        for _ in range(50):
            res = env.step(np.zeros(6))
        assert not np.any(np.isnan(res.obs)), f"{variant}: NaN in obs"
        assert not np.any(np.isinf(res.obs)), f"{variant}: Inf in obs"

    def test_position_drift_bounded(self, env, variant):
        ee_before = env.data.site_xpos[env.ee_site].copy()
        for _ in range(50):
            env.step(np.zeros(6))
        ee_after = env.data.site_xpos[env.ee_site].copy()
        drift = np.linalg.norm(ee_after - ee_before)
        if variant == "ik_abs":
            # ik_abs: zero action = "go to workspace center", drift expected
            assert drift < 1.0, f"ik_abs drift {drift:.4f} m"
        else:
            assert drift < 0.05, f"{variant} drift {drift:.5f} m"

    def test_joint_velocities_bounded(self, env, variant):
        for _ in range(50):
            env.step(np.zeros(6))
        max_qvel = max(abs(env.data.qvel[d]) for d in env.robot_dofs)
        assert max_qvel < 10.0, f"max joint vel {max_qvel:.3f} rad/s"


class TestProportionalReach:
    """A proportional controller should converge toward the goal."""

    N_STEPS = 200

    def _run_proportional(self, env, variant):
        dists, ori_errs = [], []
        for _ in range(self.N_STEPS):
            if variant == "ik_rel":
                ee_pos = env.data.site_xpos[env.ee_site].copy()
                diff = env.goal_pos - ee_pos
                d = np.linalg.norm(diff)
                pos_act = (diff / max(d, 1e-6)) * min(1.0, d / env.ee_step) if d > 1e-6 else np.zeros(3)
                ori_err_vec = orientation_error_axis_angle(env._ee_quat(), env.goal_quat)
                ori_mag = np.linalg.norm(ori_err_vec)
                ori_act = (ori_err_vec / max(ori_mag, 1e-6)) * min(1.0, ori_mag / env.ori_step) if ori_mag > 1e-6 else np.zeros(3)
                act = np.concatenate([pos_act, ori_act])
            elif variant == "ik_abs":
                lo, hi = env.ee_bounds[:, 0], env.ee_bounds[:, 1]
                pos_act = 2.0 * (env.goal_pos - lo) / (hi - lo) - 1.0
                pos_act = np.clip(pos_act, -1, 1)
                q_err = quat_multiply(env.goal_quat, quat_conjugate(env._home_quat))
                q_err = quat_unique(q_err)
                aa = axis_angle_from_quat(q_err)
                ori_act = np.clip(aa / env.ori_abs_max, -1, 1)
                act = np.concatenate([pos_act, ori_act])
            else:  # joint_pos
                qvel_cmd = env._ik.solve(env.goal_pos, env.goal_quat)
                act = np.clip(qvel_cmd * 0.3 / env.joint_action_scale, -1, 1)

            res = env.step(act)
            dists.append(res.info["dist"])
            ori_errs.append(res.info["ori_err"])
        return dists, ori_errs

    def test_min_distance_close(self, env, variant):
        dists, _ = self._run_proportional(env, variant)
        assert min(dists) < 0.10, f"{variant}: min dist {min(dists):.4f}"

    def test_distance_decreasing(self, env, variant):
        dists, _ = self._run_proportional(env, variant)
        if variant == "ik_abs":
            pytest.skip("ik_abs oscillates by design (absolute target mapping)")
        q1 = np.mean(dists[: self.N_STEPS // 4])
        q4 = np.mean(dists[-self.N_STEPS // 4 :])
        assert q4 < q1, f"{variant}: first_q={q1:.4f} last_q={q4:.4f}"


class TestOrientationError:
    """Orientation error magnitude should match axis-angle norm."""

    def test_magnitude_consistency(self, env):
        ori_err_vec = env._orientation_error()
        mag_vec = np.linalg.norm(ori_err_vec)
        mag_quat = env._orientation_error_magnitude()
        assert abs(mag_vec - mag_quat) < 1e-5


class TestRewardSanity:
    """Basic reward boundedness checks."""

    def test_reward_is_finite(self, variant):
        env = VARIANT_CLS[variant](
            robot=ROBOT, time_limit=50, randomize_init=False, obs_noise=0.0, seed=SEED,
        )
        env.reset(seed=SEED)
        res = env.step(np.zeros(6))
        assert np.isfinite(res.reward), f"Reward not finite: {res.reward}"
        env.close()

    def test_reward_not_excessive_when_far(self, variant):
        env = VARIANT_CLS[variant](
            robot=ROBOT, time_limit=50, randomize_init=False, obs_noise=0.0, seed=SEED,
        )
        env.reset(seed=SEED)
        res = env.step(np.zeros(6))
        assert res.reward < 0.5, f"Reward too high when far from goal: {res.reward}"
        env.close()


class TestIKControllerSanity:
    """IK controller produces sensible joint velocities (IK variants only)."""

    @pytest.fixture(params=["ik_rel", "ik_abs"])
    def ik_env(self, request):
        e = _make(request.param)
        e.reset(seed=SEED)
        yield e
        e.close()

    def test_no_nans(self, ik_env):
        qvel = ik_env._ik.solve(ik_env.goal_pos, ik_env.goal_quat)
        assert not np.any(np.isnan(qvel))

    def test_no_infs(self, ik_env):
        qvel = ik_env._ik.solve(ik_env.goal_pos, ik_env.goal_quat)
        assert not np.any(np.isinf(qvel))

    def test_nonzero(self, ik_env):
        qvel = ik_env._ik.solve(ik_env.goal_pos, ik_env.goal_quat)
        assert np.linalg.norm(qvel) > 1e-6

    def test_not_wild(self, ik_env):
        qvel = ik_env._ik.solve(ik_env.goal_pos, ik_env.goal_quat)
        assert np.max(np.abs(qvel)) < 50


class TestLastAction:
    """Observation's ``last_action`` slot must track sent actions."""

    def test_tracks_sent_action(self, env):
        act = np.array([0.5, -0.3, 0.1, 0.0, 0.2, -0.8], dtype=np.float32)
        res = env.step(act)
        np.testing.assert_allclose(res.obs[22:28], act, atol=1e-5)

    def test_updates_each_step(self, env):
        env.step(np.array([0.5, -0.3, 0.1, 0.0, 0.2, -0.8], dtype=np.float32))
        act2 = np.array([-0.1, 0.4, 0.0, 0.3, -0.5, 0.2], dtype=np.float32)
        res2 = env.step(act2)
        np.testing.assert_allclose(res2.obs[22:28], act2, atol=1e-5)


class TestGoalConsistency:
    """Goal pose in observation must remain consistent after stepping."""

    def test_goal_quat_stable(self, env):
        env.step(np.zeros(6))
        obs = env._observe()
        np.testing.assert_allclose(obs[15:19], env.goal_quat, atol=1e-5)

    def test_goal_pos_stable(self, env):
        env.step(np.zeros(6))
        obs = env._observe()
        expected = env.goal_pos - env._BASE_POS
        np.testing.assert_allclose(obs[12:15], expected, atol=1e-5)

    def test_goal_quat_unit_after_step(self, env):
        env.step(np.zeros(6))
        obs = env._observe()
        assert abs(np.linalg.norm(obs[15:19]) - 1.0) < 1e-5


class TestCrossVariantConsistency:
    """Same seed → same initial base observation across all variants."""

    def test_base_obs_identical(self):
        observations = {}
        for v in ["ik_rel", "ik_abs", "joint_pos"]:
            e = _make(v)
            obs = e.reset(seed=SEED)
            observations[v] = obs[:22].copy()
            e.close()

        np.testing.assert_allclose(
            observations["ik_rel"], observations["ik_abs"], atol=1e-5,
            err_msg="ik_rel vs ik_abs base obs differ",
        )
        np.testing.assert_allclose(
            observations["ik_rel"], observations["joint_pos"], atol=1e-5,
            err_msg="ik_rel vs joint_pos base obs differ",
        )
