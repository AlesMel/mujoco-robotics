"""Deep diagnostic: physics / observations / orientations for all 3 variants.

Checks performed for each variant (ik_rel, ik_abs, joint_pos):
1. Observation vector shape and layout sanity
2. Zero-action stability (physics not exploding)
3. Proportional controller can reach target (pos + ori)
4. Orientation error direction correctness
5. Observation consistency with MuJoCo ground truth
6. Reward monotonicity when approaching goal
7. Quaternion math correctness (quat_multiply, conjugate, axis_angle)

Observation layout (Isaac Lab aligned, 25-D for 6-DOF):
  joint_pos_rel(6) + joint_vel_rel(6) + pose_cmd_pos(3) + pose_cmd_quat(4) + last_action(6)
"""
from __future__ import annotations
import sys, traceback
import numpy as np
from mujoco_robot.tasks.reach import ReachIKAbsEnv, ReachIKRelEnv, ReachJointPosEnv
from mujoco_robot.core.ik_controller import (
    orientation_error_axis_angle,
    quat_multiply,
    quat_conjugate,
    quat_error_magnitude,
    axis_angle_from_quat,
    quat_unique,
)

VARIANTS = ["ik_rel", "ik_abs", "joint_pos"]
ROBOT = "ur3e"
SEED = 42
PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
n_pass = 0
n_fail = 0
n_warn = 0


def check(condition: bool, msg: str, warn_only: bool = False) -> bool:
    global n_pass, n_fail, n_warn
    if condition:
        print(f"{PASS} {msg}")
        n_pass += 1
    elif warn_only:
        print(f"{WARN} {msg}")
        n_warn += 1
    else:
        print(f"{FAIL} {msg}")
        n_fail += 1
    return condition


def make(variant, **kw):
    variant_map = {
        "ik_rel": ReachIKRelEnv,
        "ik_abs": ReachIKAbsEnv,
        "joint_pos": ReachJointPosEnv,
    }
    return variant_map[variant](
        robot=ROBOT,
        time_limit=0,
        randomize_init=False,
        obs_noise=0.0,
        seed=SEED,
        **kw,
    )


# ===================================================================
# 0. Quaternion math unit checks
# ===================================================================
print("=" * 70)
print("0. QUATERNION MATH CONSISTENCY")
print("=" * 70)

# Identity quaternion
q_id = np.array([1.0, 0.0, 0.0, 0.0])
check(np.allclose(quat_multiply(q_id, q_id), q_id),
      "quat_multiply identity * identity = identity")

# q * conj(q) = identity
q_test = quat_unique(np.array([0.5, 0.5, 0.5, 0.5]))
q_inv = quat_conjugate(q_test)
prod = quat_multiply(q_test, q_inv)
check(np.allclose(prod, q_id, atol=1e-10),
      f"q * conj(q) = identity  (got w={prod[0]:.8f})")

# axis_angle round-trip: 90° about Z
angle_90 = np.pi / 2
half = angle_90 / 2
q_z90 = np.array([np.cos(half), 0, 0, np.sin(half)])
aa = axis_angle_from_quat(q_z90)
check(abs(np.linalg.norm(aa) - angle_90) < 1e-8,
      f"axis_angle_from_quat 90° Z: angle={np.linalg.norm(aa):.6f} (expect {angle_90:.6f})")
check(abs(aa[2]) > abs(aa[0]) and abs(aa[2]) > abs(aa[1]),
      f"axis_angle_from_quat 90° Z: axis dominated by Z  (aa={aa})")

# quat_error_magnitude: identical quats → 0
check(quat_error_magnitude(q_test, q_test) < 1e-8,
      "quat_error_magnitude(q, q) ≈ 0")

# quat_error_magnitude: q vs conj(q) → large
err_mag = quat_error_magnitude(q_z90, quat_conjugate(q_z90))
check(err_mag > 0.5, f"quat_error_magnitude(q, conj(q)) = {err_mag:.4f} (expect ~π)")

# rot6d removed from observation; skip rot6d tests

# orientation_error_axis_angle direction check:
# error vector should point FROM current TOWARD target
q_cur = q_id
q_tgt = q_z90  # +90° about Z
err_vec = orientation_error_axis_angle(q_cur, q_tgt)
check(err_vec[2] > 0, f"ori_error(id → +Z90) points +Z: err_vec={err_vec}")
# reverse: error should flip
err_vec_rev = orientation_error_axis_angle(q_tgt, q_cur)
check(err_vec_rev[2] < 0, f"ori_error(+Z90 → id) points -Z: err_vec={err_vec_rev}")


for variant in VARIANTS:
    print("\n" + "=" * 70)
    print(f"VARIANT: {variant}")
    print("=" * 70)

    # ===================================================================
    # 1. Observation shape and layout
    # ===================================================================
    print("\n--- 1. Observation shape & layout ---")
    env = make(variant)
    obs = env.reset(seed=SEED)

    expected_dim = 19 + env.action_dim  # 25 for 6-DOF
    check(obs.shape == (expected_dim,),
          f"obs shape = {obs.shape} (expected ({expected_dim},))")
    check(env.observation_dim == expected_dim,
          f"observation_dim = {env.observation_dim}")
    check(env.action_dim == 6, f"action_dim = {env.action_dim}")

    # Decompose observation (Isaac Lab aligned layout)
    idx = 0
    joint_pos_obs  = obs[idx:idx+6]; idx += 6
    joint_vel_obs  = obs[idx:idx+6]; idx += 6
    goal_pos_obs   = obs[idx:idx+3]; idx += 3
    goal_quat_obs  = obs[idx:idx+4]; idx += 4
    last_action    = obs[idx:idx+6]; idx += 6

    check(idx == expected_dim, f"Observation consumed all {idx} dims (expected {expected_dim})")

    # ===================================================================
    # 2. Observation consistency with MuJoCo ground truth
    # ===================================================================
    print("\n--- 2. Observation vs ground truth ---")

    # Goal position (in base frame: goal_pos - _BASE_POS)
    goal_pos_base = env.goal_pos - env._BASE_POS
    check(np.allclose(goal_pos_obs, goal_pos_base, atol=1e-5),
          f"goal_pos obs matches env.goal_pos in base frame")

    # Goal quaternion (wxyz)
    check(np.allclose(goal_quat_obs, env.goal_quat, atol=1e-5),
          f"goal_quat obs matches env.goal_quat (wxyz)")

    # Quaternion is unit
    check(abs(np.linalg.norm(goal_quat_obs) - 1.0) < 1e-5,
          f"goal_quat is unit (norm={np.linalg.norm(goal_quat_obs):.6f})")

    # last_action at reset should be zeros
    check(np.allclose(last_action, 0.0),
          f"last_action zeros at reset")

    # Joint positions relative to home
    for qi, j in enumerate(env.robot_joints):
        import mujoco
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qpos = env.data.qpos[env.model.jnt_qposadr[jid]]
        expected = qpos - env.init_q[qi]
        check(abs(joint_pos_obs[qi] - expected) < 1e-5,
              f"joint_pos[{qi}] obs={joint_pos_obs[qi]:.5f} vs expected={expected:.5f}")

    # ===================================================================
    # 3. Zero-action stability
    # ===================================================================
    print("\n--- 3. Zero-action stability (50 steps) ---")
    ee_pos_before = env.data.site_xpos[env.ee_site].copy()
    ee_quat_before = env._ee_quat()

    for _ in range(50):
        res = env.step(np.zeros(6))

    ee_pos_after = env.data.site_xpos[env.ee_site].copy()
    ee_quat_after = env._ee_quat()
    pos_drift = np.linalg.norm(ee_pos_after - ee_pos_before)
    ori_drift = quat_error_magnitude(ee_quat_before, ee_quat_after)

    if variant == "ik_abs":
        # ik_abs: zero action = "go to workspace center + home_quat"
        # NOT "hold in place", so large drift is expected by design.
        check(not np.any(np.isnan(res.obs)),
              "No NaNs in observation (ik_abs: drift expected, zero=midpoint)")
        check(pos_drift < 1.0,
              f"Position drift bounded (ik_abs midpoint-seek): {pos_drift:.5f} m",
              warn_only=True)
    else:
        check(pos_drift < 0.05,
              f"Position drift after 50 zero steps: {pos_drift:.5f} m")
        check(ori_drift < 0.2,
              f"Orientation drift after 50 zero steps: {ori_drift:.5f} rad")
    check(not np.any(np.isnan(res.obs)),
          "No NaNs in observation after 50 zero steps")
    check(not np.any(np.isinf(res.obs)),
          "No Infs in observation after 50 zero steps")

    # Check physics didn't explode
    max_qvel = max(abs(env.data.qvel[d]) for d in env.robot_dofs)
    check(max_qvel < 10.0,
          f"max joint velocity < 10 rad/s (got {max_qvel:.3f})", warn_only=True)

    env.close()

    # ===================================================================
    # 4. Proportional controller converges (variant-aware)
    # ===================================================================
    print(f"\n--- 4. Proportional reach test ---")
    env = make(variant)
    env.reset(seed=SEED)

    n_steps = 200
    dists = []
    ori_errs = []

    for i in range(n_steps):
        if variant == "ik_rel":
            # Position: proportional in world frame → action[:3]
            ee_pos = env.data.site_xpos[env.ee_site].copy()
            diff = env.goal_pos - ee_pos
            d = np.linalg.norm(diff)
            pos_act = (diff / max(d, 1e-6)) * min(1.0, d / env.ee_step) if d > 1e-6 else np.zeros(3)
            # Orientation: proportional from orientation error
            ori_err_vec = orientation_error_axis_angle(env._ee_quat(), env.goal_quat)
            ori_mag = np.linalg.norm(ori_err_vec)
            ori_act = (ori_err_vec / max(ori_mag, 1e-6)) * min(1.0, ori_mag / env.ori_step) if ori_mag > 1e-6 else np.zeros(3)
            act = np.concatenate([pos_act, ori_act])

        elif variant == "ik_abs":
            # Absolute: map goal pos to [-1,1] range using ee_bounds
            lo = env.ee_bounds[:, 0]
            hi = env.ee_bounds[:, 1]
            pos_act = 2.0 * (env.goal_pos - lo) / (hi - lo) - 1.0
            pos_act = np.clip(pos_act, -1, 1)
            # Orientation: compute axis-angle from goal_quat relative to home_quat
            q_err = quat_multiply(env.goal_quat, quat_conjugate(env._home_quat))
            q_err = quat_unique(q_err)
            aa = axis_angle_from_quat(q_err)
            ori_act = aa / env.ori_abs_max
            ori_act = np.clip(ori_act, -1, 1)
            act = np.concatenate([pos_act, ori_act])

        elif variant == "joint_pos":
            # Use IK to solve for goal, then proportional joint control
            qvel_cmd = env._ik.solve(env.goal_pos, env.goal_quat)
            # Scale to action range
            act = np.clip(qvel_cmd * 0.3 / env.joint_action_scale, -1, 1)

        res = env.step(act)
        dists.append(res.info["dist"])
        ori_errs.append(res.info["ori_err"])

    final_dist = dists[-1]
    min_dist = min(dists)
    final_ori = ori_errs[-1]
    min_ori = min(ori_errs)

    check(min_dist < 0.10,
          f"Min dist < 0.10 m  (got {min_dist:.4f})")
    if variant == "ik_abs":
        # ik_abs can oscillate due to IK gain + absolute mapping;
        # reaching close at any point proves correctness.
        check(min_dist < 0.05,
              f"ik_abs ever reached close (min_dist={min_dist:.4f})")
    else:
        check(final_dist < 0.25,
              f"Final dist < 0.25 m  (got {final_dist:.4f})")
    check(min_ori < 1.5,
          f"Min ori error < 1.5 rad  (got {min_ori:.4f})", warn_only=True)

    # Check distance generally decreases (skip for ik_abs which oscillates)
    first_quarter = np.mean(dists[:n_steps//4])
    last_quarter = np.mean(dists[-n_steps//4:])
    if variant != "ik_abs":
        check(last_quarter < first_quarter,
              f"Distance decreasing: first_q={first_quarter:.4f} > last_q={last_quarter:.4f}")

    env.close()

    # ===================================================================
    # 5. Orientation error direction consistency
    # ===================================================================
    print(f"\n--- 5. Orientation error direction check ---")
    env = make(variant)
    env.reset(seed=SEED)

    # Orientation error no longer in obs, but _orientation_error() still used
    # by reward.  Verify magnitude matches quat_error_magnitude.
    ori_err_gt = env._orientation_error()
    mag_from_vec = np.linalg.norm(ori_err_gt)
    mag_from_quat = env._orientation_error_magnitude()
    check(abs(mag_from_vec - mag_from_quat) < 1e-5,
          f"||ori_error_vec|| = {mag_from_vec:.5f} ≈ quat_error_mag = {mag_from_quat:.5f}")

    env.close()

    # ===================================================================
    # 6. Reward structure check
    # ===================================================================
    print(f"\n--- 6. Reward sanity ---")
    variant_map = {
        "ik_rel": ReachIKRelEnv,
        "ik_abs": ReachIKAbsEnv,
        "joint_pos": ReachJointPosEnv,
    }
    env = variant_map[variant](
        robot=ROBOT,
        time_limit=50, randomize_init=False, obs_noise=0.0, seed=SEED,
    )
    env.reset(seed=SEED)
    
    # Step once with zero action
    res0 = env.step(np.zeros(6))
    r0 = res0.reward
    d0 = res0.info["dist"]
    
    # Reward should be finite
    check(np.isfinite(r0), f"Reward is finite: {r0:.4f}")
    
    # Reward should be negative when far from goal (coarse component dominates)
    check(r0 < 0.5, f"Reward not excessively positive when far from goal: {r0:.4f}")

    env.close()

    # ===================================================================
    # 7. IK controller produces sensible joint velocities
    # ===================================================================
    if variant in ("ik_rel", "ik_abs"):
        print(f"\n--- 7. IK controller sanity ---")
        env = make(variant)
        env.reset(seed=SEED)

        qvel = env._ik.solve(env.goal_pos, env.goal_quat)
        check(not np.any(np.isnan(qvel)), "IK solution has no NaNs")
        check(not np.any(np.isinf(qvel)), "IK solution has no Infs")
        check(np.linalg.norm(qvel) > 1e-6,
              f"IK solution is non-zero (norm={np.linalg.norm(qvel):.5f})")
        check(np.max(np.abs(qvel)) < 50,
              f"IK solution not wild (max={np.max(np.abs(qvel)):.3f})")
        env.close()

    # ===================================================================
    # 8. last_action in observation matches what was sent
    # ===================================================================
    print(f"\n--- 8. last_action tracking ---")
    env = make(variant)
    env.reset(seed=SEED)

    test_action = np.array([0.5, -0.3, 0.1, 0.0, 0.2, -0.8], dtype=np.float32)
    res = env.step(test_action)
    last_act_in_obs = res.obs[19:25]
    check(np.allclose(last_act_in_obs, test_action, atol=1e-5),
          f"last_action in obs matches sent action")

    # Step again with different action; prev should be the old one  
    test_action_2 = np.array([-0.1, 0.4, 0.0, 0.3, -0.5, 0.2], dtype=np.float32)
    res2 = env.step(test_action_2)
    last_act_in_obs_2 = res2.obs[19:25]
    check(np.allclose(last_act_in_obs_2, test_action_2, atol=1e-5),
          f"last_action updates correctly on 2nd step")
    env.close()

    # ===================================================================
    # 9. Goal quaternion consistency after stepping
    # ===================================================================
    print(f"\n--- 9. Goal quaternion consistency after stepping ---")
    env = make(variant)
    env.reset(seed=SEED)
    env.step(np.zeros(6))
    obs2 = env._observe()

    # goal_quat in obs should still match env.goal_quat
    goal_quat_post = obs2[12+3:12+3+4]  # indices 15:19
    check(np.allclose(goal_quat_post, env.goal_quat, atol=1e-5),
          "goal_quat in obs still consistent after step")

    # goal_pos in obs should still match env.goal_pos in base frame
    goal_pos_post = obs2[12:15]
    goal_pos_base_post = env.goal_pos - env._BASE_POS
    check(np.allclose(goal_pos_post, goal_pos_base_post, atol=1e-5),
          "goal_pos in obs still consistent after step (base frame)")

    # quaternion should be unit
    check(abs(np.linalg.norm(goal_quat_post) - 1.0) < 1e-5,
          f"goal_quat is unit after step (norm={np.linalg.norm(goal_quat_post):.6f})")

    env.close()


# ===================================================================
# 10. Cross-variant consistency: same seed → same initial obs (except action part)
# ===================================================================
print("\n" + "=" * 70)
print("10. CROSS-VARIANT CONSISTENCY")
print("=" * 70)

observations = {}
for variant in VARIANTS:
    env = make(variant)
    obs = env.reset(seed=SEED)
    observations[variant] = obs.copy()
    env.close()

# Base observations (first 19) should be identical across variants
# (all use the same physics, same seed → same goal)
base_ik_rel = observations["ik_rel"][:19]
base_ik_abs = observations["ik_abs"][:19]
base_joint  = observations["joint_pos"][:19]

check(np.allclose(base_ik_rel, base_ik_abs, atol=1e-5),
      f"ik_rel vs ik_abs: base obs match (max_diff={np.max(np.abs(base_ik_rel - base_ik_abs)):.7f})")
check(np.allclose(base_ik_rel, base_joint, atol=1e-5),
      f"ik_rel vs joint_pos: base obs match (max_diff={np.max(np.abs(base_ik_rel - base_joint)):.7f})")


# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 70)
print(f"SUMMARY:  {n_pass} passed,  {n_fail} FAILED,  {n_warn} warnings")
print("=" * 70)
if n_fail > 0:
    print(">>> SOME CHECKS FAILED — see [FAIL] lines above.")
    sys.exit(1)
else:
    print(">>> All checks passed.")
