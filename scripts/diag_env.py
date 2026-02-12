"""Diagnostic: check if the environment is working properly."""
from mujoco_robot.tasks.reach import ReachIKRelEnv, ReachJointPosEnv
import numpy as np

# Test with JOINT mode (default for training)
env = ReachJointPosEnv(robot="ur5e", time_limit=200, seed=42, randomize_init=False)
obs = env.reset()
print("=== JOINT MODE ===")
print(f"EE pos:  {env.data.site_xpos[env.ee_site]}")
print(f"Goal pos: {env.goal_pos}")
print(f"Init dist: {env._ee_goal_dist():.4f}")
print(f"Init ori err: {env._orientation_error_magnitude():.4f}")

# Step with zero action — should be stable
for i in range(50):
    res = env.step(np.zeros(6))

print(f"\nAfter 50 zero-action steps:")
print(f"  dist: {res.info['dist']:.4f}")
print(f"  ori_err: {res.info['ori_err']:.4f}")
print(f"  reward: {res.reward:.4f}")
env.close()

# Test CARTESIAN mode with proportional position control
print("\n=== CARTESIAN MODE (proportional position-only control) ===")
env3 = ReachIKRelEnv(robot="ur5e", time_limit=300, seed=42, randomize_init=False)
obs = env3.reset()
print(f"EE pos:  {env3.data.site_xpos[env3.ee_site]}")
print(f"Goal pos: {env3.goal_pos}")
print(f"Init dist: {env3._ee_goal_dist():.4f}")
print(f"Init ori err: {env3._orientation_error_magnitude():.4f}")

dists = []
ori_errs = []
for i in range(300):
    ee_pos = env3.data.site_xpos[env3.ee_site].copy()
    direction = env3.goal_pos - ee_pos
    d = np.linalg.norm(direction)
    if d > 1e-6:
        pos_act = (direction / d) * min(1.0, d / 0.06)
    else:
        pos_act = np.zeros(3)
    
    act = np.concatenate([pos_act, np.zeros(3)])
    res = env3.step(act)
    dists.append(res.info["dist"])
    ori_errs.append(res.info["ori_err"])
    
    if i < 5 or i % 50 == 0:
        print(f"  step {i:3d}: dist={dists[-1]:.4f}  ori_err={ori_errs[-1]:.4f}  rew={res.reward:.4f}")

print(f"\nFinal: dist={dists[-1]:.4f}  min_dist={min(dists):.4f}  ori_err={ori_errs[-1]:.4f}")
env3.close()

# Test CARTESIAN mode with pos + ori control
print("\n=== CARTESIAN MODE (position + orientation control) ===")
env4 = ReachIKRelEnv(robot="ur5e", time_limit=300, seed=42, randomize_init=False)
obs = env4.reset()

from mujoco_robot.core.ik_controller import orientation_error_axis_angle
dists = []
ori_errs = []
for i in range(300):
    ee_pos = env4.data.site_xpos[env4.ee_site].copy()
    
    # Position proportional control
    direction = env4.goal_pos - ee_pos
    d = np.linalg.norm(direction)
    if d > 1e-6:
        pos_act = (direction / d) * min(1.0, d / 0.06)
    else:
        pos_act = np.zeros(3)
    
    # Orientation proportional control
    ori_err_vec = orientation_error_axis_angle(env4._ee_quat(), env4.goal_quat)
    ori_err_mag = np.linalg.norm(ori_err_vec)
    if ori_err_mag > 1e-6:
        ori_act = (ori_err_vec / ori_err_mag) * min(1.0, ori_err_mag / 0.5)
    else:
        ori_act = np.zeros(3)
    
    act = np.concatenate([pos_act, ori_act])
    res = env4.step(act)
    dists.append(res.info["dist"])
    ori_errs.append(res.info["ori_err"])
    
    if i < 5 or i % 50 == 0:
        print(f"  step {i:3d}: dist={dists[-1]:.4f}  ori_err={ori_errs[-1]:.4f}  rew={res.reward:.4f}")

print(f"\nFinal: dist={dists[-1]:.4f}  min_dist={min(dists):.4f}  ori_err={ori_errs[-1]:.4f}  min_ori_err={min(ori_errs):.4f}")
env4.close()

# Direct IK controller test
print("\n=== IK CONTROLLER DIRECT TEST ===")
env5 = ReachIKRelEnv(robot="ur5e", time_limit=200, seed=42, randomize_init=False)
obs = env5.reset()
qvel = env5._ik.solve(env5.goal_pos, env5.goal_quat)
pos_err = env5.goal_pos - env5.data.site_xpos[env5.ee_site]
ori_err_vec = orientation_error_axis_angle(env5._ee_quat(), env5.goal_quat)
print(f"Pos err norm: {np.linalg.norm(pos_err):.4f}  Ori err norm: {np.linalg.norm(ori_err_vec):.4f}")
print(f"IK qvel: {qvel}")
print(f"IK qvel norm: {np.linalg.norm(qvel):.4f}")
print(f"Position weight: {env5._ik.position_weight}")
env5.close()
print("\nDone!")

print(f"\nRatio of orientation to position in error vector:")
print(f"  pos_err norm: {np.linalg.norm(pos_err):.4f}")
print(f"  ori_err norm: {np.linalg.norm(ori_err_vec):.4f}")
print(f"  ratio ori/pos: {np.linalg.norm(ori_err_vec)/max(np.linalg.norm(pos_err),1e-10):.2f}")
env4.close()
print("\nDone!")
