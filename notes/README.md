# 📚 Learning Notes — MuJoCo Robot RL

Welcome! These notes explain **everything** that powers this project — from the math behind 3D rotations to how the robot actually learns. They're written for students and beginners, with lots of intuition, diagrams, and references to our actual code.

**No prior robotics or RL knowledge required.** Start from the top and work your way down.

---

## Guide Index

### [01 — 3D Rotations & Quaternions](01_3d_rotations_and_quaternions.md)
**The math of orientation.** How do we represent "which way something is pointing" in 3D? Covers rotation matrices, Euler angles (and why they're dangerous), quaternions (and why they're awesome), and axis-angle representation. All linked to our code's quaternion helper functions.

### [02 — The Jacobian & Inverse Kinematics](02_jacobian_and_inverse_kinematics.md)
**How to control a robot arm.** Given a target pose, how do we figure out what to do with each joint? Covers forward kinematics, the Jacobian matrix, pseudo-inverse, and damped least squares (DLS). Walks through our `IKController.solve()` method line by line.

### [03 — The RL Environment](03_rl_environment.md)
**Turning robot reaching into a learning problem.** How we formulate the task as a Markov Decision Process. Detailed breakdown of the 39-dimensional observation space, 6-dimensional action space, and every component of the reward function. Explains hold-then-resample, proximity damping, and EMA smoothing.

### [04 — MuJoCo Physics Simulation](04_mujoco_physics.md)
**How the virtual robot works.** The MJCF robot description format, timesteps and substeps, position servo actuators, contact/collision handling, and how we build our simulation scene through XML injection.

### [05 — PPO Training](05_ppo_training.md)
**How the neural network learns.** Policy gradients from first principles, the PPO clipping mechanism, our network architecture ([64,64] MLP), parallel environments (16 SubprocVecEnv), VecNormalize, hyperparameter guide, curriculum learning, and TensorBoard monitoring.

---

## Reading Order

These guides build on each other. Recommended order:

```
01 Rotations ──→ 02 Jacobian & IK ──→ 03 RL Environment
                                              │
                                              ▼
                        04 MuJoCo Physics ──→ 05 PPO Training
```

Guides 01–02 are the **math/robotics foundation**. Guide 03 is the **core environment design**. Guides 04–05 cover the **simulation and training infrastructure**.

---

## Quick Reference: Key Code Files

| File | What it does | Related guide(s) |
|------|-------------|-------------------|
| `src/mujoco_robot/core/ik_controller.py` | Quaternion math + DLS IK | [01](01_3d_rotations_and_quaternions.md), [02](02_jacobian_and_inverse_kinematics.md) |
| `src/mujoco_robot/envs/reach/reach_env_base.py` | Core reach task logic (obs, rewards, stepping) | [03](03_rl_environment.md) |
| `src/mujoco_robot/core/xml_builder.py` | MJCF XML construction | [04](04_mujoco_physics.md) |
| `src/mujoco_robot/core/collision.py` | Self-collision detection | [04](04_mujoco_physics.md) |
| `src/mujoco_robot/robots/configs.py` | Robot configurations (UR5e, UR3e) | [04](04_mujoco_physics.md) |
| `src/mujoco_robot/training/train_reach.py` | PPO training script | [05](05_ppo_training.md) |
