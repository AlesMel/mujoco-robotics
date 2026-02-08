# The Jacobian & Inverse Kinematics — Moving a Robot Arm

> **Where this is used in our code:**
> [`ik_controller.py`](../src/mujoco_robot/core/ik_controller.py) — the `IKController.solve()` method
> [`reach_env.py`](../src/mujoco_robot/envs/reach_env.py) — calls `ik.solve()` every control step

> **Prerequisites:** [01 — 3D Rotations & Quaternions](01_3d_rotations_and_quaternions.md)

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Forward Kinematics (FK) — "If I Move This Joint, Where Does the Hand Go?"](#2-forward-kinematics-fk)
3. [The Jacobian — Connecting Joint Velocities to EE Velocities](#3-the-jacobian)
4. [Inverse Kinematics (IK) — "How Do I Move Joints to Reach There?"](#4-inverse-kinematics-ik)
5. [The Pseudo-Inverse — A First Solution](#5-the-pseudo-inverse)
6. [Damped Least Squares (DLS) — The Robust Solution](#6-damped-least-squares-dls)
7. [How Our Code Puts It All Together](#7-how-our-code-puts-it-all-together)
8. [Common Pitfalls & Intuitions](#8-common-pitfalls--intuitions)

---

## 1. The Big Picture

A robot arm has **joints** (things that rotate) and an **end-effector** (the tool at the tip). We have two fundamental questions:

| Question | Name | Easy? |
|----------|------|-------|
| "If I set each joint angle to X, where is the hand?" | **Forward Kinematics (FK)** | ✅ Straightforward |
| "I want the hand HERE — what joint angles do I need?" | **Inverse Kinematics (IK)** | ❌ Hard! |

FK is like pushing dominoes forward — it's a simple chain of calculations. IK is like trying to figure out which dominoes to push to make the last one land in a specific spot — it's much harder, and there might be **multiple solutions** or **no solution** at all.

---

## 2. Forward Kinematics (FK)

### The chain of transformations

A robot arm is a chain of rigid links connected by joints. Each joint rotates by some angle $q_i$. To find where the end-effector is, we multiply together all the transformation matrices:

$$
T_{ee} = T_0 \cdot T_1(q_1) \cdot T_2(q_2) \cdot \ldots \cdot T_n(q_n)
$$

Each $T_i$ is a 4×4 **homogeneous transformation matrix** that encodes both rotation and translation:

$$
T = \begin{bmatrix} R_{3\times3} & \mathbf{p}_{3\times1} \\ \mathbf{0}_{1\times3} & 1 \end{bmatrix}
$$

where $R$ is the rotation matrix and $\mathbf{p}$ is the translation vector.

### What FK gives us

The result $T_{ee}$ tells us:
- **Position**: the translation part $\mathbf{p}_{ee}$ = (x, y, z) in the world
- **Orientation**: the rotation part $R_{ee}$ = 3×3 matrix (which we convert to a quaternion)

### MuJoCo does FK for us

We don't compute FK manually — MuJoCo does it every simulation step. We just read the results:

```python
# Position — MuJoCo computed this from joint angles via FK
pos = data.site_xpos[ee_site]  # (3,) array

# Orientation — 3×3 rotation matrix, also from FK
mat = data.site_xmat[ee_site].reshape(3, 3)
quat = _mat_to_quat(mat)       # convert to quaternion
```

---

## 3. The Jacobian

### The key question

If we **slightly change** the joint angles, how does the end-effector move?

This is a question about **derivatives** — and the answer is the **Jacobian matrix**.

### Definition

The Jacobian $J$ is a matrix that maps joint velocities $\dot{q}$ to end-effector velocities $\dot{x}$:

$$
\dot{x} = J(q) \cdot \dot{q}
$$

where:
- $\dot{q} \in \mathbb{R}^n$ = joint velocities (for a 6-joint arm, n=6)
- $\dot{x} \in \mathbb{R}^m$ = end-effector velocity (for us, m=6: 3 linear + 3 angular)
- $J \in \mathbb{R}^{m \times n}$ = the Jacobian matrix

### What each row and column means

For our 6-DOF arm with full position + orientation control:

$$
J = \begin{bmatrix}
\frac{\partial p_x}{\partial q_1} & \frac{\partial p_x}{\partial q_2} & \cdots & \frac{\partial p_x}{\partial q_6} \\[4pt]
\frac{\partial p_y}{\partial q_1} & \frac{\partial p_y}{\partial q_2} & \cdots & \frac{\partial p_y}{\partial q_6} \\[4pt]
\frac{\partial p_z}{\partial q_1} & \frac{\partial p_z}{\partial q_2} & \cdots & \frac{\partial p_z}{\partial q_6} \\[4pt]
\frac{\partial \omega_x}{\partial q_1} & \frac{\partial \omega_x}{\partial q_2} & \cdots & \frac{\partial \omega_x}{\partial q_6} \\[4pt]
\frac{\partial \omega_y}{\partial q_1} & \frac{\partial \omega_y}{\partial q_2} & \cdots & \frac{\partial \omega_y}{\partial q_6} \\[4pt]
\frac{\partial \omega_z}{\partial q_1} & \frac{\partial \omega_z}{\partial q_2} & \cdots & \frac{\partial \omega_z}{\partial q_6}
\end{bmatrix}
$$

- **Rows 1-3** (translational Jacobian $J_p$): How each joint affects EE **position**
- **Rows 4-6** (rotational Jacobian $J_r$): How each joint affects EE **angular velocity**
- **Each column**: The effect of one joint on the entire EE velocity

### Physical intuition: columns as "effect vectors"

Column $j$ of the Jacobian is the EE velocity you'd get if **only** joint $j$ moved at unit speed. Think of it as "the influence of joint $j$ on the end-effector."

- **Base joint** (shoulder): turning it moves the EE in a large arc → large entries in $J_p$
- **Wrist joint**: barely moves the EE position (small $J_p$ entries) but strongly rotates it (large $J_r$ entries)

### MuJoCo computes the Jacobian for us

```python
jacp = np.zeros((3, model.nv))  # translational Jacobian
jacr = np.zeros((3, model.nv))  # rotational Jacobian
mujoco.mj_jacSite(model, data, jacp, jacr, ee_site)

# Stack them into the full 6×n Jacobian
J = np.vstack([jacp[:, robot_dofs], jacr[:, robot_dofs]])  # (6, 6)
```

Note: `model.nv` is the total number of velocity DOFs in the model (might include a free-floating base, etc.), so we select only our robot's joint columns with `robot_dofs`.

---

## 4. Inverse Kinematics (IK)

### The problem

We want to find $\dot{q}$ such that the EE moves toward a **target pose**. We know:
- **Where we are**: current EE pose (from FK)
- **Where we want to be**: target position + target quaternion
- **The Jacobian**: $J$ at the current configuration

The **error** between current and target is:

$$
\mathbf{e} = \begin{bmatrix} \mathbf{p}_{target} - \mathbf{p}_{current} \\ \text{axis\_angle\_error}(q_{current}, q_{target}) \end{bmatrix}
$$

This is a 6-D vector: 3 for position error + 3 for orientation error (as axis-angle, see previous guide).

### The ideal equation

We want: $J \dot{q} = \mathbf{e}$

If $J$ is square (6×6) and non-singular: $\dot{q} = J^{-1} \mathbf{e}$

But there are problems:
1. $J$ might not be square (more joints than needed → redundancy, or fewer → under-actuated)
2. $J$ might be **singular** (at certain configurations, some directions become unreachable)
3. Direct inversion is **numerically unstable** near singularities

---

## 5. The Pseudo-Inverse — A First Solution

### Least-squares solution

When $J$ is not invertible, we find the $\dot{q}$ that minimizes $\|J\dot{q} - \mathbf{e}\|^2$ (least squares):

$$
\dot{q} = J^T (J J^T)^{-1} \mathbf{e}
$$

This is the **right pseudo-inverse** $J^\dagger = J^T (J J^T)^{-1}$.

For our square 6×6 Jacobian, this gives the same result as $J^{-1}$ when $J$ is invertible.

### The singularity problem 💥

When the robot reaches certain configurations (called **singularities**), $J J^T$ becomes singular (determinant → 0). Physically, this means some EE directions become unreachable — no matter what joint velocities you apply, the EE can't move in that direction.

**Example**: A fully stretched-out arm. It can't move further outward — the Jacobian has no component in the radial direction. At this point, $J J^T$ is nearly singular, and $(J J^T)^{-1}$ produces **enormous** (or infinite) joint velocities.

This is bad — the robot would wildly spin its joints trying to achieve an impossible motion.

---

## 6. Damped Least Squares (DLS) — The Robust Solution

### The fix: add damping

Instead of $(J J^T)^{-1}$, we compute:

$$
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} \mathbf{e}
$$

where $\lambda$ is a small damping factor (we use $\lambda = 0.02$).

### What does the damping do?

The term $\lambda^2 I$ adds $\lambda^2$ to the diagonal of $J J^T$ **before** inverting. This:
- **Prevents division by zero** at singularities
- **Limits maximum joint velocities** — even at singularities, $\dot{q}$ stays bounded
- **Trades accuracy for robustness** — near singularities, the EE won't move exactly where we want, but the robot won't go crazy either

### The damping trade-off

| $\lambda$ too small | $\lambda$ too large |
|---------------------|---------------------|
| Near-perfect tracking | Sluggish tracking |
| Wild joints at singularities | Smooth joints everywhere |
| Numerically unstable | Very stable |

Our value of $\lambda = 0.02$ is a good balance — it's small enough for accurate tracking but large enough to prevent joint explosions near singularities.

### Optimization perspective

DLS minimizes a trade-off between tracking error and joint effort:

$$
\min_{\dot{q}} \left( \|J\dot{q} - \mathbf{e}\|^2 + \lambda^2 \|\dot{q}\|^2 \right)
$$

The first term wants accurate EE tracking. The second penalizes large joint velocities. $\lambda$ controls the balance.

---

## 7. How Our Code Puts It All Together

Here's our `IKController.solve()` method, annotated step by step:

```python
def solve(self, target_pos, target_quat):
    # Step 1: Get the Jacobian from MuJoCo
    jacp = np.zeros((3, self.model.nv))   # translational (3 × nv)
    jacr = np.zeros((3, self.model.nv))   # rotational    (3 × nv)
    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)

    # Step 2: Compute the 6-D error vector
    pos_err = target_pos - self.data.site_xpos[self.ee_site]  # (3,)
    ori_err = orientation_error_axis_angle(self.ee_quat(), target_quat)  # (3,)
    target_vec = np.concatenate([pos_err, ori_err])  # (6,)

    # Step 3: Build the full 6×n Jacobian (only our robot's joints)
    cols = self.robot_dofs
    J = np.vstack([jacp[:, cols], jacr[:, cols]])  # (6, n_joints)

    # Step 4: Damped least-squares solve
    lam = self.damping
    JJT = J @ J.T + (lam ** 2) * np.eye(6)   # (6, 6) — always invertible!
    return J.T @ np.linalg.solve(JJT, target_vec)  # (n_joints,)
```

### What happens at each control step

```
    ┌─────────────────────────────────────────────────┐
    │                  CONTROL LOOP                    │
    │                                                  │
    │  1. RL policy outputs action  →  [dx,dy,dz,     │
    │                                   dwx,dwy,dwz]  │
    │                                                  │
    │  2. Action is integrated into →  target_pos,     │
    │     a Cartesian target pose      target_quat     │
    │                                                  │
    │  3. IK controller computes    →  joint velocities│
    │     J^T(JJ^T + λ²I)⁻¹ error     (6 numbers)    │
    │                                                  │
    │  4. Joint velocities are      →  MuJoCo actuator │
    │     converted to position        targets         │
    │     targets: q_new = q + dt*dq                   │
    │                                                  │
    │  5. MuJoCo simulates physics  →  new joint       │
    │     (PD control + contacts)      angles & EE pose│
    │                                                  │
    │  6. New observation built     →  back to step 1  │
    └─────────────────────────────────────────────────┘
```

---

## 8. Common Pitfalls & Intuitions

### Singularities — when the arm gets "stuck"

**Fully extended arm**: Can't move further outward. The Jacobian row for radial motion becomes zero.

**Folded arm with wrist aligned to shoulder**: Two joints become equivalent — they both rotate around the same effective axis. The Jacobian loses rank (two columns become linearly dependent).

DLS handles both cases gracefully by sacrificing perfect tracking near singularities.

### Redundancy — when there are too many solutions

A 6-DOF arm has exactly 6 joints and 6 EE DOFs (3 position + 3 orientation). This means the system is **square** — typically one solution.

A 7-DOF arm (like many humanoid arms) has an extra joint — it's **redundant**. There are infinitely many joint configurations that achieve the same EE pose. The pseudo-inverse gives the "minimum norm" solution (smallest joint velocities), but you can add motions in the **null space** (joint motions that don't affect the EE at all) for secondary objectives like avoiding joint limits.

### Why position control, not velocity control?

Our code converts IK velocity outputs into **position targets**:

```python
q_target = q_current + dt * q_vel_from_ik
```

This is because MuJoCo's actuators are **position servos** — they use internal PD controllers to track position targets. This is how real UR robots work too: you send joint position commands, and the robot's built-in controller handles the low-level torques.

### Scale of position vs orientation errors

Position errors are in **meters** and orientation errors are in **radians**. These have different physical scales:
- 0.01m = 1cm (small position error)
- 0.01 rad ≈ 0.57° (very small angle)

If the scales are very different, the IK might focus on one at the expense of the other. Our reward function handles this by using separate weights: $-0.2 \times \text{dist}$ for position and $-0.1 \times \text{ori\_err}$ for orientation.

---

## Mathematical Summary

| Symbol | Meaning | Size |
|--------|---------|------|
| $q$ | Joint angles | $(n,)$ |
| $\dot{q}$ | Joint velocities | $(n,)$ |
| $\mathbf{p}$ | EE position | $(3,)$ |
| $\omega$ | EE angular velocity | $(3,)$ |
| $\mathbf{e}$ | 6-D pose error (position + orientation) | $(6,)$ |
| $J$ | Full Jacobian | $(6 \times n)$ |
| $J_p$ | Translational Jacobian | $(3 \times n)$ |
| $J_r$ | Rotational Jacobian | $(3 \times n)$ |
| $\lambda$ | DLS damping factor | scalar |
| $I$ | Identity matrix | $(6 \times 6)$ |

**DLS formula:**

$$
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} \mathbf{e}
$$

---

**Previous:** [01 — 3D Rotations & Quaternions](01_3d_rotations_and_quaternions.md)
**Next:** [03 — The RL Environment](03_rl_environment.md) — how we formulate robot reaching as a reinforcement learning problem.
