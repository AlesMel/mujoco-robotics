# 3D Rotations & Quaternions — From Zero to Hero

> **Where this is used in our code:**
> [`ik_controller.py`](../src/mujoco_robot/core/ik_controller.py) — all the quaternion math helpers
> [`reach_env.py`](../src/mujoco_robot/envs/reach_env.py) — goal orientation, EE orientation, orientation error

---

## Table of Contents

1. [Why Do We Need Rotations?](#1-why-do-we-need-rotations)
2. [2D Rotation — The Intuitive Start](#2-2d-rotation--the-intuitive-start)
3. [3D Rotation Matrices](#3-3d-rotation-matrices)
4. [Euler Angles — The Intuitive (But Problematic) Way](#4-euler-angles--the-intuitive-but-problematic-way)
5. [Quaternions — The Robust Way](#5-quaternions--the-robust-way)
6. [Axis-Angle Representation](#6-axis-angle-representation)
7. [Converting Between Representations](#7-converting-between-representations)
8. [How Our Code Uses All of This](#8-how-our-code-uses-all-of-this)

---

## 1. Why Do We Need Rotations?

A robot arm needs to reach a **pose** — that's a position (where) plus an orientation (which way it's pointing). Position is easy: just 3 numbers (x, y, z). But orientation is trickier.

Imagine you're holding a screwdriver. You can:
- **Point it** in any direction (that's 2 degrees of freedom — like latitude and longitude)
- **Twist it** around its own axis (that's 1 more degree of freedom)

That's **3 degrees of freedom** (DOF) for orientation. But representing those 3 DOF without problems turns out to be surprisingly hard.

---

## 2. 2D Rotation — The Intuitive Start

Before jumping to 3D, let's build intuition in 2D.

### A single angle

In 2D, rotation is simple — one angle θ:

```
Rotate point (x, y) by angle θ:

    x' = x·cos(θ) - y·sin(θ)
    y' = x·sin(θ) + y·cos(θ)
```

### As a matrix

We can write this as a **rotation matrix**:

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

So rotating a vector $\mathbf{v}$ is just matrix multiplication: $\mathbf{v'} = R \cdot \mathbf{v}$.

**Key property:** $R^T = R^{-1}$ (the transpose IS the inverse). This is what makes a matrix a *rotation* matrix — it's **orthogonal**.

---

## 3. 3D Rotation Matrices

### Basic rotations around each axis

In 3D, we can rotate around any of the three coordinate axes:

**Rotation around X-axis** (roll):

$$
R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}
$$

**Rotation around Y-axis** (pitch):

$$
R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}
$$

**Rotation around Z-axis** (yaw):

$$
R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

### Properties of 3×3 rotation matrices

A valid rotation matrix $R$ satisfies:
- $R^T R = I$ (orthogonal — columns are unit vectors and mutually perpendicular)
- $\det(R) = +1$ (proper rotation — no reflection)
- The **columns** of $R$ are the rotated coordinate axes

### What the columns mean physically

If you have a rotation matrix for a robot's end-effector:

$$
R = \begin{bmatrix} | & | & | \\ \mathbf{x}_{ee} & \mathbf{y}_{ee} & \mathbf{z}_{ee} \\ | & | & | \end{bmatrix}
$$

- **Column 1** ($\mathbf{x}_{ee}$) = where the EE's X-axis (red) points in the world
- **Column 2** ($\mathbf{y}_{ee}$) = where the EE's Y-axis (green) points in the world
- **Column 3** ($\mathbf{z}_{ee}$) = where the EE's Z-axis (blue) points in the world

This is exactly what the **RGB axes** in our MuJoCo visualization show!

### Composing rotations

To apply rotation $R_1$ first, then $R_2$:

$$
R_{total} = R_2 \cdot R_1
$$

⚠️ **Order matters!** $R_2 \cdot R_1 \neq R_1 \cdot R_2$ in general. Try rotating a book: 90° around X then 90° around Z gives a different result than Z-then-X.

---

## 4. Euler Angles — The Intuitive (But Problematic) Way

Euler angles describe a rotation as three sequential rotations around coordinate axes. For example, **ZYX convention** (yaw-pitch-roll):

$$
R = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)
$$

where $\psi$ = yaw, $\theta$ = pitch, $\phi$ = roll.

### Why Euler angles are tempting

- **Easy to visualize**: "rotate 30° left, tilt 15° forward"
- **Compact**: just 3 numbers
- **Human-friendly**: pilots and game devs use them daily

### Why Euler angles are DANGEROUS

#### Gimbal Lock 🔒

When the middle rotation is ±90°, you lose one degree of freedom. The first and third rotations become equivalent — they rotate around the same axis.

**Example**: In ZYX convention, if pitch = 90°, then yaw and roll both rotate around the same axis. You can't distinguish yaw from roll anymore!

This is not just a math curiosity — it causes real problems:
- **Interpolation breaks** near gimbal lock
- **Control becomes singular** — the Jacobian loses rank
- **Numerical instability** near ±90° pitch

#### Discontinuities

Euler angles wrap around (e.g., 359° and 1° are close, but numerically far apart). This makes computing "how far apart are two orientations?" unreliable.

### 💡 This is why we switched from yaw-only to quaternions

Our earlier code only controlled yaw (one Euler angle). That worked okay because we avoided gimbal lock by only using one rotation. But to control **all 3 axes**, we need quaternions.

---

## 5. Quaternions — The Robust Way

### What IS a quaternion?

A quaternion is a 4-number representation of a 3D rotation:

$$
q = w + xi + yj + zk = (w, x, y, z)
$$

where:
- $w$ is the **scalar** (real) part
- $(x, y, z)$ is the **vector** (imaginary) part
- $i, j, k$ are imaginary units with special multiplication rules

### The geometric meaning

A unit quaternion $(w, x, y, z)$ represents a rotation of angle $\theta$ around axis $\hat{n} = (n_x, n_y, n_z)$:

$$
q = \left(\cos\frac{\theta}{2},\;\; n_x\sin\frac{\theta}{2},\;\; n_y\sin\frac{\theta}{2},\;\; n_z\sin\frac{\theta}{2}\right)
$$

**Examples:**
- **No rotation** (identity): $q = (1, 0, 0, 0)$ → $\theta = 0$
- **90° around Z**: $q = (\cos 45°, 0, 0, \sin 45°) = (0.707, 0, 0, 0.707)$
- **180° around X**: $q = (\cos 90°, \sin 90°, 0, 0) = (0, 1, 0, 0)$

### Why the half angle?

The factor of $\frac{\theta}{2}$ is what makes quaternion algebra work correctly. It's not arbitrary — it comes from the mathematical structure of the rotation group SO(3).

### Unit quaternion constraint

A **rotation** quaternion must have unit norm:

$$
\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2} = 1
$$

This is a constraint our code enforces (see `_mat_to_quat` which normalizes at the end).

### Quaternion multiplication (Hamilton product)

To compose two rotations, we multiply their quaternions:

$$
q_1 \otimes q_2 = \begin{pmatrix}
w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
\end{pmatrix}
$$

This is implemented in our `quat_multiply()` function.

⚠️ **Order matters!** $q_1 \otimes q_2 \neq q_2 \otimes q_1$ (just like rotation matrices).

### Quaternion conjugate (inverse rotation)

For a unit quaternion, the **conjugate** is the inverse rotation:

$$
q^* = (w, -x, -y, -z)
$$

This is `quat_conjugate()` in our code. If $q$ rotates 90° clockwise around Z, then $q^*$ rotates 90° counterclockwise around Z.

### The double cover problem: q and -q

Here's a subtle but critical point: **$q$ and $-q$ represent the same rotation!**

$$
(w, x, y, z) \text{ and } (-w, -x, -y, -z) \text{ produce identical rotations}
$$

This is because the rotation formula involves $q \mathbf{v} q^*$, and the double negation cancels out.

This creates problems for computing errors (the "distance" between two quaternions could take the short or long path). Our `quat_unique()` function handles this by enforcing $w \geq 0$:

```python
def quat_unique(q):
    """Ensure w ≥ 0 (resolve q / -q ambiguity)."""
    return -q if q[0] < 0 else q.copy()
```

### Convention: WXYZ vs XYZW

⚠️ Different libraries use different ordering:
- **MuJoCo, our code**: $(w, x, y, z)$ — scalar first
- **PyTorch3D, SciPy**: $(x, y, z, w)$ — scalar last

Always check which convention a library uses!

### Why quaternions are better than Euler angles

| Feature | Euler Angles | Quaternions |
|---------|-------------|-------------|
| Parameters | 3 | 4 |
| Gimbal lock? | YES ❌ | NO ✅ |
| Smooth interpolation? | NO ❌ | YES ✅ (SLERP) |
| Composition | Messy trig | Simple multiplication |
| Error computation | Wrap-around issues | Clean axis-angle |
| Numerical stability | Poor near singularities | Excellent ✅ |

---

## 6. Axis-Angle Representation

### What is axis-angle?

Any 3D rotation can be described as a rotation of angle $\theta$ around a unit axis $\hat{n}$:

$$
\text{axis-angle vector} = \theta \cdot \hat{n} = (\theta n_x, \theta n_y, \theta n_z)
$$

This is a **3-D vector** where:
- **Direction** = the rotation axis
- **Magnitude** = the rotation angle (in radians)

**Example**: Rotating 90° around the Z-axis → axis-angle = $(0, 0, \frac{\pi}{2})$

### Converting quaternion → axis-angle

This is `axis_angle_from_quat()` in our code:

```python
def axis_angle_from_quat(q):
    q = quat_unique(q)            # ensure w ≥ 0
    sin_half = ||q[1:4]||         # magnitude of vector part
    half_angle = atan2(sin_half, q[0])
    axis = q[1:4] / sin_half      # unit rotation axis
    return axis * (2 * half_angle) # angle × axis
```

The math:
- Since $q = (\cos\frac{\theta}{2}, \hat{n}\sin\frac{\theta}{2})$
- The vector part norm is $\sin\frac{\theta}{2}$
- The half angle is $\frac{\theta}{2} = \text{atan2}(\sin\frac{\theta}{2}, \cos\frac{\theta}{2})$
- The axis is the normalized vector part

### Converting axis-angle → quaternion

Given axis-angle vector $\mathbf{a} = \theta \hat{n}$:

$$
q = \left(\cos\frac{\theta}{2},\;\; \hat{n}\sin\frac{\theta}{2}\right)
$$

This is used in `_desired_ee()` to integrate orientation increments:

```python
angle = np.linalg.norm(delta_ori)    # θ
axis = delta_ori / angle             # n̂
half = angle / 2.0
dq = [cos(half), axis * sin(half)]   # quaternion from axis-angle
```

### Why axis-angle is great for errors

The **orientation error** between current and target orientations is naturally expressed as an axis-angle vector:

$$
\mathbf{e}_{ori} = \text{axis\_angle}(q_{target} \otimes q_{current}^*)
$$

This gives us:
- A **3-D vector** we can feed to the IK controller
- Its **magnitude** is the angular error in radians
- Its **direction** tells us WHICH WAY to rotate

This is exactly what `orientation_error_axis_angle()` computes:

```python
def orientation_error_axis_angle(current_quat, target_quat):
    q_err = quat_multiply(target_quat, quat_conjugate(current_quat))
    return axis_angle_from_quat(q_err)
```

---

## 7. Converting Between Representations

### Rotation matrix → Quaternion (Shepperd's method)

This is `_mat_to_quat()` in our code. It's more complex than you'd expect because naive formulas have numerical issues.

The idea: from a rotation matrix $R$, we can extract:

$$
w = \frac{1}{2}\sqrt{1 + R_{00} + R_{11} + R_{22}}
$$

But when the trace $(R_{00} + R_{11} + R_{22})$ is negative, this involves the square root of a negative-ish number. **Shepperd's method** checks which diagonal element is largest and uses a numerically stable formula for each case.

### Quaternion → Rotation matrix

Given $q = (w, x, y, z)$:

$$
R = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}
$$

### Summary of all representations

| Representation | # Numbers | Pros | Cons |
|---------------|-----------|------|------|
| Rotation Matrix | 9 (3×3) | Compose = multiply, columns = axes | Redundant (6 constraints) |
| Euler Angles | 3 | Human-readable | Gimbal lock, discontinuous |
| Quaternion | 4 | No gimbal lock, smooth, fast | Double cover (q = -q), less intuitive |
| Axis-Angle | 3 | Minimal, physical meaning | Singular at θ=0 (axis undefined) |

---

## 8. How Our Code Uses All of This

### The full pipeline

1. **MuJoCo gives us** a 3×3 rotation matrix for the EE site (`data.site_xmat`)

2. **We convert** that matrix → quaternion using `_mat_to_quat()` (Shepperd's method)

3. **Goal orientation** is stored as a quaternion (`goal_quat`), sampled uniformly via the Shoemake method

4. **Orientation error** is computed as an axis-angle vector:
   - $q_{err} = q_{goal} \otimes q_{ee}^*$
   - $\mathbf{e}_{ori} = \text{axis\_angle\_from\_quat}(q_{err})$
   - This 3-D vector is both an **observation** for the RL policy and part of the **IK target**

5. **Error magnitude** (scalar) is used in the reward:
   - $\text{ori\_err} = \|\mathbf{e}_{ori}\|$ (in radians, 0 to π)
   - Reward contribution: $-0.1 \times \text{ori\_err}$

6. **IK controller** uses the 3-D axis-angle error as the rotational component of its 6-D target vector (see the [Jacobian & IK guide](02_jacobian_and_inverse_kinematics.md))

### Uniform random quaternion sampling (Shoemake method)

When we sample a random goal orientation, we need it to be **uniformly distributed** over all possible rotations. Simply randomizing 4 numbers and normalizing does NOT give uniform rotations (it clusters near the poles).

The **Shoemake method** uses 3 uniform random numbers $(u_1, u_2, u_3) \in [0,1)$:

$$
q = \begin{pmatrix}
\sqrt{1-u_1}\sin(2\pi u_2) \\
\sqrt{1-u_1}\cos(2\pi u_2) \\
\sqrt{u_1}\sin(2\pi u_3) \\
\sqrt{u_1}\cos(2\pi u_3)
\end{pmatrix}
$$

This produces a **perfectly uniform distribution** over SO(3) — every possible rotation is equally likely. This matters for RL because the agent needs to see all possible orientations during training.

---

## Quick Reference

```python
# Our quaternion convention: (w, x, y, z) — scalar first, like MuJoCo

# Identity (no rotation)
q_identity = [1, 0, 0, 0]

# 90° around Z-axis
q_90z = [cos(π/4), 0, 0, sin(π/4)] = [0.707, 0, 0, 0.707]

# Compose: first rotate by q1, then by q2
q_total = quat_multiply(q2, q1)

# Inverse rotation
q_inv = quat_conjugate(q)  # = (w, -x, -y, -z)

# Error from current to target
q_err = quat_multiply(q_target, quat_conjugate(q_current))
axis_angle_err = axis_angle_from_quat(q_err)  # 3-D vector
angular_distance = np.linalg.norm(axis_angle_err)  # scalar in [0, π]
```

---

**Next:** [02 — The Jacobian & Inverse Kinematics](02_jacobian_and_inverse_kinematics.md) — how we use these rotations to control the robot arm.
