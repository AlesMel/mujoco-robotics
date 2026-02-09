# The RL Environment — Turning Robot Reaching into a Learning Problem

> **Where this is used in our code:**
> [`reach_env.py`](../src/mujoco_robot/envs/reach_env.py) — the full environment
> [`__init__.py`](../src/mujoco_robot/__init__.py) — Gymnasium registration

> **Prerequisites:** [01 — 3D Rotations](01_3d_rotations_and_quaternions.md), [02 — Jacobian & IK](02_jacobian_and_inverse_kinematics.md)

---

## Table of Contents

1. [What Is Reinforcement Learning?](#1-what-is-reinforcement-learning)
2. [The Markov Decision Process (MDP)](#2-the-markov-decision-process-mdp)
3. [Our Observation Space (What the Agent Sees)](#3-our-observation-space-what-the-agent-sees)
4. [Our Action Space (What the Agent Can Do)](#4-our-action-space-what-the-agent-can-do)
5. [The Reward Function (What "Good" Means)](#5-the-reward-function-what-good-means)
6. [Hold-Then-Resample (Curriculum Within an Episode)](#6-hold-then-resample-curriculum-within-an-episode)
7. [Smoothing & Damping (Making Motion Physically Realistic)](#7-smoothing--damping-making-motion-physically-realistic)
8. [The Full Step Loop](#8-the-full-step-loop)
9. [Gymnasium API & Registration](#9-gymnasium-api--registration)

---

## 1. What Is Reinforcement Learning?

Imagine training a dog. You can't explain calculus to the dog — instead, you:
1. Let the dog **try things** (actions)
2. **Reward** good behaviour (treats!) and **penalise** bad behaviour (no treat!)
3. Over time, the dog learns which actions lead to rewards

**Reinforcement Learning (RL)** works the same way:
- An **agent** (the neural network) interacts with an **environment** (the robot simulation)
- At each step, the agent observes the state, picks an action, and receives a reward
- Over millions of steps, the agent learns a **policy** — a mapping from observations to actions that maximizes cumulative reward

```
    ┌─────────┐    action     ┌─────────────┐
    │  Agent  │──────────────→│ Environment │
    │ (Policy)│               │  (MuJoCo    │
    │         │←──────────────│   Robot)    │
    └─────────┘ observation,  └─────────────┘
                  reward
```

---

## 2. The Markov Decision Process (MDP)

RL is formally described as a **Markov Decision Process**:

| Component | Symbol | Meaning |
|-----------|--------|---------|
| **States** | $S$ | All possible situations the robot can be in |
| **Actions** | $A$ | All possible commands the agent can give |
| **Transition** | $P(s'|s,a)$ | Physics: "if I do action $a$ in state $s$, what's the next state $s'$?" |
| **Reward** | $R(s,a)$ | How good was this action in this state? |
| **Discount** | $\gamma$ | How much to care about future vs. immediate rewards (0.99 in our case) |

The **Markov property** means: the future depends only on the **current state**, not on the history. This is why our observation includes everything the agent needs (joint positions, velocities, goal, etc.) — no memory required.

The agent's goal is to learn a policy $\pi(a|s)$ that maximises the **expected cumulative discounted reward**:

$$
\max_\pi \; \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, R(s_t, a_t)\right]
$$

---

## 3. Our Observation Space (What the Agent Sees)

The observation is a **39-dimensional vector** — a flat array of numbers that tells the agent everything it needs to know. Here's what each part means:

| Component | Dimensions | Range | Description |
|-----------|-----------|-------|-------------|
| `joint_pos` | 6 | ~[-π, π] | Joint angles **relative to home pose** (so home = all zeros) |
| `joint_vel` | 6 | ~[-4, 4] | How fast each joint is moving (rad/s) |
| `ee_pos` | 3 | [varies] | Where the end-effector is in the world (metres) |
| `ee_quat` | 4 | [-1, 1] | End-effector orientation as quaternion (w,x,y,z) |
| `goal_pos` | 3 | [varies] | Where the goal is in the world (metres) |
| `goal_quat` | 4 | [-1, 1] | Goal orientation as quaternion (w,x,y,z) |
| `goal_direction` | 3 | [-1, 1] | **Unit vector** from EE to goal (which way to go) |
| `ori_error` | 3 | ~[-π, π] | Axis-angle error from EE orientation to goal orientation |
| `collision` | 1 | {0, 1} | Is the robot colliding with itself? |
| `last_action` | 6 | [-1, 1] | What the agent did last step (helps learn smooth actions) |
| **Total** | **39** | | |

### Why these specific observations?

- **Joint positions/velocities**: The agent needs to know its own body state (proprioception — like knowing where your arm is without looking)
- **EE position + quaternion**: Where the hand is and how it's oriented
- **Goal position + quaternion**: Where the hand should be and how it should be oriented
- **Goal direction**: A "compass" pointing toward the goal — makes learning faster because the agent doesn't have to figure out "goal minus EE" on its own
- **Orientation error**: Pre-computed axis-angle error — same idea, tells the agent directly "rotate THIS way to match"
- **Collision flag**: Tells the agent when it's doing something physically wrong
- **Last action**: The agent can learn to be smooth if it knows what it did last time

### Observation noise

We add small uniform noise (±0.01) to the first 12 dimensions (joint positions and velocities). This makes the policy robust to sensor noise in the real world — a technique called **domain randomization**.

```python
# Applied only to proprioceptive channels (joints)
noise = rng.uniform(-0.01, 0.01, size=12)
obs[:12] += noise
```

---

## 4. Our Action Space (What the Agent Can Do)

The agent outputs **6 numbers**, each in [-1, 1]. The interpretation depends on the **action mode**:

### Cartesian mode (default for IK)

```
action = [dx, dy, dz, dwx, dwy, dwz]
         ├─────────┤  ├──────────────┤
         linear vel    angular vel
         × ee_step     × ori_step
         (0.06 m)      (0.5 rad)
```

- `dx, dy, dz`: Scaled by `ee_step` (0.06m) → max 6cm per step in each direction
- `dwx, dwy, dwz`: Scaled by `ori_step` (0.5 rad) → max ~28.6° per step rotation increment
- These Cartesian commands are converted to joint commands by the **IK controller** (see [guide 02](02_jacobian_and_inverse_kinematics.md))

### Joint mode (Isaac Lab style, default for training)

```
action = [dq1, dq2, dq3, dq4, dq5, dq6]
         ├─────────────────────────────┤
         joint position offsets
         × joint_action_scale (0.125 rad)
```

- Each value is scaled by `joint_action_scale` (0.125 rad ≈ 7.2°) and added to the current joint position target
- This is simpler — no IK needed — the RL agent learns to coordinate joints directly
- Matches how Isaac Lab handles actions

### Why [-1, 1] range?

Neural networks work best with normalized inputs and outputs. By clipping actions to [-1, 1] and scaling them, we ensure the network outputs are always in a reasonable range. PPO uses a Gaussian policy that naturally produces values near 0 with occasional excursions to ±1.

---

## 5. The Reward Function (What "Good" Means)

The reward function is the **most important part** of the RL environment. It tells the agent what we want. Our reward has several components:

### Position tracking (coarse)
$$
r_{pos} = -0.2 \times \text{dist}
$$

A **linear penalty** proportional to the distance from the goal. The further away, the more negative the reward. The -0.2 weight means: "every metre away costs 0.2 reward per step."

**Intuition**: This creates a "slope" in reward space — the agent always gets more reward by moving closer.

### Position tracking (fine / tanh bonus)
$$
r_{fine} = 0.1 \times \left(1 - \tanh\left(\frac{\text{dist}}{0.1}\right)\right)
$$

The `tanh` function creates a **sharper bonus** for being very close:
- At dist = 0: bonus = +0.1 (maximum)
- At dist = 0.1m: bonus ≈ +0.024
- At dist = 0.5m: bonus ≈ +0.0001 (almost zero)

**Why both?** The coarse linear reward provides gradient everywhere (even far from the goal), while the tanh bonus creates a strong "pull" when close. Without the tanh bonus, the agent might hover at 2cm distance because the linear reward improvement is tiny. With it, there's a strong incentive to close the last few centimeters.

```
    reward
    ↑
    │  ╭─── tanh bonus (sharp near goal)
    │ ╱
    │╱
    ├──────────── ─── linear penalty (constant slope)
    │              ╲
    │               ╲
    │                ╲
    └───────────────────→ distance
    0                   far
```

### Orientation tracking
$$
r_{ori} = -0.1 \times \text{ori\_err\_mag}
$$

Same idea as position: linear penalty proportional to the angular error (in radians). Maximum error is π radians (180°), so the worst penalty is about -0.314.

### Hold bonus
$$
r_{hold} = \begin{cases} 0.3 \times (1 - \frac{\text{dist}}{\text{reach\_threshold}}) \times (1 - \frac{\text{ori\_err}}{\text{ori\_threshold}}) & \text{if inside thresholds} \\ 0 & \text{otherwise} \end{cases}
$$

**This is critical!** It rewards the agent for **staying still at the goal**. Without this, the agent might learn to oscillate through the goal (high speed, briefly touching the target). The hold bonus makes "be at the goal AND be still" strictly better than "swing through it."

### Velocity damping near goal
$$
r_{vel} = \begin{cases} -0.15 \times \text{proximity} \times \sum \dot{q}_i^2 & \text{if dist} < 3 \times \text{reach\_threshold} \\ 0 & \text{otherwise} \end{cases}
$$

Penalises fast joint motion when close to the goal. The penalty increases as the EE gets closer (proximity goes from 0 at the edge to 1 at the centre). This teaches the agent to **slow down on approach** — like how you slow your hand when reaching for a glass of water.

### Action rate penalty (with curriculum)
$$
r_{ar} = -w_{ar} \times \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2
$$

Penalises **jerky changes** in action. The weight $w_{ar}$ ramps up over training:
- Start: $w_{ar} = 0.0001$ (almost no penalty — let the agent explore)
- End: $w_{ar} = 0.005$ (moderate penalty — encourage smoothness)

This is a **curriculum**: at first, let the agent learn WHAT to do; later, teach it HOW to do it smoothly.

### Joint velocity penalty (with curriculum)
$$
r_{jv} = -w_{jv} \times \sum \dot{q}_i^2
$$

Penalises fast overall joint movement. Also ramps up over training (0.0001 → 0.001).

### Self-collision penalty
$$
r_{col} = \begin{cases} -1.0 & \text{if self-collision} \\ 0 & \text{otherwise} \end{cases}
$$

A harsh, fixed penalty for the robot hitting itself. The collision detector checks if non-adjacent links are in contact (see [`collision.py`](../src/mujoco_robot/core/collision.py)).

### Total reward

$$
r = r_{pos} + r_{fine} + r_{ori} + r_{hold} + r_{vel} - r_{ar} - r_{jv} + r_{col}
$$

---

## 6. Hold-Then-Resample (Curriculum Within an Episode)

### The mechanism

Unlike many RL environments that end the episode on success, our environment **never ends on success**. Instead:

1. Agent reaches the goal (distance < `reach_threshold` AND orientation error < `ori_threshold`)
2. A **hold counter** starts counting up
3. If the agent stays within thresholds for `hold_steps` consecutive steps (~2 seconds at 50 Hz ≈ 100 steps):
   - Goal is **resampled** to a new random position and orientation
   - Hold counter resets to 0
4. If the agent drifts outside the thresholds, the hold counter **gradually decays** (decrements by 3 per step instead of resetting to 0)
5. The episode only ends when `time_limit` steps are reached (375 steps ≈ 7.5 seconds)

### Why gradual decay?

A hard reset (counter → 0 when leaving the threshold) is frustrating for the agent — a tiny vibration could erase 2 seconds of careful holding. Gradual decay (counter -= 3) means brief drifts are forgiven, but sustained departure loses progress.

### Why hold-then-resample?

This creates a natural **curriculum**:
- **Easy goal**: agent learns to reach it, hold, gets rewarded
- **New goal**: immediately after holding, a new challenge appears
- **Multiple goals per episode**: the agent sees 3-6 different goals per episode, dramatically increasing sample efficiency
- **No success termination**: episodes always run the full length, giving consistent gradient signal

The `goals_reached` counter tracks how many goals the agent successfully reaches per episode — this is a great metric for tracking training progress.

---

## 7. Smoothing & Damping (Making Motion Physically Realistic)

### Problem: near-goal jitter

Without special handling, RL policies tend to produce **bumpy, oscillating motion** near the goal. This happens because:
- The policy is a neural network making continuous predictions — it naturally fluctuates
- Small action changes near the goal can cause overshooting (move past → correct back → move past again)
- The physics simulation amplifies high-frequency commands

### Solution 1: Proximity damping on actions

When the EE is within 2× the reach threshold of the goal:

```python
prox_t = dist / (2 * reach_threshold)        # 1 at edge → 0 at goal
prox_dampen = 0.3 + 0.7 * prox_t             # 1.0 at edge → 0.3 at goal
act = act * prox_dampen                       # scale down actions
```

This **reduces the magnitude** of actions as the EE approaches the goal. At the threshold boundary, actions are at 30% of their original strength.

### Solution 2: Hold damping

Once the EE is inside the threshold and accumulating hold time:

```python
hold_frac = hold_counter / hold_steps         # 0 → 1 over hold period
hold_dampen = 0.25 * (1 - hold_frac) + 0.02 * hold_frac  # 0.25 → 0.02
act = act * hold_dampen
```

Actions are progressively squashed from 25% to 2% of original strength as the hold progresses. By the end, the robot is essentially locked in place.

### Solution 3: EMA smoothing on joint targets

When close to the goal, new joint position targets are blended with previous ones using **Exponential Moving Average (EMA)**:

```python
alpha = 0.3 + 0.7 * (dist / approach_radius)  # 0.3 at goal → 1.0 far away
targets = alpha * new_targets + (1 - alpha) * prev_targets
```

Near the goal, only 30% of the new target is applied — the rest is the old target. This acts as a **low-pass filter**, removing high-frequency jitter.

---

## 8. The Full Step Loop

Here's what happens every time the agent calls `env.step(action)`:

```
    action (6 floats in [-1, 1])
        │
        ▼
    ┌──────────────────────────────┐
    │ 1. Clip to [-1, 1]          │
    │ 2. Apply proximity damping  │
    │ 3. Apply hold damping       │
    └──────────────┬───────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   Cartesian mode        Joint mode
        │                     │
    ┌───┴────────┐    ┌───────┴──────┐
    │ Scale by   │    │ Scale by     │
    │ ee_step,   │    │ joint_action │
    │ ori_step   │    │ _scale       │
    │            │    │              │
    │ IK solve → │    │ Add to       │
    │ joint vels │    │ current      │
    │ → position │    │ targets      │
    │ targets    │    │              │
    └───┬────────┘    └──────┬───────┘
        │                    │
        └────────┬───────────┘
                 │
        ┌────────┴────────────────┐
        │ 4. EMA smoothing        │
        │ 5. Clamp to joint limits│
        │ 6. Send to actuators    │
        └────────┬────────────────┘
                 │
        ┌────────┴────────────────┐
        │ 7. Physics simulation   │
        │    (4 substeps × 5ms    │
        │     = 20ms real time)   │
        └────────┬────────────────┘
                 │
        ┌────────┴────────────────┐
        │ 8. Collision check      │
        │    (revert if collided) │
        └────────┬────────────────┘
                 │
        ┌────────┴────────────────┐
        │ 9. Compute reward       │
        │ 10. Check hold progress │
        │ 11. Build observation   │
        └────────┬────────────────┘
                 │
                 ▼
    (obs, reward, done, info)
```

---

## 9. Gymnasium API & Registration

### What is Gymnasium?

[Gymnasium](https://gymnasium.farama.org/) (successor to OpenAI Gym) is the standard API for RL environments. It defines:
- `env.reset()` → `(observation, info)`
- `env.step(action)` → `(observation, reward, terminated, truncated, info)`
- `env.observation_space` and `env.action_space`

### Our wrapper

The `ReachGymnasium` class wraps our `URReachEnv` to comply with the Gymnasium API:

```python
class ReachGymnasium(gymnasium.Env):
    def __init__(self, robot="ur5e", ...):
        self.base = URReachEnv(robot=robot, ...)
        self.action_space = Box(-1, 1, shape=(6,))
        self.observation_space = Box(-inf, inf, shape=(39,))
    
    def reset(self, seed=None, options=None):
        obs = self.base.reset(seed=seed)
        return obs, {}
    
    def step(self, action):
        result = self.base.step(action)
        terminated = False        # never terminates on success
        truncated = time_up       # only on time limit
        return obs, reward, terminated, truncated, info
```

### Registration

We register the environment so it can be created with `gymnasium.make()`:

```python
# In __init__.py
gymnasium.register(
    id="MuJoCoRobot/Reach-v0",
    entry_point="mujoco_robot.envs:ReachGymnasium",
)

# Usage:
import gymnasium
import mujoco_robot  # triggers registration
env = gymnasium.make("MuJoCoRobot/Reach-v0", robot="ur3e")
```

### terminated vs truncated

Gymnasium distinguishes two kinds of "done":
- **terminated** = the episode ended because of something that happened (success, failure). We set this to `False` always — reaching the goal just resamples, it doesn't end the episode.
- **truncated** = the episode ended because of a time limit. We set this to `True` when `step_id >= time_limit`.

This distinction matters for value estimation in RL: if truncated, the value function should bootstrap (the state isn't terminal); if terminated, it shouldn't.

---

**Previous:** [02 — The Jacobian & Inverse Kinematics](02_jacobian_and_inverse_kinematics.md)
**Next:** [04 — MuJoCo Physics Simulation](04_mujoco_physics.md) — how the physics engine simulates the robot.
