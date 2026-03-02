# Crazyflie Beginner Guide

This guide is for students who are seeing a drone simulator for the first time.
No robotics background is required.

It explains:
- what this Crazyflie environment is,
- what the physics means in simple words,
- the exact math equations behind the simulator,
- what the agent sees (observations),
- what the agent can do (actions),
- how rewards work,
- and why PPO can learn this task.

---

## 1. What are we building?

We have a tiny virtual drone (Crazyflie) inside MuJoCo physics simulation.

There are two tasks:
- `crazyflie_hover`: stay stable near a target hover point.
- `crazyflie_reach`: fly to random 3D goals again and again.

Code:
- `src/mujoco_robot/tasks/crazyflie/crazyflie_env.py`
- `src/mujoco_robot/tasks/crazyflie/crazyflie_reach_env.py`

### 1.1 Project file map

```
src/mujoco_robot/
├── assets/
│   └── crazyflie_2_1.xml          # MuJoCo model (mass, inertia, geometry, sensors)
├── tasks/crazyflie/
│   ├── crazyflie_env.py           # CrazyflieHoverEnv + CrazyflieHoverGymnasium
│   ├── crazyflie_reach_env.py     # CrazyflieReachEnv + CrazyflieReachGymnasium
│   └── config.py                  # Named configuration profiles
├── training/
│   └── eval_crazyflie_reach.py    # Interactive evaluation with OpenCV viewer
scripts/
├── train.py                       # PPO training entry point (SB3)
├── eval_crazyflie_reach.py        # Evaluation helper
tests/
├── test_crazyflie_task.py         # Unit tests for hover env
├── test_crazyflie_reach_task.py   # Unit tests for reach env
runs/                              # Saved models, TensorBoard logs, VecNormalize stats
docs/
└── README_crazyflie_beginner.md   # This file
```

Key rule: **environment** files define physics, observations, rewards. **Training** scripts set PPO hyperparameters and launch learning. **Config** profiles connect the two with named presets.

---

## 2. Super simple mental model

Think of the agent as a child learning a video game:
1. Look at the screen (observation).
2. Press buttons (action).
3. Get points (reward).
4. Repeat thousands of times.

Over time, it learns which button presses give better points.

---

## 2.5 Coordinate frames and quaternions

Understanding rotation is essential for quadrotor work.

### 2.5.1 Two frames of reference

| Frame | Origin | Axes | Used for |
|---|---|---|---|
| **World** | arena center, floor level | X forward, Y left, Z up (MuJoCo default) | positions, gravity, goals |
| **Body** | drone center of mass | moves and rotates with drone | IMU readings, drag forces, motor torques |

Converting between frames uses the rotation matrix **R** (body → world):

```text
v_world = R * v_body
v_body  = R^T * v_world
```

### 2.5.2 Why quaternions?

A rotation can be stored as:
- 3 Euler angles (roll, pitch, yaw) — simple but has **gimbal lock** (singularity when pitch = ±90°),
- 3×3 rotation matrix **R** — no singularity but 9 numbers with constraints,
- 4-element quaternion **q** = `[w, x, y, z]` — compact, singularity-free, cheap to compose.

MuJoCo and this codebase use unit quaternions: `||q|| = 1`.

### 2.5.3 Quaternion math essentials

A unit quaternion `q = [w, x, y, z]` encodes rotation by angle θ around axis **n**:

```text
w = cos(θ/2)
[x, y, z] = sin(θ/2) * n
```

**Identity** (no rotation): `q = [1, 0, 0, 0]`.

**Tilt angle** from upright:

```text
cos(tilt) = 1 - 2*(q_x^2 + q_y^2)
```

This equals the Z-component of the body's up-vector in world frame (`body_z[2]`), which the reward uses directly.

**Body-to-world rotation matrix** from quaternion:

```text
R = [
  [1-2(y²+z²),  2(xy-wz),    2(xz+wy)  ],
  [2(xy+wz),    1-2(x²+z²),  2(yz-wx)   ],
  [2(xz-wy),    2(yz+wx),    1-2(x²+y²) ]
]
```

You rarely need to implement this yourself — MuJoCo provides `data.xmat` (the 3×3 rotation matrix) and `data.xquat` (the quaternion) per body.

### 2.5.4 Why the observation includes the full quaternion

The policy receives all 4 quaternion components because:
- the agent needs to know its orientation to apply the right motor corrections,
- a single tilt angle loses information about roll vs pitch vs combined,
- quaternions are smooth and continuous (no angle wrap-around discontinuities).

---

## 3. One environment step (what happens each control tick)

At each step:
1. Agent sends 4 action numbers in `[-1, 1]`.
2. Environment turns them into motor commands.
3. Motor speed changes with lag (motors cannot change instantly).
4. Environment computes forces and torques from propellers, drag, and disturbances.
5. MuJoCo advances physics.
6. Environment returns:
   - new observation,
   - reward,
   - done flag,
   - info dictionary.

Important timing:
- MuJoCo physics timestep: `0.001 s` (1 ms).
- Default `n_substeps = 10`.
- Control period is `0.001 * 10 = 0.01 s` (100 Hz control loop).

---

## 4. The drone physics in plain language

### 4.0 Physical parameters of the model

All values come from `crazyflie_2_1.xml` and the environment constructor defaults.

**Airframe:**

| Property | Value | Notes |
|---|---|---|
| Total mass | 0.027 kg | body 0.021 + battery 0.006 + 4 rotors ≈ 0 |
| Body inertia (diag) | `1.66e-5, 1.66e-5, 2.9e-5` kg·m² | |
| Arm length | 0.03253 m | Center to motor, X-config |
| Arm count | 4 | |
| Rotor mass (each) | 1e-5 kg | Negligible |
| Prop radius (visual) | 23 mm | |

**Motor layout (X-configuration, top view):**

```text
    m2 (CCW)     m1 (CW)       +Y
       \         /              ↑
        \       /               |
         [body]        +X  ← ──┘
        /       \
       /         \
    m3 (CW)     m4 (CCW)
```

Motor positions (meters from center):

| Motor | X | Y | Z offset | Spin |
|---|---|---|---|---|
| m1 (front-right) | +0.03253 | +0.03253 | +0.0032 | CW  (+1) |
| m2 (front-left)  | −0.03253 | +0.03253 | +0.0032 | CCW (−1) |
| m3 (back-left)   | −0.03253 | −0.03253 | +0.0032 | CW  (+1) |
| m4 (back-right)  | +0.03253 | −0.03253 | +0.0032 | CCW (−1) |

**Aerodynamic coefficients:**

| Symbol | Parameter | Default value | Unit |
|---|---|---|---|
| k_T | `thrust_coeff` | 2.45 × 10⁻⁸ | N / (rad/s)² |
| k_Q | `torque_coeff` | 1.46 × 10⁻¹⁰ | N·m / (rad/s)² |
| ω_max | `max_motor_omega` | 2600 | rad/s |
| τ_m | `tau_motor` | 0.02 | s |
| h_ge | `ground_effect_height` | 0.092 | m |
| k_ge | `ground_effect_gain` | 0.25 | — |
| c_v | `drag_coeff_lin` | (0.08, 0.08, 0.12) | N·s²/m² |
| c_ω | `drag_coeff_ang` | (2e-6, 2e-6, 4e-6) | N·m·s²/rad² |

**Hover operating point** (useful for sanity checks):

```text
omega_hover = sqrt(m * g / (4 * k_T))
            = sqrt(0.027 * 9.81 / (4 * 2.45e-8))
            ≈ 1640 rad/s  (63% of omega_max)
```

**Simulation timing:**

| Property | Value |
|---|---|
| MuJoCo timestep | 0.001 s (1 kHz) |
| n_substeps | 10 |
| Control period (dt_ctrl) | 0.01 s (100 Hz) |
| Integrator | `implicitfast` |

### 4.1 Propeller thrust

Each of 4 motors spins with angular speed `omega` (rad/s).

Each motor makes upward thrust approximately:

`thrust_i = thrust_coeff * omega_i^2 * ground_effect_multiplier`

Why squared? If propellers spin faster, air push grows nonlinearly.

### 4.2 Torques (rotation effects)

The drone rotates because:
- motors are away from center (lever arm -> roll/pitch torque),
- spinning props create reaction yaw torque.

So the model adds:
- lever-arm cross-product torques,
- plus yaw torque from rotor directions.

### 4.3 Drag

Air drag slows motion:
- linear drag opposes translation,
- angular drag opposes rotation.

In code, drag is quadratic in velocity magnitude (a common approximation).

### 4.4 Ground effect

Near the floor, downwash bounces and adds lift.

This model increases thrust near ground, then reduces this bonus when:
- drone is tilted a lot,
- drone moves sideways quickly.

So ground effect is strongest when low, upright, and slow laterally.

### 4.5 Disturbances ("wind")

Random disturbances are added as:
- force in world frame,
- torque in body frame.

This makes learning robust: the policy cannot memorize a perfectly calm world.

### 4.6 Motor lag

Motors follow a first-order lag:
- command says "go fast",
- real speed approaches target gradually with time constant `tau_motor`.

This is realistic and important for control stability.

### 4.7 Battery model

Battery state of charge (`battery_soc`) decreases over time using a simple electrical model.

It includes:
- voltage sag with lower SOC,
- approximate per-motor current drain,
- small baseline avionics power.

This is still a simplified battery model, but better than constant battery.

### 4.8 Physics math (equations with symbol meanings)

This section writes the same ideas as formulas.

#### 4.8.1 Motor command mapping

Default direct mode:

```text
u_i = clip((a_i + 1) / 2, 0, 1)
```

- `a_i`: action from policy in `[-1, 1]`
- `u_i`: normalized motor command in `[0, 1]`

Mixer mode (`use_mixer=True`) builds `[u1, u2, u3, u4]^T` from collective+attitude channels via mixer matrix `M`:

```text
u = clip(M * m, 0, 1)
```

where `m = [collective, roll, pitch, yaw]^T`.

#### 4.8.2 Motor lag dynamics

Target speed:

```text
omega_des_i = u_i * omega_max * s_v
```

with voltage scale `s_v in [0, 1]`.

First-order lag update (discrete):

```text
omega_i <- omega_i + alpha * (omega_des_i - omega_i)
alpha   = dt_ctrl / tau_motor
```

This is a standard first-order system (low-pass behavior).

#### 4.8.3 Thrust and torques

Per motor thrust:

```text
T_i = k_T * omega_i^2 * g_i
```

- `k_T`: thrust coefficient
- `g_i`: ground-effect multiplier

Total force in world frame:

```text
F_world = sum_i( R * [0, 0, T_i]^T ) + R * F_drag_body + F_dist_world
```

Total torque in body frame:

```text
tau_body =
  sum_i( r_i x [0, 0, T_i]^T )
  + sum_i( d_i * k_Q * omega_i^2 * e_z )
  + tau_drag_body
  + tau_dist_body
```

- `R`: body-to-world rotation matrix
- `r_i`: arm vector from COM to rotor `i`
- `d_i in {+1, -1}`: rotor spin direction sign
- `k_Q`: reaction-torque coefficient

Then convert to world torque:

```text
tau_world = R * tau_body
```

#### 4.8.4 Drag model

Quadratic drag in body frame:

```text
F_drag_body   = -c_v     .* abs(v_body)     .* v_body
tau_drag_body = -c_omega .* abs(omega_body) .* omega_body
```

`.*` means element-wise multiplication.

#### 4.8.5 Ground effect formula

Height term (active only near floor):

```text
r_h = 1 - h / h_ge
b   = k_ge * r_h^2
```

Tilt and lateral speed attenuation:

```text
f_tilt  = max(cos(theta), 0)^2
f_speed = exp(-1.4 * ||v_xy||)
```

Final multiplier:

```text
g_i = clip(1 + b * f_tilt * f_speed, 1, 1.25)
```

#### 4.8.6 Disturbance process (OU process)

Force disturbance:

```text
dx = theta * (-x) * dt + sigma * dW_t
```

In discrete form (used in code):

```text
x_{t+1} = x_t + theta * (-x_t) * dt + sigma * sqrt(dt) * eps
eps ~ N(0, I)
```

Same idea for disturbance torque, with scaled `sigma` and max norm.

#### 4.8.7 Battery power update

Voltage scale:

```text
s_v   = clip(0.78 + 0.22 * SOC, 0, 1)
V_bat = V_nom * s_v
```

Current approximation per motor:

```text
I_i = max((V_bat * u_i - E_i) / R_m, 0)
```

Total power:

```text
P = V_bat * sum_i(I_i) + P_base
```

SOC update:

```text
SOC <- clip(SOC - (P * dt_ctrl) / E_bat, 0, 1)
```

#### 4.8.8 Sensor noise model

Observations are corrupted with Gaussian noise (except in `privileged` mode):

```text
x_noisy = x_true + N(0, sigma) * sensor_noise_scale
```

| Sensor | σ (base) | Why this matters |
|---|---|---|
| Gyroscope | 0.02 rad/s | Simulates MEMS gyro drift |
| Accelerometer | 0.08 m/s² | Real IMU vibration noise |
| Velocity (world) | 0.015 m/s | State-estimator uncertainty |
| Barometric altitude | 0.01 m | Pressure sensor noise floor |
| Position error | 0.01 m | Applied before body-frame transform |
| Flowdeck velocity | 0.02 m/s | Optical flow estimation noise |

Multiply all σ by `sensor_noise_scale` (default 1.0). Set to 0 for noiseless training.

This matters for thesis work because **policies trained with noise generalize better to real hardware** (domain randomization of perception).

---

## 5. Actions (what the agent can control)

Action vector has 4 values.

Default mode (`use_mixer=False`):
- each action directly controls one motor command level.
- mapping: `[-1, 1] -> [0, 1]`.

Optional mixer mode (`use_mixer=True`):
- action means `[collective, roll, pitch, yaw]`,
- then mixed into 4 motor commands.

For beginners:
- direct-per-motor is harder to interpret but very flexible,
- mixer mode is closer to "pilot style" commands.

---

## 6. Observations (what the agent sees)

### Hover environment observation

Base terms:
- position error to goal (3)
- velocity (3)
- quaternion orientation (4)
- gyro (3)
- accelerometer (3)
- altitude/baro-like term (1)
- normalized motor speeds (4)
- battery SOC (1)
- last action (4)

Total:
- base: 26 values
- flowdeck mode: 28 values (adds body-frame planar flow-like velocity 2)

### Reach environment observation

Similar, but goal-oriented and with one extra progress term:
- goal error in body frame (3)
- body-frame velocity (3)
- quaternion (4)
- gyro (3)
- accelerometer (3)
- altitude (1)
- normalized motor speeds (4)
- battery SOC (1)
- last action (4)
- normalized goals reached counter (1)

Total:
- base: 27 values
- flowdeck mode: 29 values

Why include `last_action`?
- helps policy know momentum of its own command changes.

Why include `motor_norm`?
- tells policy real actuation state (important with motor lag).

---

## 7. Reward design (how the agent gets points)

### Hover reward idea

Policy gets better score for:
- being close to goal position,
- being upright,
- moving smoothly with low speed,
- holding stable hover for many steps.

Policy gets penalties for:
- large position error,
- tilt and high speeds,
- abrupt action changes,
- crash/battery depletion.

### Reach reward idea

Policy gets better score for:
- being upright and alive,
- moving toward the current goal,
- reaching and holding at goal,
- collecting goal bonuses.

Penalties:
- too much angular speed,
- abrupt action changes,
- crash/battery/out-of-bounds.

### 7.1 Hover reward equation (from code)

```text
r =
  0.03
  + 0.9 * exp(-d / 0.08)
  - 1.5 * d
  - 0.25 * abs(e_yaw)
  - 0.18 * theta
  - 0.15 * ||v||
  - 0.05 * ||omega||
  - 0.02 * ||a_t - a_{t-1}||
```

Then bonuses/penalties:
- `+0.06` if stable this step,
- `+3.0` on hover success,
- `-3.0` on crash,
- `-1.0` on battery depletion.

Where:
- `d`: position error,
- `e_yaw`: yaw error,
- `theta`: tilt angle in radians,
- `v, omega`: linear and angular velocity.

### 7.2 Reach reward equation (from code)

```text
r =
  0.25 * cos(theta)
  + 0.05
  + 0.8 * exp(-d / 0.25)
  + 0.3 * tanh(v_parallel / 0.3)
  - 0.06 * min(||omega||, 5)
  - 0.02 * ||a_t - a_{t-1}||
```

Then:
- `+0.08` if near goal,
- `+5.0` when a goal is completed,
- `-5.0` on crash,
- `-1.0` on battery depletion.

`v_parallel` is velocity projected toward the goal direction.

---

## 8. Episode end conditions

Episode can terminate/truncate when:
- crashed with floor,
- too much tilt,
- out of workspace,
- battery too low,
- time limit reached.

Reach task also keeps sampling new goals during an episode until one of those stop conditions happens.

---

## 9. Why PPO works here (beginner-friendly explanation)

PPO = Proximal Policy Optimization.

### 9.1 What PPO learns

PPO trains:
- a **policy** (the actor): "what action should I take now?"
- a **value function** (the critic): "how good is this state?"

### 9.2 How learning loop works

1. Run policy in environment, collect many transitions:
   `(obs, action, reward, next_obs, done)`
2. Estimate how good each action was (advantage).
3. Update policy to increase probability of good actions.
4. Update value function to better predict future return.
5. Repeat.

### 9.3 Why "proximal" matters

If policy changes too much in one update, training can collapse.

PPO uses a clipped objective so each update is not too large.
This gives:
- more stable learning,
- fewer catastrophic jumps,
- better reliability in continuous control tasks like drones.

### 9.5 PPO math (intuitive but exact)

Policy update uses probability ratio:

```text
r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```

Clipped objective:

```text
L_clip(theta) = E_t[
  min(
    r_t(theta) * A_hat_t,
    clip(r_t(theta), 1 - eps, 1 + eps) * A_hat_t
  )
]
```

Meaning:
- if update tries to change policy too much (`r_t` too far from 1),
- clipping limits that change.

Value loss:

```text
L_V(phi) = E_t[(V_phi(s_t) - R_hat_t)^2]
```

Entropy bonus (for exploration):

```text
L_H(theta) = E_t[H(pi_theta(. | s_t))]
```

Typical final objective (maximize):

```text
L = L_clip - c1 * L_V + c2 * L_H
```

### 9.6 Advantage and return math (how "good action" is measured)

TD residual:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

GAE advantage:

```text
A_hat_t = sum_{l=0..inf} (gamma * lambda)^l * delta_{t+l}
```

Return target:

```text
R_hat_t = A_hat_t + V(s_t)
```

So PPO does not learn from raw reward only; it learns from **advantage**:
"was this action better or worse than expected from this state?"

### 9.4 Why PPO is a good default for this drone

- Works with continuous actions.
- Stable enough for noisy physics.
- Popular and battle-tested in robotics simulators.
- Easy to scale with parallel environments.

---

## 10. How can the agent learn from random start?

At first, actions are almost random.

But even random behavior sometimes gets slightly better reward (for example: less tilt).
PPO amplifies those better actions little by little.

After many episodes:
- bad actions become less likely,
- good actions become more likely,
- policy becomes smooth and goal-directed.

This is like practicing a sport:
- mistakes happen,
- feedback is small but consistent,
- skill appears after many repetitions.

---

## 10.5 Training pipeline (practical details)

### 10.5.1 Hyperparameters

All values from `scripts/train.py`:

| Hyperparameter | Reach | Hover | What it controls |
|---|---|---|---|
| Algorithm | PPO (SB3) | PPO (SB3) | Policy gradient method |
| Total timesteps | 6,000,000 | 4,000,000 | Amount of experience |
| `n_envs` | 16 | 16 | Parallel environments |
| `n_steps` | 1024 | 1024 | Steps per env before update |
| Effective batch | 16×1024 = 16384 | 16384 | Transitions per update |
| `n_epochs` | 8 | 8 | Passes over each batch |
| `n_minibatches` | 4 | 4 | Splits per epoch |
| Minibatch size | 4096 | 4096 | 16384 / 4 |
| `learning_rate` | 3 × 10⁻⁴ | 3 × 10⁻⁴ | Step size for gradient updates |
| `gamma` (γ) | 0.995 | 0.995 | Discount factor |
| `gae_lambda` (λ) | 0.95 | 0.95 | GAE smoothing |
| `clip_range` (ε) | 0.2 | 0.2 | PPO clipping bound |
| `ent_coef` | 0.003 | 0.003 | Entropy bonus weight |
| `vf_coef` | 1.0 | 1.0 | Value loss weight |
| `max_grad_norm` | 1.0 | 1.0 | Gradient clipping |
| Device | CUDA | CUDA | GPU acceleration |

### 10.5.2 Network architecture

```text
Observation (27d) ─→ [Linear 128] ─→ Tanh ─→ [Linear 128] ─→ Tanh ─→ Action mean (4d)
                                                                    ↘ Log-std (4d, learned)

Observation (27d) ─→ [Linear 128] ─→ Tanh ─→ [Linear 128] ─→ Tanh ─→ Value (1d)
```

Reach uses `[128, 128]` for both actor and critic. Hover uses `[256, 256]`.

Why **Tanh** activation? It bounds hidden activations, which helps stability in continuous control. ReLU can diverge more easily with noisy gradients.

### 10.5.3 Observation and reward normalization (VecNormalize)

SB3's `VecNormalize` wrapper maintains running statistics:

```text
obs_normalized = (obs - running_mean) / sqrt(running_var + 1e-8)
```

Clipped to `[-10, 10]` (via `clip_obs=10.0`).

Rewards are also normalized the same way (`norm_reward=True`).

**Why this matters:**
- Raw observations span very different scales (quaternion ∈ [-1,1] vs altitude ∈ [0,1.5]).
- Without normalization, the network struggles to learn features at different magnitudes.
- The normalizer stats are saved alongside the model and **must be loaded during evaluation**.

### 10.5.4 Configuration profiles

Defined in `config.py`:

| Profile name | Task | Key settings |
|---|---|---|
| `crazyflie_hover` | Hover | All defaults, no noise |
| `crazyflie_hover_dense_stable` | Hover | `disturbance_σ=0.03`, `actuator_noise=0.015`, `sensor_noise=1.0` |
| `crazyflie_hover_flowdeck` | Hover | Same + `observation_mode="flowdeck"` (29d obs) |
| `crazyflie_reach` | Reach | All defaults, no noise |
| `crazyflie_reach_dense_stable` | Reach | `disturbance_σ=0.03`, `actuator_noise=0.015`, `sensor_noise=1.0` |

Use `dense_stable` profiles for thesis-quality training (noise forces generalization).

### 10.5.5 How to train

```bash
# Reach task (recommended starting point):
python scripts/train.py --task crazyflie_reach --config crazyflie_reach_dense_stable

# Hover task:
python scripts/train.py --task crazyflie_hover --config crazyflie_hover_dense_stable
```

Models are saved to `runs/<run_name>/` with:
- `best_model.zip` — highest mean reward checkpoint,
- `vecnormalize.pkl` — observation/reward running statistics,
- TensorBoard event files.

### 10.5.6 How to evaluate

```bash
python scripts/eval_crazyflie_reach.py --model runs/<run_name>/best_model.zip
```

Interactive controls in the OpenCV window:
- **R**: reset episode,
- **M**: toggle mouse follow mode (click/scroll to move goal),
- **V**: start/stop video recording,
- **ESC**: quit.

---

## 11. Beginner experiments to try

1. Start with hover, no disturbance:
   set `disturbance_sigma=0.0` and observe easier stability.
2. Increase disturbance slowly:
   compare policy robustness.
3. Toggle `use_mixer=True`:
   compare learning speed vs direct motor control.
4. Change `reach_threshold` and `reach_hold_steps`:
   see how "strict success" changes behavior.
5. Plot `battery_soc` over time:
   understand energy constraints.

---

## 11.5 Reading training curves (TensorBoard)

Launch TensorBoard:

```bash
tensorboard --logdir runs/
```

**Key metrics to watch:**

| Metric | Healthy sign | Warning sign |
|---|---|---|
| `rollout/ep_rew_mean` | Steady upward trend | Flat or decreasing after 1M steps |
| `rollout/ep_len_mean` | Increases (survives longer) | Stuck near 50–100 (crashes early) |
| `train/entropy_loss` | Slowly decreases | Collapses to near-zero (policy too deterministic too early) |
| `train/policy_gradient_loss` | Small magnitude, stable | Wild oscillations |
| `train/value_loss` | Decreases over time | Increasing = value function diverging |
| `train/clip_fraction` | 0.05–0.15 | > 0.3 = updates too aggressive |
| `train/approx_kl` | < 0.02 | > 0.05 = policy changing too fast |

**Typical training timeline (reach task, 6M steps):**

1. **0–500k steps**: Random exploration, reward ≈ 0, frequent crashes.
2. **500k–2M**: Policy learns to stay upright and alive, reward rises.
3. **2M–4M**: Navigation emerges, goals start being reached.
4. **4M–6M**: Fine-tuning, reward plateaus, action smoothness improves.

If reward is still flat at 2M steps, check: (1) reward scale, (2) observation normalization loaded, (3) environment not broken.

---

## 12. Important simplifications (honest realism notes)

This simulator is useful and educational, but still simplified:
- no full propeller aerodynamics (blade flapping/inflow interaction are not explicit),
- battery/electrical model is approximate,
- disturbances are stochastic processes, not CFD wind fields.

This is normal for RL training: we choose a model that is fast, stable, and informative.

### 12.1 Sim-to-real gap analysis

For thesis students considering real-world deployment, here is what the sim approximates vs what reality adds:

| Aspect | Simulator | Reality | Gap severity |
|---|---|---|---|
| **Thrust model** | k_T · ω² (ideal quadratic) | Varies with airspeed, blade pitch, inflow | Medium |
| **Motor dynamics** | 1st-order lag (τ = 20 ms) | 2nd-order with nonlinear friction, ESC PWM | Low–Medium |
| **Ground effect** | Analytical cos²·exp formula | Complex recirculation, depends on surface | Low (close range only) |
| **Drag** | Quadratic, 3-axis independent | Coupled, varies with angle of attack | Low |
| **IMU noise** | Gaussian i.i.d. | Bias drift, temperature-dependent, correlated | Medium |
| **State estimation** | Direct sensor read + noise | EKF/complementary filter with latency | High |
| **Communication** | Instant (same process) | Radio link with 4–10 ms latency, packet loss | High |
| **Battery** | Heuristic power drain | Nonlinear LiPo chemistry, temperature effects | Low |

**Domain randomization** strategies already in the code:
- Actuator noise (`actuator_noise_std`): jitters motor commands.
- Sensor noise (`sensor_noise_scale`): corrupts observations.
- OU disturbances (`disturbance_sigma`): simulates gusts.
- Spawn noise: randomizes initial position and orientation.
- Battery SOC: starts between 0.95–1.0.

To improve sim-to-real transfer, consider adding:
1. **Latency injection** (delay observations by 1–3 steps).
2. **Mass/inertia randomization** (±10% of nominal).
3. **Motor constant randomization** (k_T ± 5%).
4. **Communication dropout** (repeat stale observations occasionally).

---

## 13. Thesis project ideas

These are concrete directions a student can pursue using this codebase:

### 13.1 Sim-to-real transfer

**Goal:** Deploy a policy trained in this simulator onto a real Crazyflie 2.1.

**Steps:** Add observation latency, mass randomization, and motor constant randomization as domain randomization. Train with aggressive noise. Evaluate on real hardware via Crazyradio PA.

**Thesis contribution:** Quantify the reality gap and which randomization parameters matter most.

### 13.2 Reward shaping comparison

**Goal:** Compare different reward functions on the same task.

**Steps:** Implement 3–4 reward variants (sparse, dense, curriculum-based, hindsight). Train each for the same budget. Compare sample efficiency and final performance.

**Thesis contribution:** Principled reward engineering guidelines for quadrotor tasks.

### 13.3 Alternative RL algorithms

**Goal:** Benchmark PPO against SAC, TD3, or DDPG on the reach task.

**Steps:** Use Stable-Baselines3 implementations. Keep environment and hyperparameter search budget equal. Compare reward curves, wall-clock time, and robustness to noise.

**Thesis contribution:** Algorithm selection guidelines for continuous quadrotor control.

### 13.4 Curriculum learning for harder tasks

**Goal:** Train incrementally harder variants (hover → reach → obstacle avoidance → trajectory tracking).

**Steps:** Start with `crazyflie_hover`, transfer policy to `crazyflie_reach`, then add obstacles. Use weight initialization from previous stage.

**Thesis contribution:** Does curriculum transfer improve sample efficiency vs training from scratch?

### 13.5 Battery-aware control

**Goal:** Policy that adapts behavior as battery depletes.

**Steps:** Train with battery_soc in observation. Compare against a policy trained without battery info. Evaluate: does the battery-aware policy land safely when SOC is critical?

**Thesis contribution:** Energy-constrained RL for micro-UAVs.

### 13.6 Disturbance rejection analysis

**Goal:** Characterize how disturbance intensity affects learned policies.

**Steps:** Train separate policies at `disturbance_sigma` ∈ {0, 0.01, 0.03, 0.06, 0.12}. Cross-evaluate each at all disturbance levels.

**Thesis contribution:** Robustness-performance trade-off curve.

### 13.7 Observation ablation study

**Goal:** Which observation components are essential for learning?

**Steps:** Remove one sensor group at a time (gyro, accel, motor_norm, battery_soc, last_action). Train and compare performance degradation.

**Thesis contribution:** Minimum sensor set for quadrotor RL.

---

## 14. Glossary (very short)

- **Observation**: numbers the agent reads.
- **Action**: numbers the agent outputs.
- **Reward**: score signal used for learning.
- **Policy**: mapping from observation to action.
- **Episode**: one run from reset to termination/truncation.
- **PPO**: stable policy-gradient RL algorithm.
- **Quaternion**: rotation representation used by MuJoCo (4 numbers, no gimbal lock).
- **GAE**: Generalized Advantage Estimation — smooths noisy advantage estimates.
- **VecNormalize**: SB3 wrapper that standardizes observations and rewards using running statistics.
- **Domain randomization**: injecting noise/variation during training so the policy generalizes.
- **Ornstein-Uhlenbeck (OU)**: mean-reverting stochastic process used for disturbance forces.
- **Sim-to-real**: transferring a policy trained in simulation to a real robot.

---

## 15. Hover vs Reach side-by-side comparison

| Aspect | Hover | Reach |
|---|---|---|
| **Class** | `CrazyflieHoverEnv` | `CrazyflieReachEnv` |
| **Goal type** | Single fixed hover point + yaw | Sequence of random 3D positions |
| **Goal range (XY)** | ±0.18 m | ±0.55 m |
| **Goal range (Z)** | 0.22–0.55 m | 0.20–0.65 m |
| **Yaw goal** | Yes (random per episode) | No |
| **Spawn location** | Near goal (±3 cm XY, ±2 cm Z) | Fixed center (0, 0, 0.35) + noise |
| **Observation dim** | 26 (28 flowdeck) | 27 (29 flowdeck) |
| **Pos/vel frame** | World frame | Body frame |
| **Extra obs** | — | `goals_norm` (count / 20) |
| **Reward focus** | Tight hover precision | Navigation + goal collection |
| **Alive bonus** | 0.03 | 0.05 |
| **Position reward** | 0.9 · exp(−d/0.08) − 1.5d | 0.8 · exp(−d/0.25) |
| **Yaw penalty** | −0.25 · \|e_yaw\| | None |
| **Velocity toward goal** | None | 0.3 · tanh(v_∥/0.3) |
| **Success condition** | d < 5cm, tilt < 8°, speed < 0.2 m/s for 40 steps | d < 6cm for 15 steps |
| **Success outcome** | Episode terminates (+3.0 bonus) | New goal sampled (+5.0 bonus) |
| **Crash penalty** | −3.0 | −5.0 |
| **Time limit** | 600 steps (6 s) | 800 steps (8 s) |
| **Network size** | [256, 256] | [128, 128] |
| **Training budget** | 4M steps | 6M steps |

**Key design insight:** Hover rewards precision (tight exponential decay, yaw control, strict success criteria). Reach rewards mobility (wider approach reward, velocity reward toward goal, multiple goals per episode).

---