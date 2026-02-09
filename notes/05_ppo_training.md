# PPO Training — How the Neural Network Learns

> **Where this is used in our code:**
> [`train_reach.py`](../src/mujoco_robot/training/train_reach.py) — training script
> We use [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3), a popular RL library.

> **Prerequisites:** [03 — The RL Environment](03_rl_environment.md)

---

## Table of Contents

1. [From RL Theory to Practice](#1-from-rl-theory-to-practice)
2. [Policy Gradient Methods — The Core Idea](#2-policy-gradient-methods--the-core-idea)
3. [PPO — Proximal Policy Optimization](#3-ppo--proximal-policy-optimization)
4. [The Neural Network Architecture](#4-the-neural-network-architecture)
5. [Parallel Environments & VecNormalize](#5-parallel-environments--vecnormalize)
6. [Hyperparameter Guide](#6-hyperparameter-guide)
7. [Curriculum Learning](#7-curriculum-learning)
8. [Our Training Script Explained](#8-our-training-script-explained)
9. [Monitoring & TensorBoard](#9-monitoring--tensorboard)

---

## 1. From RL Theory to Practice

In the [previous guide](03_rl_environment.md), we defined the MDP: states, actions, rewards, transitions. Now we need an algorithm that **actually learns** a good policy from experience.

There are two main families of RL algorithms:

| Family | How it works | Examples |
|--------|-------------|---------|
| **Value-based** | Learn "how good is each state/action?" then act greedily | DQN, DDQN |
| **Policy-based** | Directly learn "what action to take in each state" | REINFORCE, PPO, SAC |

We use **PPO** (Proximal Policy Optimization), which is policy-based. PPO is the most popular algorithm for continuous-action robotics because:
- Works with **continuous actions** (our 6-D action space)
- **Stable** training (doesn't diverge easily)
- **Sample efficient** enough for simulation
- **Simple** to implement and tune

---

## 2. Policy Gradient Methods — The Core Idea

### The policy

A **policy** $\pi_\theta(a|s)$ is a neural network parameterised by weights $\theta$ that maps observations to action probabilities. For continuous actions, it outputs a **Gaussian distribution**:

$$
\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))
$$

- $\mu_\theta(s)$ = mean action (the "best guess")
- $\sigma_\theta(s)$ = standard deviation (how much to explore)

When the agent acts, it **samples** from this distribution. Early in training, $\sigma$ is large (lots of random exploration). As training progresses, $\sigma$ shrinks (the agent becomes more confident).

### The objective

We want to maximise expected cumulative reward:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]
$$

### The policy gradient theorem

The key insight: we can compute the **gradient** of $J$ with respect to the policy parameters:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]
$$

where $A_t$ is the **advantage** — "how much better was this action than average?"

**Intuitive reading:**
- If $A_t > 0$ (action was better than average): increase the probability of that action
- If $A_t < 0$ (action was worse than average): decrease the probability of that action
- $\nabla_\theta \log \pi_\theta$ is the direction in parameter space that makes the action more likely

### The advantage function

The advantage $A_t$ tells us: "was this action better or worse than expected?"

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

- $Q(s_t, a_t)$ = expected return if we take action $a_t$ in state $s_t$
- $V(s_t)$ = expected return from state $s_t$ under the current policy (the "baseline")

We estimate $A_t$ using **Generalised Advantage Estimation (GAE)**:

$$
A_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error**.

The parameter $\lambda$ (GAE lambda, we use 0.95) controls the bias-variance trade-off:
- $\lambda = 0$: low variance, high bias (only uses 1-step returns)
- $\lambda = 1$: high variance, low bias (uses full Monte Carlo returns)

---

## 3. PPO — Proximal Policy Optimization

### The problem with vanilla policy gradients

Vanilla policy gradient (REINFORCE) is unstable: a single bad update can make the policy much worse, and it's hard to recover. The gradient step might be too large, causing the policy to "jump" to a very different behaviour.

### PPO's solution: clipping

PPO constrains how much the policy can change in one update. It uses a **clipped surrogate objective**:

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the **probability ratio** (how much more/less likely is this action under the new vs. old policy?)
- $\epsilon = 0.2$ is the **clip range** (we use 0.2)

### What does clipping do?

```
    L^CLIP
    ↑
    │      ╭──── clipped (can't go higher)
    │     ╱
    │    ╱
    │───╱────────── unclipped objective
    │  ╱
    │ ╱
    │╱ ← clipped (can't go lower)
    └───────────────→ r(θ)
    0  0.8  1  1.2
       ↑       ↑
    1-ε      1+ε
```

- If the advantage is **positive** (good action): the objective wants to increase $r$, but clipping caps it at $1 + \epsilon$. The policy can become at most 20% more likely to take this action.
- If the advantage is **negative** (bad action): the objective wants to decrease $r$, but clipping caps it at $1 - \epsilon$. The policy can become at most 20% less likely.

This prevents catastrophic updates: the policy changes gradually, step by step.

### The value function

PPO simultaneously learns a **value function** $V_\phi(s)$ that predicts the expected return from each state. This serves two purposes:
1. **Baseline** for advantage estimation (reduces variance)
2. **Critic** for detecting overestimation

The value function loss is simply MSE:

$$
L^{VF}(\phi) = \mathbb{E}_t\left[(V_\phi(s_t) - R_t^{target})^2\right]
$$

### Entropy bonus

PPO adds an **entropy bonus** to encourage exploration:

$$
L^{total} = L^{CLIP} - c_1 \cdot L^{VF} + c_2 \cdot H[\pi_\theta]
$$

where $H[\pi_\theta]$ is the entropy of the policy (higher = more random). We use $c_2 = 0.01$, which provides mild exploration pressure.

---

## 4. The Neural Network Architecture

### Our architecture

```
    Observation (39 floats)
         │
         ▼
    ┌──────────┐
    │ Layer 1  │ 39 → 64 neurons + tanh
    └────┬─────┘
         │
    ┌────┴─────┐
    │ Layer 2  │ 64 → 64 neurons + tanh
    └────┬─────┘
         │
    ┌────┴─────────────────────┐
    │     Split into two heads │
    ├──────────┬───────────────┤
    │          │               │
    ▼          ▼               │
  Policy     Value             │
  head       head              │
  64→6       64→1              │
  (mean μ)   (V(s))           │
    │                          │
    + σ (learned per-action    │
    │   log std deviation)     │
    ▼                          │
  Action ~ N(μ, σ²)           │
```

### Why [64, 64]?

- Our observation is 39-D and action is 6-D — this is a relatively small problem
- Two layers of 64 neurons is sufficient for learning the nonlinear mapping
- Larger networks (256, 256) would work but train slower with no benefit for this task size
- This matches Isaac Lab's default for reach tasks

### Why tanh activation?

- PPO typically uses `tanh` (not `relu`) because it's bounded and smooth
- Outputs are naturally in [-1, 1], which matches our action range
- Smoother gradients lead to more stable training

---

## 5. Parallel Environments & VecNormalize

### SubprocVecEnv — training with 16 robots simultaneously

Instead of training with one robot, we run **16 copies of the environment in parallel**:

```python
vec_env = SubprocVecEnv([make_env(i) for i in range(16)])
```

Each environment runs in its own **subprocess** (separate Python process). This means:
- 16 robots explore simultaneously → 16× more experience per second
- Different random seeds → diverse experiences
- Utilises all CPU cores (we have 16)

```
    ┌────────┐  ┌────────┐  ┌────────┐       ┌────────┐
    │ Env 0  │  │ Env 1  │  │ Env 2  │  ...  │ Env 15 │
    │ seed=0 │  │ seed=1 │  │ seed=2 │       │ seed=15│
    └───┬────┘  └───┬────┘  └───┬────┘       └───┬────┘
        │           │           │                 │
        └───────────┴───────────┴────────┬────────┘
                                         │
                                    ┌────┴────┐
                                    │   PPO   │
                                    │ (single │
                                    │ policy) │
                                    └─────────┘
```

### VecNormalize — normalising observations and rewards

Raw observations have very different scales:
- Joint angles: ~[-π, π] ≈ [-3.14, 3.14]
- Joint velocities: ~[-4, 4]
- EE position: ~[0, 1.5] metres
- Collision: {0, 1}

Neural networks learn best when all inputs are roughly the same scale (~0 mean, ~1 std). `VecNormalize` maintains **running statistics** and normalises:

```python
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
```

- **Observation normalisation**: subtracts running mean, divides by running std
- **Reward normalisation**: similarly normalises rewards (prevents large rewards from causing huge gradient steps)
- **Clip at ±10**: prevents extreme outliers from destabilising training

⚠️ **Critical**: The normalisation stats must be saved with the model and loaded at inference time. If you use raw (unnormalised) observations with a model trained on normalised ones, it will fail completely.

---

## 6. Hyperparameter Guide

Here are all our PPO hyperparameters and what they do:

### Collection parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_steps` | 2048 | Steps collected per environment before updating |
| `n_envs` | 16 | Parallel environments |
| **Batch size** | 2048 × 16 = **32,768** | Total steps per rollout |

Every 2048 steps per env (32,768 total), PPO does an update.

### Optimisation parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `batch_size` | 8192 | Minibatch size (32,768 / 4 minibatches) |
| `n_epochs` | 8 | Times to reuse each batch of data |
| `learning_rate` | 1e-3 | SGD step size (relatively large — fast learning) |
| `max_grad_norm` | 1.0 | Gradient clipping (prevents exploding gradients) |

### PPO-specific parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `gamma` | 0.99 | Discount factor (care about future ~100 steps) |
| `gae_lambda` | 0.95 | GAE parameter (moderate bias-variance trade-off) |
| `clip_range` | 0.2 | PPO clipping (policy can change ≤20% per update) |
| `ent_coef` | 0.01 | Entropy bonus (mild exploration encouragement) |
| `vf_coef` | 1.0 | Value function loss weight |

### Intuition for key parameters

**`gamma = 0.99`**: The agent cares about rewards ~100 steps into the future. With 50 Hz control, that's about 2.0 seconds — enough to plan a reaching motion.

$$
\text{effective horizon} = \frac{1}{1 - \gamma} = \frac{1}{0.01} = 100 \text{ steps} \approx 2.0 \text{ seconds}
$$

**`learning_rate = 1e-3`**: Relatively high — we want fast learning. For larger, more complex tasks, you'd use 3e-4 or lower.

**`ent_coef = 0.01`**: Small entropy bonus. Without it, the policy might converge prematurely to a suboptimal solution (always reaching the same way). With it, the policy keeps exploring different approaches.

**`n_epochs = 8`**: Each batch of experience is reused 8 times. More epochs = better sample efficiency but risk of overfitting to stale data. 8 is a common sweet spot.

---

## 7. Curriculum Learning

### What is curriculum learning?

Like teaching a child — start with easy material, gradually increase difficulty. In our environment, we ramp up penalty weights over training:

```python
def _curriculum_weight(self, base, target):
    progress = min(1.0, total_episodes / curriculum_steps)
    return base + (target - base) * progress
```

### Our curriculum schedule

| Penalty | Start | End | Ramp over |
|---------|-------|-----|-----------|
| Action rate | 0.0001 | 0.005 | 4500 episodes |
| Joint velocity | 0.0001 | 0.001 | 4500 episodes |

**Why curriculum?**

Phase 1 (early training): Low penalties → the agent is free to make large, jerky movements. It learns WHAT to do (reach the goal) without being punished for HOW it moves.

Phase 2 (later training): Higher penalties → the agent learns to reach the goal SMOOTHLY, with minimal joint velocity and smooth action changes.

If we started with high penalties from the beginning, the agent might learn to "do nothing" (zero actions → zero penalties → positive reward from proximity), which is a local optimum.

---

## 8. Our Training Script Explained

Here's a walkthrough of `train_reach.py`:

```python
# 1. Create 16 parallel environments
vec_env = SubprocVecEnv([make_env(i) for i in range(16)])

# 2. Wrap with observation/reward normalisation
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# 3. Create PPO agent with our hyperparameters
model = PPO(
    "MlpPolicy",          # Multi-Layer Perceptron (not CNN, not RNN)
    vec_env,
    n_steps=2048,          # Steps per env before update
    batch_size=8192,       # Minibatch size
    n_epochs=8,            # Reuse epochs
    learning_rate=1e-3,    # SGD step size
    gamma=0.99,            # Discount factor
    gae_lambda=0.95,       # GAE parameter
    ent_coef=0.01,         # Entropy bonus
    clip_range=0.2,        # PPO clipping
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
)

# 4. Train!
model.learn(total_timesteps=10_000_000, callback=callbacks)

# 5. Save model + normalisation stats
model.save("ppo_reach_ur3e")
vec_env.save("ppo_reach_ur3e_vecnorm.pkl")
```

### Command-line usage

```bash
# Quick test (500K steps ≈ 5 minutes)
python -m mujoco_robot.training.train_reach --robot ur3e --total-timesteps 500000

# Full training (10M steps ≈ 1 hour)
python -m mujoco_robot.training.train_reach --robot ur3e --total-timesteps 10000000

# With custom thresholds
python -m mujoco_robot.training.train_reach \
    --robot ur5e \
    --reach-threshold 0.03 \
    --ori-threshold 0.25 \
    --hold-seconds 3.0
```

---

## 9. Monitoring & TensorBoard

### What to look at during training

Training logs are saved to the `runs/` directory. View them with TensorBoard:

```bash
tensorboard --logdir runs
```

### Key metrics

| Metric | What it means | Healthy values |
|--------|--------------|----------------|
| `rollout/ep_rew_mean` | Average episode reward | Should **increase** over time |
| `rollout/ep_len_mean` | Average episode length | Should stay near `time_limit` (375) since episodes only end on timeout |
| `train/approx_kl` | How much the policy changed | Should stay **below 0.02** |
| `train/clip_fraction` | Fraction of clipped updates | Should be **0.05–0.20** |
| `train/entropy_loss` | Policy entropy | Should **decrease slowly** (getting more confident) |
| `train/policy_gradient_loss` | Policy loss | Should fluctuate near 0 |
| `train/value_loss` | Value function MSE | Should **decrease** |

### Warning signs

- **`ep_rew_mean` flat or decreasing**: Policy isn't learning — check reward function, try different hyperparameters
- **`approx_kl` > 0.05**: Policy is changing too fast — reduce learning rate
- **`entropy` drops to 0 quickly**: Policy collapsed to deterministic — increase `ent_coef`
- **`clip_fraction` > 0.3**: Too many clipped updates — reduce learning rate or increase `clip_range`

### Videos

If `--save-video` is enabled, evaluation videos are saved to `videos/reach_{robot}/`. These are the **best way** to assess training progress — you can actually see the robot's behaviour improve over time.

---

## Summary: The Full Training Pipeline

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    TRAINING LOOP                            │
    │                                                             │
    │  for each rollout (2048 steps × 16 envs = 32,768 steps):  │
    │                                                             │
    │    1. Collect experience                                    │
    │       ├── obs → policy → action (sample from Gaussian)     │
    │       ├── env.step(action) → obs', reward                  │
    │       └── store (obs, action, reward, obs', done)          │
    │                                                             │
    │    2. Compute advantages (GAE)                              │
    │       ├── For each step, compute δ = r + γV(s') - V(s)    │
    │       └── A_t = Σ (γλ)^l δ_{t+l}                          │
    │                                                             │
    │    3. Optimise (8 epochs × 4 minibatches = 32 SGD steps)   │
    │       ├── Clip policy ratio to [0.8, 1.2]                  │
    │       ├── Minimise clipped surrogate loss                   │
    │       ├── Minimise value function MSE                       │
    │       └── Add entropy bonus                                 │
    │                                                             │
    │    4. Log metrics, maybe record video                       │
    │                                                             │
    │  Repeat until total_timesteps reached                       │
    └─────────────────────────────────────────────────────────────┘
```

---

**Previous:** [04 — MuJoCo Physics](04_mujoco_physics.md)
**Back to index:** [📚 Notes Index](README.md)
