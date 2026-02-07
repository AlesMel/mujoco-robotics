# 🤖 MuJoCo Robot — UR Arm RL Environments

Modular reinforcement-learning environments for **UR5e** and **UR3e** robot arms
in [MuJoCo](https://mujoco.readthedocs.io/), with [Gymnasium](https://gymnasium.farama.org/)
wrappers for training and keyboard/gamepad teleop for interactive testing.

---

## 📁 Project Structure

```
mujoco-robot/
├── mujoco_robot/                 # Main Python package
│   ├── __init__.py               # Package root (version, top-level imports)
│   ├── robots/                   # Robot models & configuration
│   │   ├── configs.py            # RobotConfig dataclass + registry
│   │   ├── ur5e.xml              # UR5e MJCF (dual-geom collision)
│   │   └── ur3e.xml              # UR3e MJCF (dual-geom collision)
│   ├── core/                     # Reusable engine modules
│   │   ├── ik_controller.py      # Damped-least-squares IK solver
│   │   ├── collision.py          # Self-collision detector
│   │   └── xml_builder.py        # MJCF XML injection utilities
│   ├── envs/                     # Gymnasium-ready environments
│   │   ├── reach_env.py          # URReachEnv + ReachGymnasium
│   │   └── slot_sorter_env.py    # URSlotSorterEnv + SlotSorterGymnasium
│   ├── training/                 # RL training utilities
│   │   ├── callbacks.py          # BestEpisodeVideoCallback (SB3)
│   │   ├── train_reach.py        # PPO training for reach task
│   │   └── train_slot_sorter.py  # PPO training for slot sorter
│   └── teleop/                   # Interactive controllers
│       ├── keyboard.py           # Keyboard teleop (both tasks)
│       └── gamepad.py            # DualShock/DualSense gamepad
├── scripts/                      # CLI entry points
│   ├── teleop.py                 # Unified teleop launcher
│   ├── train.py                  # Unified training launcher
│   └── visual_smoke.py           # Scripted rollout video
├── tests/                        # Pytest suite
│   ├── test_reach_env.py         # 11 reach-env tests
│   └── test_slot_sorter.py       # 3 slot-sorter tests
├── assets/                       # Original XML files (kept for reference)
├── pyproject.toml                # Package metadata & dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Core (MuJoCo + Gymnasium)
pip install mujoco numpy gymnasium

# Training (optional)
pip install stable-baselines3 imageio[ffmpeg] tensorboard

# Gamepad (optional)
pip install pygame
```

Or install everything at once:

```bash
pip install -e ".[dev]"
```

### 2. Run teleop (keyboard)

```bash
# Reach task with UR5e
python scripts/teleop.py --task reach --robot ur5e

# Slot sorter
python scripts/teleop.py --task slot_sorter

# Slot sorter with gamepad
python scripts/teleop.py --task slot_sorter --gamepad
```

**Keyboard controls:**

| Key     | Action      |
|---------|-------------|
| W / S   | ±Y movement |
| A / D   | ±X movement |
| R / F   | ±Z movement |
| Q / E   | ±Yaw        |
| SPACE   | Grip toggle (slot sorter only) |
| X       | Emergency stop |

### 3. Train with PPO

```bash
# Reach task
python scripts/train.py --task reach --robot ur5e --total-timesteps 500000

# Slot sorter
python scripts/train.py --task slot_sorter --total-timesteps 1000000

# Monitor in TensorBoard
tensorboard --logdir runs
```

### 4. Use as a Python library

```python
# Gymnasium API (compatible with SB3, CleanRL, etc.)
from mujoco_robot.envs import ReachGymnasium

env = ReachGymnasium(robot="ur5e")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

```python
# Low-level API (for custom loops / teleop)
from mujoco_robot.envs import URReachEnv

env = URReachEnv(robot="ur3e", time_limit=0)
obs = env.reset()
result = env.step([0.5, 0.0, 0.0, 0.0])  # returns StepResult dataclass
print(f"EE pos: {result.info['ee_pos']}, dist: {result.info['dist']:.3f}")
```

---

## 🏗️ Architecture

### Robot Models (dual-geom collision)

Each robot MJCF uses a **dual-geom architecture** for robust collision handling:

- **`viz` class** geoms — visual only (`contype=0`), provide the rendered appearance.
- **`col` class** geoms — collision only (`contype=1`), used for physics contacts.
- Only **6 adjacent body pairs** are excluded from collision (shoulder↔base, etc.).

### Environments

| Environment | Action Dim | Obs Dim | Description |
|-------------|-----------|---------|-------------|
| `URReachEnv` | 4 | 20 | Move EE to random 3-D goals |
| `URSlotSorterEnv` | 5 | 71 | Pick colored objects → matching slots |

Both environments use:
- **Position servo actuators** (kp=200) for stable joint control.
- **Damped-least-squares IK** for Cartesian end-effector commands.
- **Dense reward shaping** to help RL exploration.

### Core Modules

| Module | Purpose |
|--------|---------|
| `IKController` | Cartesian → joint velocity via Jacobian pseudo-inverse |
| `CollisionDetector` | Counts non-adjacent robot link contacts |
| `xml_builder` | Programmatic MJCF injection (goals, cameras, etc.) |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected: **14 tests**, all passing.

---

## 📊 Supported Robots

| Robot | Reach | Link Lengths Source |
|-------|-------|---------------------|
| UR5e  | ~0.85 m | Official UR ROS2 description |
| UR3e  | ~0.50 m | Official UR ROS2 description |

To add a new robot:
1. Create an MJCF XML in `mujoco_robot/robots/`.
2. Register it in `mujoco_robot/robots/configs.py` with a `RobotConfig` entry.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mujoco` | ≥ 3.1 | Physics simulation |
| `numpy` | any | Numerical computation |
| `gymnasium` | ≥ 1.0 | RL environment API |
| `stable-baselines3` | ≥ 2.0 | PPO training (optional) |
| `imageio` | any | Video recording (optional) |
| `pygame` | any | Gamepad input (optional) |

---

## 📝 License

MIT
