# Reach Task Parameter Guide

This guide shows how to tune the reach task without editing core logic.

## 1) Common Training Knobs (SB3)

Use the unified script:

```bash
python -m mujoco_robot.scripts.train \
  --task reach \
  --robot ur3e \
  --control-variant joint_pos \
  --total-timesteps 500000 \
  --reach-threshold 0.03 \
  --ori-threshold 0.25 \
  --progress-bar \
  --sb3-verbose 0
```

For quiet terminal output with only new-best updates:
- keep `--progress-bar`
- keep `--sb3-verbose 0`
- use `train_reach.py` default `--callback-new-best-only`

## 2) Main Reach Environment Parameters

The base defaults live in `src/mujoco_robot/envs/reach/reach_env_base.py`.

- `reach_threshold` (default `0.03`): position success tolerance in meters.
- `ori_threshold` (default `0.25`): orientation success tolerance in radians.
- `goal_resample_time_range_s` (default `(4.0, 4.0)`): command resampling window.
- `time_limit` (default `360`): episode length in control steps.
- `joint_action_scale`, `ee_step`, `ori_step`, `ori_abs_max`: action sensitivity.
- `action_rate_weight`, `joint_vel_weight`: regularization terms.

## 3) Stability / Jitter Tuning

Also in `src/mujoco_robot/envs/reach/reach_env_base.py`:

- `actuator_kp` (default `250.0`): position-servo stiffness.
- `min_joint_damping` (default `20.0`): passive damping floor.
- `min_joint_frictionloss` (default `1.0`): passive friction floor.

Lower `actuator_kp` and/or higher `min_joint_damping` reduce oscillation near goals.

Actuator application logic is in:
- `src/mujoco_robot/robots/actuators.py`

## 4) Reward/MDP Term Overrides

Manager-style defaults:
- `src/mujoco_robot/envs/reach/mdp/config.py`

Term implementations:
- `src/mujoco_robot/envs/reach/mdp/rewards.py`
- `src/mujoco_robot/envs/reach/mdp/observations.py`
- `src/mujoco_robot/envs/reach/mdp/terminations.py`
- `src/mujoco_robot/envs/reach/mdp/commands.py`

You can pass a custom `mdp_cfg` to `ReachGymnasium(...)` or `URReachEnv(...)`.

## 5) Programmatic Example

```python
from mujoco_robot.envs.reach_env import ReachGymnasium

env = ReachGymnasium(
    robot="ur3e",
    control_variant="joint_pos",
    reach_threshold=0.025,
    ori_threshold=0.20,
    goal_resample_time_range_s=(3.0, 5.0),
    actuator_kp=220.0,
    min_joint_damping=24.0,
    min_joint_frictionloss=1.2,
)
```

