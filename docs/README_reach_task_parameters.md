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
  --success-hold-steps 10 \
  --success-bonus 0.25 \
  --stay-reward-weight 0.05 \
  --resample-on-success \
  --progress-bar \
  --sb3-verbose 0
```

For quiet terminal output with only new-best updates:
- keep `--progress-bar`
- keep `--sb3-verbose 0`
- use `train_reach.py` default `--callback-new-best-only`

## 2) Main Reach Environment Parameters

The base defaults live in `src/mujoco_robot/envs/reach/reach_env_base.py`.

- `reach_threshold` (default `0.001`): position success tolerance in meters.
- `ori_threshold` (default `0.01`): orientation success tolerance in radians.
- `goal_resample_time_range_s` (default `None`): auto-set to a value greater than episode length when `time_limit > 0`, so goals do not resample mid-episode by default.
- `time_limit` (default `360`): episode length in control steps.
- `success_hold_steps` (default `10`): consecutive in-threshold steps needed for hold-success.
- `success_bonus` (default `0.25`): one-time bonus when hold-success is first reached.
- `stay_reward_weight` (default `0.05`): per-second reward while hold-success is maintained.
- `resample_on_success` (default `False`): if true, samples a new goal immediately after hold-success.
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

### Easier reward customization with `ReachRewardCfg`

```python
from mujoco_robot.envs.reach_env import URReachEnv
from mujoco_robot.envs.reach.mdp import ReachRewardCfg

reward_cfg = ReachRewardCfg(
    position_error_weight=-0.15,
    position_tanh_weight=0.08,
    position_tanh_std=0.08,
    orientation_error_weight=-0.08,
    include_action_rate=True,
    include_joint_vel=False,
)

env = URReachEnv(
    robot="ur3e",
    control_variant="joint_pos",
    reward_cfg=reward_cfg,
)
```

## 5) Programmatic Example

```python
from mujoco_robot.envs.reach_env import ReachGymnasium

env = ReachGymnasium(
    robot="ur3e",
    control_variant="joint_pos",
    reach_threshold=0.025,
    ori_threshold=0.20,
    goal_resample_time_range_s=(20.0, 20.0),
    success_hold_steps=12,
    success_bonus=0.3,
    stay_reward_weight=0.08,
    resample_on_success=True,
    actuator_kp=220.0,
    min_joint_damping=24.0,
    min_joint_frictionloss=1.2,
)
```
