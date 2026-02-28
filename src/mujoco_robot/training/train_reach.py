"""Train PPO on the manager-based reach task.

Usage::

    python -m mujoco_robot.training.train_reach --cfg-name ur5e_joint_pos --total-timesteps 500000
    python -m mujoco_robot.training.train_reach --cfg-name ur5e_joint_pos

Or from Python::

    from mujoco_robot.training.train_reach import train_reach_ppo
    model = train_reach_ppo(cfg_name="ur5e_joint_pos", total_timesteps=500_000)
"""
from __future__ import annotations

import argparse

from stable_baselines3.common.monitor import Monitor

from mujoco_robot.tasks.reach import (
    get_reach_cfg,
    make_reach_manager_based_gymnasium,
)
from mujoco_robot.training.ppo_config import (
    PPOTrainConfig,
    add_common_cli_args,
    config_from_cli,
    train_ppo,
)


DEFAULT_CFG_NAME = "ur3e_joint_pos_dense_stable"


def train_reach_ppo(
    cfg_name: str = DEFAULT_CFG_NAME,
    **overrides,
):
    """Quick-start PPO training on the manager-based reach task."""
    profile_name = cfg_name

    def build_cfg(seed: int | None, render_mode: str | None):
        cfg = get_reach_cfg(profile_name)
        cfg.scene.render_mode = render_mode
        cfg.episode.seed = seed
        return cfg

    preview_cfg = build_cfg(seed=0, render_mode=None)

    print(f"\n{'='*50}")
    print("  Reach training config")
    print(f"  Config profile:   {profile_name}")
    print(f"  Robot:            {preview_cfg.scene.robot}")
    print(f"  Control variant:  {preview_cfg.actions.control_variant}")
    print(f"  Joint act scale:  {preview_cfg.actions.joint_action_scale}")
    print(f"  Reach threshold:  {preview_cfg.success.reach_threshold:.3f} m")
    print(f"  Ori threshold:    {preview_cfg.success.ori_threshold:.2f} rad")
    print(f"  Hold steps:       {preview_cfg.success.success_hold_steps}")
    print(f"  Success bonus:    {preview_cfg.success.success_bonus:.3f}")
    print(f"  Stay reward w:    {preview_cfg.success.stay_reward_weight:.3f} /s")
    print(f"  Resample success: {preview_cfg.success.resample_on_success}")
    print(f"  Train obs noise:  {preview_cfg.physics.obs_noise:.4f}")
    print("  Video obs noise:  0.0000")
    if preview_cfg.actions.control_variant == "joint_pos_isaac_reward":
        print("  Episode setup:    built-in defaults (12s, no in-episode goal resample)")
    print(f"{'='*50}\n")

    env_name = f"reach_{preview_cfg.scene.robot}_{cfg_name}".replace("/", "_")

    def make_env(rank: int):
        def _init():
            cfg = build_cfg(seed=rank, render_mode=None)
            return Monitor(make_reach_manager_based_gymnasium(cfg))
        return _init

    def make_eval_env():
        cfg = build_cfg(seed=None, render_mode="rgb_array")
        cfg.physics.obs_noise = 0.0
        return Monitor(make_reach_manager_based_gymnasium(cfg))

    train_cfg = PPOTrainConfig(
        env_factory=make_env,
        eval_env_factory=make_eval_env,
        env_name=env_name,
        log_name="reach_ppo",
        enable_reach_eval_metrics=True,
        **overrides,
    )
    return train_ppo(train_cfg)


def main():
    p = argparse.ArgumentParser(description="Train PPO on manager-based reach.")
    p.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    add_common_cli_args(p, default_timesteps=10_000_000, default_n_envs=16)
    args = p.parse_args()

    train_reach_ppo(
        cfg_name=args.cfg_name,
        **config_from_cli(args),
    )


if __name__ == "__main__":
    main()
