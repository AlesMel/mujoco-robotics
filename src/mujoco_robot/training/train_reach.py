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
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from mujoco_robot.tasks.reach import (
    get_reach_cfg,
    make_reach_manager_based_gymnasium,
)
from mujoco_robot.training.callbacks import BestEpisodeVideoCallback


DEFAULT_CFG_NAME = "ur3e_joint_pos_dense_stable"


def train_reach_ppo(
    total_timesteps: int = 10_000_000,
    n_envs: int = 16,
    log_dir: str = "runs",
    log_name: str = "reach_ppo",
    save_video: bool = True,
    save_video_every: int = 50_000,
    progress_bar: bool = True,
    sb3_verbose: int = 0,
    callback_new_best_only: bool = True,
    cfg_name: str = DEFAULT_CFG_NAME,
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
    print(f"  Joint act scale:  {preview_cfg.actions.joint_action_scale:.3f} (IsaacLab scale)")
    print(f"  Reach threshold:  {preview_cfg.success.reach_threshold:.3f} m")
    print(f"  Ori threshold:    {preview_cfg.success.ori_threshold:.2f} rad")
    print(f"  Hold steps:       {preview_cfg.success.success_hold_steps}")
    print(f"  Success bonus:    {preview_cfg.success.success_bonus:.3f}")
    print(f"  Stay reward w:    {preview_cfg.success.stay_reward_weight:.3f} /s")
    print(f"  Resample success: {preview_cfg.success.resample_on_success}")
    print(f"  Train obs noise:  {preview_cfg.physics.obs_noise:.4f}")
    print("  Video obs noise:  0.0000")
    print(f"  Progress bar:     {progress_bar}")
    if preview_cfg.actions.control_variant == "joint_pos_isaac_reward":
        print("  Episode setup:    built-in defaults (12s, no in-episode goal resample)")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"{'='*50}\n")

    def make_env(rank):
        def _init():
            cfg = build_cfg(seed=rank, render_mode=None)
            return Monitor(make_reach_manager_based_gymnasium(cfg))

        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    env_name = f"reach_{preview_cfg.scene.robot}_{cfg_name}".replace("/", "_")
    callbacks = []
    if save_video:

        def make_eval_env():
            cfg = build_cfg(seed=0, render_mode="rgb_array")
            # Keep policy-evaluation videos noise-free for stable diagnostics.
            cfg.physics.obs_noise = 0.0
            return Monitor(make_reach_manager_based_gymnasium(cfg))

        video_cb = BestEpisodeVideoCallback(
            make_eval_env=make_eval_env,
            save_every_timesteps=save_video_every,
            video_dir="videos",
            env_name=env_name,
            deterministic=True,
            vec_norm=vec_env,
            verbose=1,
            log_new_best_only=callback_new_best_only,
        )
        callbacks.append(video_cb)

    n_steps = 1024
    n_minibatches = 4
    batch_size = (n_steps * n_envs) // n_minibatches

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=8,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=1.0,
        max_grad_norm=1.0,
        device="cuda",
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=sb3_verbose,
        tensorboard_log=log_dir,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        tb_log_name=log_name,
        progress_bar=progress_bar,
    )

    model_path = f"ppo_{env_name}"
    model.save(model_path)
    vec_norm_path = f"{model_path}_vecnorm.pkl"
    vec_env.save(vec_norm_path)
    print(f"Model saved to {model_path}.zip")
    return model


def main():
    p = argparse.ArgumentParser(description="Train PPO on manager-based reach.")
    p.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video-every", type=int, default=50_000)
    p.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Stable-Baselines3 tqdm/rich progress bar.",
    )
    p.add_argument(
        "--sb3-verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity (0 recommended with progress bar).",
    )
    p.add_argument(
        "--callback-new-best-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, callback prints only when eval return reaches a new best.",
    )
    args = p.parse_args()
    train_reach_ppo(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
        progress_bar=args.progress_bar,
        sb3_verbose=args.sb3_verbose,
        callback_new_best_only=args.callback_new_best_only,
        cfg_name=args.cfg_name,
    )


if __name__ == "__main__":
    main()
