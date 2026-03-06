"""Train PPO on the Crazyflie wall-maze obstacle-avoidance reach task.

Usage::

    python -m mujoco_robot.training.train_crazyflie_obstacle \
        --cfg-name crazyflie_obstacle_dense_stable
"""
from __future__ import annotations

import argparse

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from mujoco_robot.tasks import (
    get_crazyflie_cfg,
    list_crazyflie_cfgs,
    make_crazyflie_obstacle_gymnasium,
)
from mujoco_robot.training.callbacks import (
    BestEpisodeVideoCallback,
    RunDirCheckpointCallback,
)


DEFAULT_CFG_NAME = "crazyflie_obstacle_dense_stable"


def train_crazyflie_obstacle_ppo(
    total_timesteps: int = 10_000_000,
    n_envs: int = 32,
    log_dir: str = "runs",
    log_name: str = "crazyflie_obstacle_ppo",
    save_video: bool = True,
    save_video_every: int = 2_000_000,
    progress_bar: bool = True,
    sb3_verbose: int = 0,
    callback_new_best_only: bool = True,
    cfg_name: str = DEFAULT_CFG_NAME,
    realtime_render: bool = False,
):
    """Quick-start PPO training on the Crazyflie obstacle-avoidance task."""

    def build_cfg(seed: int | None, render_mode: str | None):
        cfg = get_crazyflie_cfg(cfg_name)
        cfg.seed = seed
        cfg.render_mode = render_mode
        return cfg

    preview_cfg = build_cfg(seed=0, render_mode=None)

    if realtime_render and n_envs != 1:
        print("[train] realtime render requested, forcing n_envs=1.")
        n_envs = 1

    train_render_mode = "human" if realtime_render else None

    print(f"\n{'='*58}")
    print("  Crazyflie WALL-MAZE training config")
    print(f"  Config profile:   {cfg_name}")
    print(f"  Time limit:       {preview_cfg.time_limit}")
    print(f"  Obs mode:         {preview_cfg.env_kwargs.get('observation_mode', 'state_estimate')}")
    print(f"  Rangefinder:      {preview_cfg.env_kwargs.get('rangefinder_mode', 'lidar')}")
    print(f"  Lidar rays:       {preview_cfg.env_kwargs.get('n_lidar_rays', 16)}")
    print(f"  Barriers:         {preview_cfg.env_kwargs.get('n_barriers', (2, 3))}")
    print(f"  Gap width:        {preview_cfg.env_kwargs.get('gap_width', 0.22)}")
    print(f"  Partial wall %:   {preview_cfg.env_kwargs.get('partial_wall_prob', 0.30)}")
    print(f"  Collision pen.:   {preview_cfg.env_kwargs.get('collision_penalty', 10.0)}")
    print(f"  Goal bonus:       {preview_cfg.env_kwargs.get('goal_bonus', 10.0)}")
    print(f"  Weights (P/O/E/S/T): "
          f"{preview_cfg.env_kwargs.get('w_progress', 5.0)}/"
          f"{preview_cfg.env_kwargs.get('w_obstacle', 0.5)}/"
          f"{preview_cfg.env_kwargs.get('w_energy', 0.2)}/"
          f"{preview_cfg.env_kwargs.get('w_stability', 0.3)}/"
          f"{preview_cfg.env_kwargs.get('w_task', 1.0)}")
    print(f"  Realtime render:  {realtime_render}")
    print(f"  Progress bar:     {progress_bar}")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"{'='*58}\n")

    def make_env(rank: int):
        def _init():
            cfg = build_cfg(seed=rank, render_mode=train_render_mode)
            return Monitor(make_crazyflie_obstacle_gymnasium(cfg))
        return _init

    if realtime_render:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    n_steps = 2048

    env_name = f"crazyflie_obstacle_{cfg_name}".replace("/", "_")
    callbacks = [
        RunDirCheckpointCallback(
            env_name=env_name,
            vec_norm=vec_env,
            save_every_timesteps=500_000,
        ),
    ]
    if save_video:

        def make_eval_env():
            cfg = build_cfg(seed=None, render_mode="rgb_array")
            return Monitor(make_crazyflie_obstacle_gymnasium(cfg))

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

    # Larger network for obstacle avoidance (rangefinder obs is bigger)
    n_minibatches = 4
    batch_size = (n_steps * n_envs) // n_minibatches

    def lr_schedule(progress_remaining: float) -> float:
        """Linear decay from 3e-4 → 5e-5."""
        return 5e-5 + (3e-4 - 5e-5) * progress_remaining

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=lr_schedule,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            activation_fn=nn.Tanh,
        ),
        verbose=sb3_verbose,
        tensorboard_log=log_dir,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=log_name,
        progress_bar=progress_bar,
    )

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on Crazyflie obstacle-avoidance reach task."
    )
    parser.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    parser.add_argument("--list-cfgs", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-video-every", type=int, default=2_000_000)
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sb3-verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
    )
    parser.add_argument(
        "--callback-new-best-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--realtime-render",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.list_cfgs:
        print("\n".join(list_crazyflie_cfgs()))
        return

    train_crazyflie_obstacle_ppo(
        cfg_name=args.cfg_name,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
        progress_bar=args.progress_bar,
        sb3_verbose=args.sb3_verbose,
        callback_new_best_only=args.callback_new_best_only,
        realtime_render=args.realtime_render,
    )


if __name__ == "__main__":
    main()
