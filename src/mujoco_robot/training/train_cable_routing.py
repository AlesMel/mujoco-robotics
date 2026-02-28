"""Train PPO on the cable-routing task.

Usage::

    python -m mujoco_robot.training.train_cable_routing \
        --cfg-name ur3e_cable_routing_dense_stable
"""
from __future__ import annotations

import argparse
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from mujoco_robot.tasks import (
    get_cable_routing_cfg,
    list_cable_routing_cfgs,
    make_cable_routing_gymnasium,
)
from mujoco_robot.training.callbacks import (
    BestEpisodeVideoCallback,
    CableRoutingCurriculumCallback,
)


DEFAULT_CFG_NAME = "ur3e_cable_grasp"


def train_cable_routing_ppo(
    total_timesteps: int = 1_500_000,
    n_envs: int = 8,
    log_dir: str = "runs",
    log_name: str = "cable_routing_ppo",
    save_video: bool = True,
    save_video_every: int = 25_000,
    progress_bar: bool = True,
    sb3_verbose: int = 0,
    callback_new_best_only: bool = True,
    cfg_name: str = DEFAULT_CFG_NAME,
    use_curriculum: bool = True,
):
    """Quick-start PPO training on the cable-routing task."""

    def build_cfg(seed: int | None, render_mode: str | None):
        cfg = get_cable_routing_cfg(cfg_name)
        cfg.seed = seed
        cfg.render_mode = render_mode
        return cfg

    preview_cfg = build_cfg(seed=0, render_mode=None)

    # Auto-detect task mode from env_kwargs.
    task_mode = preview_cfg.env_kwargs.get("task_mode", "route")
    if task_mode == "grasp" and use_curriculum:
        use_curriculum = False  # curriculum is meaningless for grasp subtask

    print(f"\n{'='*50}")
    print("  Cable-routing training config")
    print(f"  Config profile:   {cfg_name}")
    print(f"  Task mode:        {task_mode}")
    print(f"  Robot profile:    {preview_cfg.actuator_profile}")
    print(f"  Time limit:       {preview_cfg.time_limit}")
    print(f"  Progress bar:     {progress_bar}")
    print(f"  Curriculum:       {use_curriculum}")
    print(f"  Save video:       {save_video}")
    print(f"  Save video every: {save_video_every:,}")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"{'='*50}\n")

    def make_env(rank: int):
        def _init():
            cfg = build_cfg(seed=rank, render_mode=None)
            return Monitor(make_cable_routing_gymnasium(cfg))

        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    # Disable reward normalization for grasp mode -- the raw distance/velocity
    # reward is already well-scaled and normalization destroys the gradient.
    norm_reward = task_mode != "grasp"
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)

    env_name = f"cable_routing_{preview_cfg.actuator_profile}_{cfg_name}".replace("/", "_")
    callbacks = []
    if use_curriculum:
        callbacks.append(
            CableRoutingCurriculumCallback(
                total_timesteps=total_timesteps,
                stage1_frac=0.35,
                stage2_frac=0.70,
                verbose=1 if sb3_verbose > 0 else 0,
            )
        )

    if save_video:

        def make_eval_env():
            cfg = build_cfg(seed=None, render_mode="rgb_array")
            return Monitor(make_cable_routing_gymnasium(cfg))

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
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=nn.Tanh,
        ),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on cable-routing task.")
    parser.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    parser.add_argument("--list-cfgs", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video-every", type=int, default=25_000)
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Stable-Baselines3 tqdm/rich progress bar.",
    )
    parser.add_argument(
        "--sb3-verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity (0 recommended with progress bar).",
    )
    parser.add_argument(
        "--callback-new-best-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, callback prints only when eval return reaches a new best.",
    )
    parser.add_argument(
        "--use-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a single-run staged curriculum (easy -> mid -> full).",
    )
    args = parser.parse_args()

    if args.list_cfgs:
        print("\n".join(list_cable_routing_cfgs()))
        return

    train_cable_routing_ppo(
        cfg_name=args.cfg_name,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
        progress_bar=args.progress_bar,
        sb3_verbose=args.sb3_verbose,
        callback_new_best_only=args.callback_new_best_only,
        use_curriculum=args.use_curriculum,
    )


if __name__ == "__main__":
    main()
