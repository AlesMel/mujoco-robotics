"""Shared PPO training configuration — eliminates copy-paste across trainers.

Every SB3-based training script should build a :class:`PPOTrainConfig` and
hand it to :func:`train_ppo`.  Task-specific logic (env factories, pre-train
banners, extra callbacks) is injected via the ``env_factory`` / ``eval_env_factory``
/ ``extra_callbacks`` fields.

Usage::

    from mujoco_robot.training.ppo_config import PPOTrainConfig, train_ppo

    cfg = PPOTrainConfig(
        env_factory=lambda rank: my_make_env(rank),
        eval_env_factory=lambda: my_make_eval_env(),
        env_name="reach_ur3e",
        total_timesteps=10_000_000,
    )
    model = train_ppo(cfg)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from mujoco_robot.training.callbacks import (
    BestEpisodeVideoCallback,
    ReachEvalMetricsCallback,
)


@dataclass
class PPOTrainConfig:
    """Declarative PPO training configuration.

    Parameters
    ----------
    env_factory : callable
        ``(rank: int) -> gymnasium.Env`` — returns a *single*, already
        ``Monitor``-wrapped training env for the given parallel rank.
    eval_env_factory : callable or None
        ``() -> gymnasium.Env`` — returns a ``Monitor``-wrapped eval env
        with ``render_mode="rgb_array"``.  ``None`` disables video recording.
    env_name : str
        Human-readable name used for log sub-folders and model filenames.

    PPO hyper-parameters
    --------------------
    total_timesteps, n_envs, n_steps, n_epochs, learning_rate, gamma,
    gae_lambda, ent_coef, clip_range, vf_coef, max_grad_norm, device,
    net_arch, activation_fn — all forwarded to ``stable_baselines3.PPO``.

    Training logistics
    ------------------
    log_dir, log_name, save_video, save_video_every, eval_every, checkpoint_every,
    progress_bar, sb3_verbose, log_new_best_only, enable_reach_eval_metrics, norm_obs,
    norm_reward, clip_obs.
    """

    # ── env factories (must be set per-task) ──────────────────────────
    env_factory: Callable[[int], Any] = field(default=None)  # type: ignore[assignment]
    eval_env_factory: Callable[[], Any] | None = None
    env_name: str = "default"

    # ── PPO hyper-parameters ──────────────────────────────────────────
    total_timesteps: int = 10_000_000
    n_envs: int = 32
    n_steps: int = 1024
    n_epochs: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    device: str = "auto"
    net_arch: dict = field(default_factory=lambda: {"pi": [128, 128], "vf": [128, 128]})
    activation_fn: type = nn.Tanh

    # ── training logistics ────────────────────────────────────────────
    log_dir: str = "runs"
    log_name: str = "ppo"
    save_video: bool = True
    save_video_every: int = 500_000
    eval_every: int = 100_000
    checkpoint_every: int = 500_000
    progress_bar: bool = True
    sb3_verbose: int = 0
    log_new_best_only: bool = False
    enable_reach_eval_metrics: bool = False

    # ── VecNormalize settings ─────────────────────────────────────────
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0

    # ── extra callbacks (e.g. curriculum) ─────────────────────────────
    extra_callbacks: list[BaseCallback] = field(default_factory=list)

    # ── computed ──────────────────────────────────────────────────────
    @property
    def batch_size(self) -> int:
        n_minibatches = 4
        return (self.n_steps * self.n_envs) // n_minibatches


def train_ppo(cfg: PPOTrainConfig) -> PPO:
    """Run PPO training from a fully-specified :class:`PPOTrainConfig`.

    Returns the trained ``PPO`` model.
    """
    if cfg.env_factory is None:
        raise ValueError("PPOTrainConfig.env_factory must be set")

    # ── vectorised training env ───────────────────────────────────────
    vec_env = SubprocVecEnv([cfg.env_factory(i) for i in range(cfg.n_envs)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=cfg.norm_obs,
        norm_reward=cfg.norm_reward,
        clip_obs=cfg.clip_obs,
    )

    # ── callbacks ─────────────────────────────────────────────────────
    callbacks: list[BaseCallback] = list(cfg.extra_callbacks)

    # Periodic model checkpoints
    if cfg.checkpoint_every > 0:
        ckpt_dir = os.path.join(cfg.log_dir, cfg.log_name + "_checkpoints", cfg.env_name)
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, cfg.checkpoint_every // cfg.n_envs),
                save_path=ckpt_dir,
                name_prefix="ppo",
                save_vecnormalize=True,
                verbose=1 if cfg.sb3_verbose > 0 else 0,
            )
        )

    # Video recording
    if cfg.save_video and cfg.eval_env_factory is not None:
        callbacks.append(
            BestEpisodeVideoCallback(
                make_eval_env=cfg.eval_env_factory,
                save_every_timesteps=cfg.save_video_every,
                video_dir="videos",
                env_name=cfg.env_name,
                deterministic=True,
                vec_norm=vec_env,
                verbose=1,
                log_new_best_only=cfg.log_new_best_only,
            )
        )
    if cfg.enable_reach_eval_metrics and cfg.eval_every > 0 and cfg.eval_env_factory is not None:
        callbacks.append(
            ReachEvalMetricsCallback(
                make_eval_env=cfg.eval_env_factory,
                eval_every_timesteps=cfg.eval_every,
                n_eval_episodes=3,
                deterministic=True,
                vec_norm=vec_env,
                verbose=1 if cfg.sb3_verbose > 0 else 0,
            )
        )

    # ── PPO model ─────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        clip_range=cfg.clip_range,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
        policy_kwargs=dict(
            net_arch=cfg.net_arch,
            activation_fn=cfg.activation_fn,
        ),
        verbose=cfg.sb3_verbose,
        tensorboard_log=cfg.log_dir,
    )

    # ── learn ─────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks if callbacks else None,
        tb_log_name=cfg.log_name,
        progress_bar=cfg.progress_bar,
    )

    # ── save ──────────────────────────────────────────────────────────
    model_path = f"ppo_{cfg.env_name}"
    model.save(model_path)
    vec_norm_path = f"{model_path}_vecnorm.pkl"
    vec_env.save(vec_norm_path)
    print(f"Model saved to {model_path}.zip")
    return model


def add_common_cli_args(parser, *, default_timesteps: int = 10_000_000, default_n_envs: int = 32) -> None:
    """Add the standard training CLI arguments to *parser*."""
    parser.add_argument("--total-timesteps", type=int, default=default_timesteps)
    parser.add_argument("--n-envs", type=int, default=default_n_envs)
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument("--save-video-every", type=int, default=500_000)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100_000,
        help="Run deterministic eval metrics every N training timesteps (0 to disable).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=500_000,
                        help="Save a checkpoint every N training timesteps (0 to disable).")
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        default=True,
        help="Use Stable-Baselines3 tqdm/rich progress bar.",
    )
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.add_argument(
        "--sb3-verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity (0 recommended with progress bar).",
    )
    parser.add_argument(
        "--log-new-best-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, video callback prints only when eval return reaches a new best.",
    )
    parser.add_argument(
        "--callback-new-best-only",
        action="store_true",
        dest="log_new_best_only",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-callback-new-best-only",
        dest="log_new_best_only",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device (default: auto).")


def config_from_cli(args, **overrides) -> dict:
    """Extract CLI-relevant PPO overrides from parsed args.

    Returns a plain dict suitable for unpacking into a task-specific
    training function (e.g. ``train_reach_ppo(**config_from_cli(args))``).
    Only includes fields that the CLI controls — task-specific fields
    like ``env_factory`` and ``env_name`` are left for the caller.
    """
    kw: dict = dict(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        progress_bar=args.progress_bar,
        sb3_verbose=args.sb3_verbose,
        log_new_best_only=args.log_new_best_only,
        device=args.device,
    )
    kw.update(overrides)
    return kw
