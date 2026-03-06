"""Train PPO on the Crazyflie wall-maze obstacle-avoidance task using SKRL.

Usage::

    python -m mujoco_robot.training.train_crazyflie_obstacle_skrl \
        --cfg-name crazyflie_obstacle_skrl_stable \
        --total-timesteps 10000000 \
        --n-envs 16 \
        --device cuda:0

    # Record an evaluation video every 100 000 training timesteps:
    python -m mujoco_robot.training.train_crazyflie_obstacle_skrl \
        --video-every 100000 --video-dir videos

This script follows the same pattern as :mod:`mujoco_robot.training.train_reach_skrl`
but is tuned for the higher-dimensional obstacle task (45-dim obs: 27 state + 18 lidar).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from mujoco_robot.tasks import (
    get_crazyflie_cfg,
    list_crazyflie_cfgs,
    make_crazyflie_obstacle_gymnasium,
)


DEFAULT_CFG_NAME = "crazyflie_obstacle_skrl_stable"


def _require_skrl() -> tuple[Any, Any]:
    """Import SKRL lazily so the package remains usable without it."""
    try:
        from skrl.envs.wrappers.torch import wrap_env
        from skrl.utils.runner.torch import Runner
    except ImportError as exc:
        raise ImportError(
            "SKRL is required for this trainer. "
            "Install it with `pip install skrl` or "
            "`pip install mujoco-robot[train-skrl]`."
        ) from exc
    return wrap_env, Runner


# ------------------------------------------------------------------
#  Eval-video helper
# ------------------------------------------------------------------

def _record_eval_video(
    agent: Any,
    cfg_name: str,
    seed: int,
    video_dir: Path,
    timestep: int,
    device: str = "cuda:0",
) -> None:
    """Run one deterministic eval episode and save an MP4.

    A fresh single environment is created with ``render_mode='rgb_array'``
    so the training envs are not affected.
    """
    import torch

    try:
        import imageio.v3 as iio
    except ImportError:
        print("[video] imageio not installed – skipping video recording")
        return

    # Build a single rendering eval env
    eval_cfg = get_crazyflie_cfg(cfg_name)
    eval_cfg.seed = seed + 9999
    eval_cfg.render = True
    eval_cfg.render_mode = "rgb_array"
    eval_env = make_crazyflie_obstacle_gymnasium(eval_cfg)

    obs, _ = eval_env.reset()
    frames: list[np.ndarray] = []
    ep_return = 0.0
    done = False

    while not done:
        # SKRL PPO.act() takes a raw tensor and wraps it internally as
        # {"states": preprocessor(obs)}, so we must NOT pass a dict here.
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t = agent.act(obs_t, timestep=0, timesteps=0)[0]
        action = action_t.cpu().numpy().squeeze(0)

        obs, reward, terminated, truncated, _info = eval_env.step(action)
        ep_return += float(reward)
        done = terminated or truncated

        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)

    eval_env.close()

    if not frames:
        return

    # Derive FPS from env control period
    try:
        base = eval_env.base
        control_dt = float(base.model.opt.timestep) * float(base.n_substeps)
    except Exception:
        control_dt = 0.01
    fps = max(1, int(round(1.0 / control_dt)))

    # Duplicate single-frame videos so codecs don't choke
    if len(frames) == 1:
        frames.append(frames[0].copy())

    video_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    fname = video_dir / f"eval_step_{timestep:010d}_{ts}.mp4"
    try:
        iio.imwrite(
            fname,
            frames,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_log_level="error",
        )
    except TypeError:
        iio.imwrite(fname, frames, fps=fps)

    print(f"[video] step {timestep:,} | return {ep_return:.3f} | "
          f"fps={fps} | {fname}")


def _build_obstacle_skrl_cfg(
    *,
    total_timesteps: int,
    seed: int,
    log_dir: str,
    experiment_name: str,
    device: str,
) -> dict[str, Any]:
    """Build SKRL Runner config for the Crazyflie obstacle-avoidance PPO task.

    Network size [256, 256, 128] matches the larger observation space (45 dims
    for default lidar config) compared with the reach task (27 dims).
    KL-adaptive LR scheduler replaces the manual linear decay used in the SB3
    version.
    """
    return {
        "seed": seed,
        "models": {
            "separate": True,
            "policy": {
                "class": "GaussianMixin",
                "clip_actions": False,
                "clip_log_std": True,
                "min_log_std": -20.0,
                "max_log_std": 2.0,
                "initial_log_std": 0.0,
                "network": [
                    {
                        "name": "net",
                        "input": "STATES",
                        "layers": [256, 256, 128],
                        "activations": "elu",
                    }
                ],
                "output": "ACTIONS",
            },
            "value": {
                "class": "DeterministicMixin",
                "clip_actions": False,
                "network": [
                    {
                        "name": "net",
                        "input": "STATES",
                        "layers": [256, 256, 128],
                        "activations": "elu",
                    }
                ],
                "output": "ONE",
            },
        },
        "memory": {
            "class": "RandomMemory",
            "memory_size": -1,
        },
        "agent": {
            "class": "PPO",
            "rollouts": 2048,
            "learning_epochs": 10,
            "mini_batches": 4,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "learning_rate": 3.0e-4,
            "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
            "random_timesteps": 0,
            "learning_starts": 0,
            "grad_norm_clip": 0.5,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.005,
            "value_loss_scale": 0.5,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            # Bootstrap value at truncation so timeout episodes do not bias
            # the value estimate downward.
            "time_limit_bootstrap": True,
            "state_preprocessor": "RunningStandardScaler",
            "state_preprocessor_kwargs": {"size": "STATES", "device": device},
            "value_preprocessor": "RunningStandardScaler",
            "value_preprocessor_kwargs": {"size": 1, "device": device},
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": int(total_timesteps),
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
        "experiment": {
            "directory": log_dir,
            "experiment_name": experiment_name,
            "write_interval": "auto",
            "checkpoint_interval": "auto",
        },
    }


def train_crazyflie_obstacle_skrl_ppo(
    cfg_name: str = DEFAULT_CFG_NAME,
    total_timesteps: int = 10_000_000,
    n_envs: int = 16,
    log_dir: str = "runs_skrl",
    experiment_name: str = "crazyflie_obstacle_skrl_ppo",
    seed: int = 42,
    device: str = "cuda:0",
    video_every: int = 0,
    video_dir: str = "videos",
) -> Any:
    """Train the Crazyflie obstacle task with SKRL PPO.

    Parameters
    ----------
    video_every : int
        Record an evaluation video every *video_every* training timesteps.
        Set to ``0`` (default) to disable video recording.
    video_dir : str
        Root directory for saved videos.
    """
    wrap_env, Runner = _require_skrl()

    def build_cfg(rank: int):
        cfg = get_crazyflie_cfg(cfg_name)
        cfg.seed = seed + rank
        return cfg

    preview_cfg = build_cfg(0)

    print(f"\n{'='*60}")
    print("  Crazyflie WALL-MAZE training config (SKRL PPO)")
    print(f"  Config profile:   {cfg_name}")
    print(f"  Time limit:       {preview_cfg.time_limit}")
    print(f"  Obs mode:         {preview_cfg.env_kwargs.get('observation_mode', 'state_estimate')}")
    print(f"  Rangefinder:      {preview_cfg.env_kwargs.get('rangefinder_mode', 'lidar')}")
    print(f"  Lidar rays:       {preview_cfg.env_kwargs.get('n_lidar_rays', 16)}")
    print(f"  Barriers:         {preview_cfg.env_kwargs.get('n_barriers', (2, 3))}")
    print(f"  Weights (P/O/E/S/T): "
          f"{preview_cfg.env_kwargs.get('w_progress', 3.0)}/"
          f"{preview_cfg.env_kwargs.get('w_obstacle', 1.5)}/"
          f"{preview_cfg.env_kwargs.get('w_energy', 0.2)}/"
          f"{preview_cfg.env_kwargs.get('w_stability', 0.5)}/"
          f"{preview_cfg.env_kwargs.get('w_task', 1.0)}")
    print(f"  Num envs:         {n_envs}")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"  Device:           {device}")
    print(f"{'='*60}\n")

    def make_env(rank: int):
        return lambda: make_crazyflie_obstacle_gymnasium(build_cfg(rank))

    # AsyncVectorEnv runs each env in its own subprocess (like SB3's
    # SubprocVecEnv), giving true parallelism on multi-core machines.
    # SyncVectorEnv steps envs serially in one process and would be
    # ~n_envs times slower for a compute-heavy environment like this one.
    base_env = gym.vector.AsyncVectorEnv([make_env(i) for i in range(n_envs)])
    env = wrap_env(base_env, wrapper="gymnasium")

    cfg = _build_obstacle_skrl_cfg(
        total_timesteps=total_timesteps,
        seed=seed,
        log_dir=log_dir,
        experiment_name=experiment_name,
        device=device,
    )

    runner = Runner(env, cfg)

    # ---- Patch agent to record videos during training ----
    if video_every > 0:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vid_dir = Path(video_dir) / f"crazyflie_obstacle_{cfg_name}" / run_stamp

        _orig_post = runner.agent.post_interaction  # bound method
        _next_video = [video_every]  # mutable cell for closure

        def _post_interaction_with_video(*, timestep: int, timesteps: int) -> None:
            _orig_post(timestep=timestep, timesteps=timesteps)
            if timestep + 1 >= _next_video[0]:
                print(f"\n[video] Recording eval video "
                      f"at timestep {timestep + 1:,} …")
                runner.agent.set_mode("eval")
                _record_eval_video(
                    agent=runner.agent,
                    cfg_name=cfg_name,
                    seed=seed,
                    video_dir=vid_dir,
                    timestep=timestep + 1,
                    device=device,
                )
                runner.agent.set_mode("train")
                _next_video[0] += video_every

        runner.agent.post_interaction = _post_interaction_with_video

    try:
        runner.run(mode="train")
    finally:
        base_env.close()
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SKRL PPO on Crazyflie obstacle-avoidance reach task."
    )
    parser.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    parser.add_argument("--list-cfgs", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--log-dir", type=str, default="runs_skrl")
    parser.add_argument("--experiment-name", type=str, default="crazyflie_obstacle_skrl_ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--video-every", type=int, default=0,
        help="Record an evaluation video every N training timesteps (0 = disabled).",
    )
    parser.add_argument(
        "--video-dir", type=str, default="videos",
        help="Root directory for saved evaluation videos.",
    )
    args = parser.parse_args()

    if args.list_cfgs:
        print("\n".join(list_crazyflie_cfgs()))
        return

    train_crazyflie_obstacle_skrl_ppo(
        cfg_name=args.cfg_name,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        device=args.device,
        video_every=args.video_every,
        video_dir=args.video_dir,
    )


if __name__ == "__main__":
    main()
