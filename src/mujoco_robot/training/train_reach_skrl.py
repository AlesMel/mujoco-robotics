"""Train reach task with SKRL PPO using Isaac Lab reach-style parameters.

Usage::

    python -m mujoco_robot.training.train_reach_skrl --robot ur3e --total-timesteps 24000

This script keeps the environment side from :mod:`mujoco_robot` and swaps the
RL backend to SKRL. The PPO hyperparameters mirror the Isaac Lab reach SKRL
configuration pattern (rollouts, epochs, minibatches, KL scheduler, etc.).
"""
from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym

from mujoco_robot.envs.reach import REACH_VARIANTS
from mujoco_robot.envs.reach_env import ReachGymnasium


def _resolve_control_variant(
    control_variant: str | None,
    action_mode: str | None,
) -> str:
    """Resolve backward-compatible control aliases."""
    if action_mode is not None:
        alias = {"cartesian": "ik_rel", "joint": "joint_pos"}
        if action_mode not in alias:
            raise ValueError(
                f"action_mode must be one of {tuple(alias)}, got '{action_mode}'"
            )
        mapped = alias[action_mode]
        if control_variant is not None and control_variant != mapped:
            raise ValueError(
                "Conflicting inputs: action_mode implies "
                f"'{mapped}' but control_variant is '{control_variant}'."
            )
        return mapped

    resolved = control_variant or "joint_pos"
    if resolved not in REACH_VARIANTS:
        raise ValueError(
            f"Unknown control_variant '{resolved}'. "
            f"Available: {sorted(REACH_VARIANTS.keys())}"
        )
    return resolved


def _require_skrl() -> tuple[Any, Any]:
    """Import SKRL lazily so the package remains usable without it."""
    try:
        from skrl.envs.wrappers.torch import wrap_env
        from skrl.utils.runner.torch import Runner
    except ImportError as exc:
        raise ImportError(
            "SKRL is required for this trainer. "
            "Install it with `pip install skrl`."
        ) from exc
    return wrap_env, Runner


def _build_isaaclab_reach_skrl_cfg(
    *,
    total_timesteps: int,
    seed: int,
    log_dir: str,
    experiment_name: str,
    device: str,
) -> dict[str, Any]:
    """Build SKRL Runner config aligned with Isaac Lab reach PPO defaults."""
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
                        "layers": [64, 64],
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
                        "layers": [64, 64],
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
            "rollouts": 24,
            "learning_epochs": 5,
            "mini_batches": 6,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "learning_rate": 5.0e-4,
            "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
            "random_timesteps": 0,
            "learning_starts": 0,
            "grad_norm_clip": 1.0,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.0,
            "value_loss_scale": 2.0,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            "time_limit_bootstrap": False,
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


def train_reach_skrl_ppo(
    robot: str = "ur3e",
    total_timesteps: int = 24_000,
    n_envs: int = 16,
    log_dir: str = "runs_skrl",
    experiment_name: str = "reach_skrl_ppo",
    control_variant: str | None = None,
    action_mode: str | None = None,
    reach_threshold: float = 0.03,
    ori_threshold: float = 0.25,
    success_hold_steps: int = 10,
    success_bonus: float = 0.25,
    stay_reward_weight: float = 0.05,
    resample_on_success: bool = False,
    seed: int = 42,
    device: str = "cuda:0",
) -> Any:
    """Train reach task with SKRL PPO using Isaac Lab-style PPO settings."""
    wrap_env, Runner = _require_skrl()
    control_variant = _resolve_control_variant(control_variant, action_mode)

    env_kwargs = dict(
        control_variant=control_variant,
        reach_threshold=reach_threshold,
        ori_threshold=ori_threshold,
        success_hold_steps=success_hold_steps,
        success_bonus=success_bonus,
        stay_reward_weight=stay_reward_weight,
        resample_on_success=resample_on_success,
    )

    print(f"\n{'='*60}")
    print("  Reach training config (SKRL PPO)")
    print(f"  Robot:            {robot}")
    print(f"  Control variant:  {control_variant}")
    print(f"  Reach threshold:  {reach_threshold:.3f} m")
    print(f"  Ori threshold:    {ori_threshold:.2f} rad")
    print(f"  Hold steps:       {success_hold_steps}")
    print(f"  Success bonus:    {success_bonus:.3f}")
    print(f"  Stay reward w:    {stay_reward_weight:.3f} /s")
    print(f"  Resample success: {resample_on_success}")
    print(f"  Num envs:         {n_envs}")
    print(f"  Timesteps:        {total_timesteps:,}")
    print(f"  Device:           {device}")
    print(f"{'='*60}\n")

    def make_env(rank: int):
        return lambda: ReachGymnasium(robot=robot, seed=seed + rank, **env_kwargs)

    base_env = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_envs)])
    env = wrap_env(base_env, wrapper="gymnasium")

    cfg = _build_isaaclab_reach_skrl_cfg(
        total_timesteps=total_timesteps,
        seed=seed,
        log_dir=log_dir,
        experiment_name=experiment_name,
        device=device,
    )

    runner = Runner(env, cfg)
    try:
        runner.run(mode="train")
    finally:
        base_env.close()
    return runner


def main() -> None:
    from mujoco_robot.robots.configs import ROBOT_CONFIGS

    p = argparse.ArgumentParser(description="Train SKRL PPO on reach task.")
    p.add_argument("--robot", type=str, default="ur3e", choices=list(ROBOT_CONFIGS))
    p.add_argument("--total-timesteps", type=int, default=24_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--log-dir", type=str, default="runs_skrl")
    p.add_argument("--experiment-name", type=str, default="reach_skrl_ppo")
    p.add_argument(
        "--control-variant",
        type=str,
        default=None,
        choices=sorted(REACH_VARIANTS.keys()),
    )
    p.add_argument(
        "--action-mode",
        type=str,
        default=None,
        choices=["cartesian", "joint"],
        help="Deprecated alias. cartesian->ik_rel, joint->joint_pos.",
    )
    p.add_argument("--reach-threshold", type=float, default=0.03)
    p.add_argument("--ori-threshold", type=float, default=0.25)
    p.add_argument("--success-hold-steps", type=int, default=10)
    p.add_argument("--success-bonus", type=float, default=0.25)
    p.add_argument("--stay-reward-weight", type=float, default=0.05)
    p.add_argument("--resample-on-success", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    train_reach_skrl_ppo(
        robot=args.robot,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        control_variant=args.control_variant,
        action_mode=args.action_mode,
        reach_threshold=args.reach_threshold,
        ori_threshold=args.ori_threshold,
        success_hold_steps=args.success_hold_steps,
        success_bonus=args.success_bonus,
        stay_reward_weight=args.stay_reward_weight,
        resample_on_success=args.resample_on_success,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
