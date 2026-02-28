"""Train PPO on the lift-suction task.

Usage::

    python -m mujoco_robot.training.train_suction --cfg-name ur3e_lift_suction_dense_stable
"""
from __future__ import annotations

import argparse

from stable_baselines3.common.monitor import Monitor

from mujoco_robot.tasks import (
    get_lift_suction_cfg,
    list_lift_suction_cfgs,
    make_lift_suction_contact_gymnasium,
    make_lift_suction_gymnasium,
)
from mujoco_robot.training.ppo_config import (
    PPOTrainConfig,
    add_common_cli_args,
    config_from_cli,
    train_ppo,
)


DEFAULT_CFG_NAME = "ur3e_lift_suction_dense_stable"


def train_suction_ppo(
    cfg_name: str = DEFAULT_CFG_NAME,
    **overrides,
):
    """Quick-start PPO training on the lift-suction task."""
    profile_name = cfg_name
    is_contact_stage = "suction_contact" in profile_name

    def build_cfg(seed: int | None, render_mode: str | None):
        cfg = get_lift_suction_cfg(profile_name)
        cfg.seed = seed
        cfg.render_mode = render_mode
        return cfg

    def make_gym(cfg):
        if is_contact_stage:
            return make_lift_suction_contact_gymnasium(cfg)
        return make_lift_suction_gymnasium(cfg)

    preview_cfg = build_cfg(seed=0, render_mode=None)
    print(f"\n{'='*50}")
    print("  Lift-suction training config")
    print(f"  Config profile:   {profile_name}")
    print(f"  Task stage:       {'suction_contact' if is_contact_stage else 'lift_suction'}")
    print(f"  Robot profile:    {preview_cfg.actuator_profile}")
    print(f"  Time limit:       {preview_cfg.time_limit}")
    print(f"{'='*50}\n")

    env_name = f"lift_suction_{preview_cfg.actuator_profile}_{cfg_name}".replace("/", "_")

    def make_env(rank: int):
        def _init():
            cfg = build_cfg(seed=rank, render_mode=None)
            return Monitor(make_gym(cfg))
        return _init

    def make_eval_env():
        cfg = build_cfg(seed=None, render_mode="rgb_array")
        return Monitor(make_gym(cfg))

    train_cfg = PPOTrainConfig(
        env_factory=make_env,
        eval_env_factory=make_eval_env,
        env_name=env_name,
        log_name="lift_suction_ppo",
        total_timesteps=overrides.pop("total_timesteps", 30_000_000),
        **overrides,
    )
    return train_ppo(train_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on lift-suction task.")
    parser.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    parser.add_argument("--list-cfgs", action="store_true")
    add_common_cli_args(parser, default_timesteps=20_000_000, default_n_envs=32)
    args = parser.parse_args()

    if args.list_cfgs:
        print("\n".join(list_lift_suction_cfgs()))
        return

    train_suction_ppo(
        cfg_name=args.cfg_name,
        **config_from_cli(args),
    )


if __name__ == "__main__":
    main()
