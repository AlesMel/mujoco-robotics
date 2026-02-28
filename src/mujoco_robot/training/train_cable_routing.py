"""Train PPO on the cable-routing task.

Usage::

    python -m mujoco_robot.training.train_cable_routing \
        --cfg-name ur3e_cable_routing_dense_stable
"""
from __future__ import annotations

import argparse

from stable_baselines3.common.monitor import Monitor

from mujoco_robot.tasks import (
    get_cable_routing_cfg,
    list_cable_routing_cfgs,
    make_cable_routing_gymnasium,
)
from mujoco_robot.training.callbacks import CableRoutingCurriculumCallback
from mujoco_robot.training.ppo_config import (
    PPOTrainConfig,
    add_common_cli_args,
    config_from_cli,
    train_ppo,
)


DEFAULT_CFG_NAME = "ur3e_cable_grasp"


def train_cable_routing_ppo(
    cfg_name: str = DEFAULT_CFG_NAME,
    use_curriculum: bool = True,
    **overrides,
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

    total_timesteps = overrides.pop("total_timesteps", 1_500_000)

    print(f"\n{'='*50}")
    print("  Cable-routing training config")
    print(f"  Config profile:   {cfg_name}")
    print(f"  Task mode:        {task_mode}")
    print(f"  Robot profile:    {preview_cfg.actuator_profile}")
    print(f"  Time limit:       {preview_cfg.time_limit}")
    print(f"  Curriculum:       {use_curriculum}")
    print(f"{'='*50}\n")

    env_name = f"cable_routing_{preview_cfg.actuator_profile}_{cfg_name}".replace("/", "_")

    def make_env(rank: int):
        def _init():
            cfg = build_cfg(seed=rank, render_mode=None)
            return Monitor(make_cable_routing_gymnasium(cfg))
        return _init

    def make_eval_env():
        cfg = build_cfg(seed=None, render_mode="rgb_array")
        return Monitor(make_cable_routing_gymnasium(cfg))

    # Disable reward normalization for grasp mode -- the raw distance/velocity
    # reward is already well-scaled and normalization destroys the gradient.
    norm_reward = task_mode != "grasp"

    extra_cbs = []
    if use_curriculum:
        sb3_verbose = overrides.get("sb3_verbose", 0)
        extra_cbs.append(
            CableRoutingCurriculumCallback(
                total_timesteps=total_timesteps,
                stage1_frac=0.35,
                stage2_frac=0.70,
                verbose=1 if sb3_verbose > 0 else 0,
            )
        )

    train_cfg = PPOTrainConfig(
        env_factory=make_env,
        eval_env_factory=make_eval_env,
        env_name=env_name,
        log_name="cable_routing_ppo",
        total_timesteps=total_timesteps,
        norm_reward=norm_reward,
        extra_callbacks=extra_cbs,
        **overrides,
    )
    return train_ppo(train_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on cable-routing task.")
    parser.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    parser.add_argument("--list-cfgs", action="store_true")
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        default=True,
        help="Use a single-run staged curriculum (easy -> mid -> full).",
    )
    parser.add_argument("--no-use-curriculum", dest="use_curriculum", action="store_false")
    add_common_cli_args(parser, default_timesteps=1_500_000, default_n_envs=8)
    args = parser.parse_args()

    if args.list_cfgs:
        print("\n".join(list_cable_routing_cfgs()))
        return

    train_cable_routing_ppo(
        cfg_name=args.cfg_name,
        use_curriculum=args.use_curriculum,
        **config_from_cli(args),
    )


if __name__ == "__main__":
    main()
