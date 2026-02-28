"""Train PPO on the URSlotSorterEnv.

Usage::

    python -m mujoco_robot.training.train_slot_sorter --total-timesteps 1000000

Or from Python::

    from mujoco_robot.training.train_slot_sorter import train_slot_sorter_ppo
    model = train_slot_sorter_ppo(total_timesteps=1_000_000)
"""
from __future__ import annotations

import argparse

from stable_baselines3.common.monitor import Monitor

from mujoco_robot.tasks.slot_sorter.slot_sorter_env import SlotSorterGymnasium
from mujoco_robot.training.ppo_config import (
    PPOTrainConfig,
    add_common_cli_args,
    config_from_cli,
    train_ppo,
)


def train_slot_sorter_ppo(**overrides):
    """Quick-start PPO training on the slot-sorter task.

    Returns
    -------
    PPO
        Trained model.
    """
    def make_env(rank: int):
        def _init():
            return Monitor(SlotSorterGymnasium(seed=rank))
        return _init

    def make_eval_env():
        return Monitor(SlotSorterGymnasium(render=True))

    train_cfg = PPOTrainConfig(
        env_factory=make_env,
        eval_env_factory=make_eval_env,
        env_name="slot_sorter",
        log_name="slot_sorter_ppo",
        total_timesteps=overrides.pop("total_timesteps", 1_000_000),
        n_envs=overrides.pop("n_envs", 8),
        **overrides,
    )
    return train_ppo(train_cfg)


def main():
    p = argparse.ArgumentParser(
        description="Train PPO on URSlotSorter with video logging."
    )
    add_common_cli_args(p, default_timesteps=1_000_000, default_n_envs=8)
    args = p.parse_args()

    train_slot_sorter_ppo(
        **config_from_cli(args),
    )


if __name__ == "__main__":
    main()
