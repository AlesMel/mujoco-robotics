"""Train PPO on the URSlotSorterEnv.

Usage::

    python -m mujoco_robot.training.train_slot_sorter --total-timesteps 1000000

Or from Python::

    from mujoco_robot.training.train_slot_sorter import train_slot_sorter_ppo
    model = train_slot_sorter_ppo(total_timesteps=1_000_000)
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_robot.envs.slot_sorter_env import SlotSorterGymnasium
from mujoco_robot.training.callbacks import BestEpisodeVideoCallback


def train_slot_sorter_ppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    log_dir: str = "runs",
    log_name: str = "slot_sorter_ppo",
    learning_rate: float = 3e-4,
    save_video: bool = True,
    save_video_every: int = 50_000,
):
    """Quick-start PPO training on the slot-sorter task.

    Parameters
    ----------
    total_timesteps : int
        Total environment steps.
    n_envs : int
        Number of parallel environments.
    log_dir : str
        TensorBoard log directory.
    log_name : str
        TensorBoard run name.
    learning_rate : float
        PPO learning rate.
    save_video : bool
        Whether to record periodic eval videos.
    save_video_every : int
        Training timesteps between video recordings.

    Returns
    -------
    PPO
        Trained model.
    """
    def make_env(rank: int):
        def _init():
            return Monitor(SlotSorterGymnasium(seed=rank))
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    callbacks = []
    if save_video:
        def make_eval_env():
            return Monitor(SlotSorterGymnasium(render=True))

        video_cb = BestEpisodeVideoCallback(
            make_eval_env=make_eval_env,
            save_every_timesteps=save_video_every,
            video_dir="videos",
            env_name="slot_sorter",
            deterministic=True,
            vec_norm=vec_env,
            verbose=1,
        )
        callbacks.append(video_cb)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=1024,
        batch_size=256,
        learning_rate=learning_rate,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        tb_log_name=log_name,
    )
    model.save("ppo_slot_sorter")
    vec_env.save("ppo_slot_sorter_vecnorm.pkl")
    print("Model saved to ppo_slot_sorter.zip")
    return model


def main():
    p = argparse.ArgumentParser(
        description="Train PPO on URSlotSorter with video logging."
    )
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--save-video-every", type=int, default=50_000)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tensorboard-log", type=str, default="runs")
    p.add_argument("--log-name", type=str, default="slot_sorter_ppo")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    args = p.parse_args()

    train_slot_sorter_ppo(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        log_dir=args.tensorboard_log,
        log_name=args.log_name,
        learning_rate=args.learning_rate,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
    )


if __name__ == "__main__":
    main()
