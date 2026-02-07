"""Train PPO on the URReachEnv.

Usage::

    python -m mujoco_robot.training.train_reach --robot ur5e --total-timesteps 500000

Or from Python::

    from mujoco_robot.training.train_reach import train_reach_ppo
    model = train_reach_ppo(robot="ur5e", total_timesteps=500_000)
"""
from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_robot.envs.reach_env import ReachGymnasium
from mujoco_robot.training.callbacks import BestEpisodeVideoCallback


def train_reach_ppo(
    robot: str = "ur5e",
    total_timesteps: int = 500_000,
    n_envs: int = 8,
    log_dir: str = "runs",
    log_name: str = "reach_ppo",
    save_video: bool = True,
    save_video_every: int = 50_000,
):
    """Quick-start PPO training on the reach task.

    Parameters
    ----------
    robot : str
        Robot model (``"ur5e"`` or ``"ur3e"``).
    total_timesteps : int
        Total environment steps.
    n_envs : int
        Number of parallel environments.
    log_dir : str
        TensorBoard log directory.
    log_name : str
        TensorBoard run name.
    save_video : bool
        Whether to record periodic eval videos.
    save_video_every : int
        Training timesteps between video recordings.

    Returns
    -------
    PPO
        Trained model.
    """
    def make_env(rank):
        def _init():
            return Monitor(ReachGymnasium(robot=robot, seed=rank))
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    env_name = f"reach_{robot}"
    callbacks = []
    if save_video:
        def make_eval_env():
            return Monitor(ReachGymnasium(robot=robot, render=True))

        video_cb = BestEpisodeVideoCallback(
            make_eval_env=make_eval_env,
            save_every_timesteps=save_video_every,
            video_dir="videos",
            env_name=env_name,
            deterministic=True,
            vec_norm=vec_env,
            verbose=1,
        )
        callbacks.append(video_cb)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        tb_log_name=log_name,
    )
    model.save(f"ppo_reach_{robot}")
    vec_env.save(f"ppo_reach_{robot}_vecnorm.pkl")
    print(f"Model saved to ppo_reach_{robot}.zip")
    return model


def main():
    from mujoco_robot.robots.configs import ROBOT_CONFIGS

    p = argparse.ArgumentParser(description="Train PPO on URReachEnv.")
    p.add_argument("--robot", type=str, default="ur3e",
                    choices=list(ROBOT_CONFIGS.keys()))
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video-every", type=int, default=50_000)
    args = p.parse_args()

    train_reach_ppo(
        robot=args.robot,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
    )


if __name__ == "__main__":
    main()
