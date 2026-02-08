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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from mujoco_robot.envs.reach_env import ReachGymnasium
from mujoco_robot.training.callbacks import BestEpisodeVideoCallback


def train_reach_ppo(
    robot: str = "ur3e",
    total_timesteps: int = 10_000_000,
    n_envs: int = 16,
    log_dir: str = "runs",
    log_name: str = "reach_ppo",
    save_video: bool = True,
    save_video_every: int = 50_000,
    action_mode: str = "joint",
    reach_threshold: float = 0.05,
    ori_threshold: float = 0.35,
    hold_seconds: float = 2.0,
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
    action_mode : str
        ``"cartesian"`` (6-D IK) or ``"joint"`` (6-D joint offsets).
    reach_threshold : float
        Position distance (m) within which the EE counts as "at goal".
    ori_threshold : float
        Orientation error (rad) within which orientation counts as matched.
    hold_seconds : float
        Seconds the EE must stay within both thresholds before the
        goal is resampled.  Set to 0 for instant resample on reach.

    Returns
    -------
    PPO
        Trained model.
    """
    env_kwargs = dict(
        reach_threshold=reach_threshold,
        ori_threshold=ori_threshold,
        hold_seconds=hold_seconds,
    )

    print(f"\n{'='*50}")
    print(f"  Reach training config")
    print(f"  Robot:            {robot}")
    print(f"  Action mode:      {action_mode}")
    print(f"  Reach threshold:  {reach_threshold:.3f} m")
    print(f"  Ori threshold:    {ori_threshold:.2f} rad")
    print(f"  Hold time:        {hold_seconds:.1f} s")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"{'='*50}\n")

    def make_env(rank):
        def _init():
            return Monitor(ReachGymnasium(
                robot=robot, seed=rank, action_mode=action_mode,
                **env_kwargs,
            ))
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    env_name = f"reach_{robot}"
    callbacks = []
    if save_video:
        def make_eval_env():
            return Monitor(ReachGymnasium(
                robot=robot, render_mode="rgb_array", action_mode=action_mode,
                **env_kwargs,
            ))

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

    # PPO config aligned with Isaac Lab reach task:
    #   - [64, 64] network (sufficient for ~30-D obs)
    #   - batch_size = n_steps * n_envs / 4  (4 minibatches)
    #   - LR 1e-3 matches Isaac Lab UR10 reach
    #   - entropy 0.01 encourages exploration early on
    n_steps = 2048
    n_minibatches = 4
    batch_size = (n_steps * n_envs) // n_minibatches

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=8,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=1.0,
        max_grad_norm=1.0,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
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
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--action-mode", type=str, default="joint",
                    choices=["cartesian", "joint"],
                    help="Action mode: 'cartesian' (6-D IK) or 'joint' (6-D offsets)")
    p.add_argument("--reach-threshold", type=float, default=0.05,
                    help="Position tolerance for goal reached (metres, default: 0.05)")
    p.add_argument("--ori-threshold", type=float, default=0.35,
                    help="Orientation tolerance for goal reached (radians, default: 0.35)")
    p.add_argument("--hold-seconds", type=float, default=2.0,
                    help="Seconds to hold at goal before resample (default: 2.0)")
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video-every", type=int, default=50_000)
    args = p.parse_args()

    train_reach_ppo(
        robot=args.robot,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
        action_mode=args.action_mode,
        reach_threshold=args.reach_threshold,
        ori_threshold=args.ori_threshold,
        hold_seconds=args.hold_seconds,
    )


if __name__ == "__main__":
    main()
