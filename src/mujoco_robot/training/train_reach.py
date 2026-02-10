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

from mujoco_robot.envs.reach import REACH_VARIANTS
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
    control_variant: str = "joint_pos",
    action_mode: str | None = None,
    reach_threshold: float = 0.03,
    ori_threshold: float = 0.25,
    progress_bar: bool = True,
    sb3_verbose: int = 0,
    callback_new_best_only: bool = True,
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
    control_variant : str
        Reach control variant key from :data:`mujoco_robot.envs.reach.REACH_VARIANTS`.
    action_mode : str | None
        Backward-compatible alias. ``"cartesian"`` maps to ``"ik_rel"``,
        ``"joint"`` maps to ``"joint_pos"``.
    reach_threshold : float
        Position distance (m) within which the EE counts as "at goal".
    ori_threshold : float
        Orientation error (rad) within which orientation counts as matched.

    Returns
    -------
    PPO
        Trained model.
    """
    if action_mode is not None:
        alias = {"cartesian": "ik_rel", "joint": "joint_pos"}
        if action_mode not in alias:
            raise ValueError(
                f"action_mode must be one of {tuple(alias)}, got '{action_mode}'"
            )
        mapped_variant = alias[action_mode]
        if control_variant != "joint_pos" and control_variant != mapped_variant:
            raise ValueError(
                "Conflicting inputs: action_mode implies "
                f"'{mapped_variant}' but control_variant is '{control_variant}'."
            )
        control_variant = mapped_variant

    env_kwargs = dict(
        control_variant=control_variant,
        reach_threshold=reach_threshold,
        ori_threshold=ori_threshold,
    )

    print(f"\n{'='*50}")
    print(f"  Reach training config")
    print(f"  Robot:            {robot}")
    print(f"  Control variant:  {control_variant}")
    print(f"  Reach threshold:  {reach_threshold:.3f} m")
    print(f"  Ori threshold:    {ori_threshold:.2f} rad")
    print(f"  Progress bar:     {progress_bar}")
    if control_variant == "joint_pos_isaac_reward":
        print("  Episode setup:    built-in Isaac-style defaults (12s, 4s command)")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"{'='*50}\n")

    def make_env(rank):
        def _init():
            return Monitor(ReachGymnasium(
                robot=robot, seed=rank,
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
                robot=robot, render_mode="rgb_array",
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
            log_new_best_only=callback_new_best_only,
        )
        callbacks.append(video_cb)

    # PPO config aligned with Isaac Lab reach task:
    #   - [128, 128] network (larger net needed for full orientation
    #     control — 43-D obs with 6-D continuous rotations)
    #   - batch_size = n_steps * n_envs / 4  (4 minibatches)
    #   - LR 3e-4 (slightly lower than before for stable orientation learning)
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
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=1.0,
        max_grad_norm=1.0,
        device="cuda",
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=sb3_verbose,
        tensorboard_log=log_dir,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks if callbacks else None,
        tb_log_name=log_name,
        progress_bar=progress_bar,
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
    p.add_argument("--control-variant", type=str, default="joint_pos",
                    choices=sorted(REACH_VARIANTS.keys()),
                    help=(
                        "Control variant. Available: "
                        + ", ".join(sorted(REACH_VARIANTS.keys()))
                    ))
    p.add_argument("--action-mode", type=str, default=None,
                    choices=["cartesian", "joint"],
                    help="Deprecated alias. cartesian->ik_rel, joint->joint_pos.")
    p.add_argument("--reach-threshold", type=float, default=0.03,
                    help="Position tolerance for goal reached (metres, default: 0.03)")
    p.add_argument("--ori-threshold", type=float, default=0.25,
                    help="Orientation tolerance for goal reached (radians, default: 0.25)")
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video-every", type=int, default=50_000)
    p.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True,
                    help="Use Stable-Baselines3 tqdm/rich progress bar.")
    p.add_argument("--sb3-verbose", type=int, default=0, choices=[0, 1, 2],
                    help="Stable-Baselines3 verbosity (0 recommended with progress bar).")
    p.add_argument(
        "--callback-new-best-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, callback prints only when eval return reaches a new best.",
    )
    args = p.parse_args()

    train_reach_ppo(
        robot=args.robot,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
        control_variant=args.control_variant,
        action_mode=args.action_mode,
        reach_threshold=args.reach_threshold,
        ori_threshold=args.ori_threshold,
        progress_bar=args.progress_bar,
        sb3_verbose=args.sb3_verbose,
        callback_new_best_only=args.callback_new_best_only,
    )


if __name__ == "__main__":
    main()
