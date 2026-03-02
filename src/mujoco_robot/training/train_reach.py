"""Train PPO on the manager-based reach task.

Usage::

    python -m mujoco_robot.training.train_reach --cfg-name ur5e_joint_pos --total-timesteps 500000
    python -m mujoco_robot.training.train_reach --cfg-name ur5e_joint_pos

Or from Python::

    from mujoco_robot.training.train_reach import train_reach_ppo
    model = train_reach_ppo(cfg_name="ur5e_joint_pos", total_timesteps=500_000)

    # Override PPO hyper-parameters per-env:
    from mujoco_robot.training.ppo_cfg import TrainCfg, PPOCfg
    cfg = TrainCfg(ppo=PPOCfg(learning_rate=1e-4), total_timesteps=5_000_000)
    model = train_reach_ppo(train_cfg=cfg)
"""
from __future__ import annotations

import argparse

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from mujoco_robot.tasks.reach import (
    get_reach_cfg,
    make_reach_manager_based_gymnasium,
)
from mujoco_robot.training.callbacks import BestEpisodeVideoCallback
from mujoco_robot.training.ppo_cfg import TrainCfg, build_sb3_ppo


DEFAULT_CFG_NAME = "ur3e_joint_pos_dense_stable"

# Default training profile for the reach task.
REACH_TRAIN_CFG = TrainCfg(
    total_timesteps=10_000_000,
    n_envs=32,
    log_name="reach_ppo",
    save_video_every=500_000,
)


def train_reach_ppo(
    cfg_name: str = DEFAULT_CFG_NAME,
    train_cfg: TrainCfg | None = None,
):
    """Quick-start PPO training on the manager-based reach task.

    Parameters
    ----------
    cfg_name:
        Registered reach-env config profile name.
    train_cfg:
        Full training config.  ``None`` uses :data:`REACH_TRAIN_CFG`.
    """
    tc = train_cfg or REACH_TRAIN_CFG
    profile_name = cfg_name

    def build_cfg(seed: int | None, render_mode: str | None):
        cfg = get_reach_cfg(profile_name)
        cfg.scene.render_mode = render_mode
        cfg.episode.seed = seed
        return cfg

    preview_cfg = build_cfg(seed=0, render_mode=None)

    print(f"\n{'='*50}")
    print("  Reach training config")
    print(f"  Config profile:   {profile_name}")
    print(f"  Robot:            {preview_cfg.scene.robot}")
    print(f"  Control variant:  {preview_cfg.actions.control_variant}")
    print(f"  Joint act scale:  {preview_cfg.actions.joint_action_scale:.3f} (IsaacLab scale)")
    print(f"  Reach threshold:  {preview_cfg.success.reach_threshold:.3f} m")
    print(f"  Ori threshold:    {preview_cfg.success.ori_threshold:.2f} rad")
    print(f"  Hold steps:       {preview_cfg.success.success_hold_steps}")
    print(f"  Success bonus:    {preview_cfg.success.success_bonus:.3f}")
    print(f"  Stay reward w:    {preview_cfg.success.stay_reward_weight:.3f} /s")
    print(f"  Resample success: {preview_cfg.success.resample_on_success}")
    print(f"  Train obs noise:  {preview_cfg.physics.obs_noise:.4f}")
    print("  Video obs noise:  0.0000")
    print(f"  PPO lr:           {tc.ppo.learning_rate}")
    print(f"  PPO ent_coef:     {tc.ppo.ent_coef}")
    print(f"  PPO n_steps:      {tc.ppo.n_steps}")
    print(f"  PPO activation:   {tc.ppo.activation}")
    print(f"  n_envs:           {tc.n_envs}")
    print(f"  Progress bar:     {tc.progress_bar}")
    if preview_cfg.actions.control_variant == "joint_pos_isaac_reward":
        print("  Episode setup:    built-in defaults (12s, no in-episode goal resample)")
    print(f"  Total timesteps:  {tc.total_timesteps:,}")
    print(f"{'='*50}\n")

    def make_env(rank):
        def _init():
            cfg = build_cfg(seed=rank, render_mode=None)
            return Monitor(make_reach_manager_based_gymnasium(cfg))

        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(tc.n_envs)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=tc.normalize.norm_obs,
        norm_reward=tc.normalize.norm_reward,
        clip_obs=tc.normalize.clip_obs,
    )

    env_name = f"reach_{preview_cfg.scene.robot}_{cfg_name}".replace("/", "_")
    callbacks = []
    if tc.save_video:

        def make_eval_env():
            # Use a stochastic eval seed so video episodes do not always start
            # from the exact same sampled goal/state.
            cfg = build_cfg(seed=None, render_mode="rgb_array")
            # Keep policy-evaluation videos noise-free for stable diagnostics.
            cfg.physics.obs_noise = 0.0
            return Monitor(make_reach_manager_based_gymnasium(cfg))

        video_cb = BestEpisodeVideoCallback(
            make_eval_env=make_eval_env,
            save_every_timesteps=tc.save_video_every,
            video_dir=tc.video_dir,
            env_name=env_name,
            deterministic=True,
            vec_norm=vec_env,
            verbose=1,
            log_new_best_only=tc.callback_new_best_only,
        )
        callbacks.append(video_cb)

    model = build_sb3_ppo(tc, vec_env)

    model.learn(
        total_timesteps=tc.total_timesteps,
        callback=callbacks if callbacks else None,
        tb_log_name=tc.log_name,
        progress_bar=tc.progress_bar,
    )

    model_path = f"ppo_{env_name}"
    model.save(model_path)
    vec_norm_path = f"{model_path}_vecnorm.pkl"
    vec_env.save(vec_norm_path)
    print(f"Model saved to {model_path}.zip")
    return model


def main():
    p = argparse.ArgumentParser(description="Train PPO on manager-based reach.")
    p.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    p.add_argument("--total-timesteps", type=int, default=REACH_TRAIN_CFG.total_timesteps)
    p.add_argument("--n-envs", type=int, default=REACH_TRAIN_CFG.n_envs)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-video-every", type=int, default=50_000)
    p.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Stable-Baselines3 tqdm/rich progress bar.",
    )
    p.add_argument(
        "--sb3-verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity (0 recommended with progress bar).",
    )
    p.add_argument(
        "--callback-new-best-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, callback prints only when eval return reaches a new best.",
    )
    p.add_argument("--lr", type=float, default=None, help="Override PPO learning rate.")
    p.add_argument("--ent-coef", type=float, default=None, help="Override entropy coef.")
    args = p.parse_args()

    # Build TrainCfg from CLI args, overlaying onto REACH_TRAIN_CFG defaults.
    from dataclasses import replace

    from mujoco_robot.training.ppo_cfg import PPOCfg

    ppo_overrides = {}
    if args.lr is not None:
        ppo_overrides["learning_rate"] = args.lr
    if args.ent_coef is not None:
        ppo_overrides["ent_coef"] = args.ent_coef

    ppo = replace(REACH_TRAIN_CFG.ppo, **ppo_overrides) if ppo_overrides else REACH_TRAIN_CFG.ppo

    tc = replace(
        REACH_TRAIN_CFG,
        ppo=ppo,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
        progress_bar=args.progress_bar,
        sb3_verbose=args.sb3_verbose,
        callback_new_best_only=args.callback_new_best_only,
    )
    train_reach_ppo(cfg_name=args.cfg_name, train_cfg=tc)


if __name__ == "__main__":
    main()
