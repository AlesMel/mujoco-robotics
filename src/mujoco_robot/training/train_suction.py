"""Train PPO on the lift-suction task.

Usage::

    python -m mujoco_robot.training.train_suction --cfg-name ur3e_lift_suction_dense_stable
"""
from __future__ import annotations

import argparse

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from mujoco_robot.tasks import (
    get_lift_suction_cfg,
    list_lift_suction_cfgs,
    make_lift_suction_contact_gymnasium,
    make_lift_suction_gymnasium,
)
from mujoco_robot.training.callbacks import BestEpisodeVideoCallback
from mujoco_robot.training.ppo_cfg import TrainCfg, build_sb3_ppo


DEFAULT_CFG_NAME = "ur3e_lift_suction_dense_stable"

# Default training profile for the lift-suction task.
SUCTION_TRAIN_CFG = TrainCfg(
    total_timesteps=30_000_000,
    n_envs=32,
    log_name="lift_suction_ppo",
    save_video_every=1_000_000,
)


def train_suction_ppo(
    cfg_name: str = DEFAULT_CFG_NAME,
    train_cfg: TrainCfg | None = None,
):
    """Quick-start PPO training on the lift-suction task."""
    tc = train_cfg or SUCTION_TRAIN_CFG
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
    print(f"  PPO lr:           {tc.ppo.learning_rate}")
    print(f"  PPO ent_coef:     {tc.ppo.ent_coef}")
    print(f"  n_envs:           {tc.n_envs}")
    print(f"  Progress bar:     {tc.progress_bar}")
    print(f"  Total timesteps:  {tc.total_timesteps:,}")
    print(f"{'='*50}\n")

    def make_env(rank: int):
        def _init():
            cfg = build_cfg(seed=rank, render_mode=None)
            return Monitor(make_gym(cfg))

        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(tc.n_envs)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=tc.normalize.norm_obs,
        norm_reward=tc.normalize.norm_reward,
        clip_obs=tc.normalize.clip_obs,
    )

    env_name = f"lift_suction_{preview_cfg.actuator_profile}_{cfg_name}".replace("/", "_")
    callbacks = []
    if tc.save_video:

        def make_eval_env():
            cfg = build_cfg(seed=None, render_mode="rgb_array")
            return Monitor(make_gym(cfg))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on lift-suction task.")
    parser.add_argument("--cfg-name", type=str, default=DEFAULT_CFG_NAME)
    parser.add_argument("--list-cfgs", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=SUCTION_TRAIN_CFG.total_timesteps)
    parser.add_argument("--n-envs", type=int, default=SUCTION_TRAIN_CFG.n_envs)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-video-every", type=int, default=SUCTION_TRAIN_CFG.save_video_every)
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Stable-Baselines3 tqdm/rich progress bar.",
    )
    parser.add_argument(
        "--sb3-verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Stable-Baselines3 verbosity (0 recommended with progress bar).",
    )
    parser.add_argument(
        "--callback-new-best-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, callback prints only when eval return reaches a new best.",
    )
    args = parser.parse_args()

    if args.list_cfgs:
        print("\n".join(list_lift_suction_cfgs()))
        return

    from dataclasses import replace

    tc = replace(
        SUCTION_TRAIN_CFG,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        save_video=args.save_video,
        save_video_every=args.save_video_every,
        progress_bar=args.progress_bar,
        sb3_verbose=args.sb3_verbose,
        callback_new_best_only=args.callback_new_best_only,
    )
    train_suction_ppo(cfg_name=args.cfg_name, train_cfg=tc)


if __name__ == "__main__":
    main()
