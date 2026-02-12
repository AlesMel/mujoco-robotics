#!/usr/bin/env python
"""Interactive evaluation for a trained SB3 reach policy.

Usage::

    python -m mujoco_robot.scripts.eval_reach --model ppo_reach_ur3e_ur3e_joint_pos_dense_stable.zip
    python -m mujoco_robot.scripts.eval_reach --model ppo_reach_ur3e_ur3e_joint_pos_dense_stable --robot ur3e
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_robot.tasks.manager_based.manipulation.reach import (
    get_reach_cfg,
    make_reach_manager_based_gymnasium,
)

_EVAL_NO_AUTO_GOAL_RESAMPLE_S = 1_000_000.0


@dataclass
class _EvalControls:
    paused: bool = False
    reset_requested: bool = False
    goal_requested: bool = False
    quit_requested: bool = False


def _resolve_model_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if path.suffix != ".zip":
        zip_path = path.with_suffix(".zip")
        if zip_path.exists():
            return zip_path
    raise FileNotFoundError(
        f"Model file not found: '{path_str}'. "
        "Pass either '<name>' or '<name>.zip'."
    )


def _default_vecnorm_path(model_path: Path) -> Path:
    if model_path.suffix == ".zip":
        stem = model_path.with_suffix("")
    else:
        stem = model_path
    return Path(f"{stem}_vecnorm.pkl")


def _load_vec_env(
    env,
    vecnorm_path: Path,
):
    vec_env = DummyVecEnv([lambda: env])
    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found: '{vecnorm_path}'. "
            "Evaluation must use the same normalization as training/video. "
            "Pass --vecnorm explicitly or ensure <model>_vecnorm.pkl exists."
        )
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print(f"[eval] Loaded VecNormalize stats: {vecnorm_path}")
    return vec_env


def _make_key_callback(ctrl: _EvalControls):
    glfw_press = int(getattr(mujoco, "GLFW_PRESS", 1))
    glfw_repeat = int(getattr(mujoco, "GLFW_REPEAT", 2))
    glfw_release = int(getattr(mujoco, "GLFW_RELEASE", 0))

    def _is_press(action: int | None) -> bool:
        if action is None:
            return True
        return int(action) in (glfw_press, glfw_repeat)

    def _match_key(key: int, glfw_name: str, char: str | None = None) -> bool:
        glfw_key = getattr(mujoco, glfw_name, None)
        if glfw_key is not None and key == int(glfw_key):
            return True
        if char is not None and key in (ord(char.lower()), ord(char.upper())):
            return True
        return False

    def _on_key(*args) -> None:
        key: int
        action: int | None

        # Handle different callback signatures across mujoco/glfw bindings.
        if len(args) == 1:
            key = int(args[0])
            action = None
        elif len(args) == 2:
            key = int(args[0])
            maybe_action = int(args[1])
            if maybe_action in (glfw_press, glfw_repeat, glfw_release):
                action = maybe_action
            else:
                action = None
        elif len(args) == 4:
            key, _scancode, action, _mods = args
            key = int(key)
            action = int(action)
        else:
            return

        if not _is_press(action):
            return

        if _match_key(key, "GLFW_KEY_G", "g"):
            ctrl.goal_requested = True
        elif _match_key(key, "GLFW_KEY_R", "r"):
            ctrl.reset_requested = True
        elif _match_key(key, "GLFW_KEY_P", "p"):
            ctrl.paused = not ctrl.paused
            print(f"[eval] {'Paused' if ctrl.paused else 'Resumed'}")
        elif _match_key(key, "GLFW_KEY_Q", "q") or _match_key(
            key, "GLFW_KEY_ESCAPE"
        ):
            ctrl.quit_requested = True

    return _on_key


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive SB3 reach policy evaluation.")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to SB3 PPO model (.zip or basename without .zip).",
    )
    p.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="VecNormalize stats .pkl path. If omitted, uses <model>_vecnorm.pkl and errors if missing.",
    )
    p.add_argument("--cfg-name", type=str, default="ur3e_joint_pos_dense_stable")
    p.add_argument("--robot", type=str, default="ur3e", choices=["ur3e", "ur5e"])
    p.add_argument(
        "--control-variant",
        type=str,
        default="joint_pos",
        choices=["joint_pos", "joint_pos_isaac_reward", "ik_rel", "ik_abs"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--time-limit",
        type=int,
        default=0,
        help="Episode length in control steps. Default 0: no timeout (manual reset only).",
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions.",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-episodes", type=int, default=0, help="0 means run until quit.")
    p.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Print metrics every N control steps.",
    )
    return p


def main() -> None:
    args = _parser().parse_args()

    model_path = _resolve_model_path(args.model)
    vecnorm_path = Path(args.vecnorm) if args.vecnorm else _default_vecnorm_path(model_path)

    cfg = get_reach_cfg(args.cfg_name)
    cfg.scene.robot = args.robot
    cfg.scene.render_mode = None
    cfg.episode.seed = args.seed
    cfg.actions.control_variant = args.control_variant
    cfg.episode.time_limit = int(args.time_limit)
    cfg.physics.obs_noise = 0.0
    
    # Evaluation should not change goals automatically.
    cfg.commands.goal_resample_time_range_s = (
        _EVAL_NO_AUTO_GOAL_RESAMPLE_S,
        _EVAL_NO_AUTO_GOAL_RESAMPLE_S,
    )
    cfg.success.resample_on_success = False
    cfg.success.terminate_on_success = False

    env = make_reach_manager_based_gymnasium(cfg)
    vec_env = _load_vec_env(env, vecnorm_path)
    model = PPO.load(str(model_path), env=vec_env, device=args.device)
    base = env.base

    ctrl_dt = float(base.model.opt.timestep * base.n_substeps)
    controls = _EvalControls()

    print("\n=== Reach Eval Controls ===")
    print("G: sample new goal")
    print("R: reset episode")
    print("P: pause/resume")
    print("Q or ESC: quit")
    print("Default: no timeout; reset manually with R")
    print("Auto goal resampling: disabled (press G for new goal)")
    print("===========================\n")

    obs = vec_env.reset()
    ep_return = 0.0
    ep_steps = 0
    n_episodes = 0
    episode_done = False

    key_callback = _make_key_callback(controls)
    with mujoco.viewer.launch_passive(base.model, base.data, key_callback=key_callback) as viewer:
        last_tick = time.perf_counter()
        while viewer.is_running() and not controls.quit_requested:
            now = time.perf_counter()
            elapsed = now - last_tick
            if elapsed < ctrl_dt:
                viewer.sync()
                time.sleep(0.001)
                continue
            last_tick = now

            if controls.reset_requested:
                obs = vec_env.reset()
                ep_return = 0.0
                ep_steps = 0
                episode_done = False
                controls.paused = False
                controls.reset_requested = False
                print("[eval] Episode reset.")

            if controls.goal_requested:
                new_goal = base.resample_goal_now()
                controls.goal_requested = False
                print(
                    "[eval] New goal sampled: "
                    f"x={new_goal[0]:+.3f}, y={new_goal[1]:+.3f}, z={new_goal[2]:+.3f}"
                )

            if controls.paused:
                viewer.sync()
                continue

            if episode_done:
                viewer.sync()
                continue

            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = vec_env.step(action)
            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]

            ep_return += reward
            ep_steps += 1

            if args.print_every > 0 and (ep_steps % args.print_every == 0):
                print(
                    "[eval] "
                    f"step={ep_steps:5d} "
                    f"reward={reward:+.4f} "
                    f"dist={float(info.get('dist', np.nan)):.4f} "
                    f"ori={float(info.get('ori_err', np.nan)):.4f} "
                    f"success={bool(info.get('success', False))}"
                )

            if done:
                n_episodes += 1
                episode_done = True
                print(
                    "[eval] Episode done: "
                    f"id={n_episodes} return={ep_return:+.4f} steps={ep_steps} "
                    f"terminated={bool(info.get('terminated', False))} "
                    f"time_out={bool(info.get('time_out', False))}"
                )
                print("[eval] Press R to reset and continue.")
                if args.max_episodes > 0 and n_episodes >= args.max_episodes:
                    print("[eval] max_episodes reached, exiting.")
                    break

            viewer.sync()

    vec_env.close()


if __name__ == "__main__":
    main()
