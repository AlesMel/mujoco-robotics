#!/usr/bin/env python
"""Interactive evaluation for a trained Crazyflie reach policy.

Renders the 4-camera composite view in an **OpenCV window** with reliable
keyboard input.  The trained PPO policy runs in real-time while the user
moves the green goal marker with the keyboard.

Controls  (click the OpenCV window first to give it focus!)
--------
    W / ↑        : goal forward  (+X)
    S / ↓        : goal backward (−X)
    A / ←        : goal left     (+Y)
    D / →        : goal right    (−Y)
    E            : goal up       (+Z)
    Q            : goal down     (−Z)
    Space        : reset episode
    P            : pause / resume
    T            : toggle auto-orbit (goal circles automatically)
    F            : snap goal to drone position
    R            : random new goal
    +  /  =      : increase goal speed
    −            : decrease goal speed
    Escape       : quit

Usage::

    python -m mujoco_robot.training.eval_crazyflie_reach \\
        --model ppo_crazyflie_reach_crazyflie_reach_dense_stable.zip

    python scripts/eval_crazyflie_reach.py \\
        --model ppo_crazyflie_reach_crazyflie_reach_dense_stable.zip \\
        --goal-speed 0.6
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mujoco_robot.tasks import (
    get_crazyflie_cfg,
    make_crazyflie_reach_gymnasium,
)


# ──────────────────────────────────────────────────────────
# OpenCV key codes  (cv2.waitKeyEx, platform-dependent)
# ──────────────────────────────────────────────────────────
_K_ESC = 27
_K_SPACE = 32
_K_PLUS = ord("+")
_K_EQUAL = ord("=")
_K_MINUS = ord("-")

# Arrow keys — Linux X11 via waitKeyEx
_K_UP = 0xFF52
_K_DOWN = 0xFF54
_K_LEFT = 0xFF51
_K_RIGHT = 0xFF53
# Fallback (some systems)
_K_UP2 = 0x00520000
_K_DOWN2 = 0x00540000
_K_LEFT2 = 0x00510000
_K_RIGHT2 = 0x00530000


def _is_key(code: int, *targets: int) -> bool:
    return code in targets


# ──────────────────────────────────────────────────────────
# Model / VecNorm utilities
# ──────────────────────────────────────────────────────────
def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    if p.suffix != ".zip":
        z = p.with_suffix(".zip")
        if z.exists():
            return z
    raise FileNotFoundError(f"Not found: '{path_str}' (tried .zip too)")


def _default_vecnorm_path(model_path: Path) -> Path:
    stem = model_path.with_suffix("") if model_path.suffix == ".zip" else model_path
    return Path(f"{stem}_vecnorm.pkl")


def _load_vec_env(env, vecnorm_path: Path) -> VecNormalize:
    vec_env = DummyVecEnv([lambda: env])
    if not vecnorm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found: '{vecnorm_path}'. "
            "Ensure <model>_vecnorm.pkl exists next to the model."
        )
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print(f"[eval] Loaded VecNormalize stats: {vecnorm_path}")
    return vec_env


# ──────────────────────────────────────────────────────────
# HUD overlay  (cv2.putText — much nicer than bitmap font)
# ──────────────────────────────────────────────────────────
def _draw_hud(
    frame: np.ndarray,
    goal: np.ndarray,
    ep_steps: int,
    ep_return: float,
    goal_speed: float,
    paused: bool,
    auto_orbit: bool,
    ep_done: bool,
    info: dict,
) -> np.ndarray:
    """Burn a translucent HUD bar onto the top and bottom of the frame."""
    h, w = frame.shape[:2]

    # ---- Top banner (semi-transparent) ----
    bar_h = 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.48
    c_white = (255, 255, 255)
    c_green = (100, 255, 120)
    c_yellow = (80, 230, 255)
    c_red = (80, 80, 255)

    dist = float(info.get("goal_dist", info.get("pos_error", 0.0)))
    goals_reached = int(info.get("goals_reached", 0))
    tilt = float(info.get("tilt_deg", 0.0))

    # Row 1
    txt1 = (
        f"Goal: [{goal[0]:+.2f}, {goal[1]:+.2f}, {goal[2]:+.2f}]   "
        f"Dist: {dist:.3f}m   Tilt: {tilt:.1f}deg   Goals: {goals_reached}"
    )
    cv2.putText(frame, txt1, (10, 18), font, fs, c_green, 1, cv2.LINE_AA)

    # Row 2
    state = "PAUSED" if paused else ("DONE - Space to reset" if ep_done else "RUNNING")
    state_color = c_yellow if paused else (c_red if ep_done else c_white)
    orbit_str = "  [ORBIT]" if auto_orbit else ""
    txt2 = (
        f"Step: {ep_steps:5d}  Return: {ep_return:+.1f}  "
        f"Speed: {goal_speed:.1f}m/s{orbit_str}   |  {state}"
    )
    cv2.putText(frame, txt2, (10, 42), font, fs, state_color, 1, cv2.LINE_AA)

    # ---- Bottom hint bar ----
    hint_h = 22
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - hint_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    hint = "WASD/Arrows:move  Q/E:up/dn  Space:reset  T:orbit  F:snap  R:random  P:pause  +/-:speed  Esc:quit"
    cv2.putText(frame, hint, (10, h - 6), font, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive Crazyflie reach eval – OpenCV window, keyboard controls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to SB3 PPO model (.zip).",
    )
    p.add_argument(
        "--vecnorm", type=str, default=None,
        help="VecNormalize .pkl. If omitted, <model>_vecnorm.pkl is used.",
    )
    p.add_argument(
        "--cfg-name", type=str, default="crazyflie_reach_dense_stable",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="Use deterministic policy actions.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--time-limit", type=int, default=0,
        help="Episode length. 0 = unlimited (manual reset only).",
    )
    p.add_argument(
        "--goal-speed", type=float, default=0.4,
        help="Keyboard goal movement speed (m/s).",
    )
    p.add_argument(
        "--print-every", type=int, default=60,
        help="Print telemetry every N control steps. 0 = off.",
    )
    p.add_argument(
        "--window-scale", type=float, default=1.0,
        help="Scale the display window (e.g. 1.5 for 150%%).",
    )
    return p


# ──────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────
_WINDOW_NAME = "Crazyflie Reach - Interactive Eval"


def main() -> None:
    args = _parser().parse_args()

    model_path = _resolve_path(args.model)
    vecnorm_path = (
        Path(args.vecnorm) if args.vecnorm else _default_vecnorm_path(model_path)
    )

    # ---- Build environment with rgb_array rendering ----
    cfg = get_crazyflie_cfg(args.cfg_name)
    cfg.seed = args.seed
    cfg.render = True
    cfg.render_mode = "rgb_array"
    if args.time_limit > 0:
        cfg.time_limit = args.time_limit
    else:
        cfg.time_limit = 100_000  # effectively unlimited

    gym_env = make_crazyflie_reach_gymnasium(cfg)
    vec_env = _load_vec_env(Monitor(gym_env), vecnorm_path)
    model = PPO.load(str(model_path), env=vec_env, device=args.device)

    # Direct access to the raw CrazyflieReachEnv
    base = gym_env.base
    ctrl_dt = float(base.model.opt.timestep) * float(base.n_substeps)

    # Workspace bounds
    xy_lim = base.workspace_xy
    z_lo, z_hi = base.workspace_z

    # ---- State ----
    goal_speed = args.goal_speed
    paused = False
    auto_orbit = False
    orbit_speed = 0.6
    orbit_phase = 0.0
    ep_return = 0.0
    ep_steps = 0
    n_episodes = 0
    ep_done = False
    last_info: dict = {}

    # ---- Print help ----
    print("\n" + "=" * 62)
    print("  Crazyflie Reach – Interactive Evaluation  (OpenCV window)")
    print("=" * 62)
    print("  WASD / Arrows  : move goal in XY")
    print("  E / Q          : goal up / down")
    print("  + / -          : faster / slower goal movement")
    print("  Space          : reset episode")
    print("  P              : pause / resume")
    print("  T              : toggle auto-orbit")
    print("  F              : snap goal to drone position")
    print("  R              : random new goal")
    print("  Escape         : quit")
    print()
    print("  >>> Click the OpenCV window to give it keyboard focus <<<")
    print("=" * 62 + "\n")

    # ---- Initial reset ----
    obs = vec_env.reset()

    # ---- OpenCV window setup ----
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    init_frame = base.render(mode="rgb_array")
    fh, fw = init_frame.shape[:2]
    disp_w = int(fw * args.window_scale)
    disp_h = int(fh * args.window_scale)
    cv2.resizeWindow(_WINDOW_NAME, disp_w, disp_h)

    running = True
    last_tick = time.perf_counter()
    need_step = True  # whether enough time has elapsed for a physics step

    while running:
        now = time.perf_counter()
        dt_real = now - last_tick
        need_step = dt_real >= ctrl_dt

        # ── Keyboard input (1ms poll — keeps the window responsive) ──
        raw_key = cv2.waitKeyEx(1)
        key = raw_key & 0xFFFFFF

        # --- One-shot actions ---
        if key == _K_ESC:
            running = False
            break

        if _is_key(key, _K_SPACE):
            obs = vec_env.reset()
            ep_return = 0.0
            ep_steps = 0
            ep_done = False
            paused = False
            last_info = {}
            print("[eval] Episode reset.")

        elif _is_key(key, ord("p"), ord("P")):
            paused = not paused
            print(f"[eval] {'Paused' if paused else 'Resumed'}")

        elif _is_key(key, ord("t"), ord("T")):
            auto_orbit = not auto_orbit
            orbit_phase = math.atan2(base._goal_pos[1], base._goal_pos[0])
            print(f"[eval] Auto-orbit: {'ON' if auto_orbit else 'OFF'}")

        elif _is_key(key, ord("f"), ord("F")):
            pos = base.data.qpos[base._cf_qpos_adr: base._cf_qpos_adr + 3].copy()
            base._goal_pos[:] = pos
            base._update_goal_marker()
            base._reach_hold_counter = 0
            print(f"[eval] Goal snapped to drone: "
                  f"[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}]")

        elif _is_key(key, ord("r"), ord("R")):
            base._goal_pos[:] = base._sample_goal()
            base._update_goal_marker()
            base._reach_hold_counter = 0
            print(f"[eval] Random goal: "
                  f"[{base._goal_pos[0]:+.2f},{base._goal_pos[1]:+.2f},"
                  f"{base._goal_pos[2]:+.2f}]")

        elif _is_key(key, _K_PLUS, _K_EQUAL):
            goal_speed = min(goal_speed + 0.1, 2.0)
            print(f"[eval] Goal speed: {goal_speed:.1f} m/s")

        elif _is_key(key, _K_MINUS):
            goal_speed = max(goal_speed - 0.1, 0.1)
            print(f"[eval] Goal speed: {goal_speed:.1f} m/s")

        # --- Continuous movement keys ---
        goal_delta = np.zeros(3, dtype=float)
        if _is_key(key, ord("w"), ord("W"), _K_UP, _K_UP2):
            goal_delta[0] += goal_speed
        if _is_key(key, ord("s"), ord("S"), _K_DOWN, _K_DOWN2):
            goal_delta[0] -= goal_speed
        if _is_key(key, ord("a"), ord("A"), _K_LEFT, _K_LEFT2):
            goal_delta[1] += goal_speed
        if _is_key(key, ord("d"), ord("D"), _K_RIGHT, _K_RIGHT2):
            goal_delta[1] -= goal_speed
        if _is_key(key, ord("e"), ord("E")):
            goal_delta[2] += goal_speed
        if _is_key(key, ord("q"), ord("Q")):
            goal_delta[2] -= goal_speed

        if np.any(np.abs(goal_delta) > 1e-6):
            # Use real dt so holding a key gives smooth movement
            dt_move = max(dt_real, ctrl_dt)
            base._goal_pos += goal_delta * dt_move
            base._goal_pos[0] = float(np.clip(base._goal_pos[0], -xy_lim, xy_lim))
            base._goal_pos[1] = float(np.clip(base._goal_pos[1], -xy_lim, xy_lim))
            base._goal_pos[2] = float(np.clip(base._goal_pos[2], z_lo, z_hi))
            base._update_goal_marker()

        # --- Auto-orbit ---
        if auto_orbit:
            dt_orbit = max(dt_real, ctrl_dt)
            orbit_phase += orbit_speed * dt_orbit
            r = 0.35
            base._goal_pos[0] = r * math.cos(orbit_phase)
            base._goal_pos[1] = r * math.sin(orbit_phase)
            base._update_goal_marker()

        # ── Physics step (rate-limited to real-time) ──
        if need_step:
            last_tick = now

            if not paused and not ep_done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, rewards, dones, infos = vec_env.step(action)
                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0]
                last_info = info

                ep_return += reward
                ep_steps += 1

                if args.print_every > 0 and ep_steps % args.print_every == 0:
                    dist = float(info.get("goal_dist", info.get("pos_error", 0)))
                    tilt = float(info.get("tilt_deg", 0))
                    goals = int(info.get("goals_reached", 0))
                    print(
                        f"[eval] step={ep_steps:5d}  "
                        f"r={reward:+.3f}  dist={dist:.3f}m  "
                        f"tilt={tilt:.1f}°  goals={goals}  "
                        f"goal=[{base._goal_pos[0]:+.2f},"
                        f"{base._goal_pos[1]:+.2f},"
                        f"{base._goal_pos[2]:+.2f}]"
                    )

                if done:
                    n_episodes += 1
                    ep_done = True
                    reason = []
                    for tag in ("crashed", "excessive_tilt", "out_of_bounds",
                                "time_out", "battery_depleted"):
                        if info.get(tag):
                            reason.append(tag)
                    print(
                        f"[eval] Episode {n_episodes} done: "
                        f"return={ep_return:+.2f}  steps={ep_steps}  "
                        f"reason={','.join(reason) or '?'}  "
                        f"goals={int(info.get('goals_reached', 0))}"
                    )
                    print("[eval] Press SPACE to reset.")

        # ── Render & display ──
        frame = base.render(mode="rgb_array")
        frame = _draw_hud(
            frame,
            goal=base._goal_pos,
            ep_steps=ep_steps,
            ep_return=ep_return,
            goal_speed=goal_speed,
            paused=paused,
            auto_orbit=auto_orbit,
            ep_done=ep_done,
            info=last_info,
        )
        # OpenCV uses BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if args.window_scale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
            )

        cv2.imshow(_WINDOW_NAME, frame_bgr)

        # Detect window closed via X button
        try:
            if cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                running = False
        except cv2.error:
            running = False

    # ---- Cleanup ----
    cv2.destroyAllWindows()
    vec_env.close()
    print("[eval] Done. Bye!")


if __name__ == "__main__":
    main()
