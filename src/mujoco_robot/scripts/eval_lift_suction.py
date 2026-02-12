#!/usr/bin/env python
"""Interactive evaluation/debug viewer for the lift-suction task.

Usage::

    python -m mujoco_robot.scripts.eval_lift_suction
    python -m mujoco_robot.scripts.eval_lift_suction --time-limit 0 --seed 0
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_robot.tasks import LiftSuctionTaskConfig, make_task


@dataclass
class _EvalControls:
    paused: bool = False
    reset_requested: bool = False
    quit_requested: bool = False


def _make_key_callback(ctrl: _EvalControls, action: np.ndarray, speed: float):
    glfw_press = int(getattr(mujoco, "GLFW_PRESS", 1))
    glfw_repeat = int(getattr(mujoco, "GLFW_REPEAT", 2))
    glfw_release = int(getattr(mujoco, "GLFW_RELEASE", 0))

    def _is_press(a: int | None) -> bool:
        if a is None:
            return True
        return int(a) in (glfw_press, glfw_repeat)

    def _match_key(key: int, glfw_name: str, char: str | None = None) -> bool:
        glfw_key = getattr(mujoco, glfw_name, None)
        if glfw_key is not None and key == int(glfw_key):
            return True
        if char is not None and key in (ord(char.lower()), ord(char.upper())):
            return True
        return False

    def _set_axis(idx: int, sign: float, pressed: bool) -> None:
        action[idx] = float(sign * speed) if pressed else 0.0

    def _on_key(*args) -> None:
        key: int
        key_action: int | None

        # Handle callback signature variations.
        if len(args) == 1:
            key = int(args[0])
            key_action = None
        elif len(args) == 2:
            key = int(args[0])
            maybe_action = int(args[1])
            if maybe_action in (glfw_press, glfw_repeat, glfw_release):
                key_action = maybe_action
            else:
                key_action = None
        elif len(args) == 4:
            key, _scancode, key_action, _mods = args
            key = int(key)
            key_action = int(key_action)
        else:
            return

        pressed = _is_press(key_action)
        released = key_action is not None and int(key_action) == glfw_release

        if _match_key(key, "GLFW_KEY_W", "w"):
            _set_axis(1, +1.0, pressed)
        elif _match_key(key, "GLFW_KEY_S", "s"):
            _set_axis(1, -1.0, pressed)
        elif _match_key(key, "GLFW_KEY_A", "a"):
            _set_axis(0, -1.0, pressed)
        elif _match_key(key, "GLFW_KEY_D", "d"):
            _set_axis(0, +1.0, pressed)
        elif _match_key(key, "GLFW_KEY_R", "r"):
            if pressed:
                _set_axis(2, +1.0, True)
            else:
                _set_axis(2, +1.0, False)
        elif _match_key(key, "GLFW_KEY_F", "f"):
            _set_axis(2, -1.0, pressed)
        elif _match_key(key, "GLFW_KEY_Q", "q"):
            _set_axis(3, +1.0, pressed)
        elif _match_key(key, "GLFW_KEY_E", "e"):
            _set_axis(3, -1.0, pressed)
        elif _match_key(key, "GLFW_KEY_SPACE"):
            action[4] = 1.0 if pressed else 0.0
        elif _match_key(key, "GLFW_KEY_X", "x") and released:
            action[:] = 0.0
        elif _match_key(key, "GLFW_KEY_P", "p") and pressed:
            ctrl.paused = not ctrl.paused
            print(f"[eval-lift] {'Paused' if ctrl.paused else 'Resumed'}")
        elif _match_key(key, "GLFW_KEY_T", "t") and pressed:
            ctrl.reset_requested = True
        elif (
            _match_key(key, "GLFW_KEY_ESCAPE")
            or _match_key(key, "GLFW_KEY_Z", "z")
        ) and pressed:
            ctrl.quit_requested = True

    return _on_key


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive lift-suction evaluation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--time-limit",
        type=int,
        default=0,
        help="Episode length in control steps. Default 0: no timeout.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Action magnitude for key presses in [0, 1].",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Print metrics every N control steps.",
    )
    p.add_argument(
        "--auto-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset automatically when episode ends.",
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    speed = float(np.clip(args.speed, 0.0, 1.0))

    cfg = LiftSuctionTaskConfig(
        time_limit=int(args.time_limit),
        seed=int(args.seed),
    )
    env = make_task("lift_suction", config=cfg)
    env.reset(seed=int(args.seed))

    ctrl_dt = float(env.model.opt.timestep * env.n_substeps)
    action = np.zeros(env.action_dim, dtype=np.float32)
    controls = _EvalControls()

    print("\n=== Lift-Suction Eval Controls ===")
    print("W/S: +/-Y")
    print("A/D: +/-X")
    print("R/F: +/-Z")
    print("Q/E: +/-Yaw")
    print("SPACE: suction on/off")
    print("X: stop action")
    print("T: reset episode")
    print("P: pause/resume")
    print("Z or ESC: quit")
    print("==================================\n")

    ep_return = 0.0
    ep_steps = 0
    ep_id = 0

    key_callback = _make_key_callback(controls, action, speed)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        last_tick = time.perf_counter()
        while viewer.is_running() and not controls.quit_requested:
            now = time.perf_counter()
            if now - last_tick < ctrl_dt:
                viewer.sync()
                time.sleep(0.001)
                continue
            last_tick = now

            if controls.reset_requested:
                env.reset()
                action[:] = 0.0
                ep_return = 0.0
                ep_steps = 0
                controls.reset_requested = False
                controls.paused = False
                print("[eval-lift] Episode reset.")

            if controls.paused:
                viewer.sync()
                continue

            step = env.step(action)
            ep_return += float(step.reward)
            ep_steps += 1

            if args.print_every > 0 and (ep_steps % int(args.print_every) == 0):
                print(
                    "[eval-lift] "
                    f"step={ep_steps:5d} "
                    f"reward={step.reward:+.4f} "
                    f"h={float(step.info.get('object_height', np.nan)):.4f} "
                    f"h_goal={float(step.info.get('goal_height', np.nan)):.4f} "
                    f"cup_obj={float(step.info.get('cup_obj_dist', np.nan)):.4f} "
                    f"obj_goal={float(step.info.get('obj_goal_dist', np.nan)):.4f} "
                    f"grasped={bool(step.info.get('grasped', False))}"
                )

            if step.done:
                ep_id += 1
                print(
                    "[eval-lift] Episode done: "
                    f"id={ep_id} return={ep_return:+.4f} steps={ep_steps} "
                    f"success={bool(step.info.get('success', False))} "
                    f"time_out={bool(step.info.get('time_out', False))}"
                )
                if args.auto_reset:
                    env.reset()
                    action[:] = 0.0
                    ep_return = 0.0
                    ep_steps = 0
                else:
                    controls.paused = True

            viewer.sync()

    env.close()


if __name__ == "__main__":
    main()

