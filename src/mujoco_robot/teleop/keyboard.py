"""Keyboard teleop controllers for UR robot environments.

Usage::

    from mujoco_robot.tasks.reach import ReachJointPosEnv
    from mujoco_robot.tasks.slot_sorter.slot_sorter_env import URSlotSorterEnv
    from mujoco_robot.teleop import ReachTeleop, SlotSorterTeleop

    # Reach task
    env = ReachJointPosEnv(robot="ur5e", time_limit=0)
    ReachTeleop(env).run()

    # Slot sorter task
    env = URSlotSorterEnv(time_limit=0)
    SlotSorterTeleop(env).run()
"""
from __future__ import annotations

import time
from typing import Dict, List

import mujoco
import mujoco.viewer
import numpy as np


class ReachTeleop:
    """Keyboard control for the reach environment.

    Controls:
        W/S — ±Y, A/D — ±X, R/F — ±Z, Q/E — ±yaw, X — stop.
    """

    def __init__(self, env, speed: float = 1.0) -> None:
        self.env = env
        self.speed = speed
        self.action = np.zeros(env.action_dim, dtype=float)

    def _on_key(self, *args) -> None:
        if len(args) == 2:
            key, action = args
        elif len(args) == 4:
            key, _, action, _ = args
        else:
            return
        press = action in (mujoco.GLFW_PRESS, getattr(mujoco, "GLFW_REPEAT", -1))

        if key == mujoco.GLFW_KEY_W:
            self.action[1] = self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_S:
            self.action[1] = -self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_A:
            self.action[0] = -self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_D:
            self.action[0] = self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_R:
            self.action[2] = self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_F:
            self.action[2] = -self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_Q:
            self.action[3] = self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_E:
            self.action[3] = -self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_X:
            if action == mujoco.GLFW_RELEASE:
                self.action[:] = 0.0

    def run(self) -> None:
        """Launch the interactive viewer loop."""
        obs = self.env.reset()
        ctrl_dt = self.env.model.opt.timestep * self.env.n_substeps
        print(f"Obs dim: {obs.shape[0]}, ctrl period: {ctrl_dt*1000:.0f} ms")
        print("Controls: W/S ±Y, A/D ±X, R/F ±Z, Q/E ±yaw, X stop")

        with mujoco.viewer.launch_passive(
            self.env.model, self.env.data, key_callback=self._on_key
        ) as viewer:
            last = time.time()
            while viewer.is_running():
                now = time.time()
                if now - last < ctrl_dt:
                    time.sleep(0.001)
                    continue
                last = now

                step = self.env.step(self.action)
                viewer.sync()

                dist = step.info["dist"]
                goals = step.info.get("goals_reached", 0)
                if step.info["reached"]:
                    status = f"REACHED! (goals: {goals})"
                else:
                    status = f"dist={dist:.3f}  goals={goals}"
                print(f"\r{status}  reward={step.reward:+.3f}", end="")

                if step.done:
                    print("\nEpisode done — resetting…")
                    time.sleep(0.5)
                    self.env.reset()
                    self.action[:] = 0.0


class SlotSorterTeleop:
    """Keyboard teleop for the slot-sorter environment.

    Controls:
        W/S or I/K — ±Y, A/D or J/L — ±X, R/F or U/O — ±Z,
        Q/E or Y/P — ±yaw, SPACE — grip, X — stop.
    """

    def __init__(self, env, speed: float = 1.0) -> None:
        self.env = env
        self.speed = speed
        self.action = np.zeros(env.action_dim, dtype=float)

    def _on_key(self, *args) -> None:
        if len(args) == 2:
            key, action = args
        elif len(args) == 4:
            key, _, action, _ = args
        else:
            return
        press = action in (mujoco.GLFW_PRESS, getattr(mujoco, "GLFW_REPEAT", -1))
        release = action == mujoco.GLFW_RELEASE

        if key in (mujoco.GLFW_KEY_I, mujoco.GLFW_KEY_W):
            self.action[1] = self.speed if press else 0.0
        elif key in (mujoco.GLFW_KEY_K, mujoco.GLFW_KEY_S):
            self.action[1] = -self.speed if press else 0.0
        elif key in (mujoco.GLFW_KEY_J, mujoco.GLFW_KEY_A):
            self.action[0] = -self.speed if press else 0.0
        elif key in (mujoco.GLFW_KEY_L, mujoco.GLFW_KEY_D):
            self.action[0] = self.speed if press else 0.0
        elif key in (mujoco.GLFW_KEY_U, mujoco.GLFW_KEY_R):
            self.action[2] = self.speed if press else 0.0
        elif key in (mujoco.GLFW_KEY_O, mujoco.GLFW_KEY_F):
            self.action[2] = -self.speed if press else 0.0
        elif key in (mujoco.GLFW_KEY_Y, mujoco.GLFW_KEY_Q):
            self.action[3] = self.speed if press else 0.0
        elif key in (mujoco.GLFW_KEY_P, mujoco.GLFW_KEY_E):
            self.action[3] = -self.speed if press else 0.0
        elif key == mujoco.GLFW_KEY_SPACE:
            self.action[4] = 1.0 if press else 0.0
        elif key == mujoco.GLFW_KEY_X and release:
            self.action[:] = 0.0

    def _hud_lines(self, info: Dict) -> List[str]:
        grasp = info["grasped"]
        status = "".join(
            "✓" if s else "_"
            for s in self.env.success_mask[: self.env.n_objects]
        )
        return [
            f"Objects placed: {info['successes']}/{info['n_objects']}",
            f"Grasp: {'none' if grasp < 0 else grasp}",
            f"Slots: {status}",
            f"Action: {self.action}",
            "Controls: W/S +/-y, A/D +/-x, R/F +/-z, Q/E +/-yaw, SPACE grip, X stop",
        ]

    def run(self) -> None:
        """Launch the interactive viewer loop."""
        obs = self.env.reset()
        ctrl_period = self.env.model.opt.timestep * self.env.n_substeps
        print(f"Observation dim: {obs.shape[0]}, control period: {ctrl_period*1000:.0f} ms")

        with mujoco.viewer.launch_passive(
            self.env.model, self.env.data, key_callback=self._on_key
        ) as viewer:
            last = time.time()
            info = {"successes": 0, "grasped": -1, "n_objects": self.env.n_objects}
            while viewer.is_running():
                now = time.time()
                if now - last < ctrl_period:
                    time.sleep(0.001)
                    continue
                last = now

                step = self.env.step(self.action)
                info = step.info
                viewer.sync()

                text = "\n".join(self._hud_lines(info))
                if hasattr(viewer, "add_overlay"):
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT, "HUD", text
                    )
                else:
                    print(f"\r{text.replace(chr(10), ' | ')}", end="")

                if step.done:
                    print("\nEpisode finished. Resetting...")
                    time.sleep(0.5)
                    obs = self.env.reset()
                    self.action[:] = 0.0
                    info = {"successes": 0, "grasped": -1, "n_objects": self.env.n_objects}
