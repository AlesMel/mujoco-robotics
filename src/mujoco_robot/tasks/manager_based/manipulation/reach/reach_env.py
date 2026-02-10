"""Manager-based reach environments (phase-3 runtime-backed)."""
from __future__ import annotations

import copy

import gymnasium
import numpy as np

from mujoco_robot.envs.manager_based import ManagerBasedEnv, ManagerBasedRLEnv
from mujoco_robot.envs.reach.reach_env_base import StepResult, URReachEnvBase

from .config import get_reach_cfg
from .mdp import ActionTermCfg, ReachMDPCfg, make_default_reach_mdp_cfg
from .mdp import actions as mdp_actions
from .reach_env_cfg import ReachEnvCfg


def resolve_reach_env_cfg(cfg: ReachEnvCfg | str | None) -> ReachEnvCfg:
    """Resolve cfg object or profile name into a concrete ReachEnvCfg."""
    if cfg is None:
        return get_reach_cfg("ur3e_joint_pos_dense_stable")
    if isinstance(cfg, str):
        return get_reach_cfg(cfg)
    return cfg


def _action_term_for_variant(control_variant: str):
    action_map = {
        "joint_pos": mdp_actions.joint_relative_targets,
        "joint_pos_isaac_reward": mdp_actions.joint_relative_targets,
        "ik_rel": mdp_actions.ik_relative_joint_targets,
        "ik_abs": mdp_actions.ik_absolute_joint_targets,
    }
    if control_variant not in action_map:
        raise ValueError(
            f"Unknown control_variant '{control_variant}'. Available: {sorted(action_map)}"
        )
    return action_map[control_variant]


def _build_runtime_mdp_cfg(cfg: ReachEnvCfg) -> ReachMDPCfg:
    if cfg.managers.mdp_cfg is None:
        mdp_cfg = make_default_reach_mdp_cfg(reward_cfg=cfg.managers.reward_cfg)
    else:
        mdp_cfg = copy.deepcopy(cfg.managers.mdp_cfg)

    action_term = mdp_cfg.action_term
    if action_term is None or action_term.fn is mdp_actions.call_variant_apply_action:
        mdp_cfg.action_term = ActionTermCfg(
            name=f"{cfg.actions.control_variant}_action",
            fn=_action_term_for_variant(cfg.actions.control_variant),
        )
    return mdp_cfg


class ReachManagerBasedEnv(URReachEnvBase, ManagerBasedEnv):
    """Raw manager-based reach env backed by shared manager runtime."""

    # Keep joint-position action behavior aligned with legacy joint-pos env.
    _ema_alpha: float = 1.0

    def __init__(self, cfg: ReachEnvCfg | str | None = None):
        resolved = resolve_reach_env_cfg(cfg)
        ManagerBasedEnv.__init__(self, cfg=resolved)
        kwargs = resolved.to_legacy_kwargs()
        kwargs["mdp_cfg"] = _build_runtime_mdp_cfg(resolved)
        super().__init__(robot=resolved.scene.robot, **kwargs)
        if resolved.actions.control_variant in {"joint_pos", "joint_pos_isaac_reward"}:
            alpha = float(resolved.actions.joint_target_ema_alpha)
            # Keep Isaac-reward variant behavior fully direct unless explicitly changed.
            if resolved.actions.control_variant == "joint_pos_isaac_reward":
                alpha = 1.0
            self._ema_alpha = float(np.clip(alpha, 0.0, 1.0))

    def _current_joint_pos(self) -> np.ndarray:
        """Read current robot joint positions from MuJoCo state."""
        q = np.empty(len(self._robot_qpos_ids))
        for i, qpos_adr in enumerate(self._robot_qpos_ids):
            q[i] = self.data.qpos[qpos_adr]
        return q


def make_reach_manager_based_env(
    cfg: ReachEnvCfg | str | None = None,
) -> ReachManagerBasedEnv:
    """Create raw reach env from manager-based cfg."""
    return ReachManagerBasedEnv(cfg)


class ReachManagerBasedRLEnv(ManagerBasedRLEnv):
    """Gymnasium-compatible manager-based reach environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: ReachEnvCfg | str | None = None):
        resolved = resolve_reach_env_cfg(cfg)
        super().__init__(cfg=resolved)
        self.render_mode = resolved.scene.render_mode

        env_cfg = copy.deepcopy(resolved)
        high_res = self.render_mode == "rgb_array"
        env_cfg.scene.render_size = (640, 480) if high_res else (160, 120)
        self.base = ReachManagerBasedEnv(env_cfg)

        self.action_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(self.base.action_dim,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(self.base.observation_dim,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options=None):
        obs = self.base.reset(seed=seed)
        return obs.astype(np.float32), {}

    def step(self, action):
        res: StepResult = self.base.step(action)
        terminated = bool(res.info.get("terminated", False))
        truncated = bool(res.info.get("time_out", False))
        if not terminated and not truncated:
            time_up = self.base.time_limit > 0 and self.base.step_id >= self.base.time_limit
            truncated = bool(time_up)
        return res.obs.astype(np.float32), res.reward, terminated, truncated, res.info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.base.render(mode="rgb_array")
        return None

    def close(self):
        self.base.close()


def make_reach_manager_based_gymnasium(
    cfg: ReachEnvCfg | str | None = None,
) -> ReachManagerBasedRLEnv:
    """Create Gymnasium reach env from manager-based cfg."""
    return ReachManagerBasedRLEnv(cfg)
