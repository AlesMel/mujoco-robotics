"""Reusable PPO training configuration dataclasses.

All SB3 PPO hyper-parameters live here so they're readable, versionable,
and overridable per-environment without touching the training loop.

Usage::

    from mujoco_robot.training.ppo_cfg import TrainCfg, PPOCfg

    # Use defaults everywhere
    cfg = TrainCfg()

    # Override per-env
    cfg = TrainCfg(
        ppo=PPOCfg(ent_coef=0.005, learning_rate=1e-4),
        total_timesteps=5_000_000,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

# ── activation look-up ──────────────────────────────────────────────
_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def _resolve_activation(name: str) -> type[nn.Module]:
    key = name.lower().replace("-", "_")
    if key not in _ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Choose from {list(_ACTIVATION_MAP)}"
        )
    return _ACTIVATION_MAP[key]


# ── PPO hyper-parameters ────────────────────────────────────────────
@dataclass
class PPOCfg:
    """Stable-Baselines3 PPO hyper-parameters.

    Defaults match the values used across reach / lift-suction / cable-routing
    tasks and work well for 6-DOF robot-arm control.
    """

    # -- rollout ----------------------------------------------------------
    n_steps: int = 1024
    """Horizon length per environment per PPO update."""

    n_minibatches: int = 4
    """Number of mini-batches.  ``batch_size = n_steps * n_envs // n_minibatches``."""

    # -- optimisation -----------------------------------------------------
    n_epochs: int = 8
    """PPO epochs per rollout."""

    learning_rate: float = 3e-4
    """Adam learning rate."""

    gamma: float = 0.99
    """Discount factor."""

    gae_lambda: float = 0.95
    """GAE lambda for advantage estimation."""

    # -- clipping / regularisation ----------------------------------------
    ent_coef: float = 0.01
    """Entropy bonus coefficient."""

    clip_range: float = 0.2
    """PPO surrogate-objective clip range."""

    vf_coef: float = 1.0
    """Value-function loss coefficient."""

    max_grad_norm: float = 1.0
    """Global gradient-norm clip."""

    # -- network architecture ---------------------------------------------
    pi_layers: Sequence[int] = (128, 128)
    """Policy network hidden-layer sizes."""

    vf_layers: Sequence[int] = (128, 128)
    """Value network hidden-layer sizes."""

    activation: str = "tanh"
    """Activation function name — ``"tanh"`` | ``"relu"`` | ``"elu"``."""

    # -- device -----------------------------------------------------------
    device: str = "cuda"
    """PyTorch device string (``"cuda"`` / ``"cpu"`` / ``"auto"``)."""

    # -- helpers ----------------------------------------------------------
    def sb3_policy_kwargs(self) -> dict:
        """Build the ``policy_kwargs`` dict for ``PPO(...)``."""
        return dict(
            net_arch=dict(pi=list(self.pi_layers), vf=list(self.vf_layers)),
            activation_fn=_resolve_activation(self.activation),
        )

    def batch_size(self, n_envs: int) -> int:
        """Compute mini-batch size for the given number of parallel envs."""
        return (self.n_steps * n_envs) // self.n_minibatches


# ── VecNormalize settings ───────────────────────────────────────────
@dataclass
class NormalizeCfg:
    """``VecNormalize`` wrapper parameters."""

    norm_obs: bool = True
    """Normalise observations with running statistics."""

    norm_reward: bool = True
    """Normalise rewards with running statistics."""

    clip_obs: float = 10.0
    """Hard clip normalised observations to ±clip_obs."""


# ── Top-level training-run config ───────────────────────────────────
@dataclass
class TrainCfg:
    """Everything needed to launch an SB3 PPO training run.

    Nest ``PPOCfg`` and ``NormalizeCfg`` for clean overrides::

        cfg = TrainCfg(
            ppo=PPOCfg(learning_rate=1e-4),
            total_timesteps=5_000_000,
        )
    """

    ppo: PPOCfg = field(default_factory=PPOCfg)
    """PPO hyper-parameters."""

    normalize: NormalizeCfg = field(default_factory=NormalizeCfg)
    """VecNormalize settings."""

    # -- run settings -----------------------------------------------------
    total_timesteps: int = 10_000_000
    """Total environment steps for training."""

    n_envs: int = 32
    """Number of parallel sub-process environments."""

    # -- logging ----------------------------------------------------------
    log_dir: str = "runs"
    """Root directory for TensorBoard logs."""

    log_name: str = "ppo"
    """TensorBoard run name prefix."""

    sb3_verbose: int = 0
    """Stable-Baselines3 verbosity (0/1/2)."""

    # -- video / evaluation -----------------------------------------------
    save_video: bool = True
    """Record periodic evaluation videos."""

    save_video_every: int = 500_000
    """Timestep interval between evaluation video recordings."""

    video_dir: str = "videos"
    """Directory for saved evaluation videos."""

    # -- misc -------------------------------------------------------------
    progress_bar: bool = True
    """Show tqdm/rich progress bar during training."""

    callback_new_best_only: bool = True
    """Only log callback messages when a new-best episode return is reached."""


# ── Builder helpers ─────────────────────────────────────────────────


def build_sb3_ppo(
    cfg: TrainCfg,
    vec_env: VecEnv,
    *,
    log_dir: str | None = None,
) -> PPO:
    """Construct an SB3 PPO model from a :class:`TrainCfg`.

    Parameters
    ----------
    cfg:
        Full training configuration.
    vec_env:
        A (wrapped) vectorised environment.
    log_dir:
        Override ``cfg.log_dir`` for tensorboard logging.  ``None``
        falls back to the value in *cfg*.
    """
    p = cfg.ppo
    return PPO(
        "MlpPolicy",
        vec_env,
        n_steps=p.n_steps,
        batch_size=p.batch_size(cfg.n_envs),
        n_epochs=p.n_epochs,
        learning_rate=p.learning_rate,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        ent_coef=p.ent_coef,
        clip_range=p.clip_range,
        vf_coef=p.vf_coef,
        max_grad_norm=p.max_grad_norm,
        device=p.device,
        policy_kwargs=p.sb3_policy_kwargs(),
        verbose=cfg.sb3_verbose,
        tensorboard_log=log_dir or cfg.log_dir,
    )
