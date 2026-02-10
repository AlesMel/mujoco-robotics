"""Manager objects for reach MDP terms."""
from __future__ import annotations

from typing import Any

import numpy as np

from .terms import (
    ActionTermCfg,
    CommandTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)


class ActionManager:
    """Process normalized actions into joint position targets."""

    def __init__(self, env: Any, term: ActionTermCfg):
        self._env = env
        self._term = term

    def compute_joint_targets(self, action: np.ndarray) -> np.ndarray:
        out = self._term.fn(self._env, action)
        return np.asarray(out, dtype=float).flatten()


class CommandManager:
    """Manage pose command sampling and time-based resampling."""

    def __init__(self, env: Any, term: CommandTermCfg):
        self._env = env
        self._term = term
        self._elapsed_s = 0.0
        self._target_s = 0.0
        self._pose_command = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._validate_time_range()

    def _validate_time_range(self) -> None:
        lo_s, hi_s = self._term.resampling_time_range_s
        lo_s = float(lo_s)
        hi_s = float(hi_s)
        if lo_s <= 0.0 or hi_s <= 0.0 or lo_s > hi_s:
            raise ValueError(
                "command resampling_time_range_s must be positive and ordered "
                f"(min <= max), got {self._term.resampling_time_range_s}"
            )

    def _sample_interval_s(self) -> float:
        lo_s, hi_s = self._term.resampling_time_range_s
        if hi_s <= lo_s:
            return float(lo_s)
        return float(self._env._rng.uniform(lo_s, hi_s))

    def _sample_pose_command(self) -> np.ndarray:
        out = np.asarray(self._term.fn(self._env), dtype=np.float32).reshape(-1)
        if out.shape[0] != 7:
            raise ValueError(
                f"Command term '{self._term.name}' must return shape (7,), got {tuple(out.shape)}"
            )
        return out

    def reset(self) -> None:
        self._elapsed_s = 0.0
        self._target_s = self._sample_interval_s()
        self._pose_command = self._sample_pose_command()

    def step(self, step_dt: float) -> bool:
        self._elapsed_s += float(step_dt)
        if self._elapsed_s < self._target_s:
            return False
        self._elapsed_s = 0.0
        self._target_s = self._sample_interval_s()
        self._pose_command = self._sample_pose_command()
        return True

    @property
    def elapsed_s(self) -> float:
        return float(self._elapsed_s)

    @property
    def target_s(self) -> float:
        return float(self._target_s)

    @property
    def pose_command(self) -> np.ndarray:
        return self._pose_command.astype(np.float32).copy()


class ObservationManager:
    """Compose observation vectors from configured terms."""

    def __init__(self, env: Any, terms: tuple[ObservationTermCfg, ...]):
        self._env = env
        self._terms = terms
        self._dim = self._infer_dim()

    @property
    def dim(self) -> int:
        return self._dim

    def _infer_dim(self) -> int:
        if not self._terms:
            return 0
        total = 0
        for term in self._terms:
            sample = np.asarray(term.fn(self._env), dtype=np.float32).ravel()
            total += int(sample.shape[0])
        return total

    def observe(self) -> np.ndarray:
        if not self._terms:
            return np.zeros((0,), dtype=np.float32)
        parts = [
            np.asarray(term.fn(self._env), dtype=np.float32).ravel()
            for term in self._terms
        ]
        return np.concatenate(parts).astype(np.float32)


class RewardManager:
    """Aggregate weighted reward terms."""

    def __init__(self, env: Any, terms: tuple[RewardTermCfg, ...]):
        self._env = env
        self._terms = terms

    def compute(self, ctx: dict[str, float]) -> tuple[float, dict[str, float], dict[str, float]]:
        total = 0.0
        raw_terms: dict[str, float] = {}
        weighted_terms: dict[str, float] = {}
        step_dt = float(self._env.model.opt.timestep * self._env.n_substeps)

        for term in self._terms:
            if not term.enabled:
                continue
            raw = float(term.fn(self._env, ctx))
            weight = term.weight(self._env) if callable(term.weight) else float(term.weight)
            weighted = weight * raw * step_dt
            total += weighted
            raw_terms[term.name] = raw
            weighted_terms[term.name] = float(weighted)

        return float(total), raw_terms, weighted_terms


class TerminationManager:
    """Compute success/failure/timeout flags."""

    def __init__(
        self,
        env: Any,
        success_term: TerminationTermCfg,
        failure_term: TerminationTermCfg,
        timeout_term: TerminationTermCfg,
    ):
        self._env = env
        self._success = success_term
        self._failure = failure_term
        self._timeout = timeout_term

    def compute(self, ctx: dict[str, float]) -> dict[str, bool]:
        success = bool(self._success.fn(self._env, ctx))
        failure = bool(self._failure.fn(self._env, ctx))
        terminated = (
            (self._env.terminate_on_success and success)
            or (self._env.terminate_on_collision and failure)
        )
        time_out = bool(self._timeout.fn(self._env, ctx))
        done = bool(terminated or time_out)
        return {
            "success": success,
            "failure": failure,
            "terminated": bool(terminated),
            "time_out": bool(time_out),
            "done": done,
        }
