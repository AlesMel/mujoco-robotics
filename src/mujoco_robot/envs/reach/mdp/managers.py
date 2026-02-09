"""Manager objects for reach MDP terms."""
from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_robot.envs.reach.mdp.terms import (
    ActionTermCfg,
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

        for term in self._terms:
            if not term.enabled:
                continue
            raw = float(term.fn(self._env, ctx))
            weight = term.weight(self._env) if callable(term.weight) else float(term.weight)
            weighted = weight * raw
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
