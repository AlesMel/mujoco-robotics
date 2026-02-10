"""Config dataclasses for manager-based MDP terms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np


ActionFn = Callable[[Any, np.ndarray], np.ndarray]
CommandFn = Callable[[Any], np.ndarray]
ObservationFn = Callable[[Any], np.ndarray]
RewardFn = Callable[[Any, Dict[str, float]], float]
TerminationFn = Callable[[Any, Dict[str, float]], bool]
WeightFn = Callable[[Any], float]


@dataclass(frozen=True)
class ActionTermCfg:
    """Configuration for the action processing term."""

    name: str
    fn: ActionFn


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for the command generation term."""

    name: str
    fn: CommandFn
    resampling_time_range_s: tuple[float, float] = (4.0, 4.0)


@dataclass(frozen=True)
class ObservationTermCfg:
    """Configuration for one observation term."""

    name: str
    fn: ObservationFn


@dataclass(frozen=True)
class RewardTermCfg:
    """Configuration for one reward term."""

    name: str
    fn: RewardFn
    weight: float | WeightFn = 1.0
    enabled: bool = True


@dataclass(frozen=True)
class TerminationTermCfg:
    """Configuration for one termination flag term."""

    name: str
    fn: TerminationFn


@dataclass
class ReachMDPCfg:
    """Manager-based MDP configuration for reach environments."""

    action_term: ActionTermCfg | None = None
    command_term: CommandTermCfg | None = None
    observation_terms: tuple[ObservationTermCfg, ...] = field(default_factory=tuple)
    reward_terms: tuple[RewardTermCfg, ...] = field(default_factory=tuple)
    success_term: TerminationTermCfg | None = None
    failure_term: TerminationTermCfg | None = None
    timeout_term: TerminationTermCfg | None = None
    include_reward_terms_in_info: bool = False
