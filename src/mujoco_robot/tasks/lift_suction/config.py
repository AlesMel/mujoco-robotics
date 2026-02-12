"""Lift-suction task configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LiftSuctionTaskConfig:
    """High-level configuration for the lift-suction task."""

    model_path: str | None = None
    actuator_profile: str = "ur3e"
    time_limit: int = 300
    seed: int | None = None
    render: bool = False
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

