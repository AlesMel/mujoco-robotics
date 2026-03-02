"""Crazyflie hover task configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict


_DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent.parent / "assets" / "crazyflie_2_1.xml"
)


@dataclass
class CrazyflieTaskConfig:
    """High-level configuration for the Crazyflie hover task."""

    model_path: str | None = _DEFAULT_MODEL
    actuator_profile: str = "crazyflie"
    time_limit: int = 600
    seed: int | None = None
    render: bool = False
    render_mode: str | None = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


def make_crazyflie_hover_cfg() -> CrazyflieTaskConfig:
    """Default Crazyflie hover profile."""
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=600,
    )


def make_crazyflie_hover_dense_stable_cfg() -> CrazyflieTaskConfig:
    """Stable baseline profile for PPO hover training."""
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=700,
        env_kwargs={
            "observation_mode": "state_estimate",
            "n_substeps": 10,
            "disturbance_sigma": 0.03,
            "actuator_noise_std": 0.015,
            "sensor_noise_scale": 1.0,
        },
    )


def make_crazyflie_hover_flowdeck_cfg() -> CrazyflieTaskConfig:
    """Profile that enables a flow-deck-like planar velocity estimate."""
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=700,
        env_kwargs={
            "observation_mode": "flowdeck",
            "n_substeps": 10,
            "disturbance_sigma": 0.03,
            "actuator_noise_std": 0.015,
            "sensor_noise_scale": 1.0,
        },
    )


# ---------- Reach profiles ----------

def make_crazyflie_reach_cfg() -> CrazyflieTaskConfig:
    """Default Crazyflie reach profile."""
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=800,
    )


def make_crazyflie_reach_dense_stable_cfg() -> CrazyflieTaskConfig:
    """Stable baseline profile for PPO reach training."""
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=800,
        env_kwargs={
            "observation_mode": "state_estimate",
            "n_substeps": 10,
            "disturbance_sigma": 0.03,
            "actuator_noise_std": 0.015,
            "sensor_noise_scale": 1.0,
            "goal_xy_range": 0.55,
            "goal_z_range": (0.20, 0.65),
            "reach_threshold": 0.06,
            "reach_hold_steps": 15,
        },
    )


_CFG_FACTORIES: dict[str, Callable[[], CrazyflieTaskConfig]] = {
    "crazyflie_hover": make_crazyflie_hover_cfg,
    "crazyflie_hover_dense_stable": make_crazyflie_hover_dense_stable_cfg,
    "crazyflie_hover_flowdeck": make_crazyflie_hover_flowdeck_cfg,
    "crazyflie_reach": make_crazyflie_reach_cfg,
    "crazyflie_reach_dense_stable": make_crazyflie_reach_dense_stable_cfg,
}


def get_crazyflie_cfg(name: str) -> CrazyflieTaskConfig:
    """Build one named Crazyflie hover config profile."""
    if name not in _CFG_FACTORIES:
        raise ValueError(
            f"Unknown crazyflie cfg '{name}'. Available: {sorted(_CFG_FACTORIES)}"
        )
    return _CFG_FACTORIES[name]()


def list_crazyflie_cfgs() -> tuple[str, ...]:
    """List available Crazyflie profile names (hover + reach)."""
    return tuple(sorted(_CFG_FACTORIES.keys()))
