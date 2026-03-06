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


# ---------- Obstacle-avoidance reach profiles ----------

def make_crazyflie_obstacle_cfg() -> CrazyflieTaskConfig:
    """Default Crazyflie wall-maze obstacle-avoidance reach profile."""
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=800,
    )


def make_crazyflie_obstacle_dense_stable_cfg() -> CrazyflieTaskConfig:
    """Stable baseline profile for PPO wall-maze training.

    Five-component reward, each sub-component normalised to ~[-1, +1]:
    * R_progress — distance shaping + alignment + approach bonus
    * R_obstacle — worst-ray penalty (normalised zones)
    * R_energy   — thrust² + balance + velocity + acceleration
    * R_stability — angular vel + tilt + jerk (all clamped)
    * R_task     — goal bonus (+10 + time), collision (−10), timeout (−5)

    Total ≈ 5·R_progress + 0.5·R_obstacle + 0.2·R_energy
          + 0.3·R_stability + 1·R_task

    Per-step reward at stable hover toward goal ≈ +1 to +3.
    """
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=1200,
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
            # -- wall-maze layout --
            "n_wall_slots": 12,
            "n_barriers": (2, 3),
            "gap_width": 0.22,
            "partial_wall_prob": 0.30,
            "wall_spawn_clearance": 0.20,
            # Randomize per reset for better generalization.
            "fixed_layout": False,
            "ceiling_height": 0.75,
            "terminate_on_goal": False,
            # -- rangefinder --
            "rangefinder_mode": "lidar",
            "n_lidar_rays": 16,
            "rangefinder_max_range": 1.0,
            "rangefinder_noise_std": 0.02,
            # -- reward weights --
            "w_progress": 5.0,
            "w_obstacle": 0.5,
            "w_energy": 0.2,
            "w_stability": 0.3,
            "w_task": 1.0,
            # -- reward thresholds --
            "goal_bonus": 10.0,
            "time_bonus_max": 5.0,
            "collision_penalty": 20.0,
            "timeout_penalty": 5.0,
            "v_optimal": 0.5,
            "v_max": 2.0,
            "a_max": 15.0,
            # -- obstacle zones (normalised fractions of rangefinder range) --
            "obstacle_collision_zone": 0.08,
            "obstacle_danger_zone": 0.25,
            "obstacle_warning_zone": 0.50,
        },
    )


_CFG_FACTORIES: dict[str, Callable[[], CrazyflieTaskConfig]] = {
    "crazyflie_hover": make_crazyflie_hover_cfg,
    "crazyflie_hover_dense_stable": make_crazyflie_hover_dense_stable_cfg,
    "crazyflie_hover_flowdeck": make_crazyflie_hover_flowdeck_cfg,
    "crazyflie_reach": make_crazyflie_reach_cfg,
    "crazyflie_reach_dense_stable": make_crazyflie_reach_dense_stable_cfg,
    "crazyflie_obstacle": make_crazyflie_obstacle_cfg,
    "crazyflie_obstacle_dense_stable": make_crazyflie_obstacle_dense_stable_cfg,
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
