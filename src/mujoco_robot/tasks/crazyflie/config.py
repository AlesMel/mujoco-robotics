"""Crazyflie hover task configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict


_DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent.parent / "assets" / "crazyflie_2_1.xml"
)

# Structured goal pool for obstacle environments.
# Goals are spread across all quadrants and altitudes to ensure the agent
# trains on a variety of navigation scenarios (left/right turns, diagonal
# crossings, fly-over, fly-under).  Pool is shuffled each episode.
# Workspace: xy in [-0.55, 0.55], z in [0.20, 0.65].
OBSTACLE_GOAL_POOL = [
    (-0.45,  0.00, 0.40),   # cross left barrier, mid height
    ( 0.45,  0.00, 0.40),   # cross right barrier, mid height
    ( 0.00,  0.45, 0.55),   # cross top barrier, fly high (over)
    ( 0.00,  0.45, 0.22),   # cross top barrier, fly low (under)
    ( 0.00, -0.45, 0.40),   # cross bottom barrier, mid height
    (-0.40,  0.40, 0.50),   # diagonal left-top, high
    ( 0.40,  0.40, 0.22),   # diagonal right-top, low
    (-0.40, -0.40, 0.35),   # diagonal left-bottom, mid
    ( 0.40, -0.40, 0.55),   # diagonal right-bottom, high
]

# ---------------------------------------------------------------------------
# Static maze layout — 11 wall segments across 4 barriers.
#
# Arena: xy in [-0.55, 0.55], ceiling at 0.75 m.  Spawn at (0, 0, ~0.28).
# All "pos" and "size" values are MuJoCo half-extents.
#
#   Barrier A  Y-aligned at x = -0.27   fly-OVER gap  →  LEFT zone
#   Barrier B  X-aligned at y = +0.25   fly-UNDER gap →  TOP zone
#   Barrier C  Y-aligned at x = +0.26   CLEAN gap     →  RIGHT zone
#   Barrier D  X-aligned at y = -0.25   fly-OVER gap  →  BOTTOM zone
#
# Skills covered: climb-over (A, D), squeeze-under (B), pure lateral (C),
# altitude-then-navigate and navigate-then-altitude combinations.
# ---------------------------------------------------------------------------
STATIC_MAZE_WALLS = [
    # --- Barrier A : Y-aligned at x = -0.27 ---
    # Gap at y in [-0.11, +0.11].  Fly-over obstacle (wall top at 0.30 m):
    # drone must climb above 0.30 m before entering the gap.
    {"pos": [-0.27, -0.305, 0.375], "size": [0.015, 0.195, 0.375], "variant": "full"},
    {"pos": [-0.27,  0.000, 0.150], "size": [0.015, 0.090, 0.150], "variant": "fly_over"},
    {"pos": [-0.27,  0.305, 0.375], "size": [0.015, 0.195, 0.375], "variant": "full"},
    # --- Barrier B : X-aligned at y = +0.25 ---
    # Gap at x in [-0.11, +0.11].  Fly-under obstacle (wall bottom at 0.40 m):
    # drone must descend below 0.40 m before entering the gap.
    {"pos": [-0.305,  0.25, 0.375], "size": [0.195, 0.015, 0.375], "variant": "full"},
    {"pos": [ 0.000,  0.25, 0.550], "size": [0.090, 0.015, 0.150], "variant": "fly_under"},
    {"pos": [ 0.305,  0.25, 0.375], "size": [0.195, 0.015, 0.375], "variant": "full"},
    # --- Barrier C : Y-aligned at x = +0.26 ---
    # Clean gap at y in [-0.18, +0.04].  No partial wall — pure lateral nav.
    {"pos": [0.26, -0.340, 0.375], "size": [0.015, 0.160, 0.375], "variant": "full"},
    {"pos": [0.26,  0.270, 0.375], "size": [0.015, 0.230, 0.375], "variant": "full"},
    # --- Barrier D : X-aligned at y = -0.25 ---
    # Gap at x in [-0.09, +0.13].  Fly-over obstacle (wall top at 0.30 m).
    {"pos": [-0.295, -0.25, 0.375], "size": [0.205, 0.015, 0.375], "variant": "full"},
    {"pos": [ 0.020, -0.25, 0.150], "size": [0.090, 0.015, 0.150], "variant": "fly_over"},
    {"pos": [ 0.315, -0.25, 0.375], "size": [0.185, 0.015, 0.375], "variant": "full"},
]

# Static goal pool — two goals per barrier zone (mid + high/low) plus a
# ceiling hover in the centre.  Each goal trains a distinct skill.
STATIC_GOAL_POOL = [
    # LEFT zone  (Barrier A, fly-over gap — climb to z > 0.30 m first)
    (-0.43,  0.00, 0.40),   # A-mid : fly over obstacle, hold mid altitude
    (-0.43,  0.00, 0.60),   # A-high: fly over obstacle, then climb high
    # TOP zone   (Barrier B, fly-under gap — descend to z < 0.40 m first)
    ( 0.00,  0.43, 0.28),   # B-low : squeeze under elevated wall, stay low
    ( 0.00,  0.43, 0.55),   # B-high: squeeze under elevated wall, then climb
    # RIGHT zone (Barrier C, clean gap — pure lateral + altitude control)
    ( 0.43, -0.07, 0.40),   # C-mid : centre of gap, mid altitude
    ( 0.43, -0.07, 0.25),   # C-low : centre of gap, low altitude
    # BOTTOM zone(Barrier D, fly-over gap — climb to z > 0.30 m first)
    ( 0.00, -0.43, 0.40),   # D-mid : fly over obstacle, hold mid altitude
    ( 0.00, -0.43, 0.55),   # D-high: fly over obstacle, then climb
    # CENTER     (no barriers — pure altitude challenge near ceiling)
    ( 0.00,  0.00, 0.65),   # centre-top: reach near-ceiling altitude
]


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
        time_limit=1600,
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
        time_limit=2000,
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
            "w_progress": 4.0,
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
            # -- structured goal pool --
            "goal_pool": OBSTACLE_GOAL_POOL,
        },
    )


def make_crazyflie_obstacle_skrl_stable_cfg() -> CrazyflieTaskConfig:
    """Balanced reward profile for SKRL PPO wall-maze training.

    Compared with ``crazyflie_obstacle_dense_stable``:

    * ``w_progress`` reduced 5 → 3 so the agent does not rush into walls.
    * ``w_obstacle`` raised 0.5 → 1.5 for a stronger obstacle-avoidance signal.
    * ``w_stability`` raised 0.3 → 0.5 — a stable hover baseline helps navigation.
    * ``collision_penalty`` raised 20 → 30 — clearer hard-stop signal.
    * Danger/warning zones widened for earlier penalty activation.
    """
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=2000,
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
            "fixed_layout": False,
            "ceiling_height": 0.75,
            "terminate_on_goal": False,
            # -- rangefinder --
            "rangefinder_mode": "lidar",
            "n_lidar_rays": 16,
            "rangefinder_max_range": 1.0,
            "rangefinder_noise_std": 0.02,
            # -- reward weights (rebalanced for SKRL) --
            "w_progress": 4.0,
            "w_obstacle": 1.0,
            "w_energy": 0.2,
            "w_stability": 0.3,
            "w_task": 1.0,
            # -- reward thresholds --
            "goal_bonus": 10.0,
            "time_bonus_max": 5.0,
            "collision_penalty": 30.0,
            "timeout_penalty": 5.0,
            "v_optimal": 0.5,
            "v_max": 2.0,
            "a_max": 15.0,
            # -- obstacle zones (wider for earlier penalty activation) --
            "obstacle_collision_zone": 0.08,
            "obstacle_danger_zone": 0.30,
            "obstacle_warning_zone": 0.60,
            # -- structured goal pool --
            "goal_pool": OBSTACLE_GOAL_POOL,
        },
    )


def make_crazyflie_obstacle_fast_cfg() -> CrazyflieTaskConfig:
    """Fast iteration profile — halved substeps and lidar rays.

    Use when compute budget is tight or for rapid hyper-parameter search.
    Physics fidelity is slightly reduced but remains suitable for RL training:

    * ``n_substeps`` 10 → 5   — same 100 Hz control rate; fewer 1 ms sub-steps
      per decision.  Error over the 5 ms window is negligible.
    * ``n_lidar_rays`` 16 → 8 — observation shrinks to 36 dims (vs 45).
      8 rays at 45° spacing still give adequate obstacle coverage.

    Reward weights match ``crazyflie_obstacle_skrl_stable``.
    """
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=2000,
        env_kwargs={
            "observation_mode": "state_estimate",
            "n_substeps": 5,
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
            "fixed_layout": False,
            "ceiling_height": 0.75,
            "terminate_on_goal": False,
            # -- rangefinder (8 rays instead of 16) --
            "rangefinder_mode": "lidar",
            "n_lidar_rays": 8,
            "rangefinder_max_range": 1.0,
            "rangefinder_noise_std": 0.02,
            # -- reward weights (same as skrl_stable) --
            "w_progress": 4.0,
            "w_obstacle": 1.0,
            "w_energy": 0.2,
            "w_stability": 0.3,
            "w_task": 1.0,
            # -- reward thresholds --
            "goal_bonus": 10.0,
            "time_bonus_max": 5.0,
            "collision_penalty": 30.0,
            "timeout_penalty": 5.0,
            "v_optimal": 0.5,
            "v_max": 2.0,
            "a_max": 15.0,
            "obstacle_collision_zone": 0.08,
            "obstacle_danger_zone": 0.30,
            "obstacle_warning_zone": 0.60,
            # -- structured goal pool --
            "goal_pool": OBSTACLE_GOAL_POOL,
        },
    )


def make_crazyflie_obstacle_static_cfg() -> CrazyflieTaskConfig:
    """Fixed 4-barrier maze with a static goal pool.

    The maze never changes between episodes — the agent must generalise
    within one layout rather than across random ones.  All 9 goals are
    drawn from :data:`STATIC_GOAL_POOL`, cycling in shuffled order.

    Barrier layout (see :data:`STATIC_MAZE_WALLS` for exact geometry):

    * **A** Y-aligned x = -0.27  fly-OVER gap → LEFT zone
    * **B** X-aligned y = +0.25  fly-UNDER gap → TOP zone
    * **C** Y-aligned x = +0.26  clean gap → RIGHT zone
    * **D** X-aligned y = -0.25  fly-OVER gap → BOTTOM zone
    """
    return CrazyflieTaskConfig(
        model_path=_DEFAULT_MODEL,
        actuator_profile="crazyflie",
        time_limit=2000,
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
            # -- static maze --
            "n_wall_slots": 12,
            "fixed_layout": True,
            "maze_walls": STATIC_MAZE_WALLS,
            "ceiling_height": 0.75,
            "terminate_on_goal": False,
            # -- rangefinder --
            "rangefinder_mode": "lidar",
            "n_lidar_rays": 16,
            "rangefinder_max_range": 1.0,
            "rangefinder_noise_std": 0.02,
            # -- reward weights --
            "w_progress": 4.0,
            "w_obstacle": 1.0,
            "w_energy": 0.2,
            "w_stability": 0.3,
            "w_task": 1.0,
            # -- reward thresholds --
            "goal_bonus": 10.0,
            "time_bonus_max": 5.0,
            "collision_penalty": 30.0,
            "timeout_penalty": 5.0,
            "v_optimal": 0.5,
            "v_max": 2.0,
            "a_max": 15.0,
            "obstacle_collision_zone": 0.08,
            "obstacle_danger_zone": 0.30,
            "obstacle_warning_zone": 0.60,
            # -- static goal pool --
            "goal_pool": STATIC_GOAL_POOL,
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
    "crazyflie_obstacle_skrl_stable": make_crazyflie_obstacle_skrl_stable_cfg,
    "crazyflie_obstacle_fast": make_crazyflie_obstacle_fast_cfg,
    "crazyflie_obstacle_static": make_crazyflie_obstacle_static_cfg,
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
