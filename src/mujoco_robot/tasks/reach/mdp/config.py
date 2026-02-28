"""Default MDP configs for manager-based reach environments."""
from __future__ import annotations

from dataclasses import dataclass

from . import actions, commands, observations, rewards, terminations
from .terms import (
    ActionTermCfg,
    CommandTermCfg,
    ObservationTermCfg,
    ReachMDPCfg,
    RewardTermCfg,
    TerminationTermCfg,
    WeightFn,
)


@dataclass(frozen=True)
class ReachRewardCfg:
    """High-level reward configuration for reach default terms.

    Two-scale position reward (``dense_bounded`` mode):
        A **broad** Gaussian (``dense_position_std``) provides gradient
        from far away (10–30 cm), while an optional **fine** Gaussian
        (``dense_position_fine_std``) adds a steep gradient in the last
        few centimetres.  ``dense_position_fine_weight`` controls how
        much of the total position budget goes to the fine term
        (0.0 = all broad, 1.0 = all fine).  Set ``dense_position_fine_std``
        to 0.0 to disable the fine term and use a single Gaussian.
    """

    reward_mode: str = "dense_bounded"
    dense_position_std: float = 0.15
    dense_position_fine_std: float = 0.025
    dense_position_fine_weight: float = 0.5
    dense_orientation_std: float = 0.3
    dense_orientation_fine_std: float = 0.15
    dense_orientation_fine_weight: float = 0.4
    dense_orientation_linear_weight: float = 0.2
    dense_position_weight: float = 0.5
    dense_orientation_weight: float = 0.5
    clip_to_unit_interval: bool = True

    # Legacy Isaac-style additive terms (used only when reward_mode="legacy_isaac")
    position_error_weight: float = -0.2
    position_tanh_weight: float = 0.1
    position_tanh_std: float = 0.1
    include_orientation: bool = True
    orientation_error_weight: float = -0.2
    orientation_tanh_weight: float = 0.05
    orientation_tanh_std: float = 0.2
    include_action_rate: bool = False
    action_rate_weight: float | WeightFn = -0.0001
    include_joint_vel: bool = False
    joint_vel_weight: float | WeightFn = -0.0001


def default_command_term() -> CommandTermCfg:
    return CommandTermCfg(
        name="ee_pose",
        fn=commands.uniform_pose_command,
        resampling_time_range_s=(4.0, 4.0),
    )


def default_observation_terms() -> tuple[ObservationTermCfg, ...]:
    return (
        ObservationTermCfg(name="joint_pos", fn=observations.joint_pos_rel),
        ObservationTermCfg(name="joint_vel", fn=observations.joint_vel_rel),
        ObservationTermCfg(name="ee_rot6d", fn=observations.ee_rot6d),
        ObservationTermCfg(name="goal_pos_base", fn=observations.goal_pos_base),
        ObservationTermCfg(name="goal_rot6d", fn=observations.goal_rot6d),
        ObservationTermCfg(name="ori_error_vec", fn=observations.orientation_error_vec),
        ObservationTermCfg(name="actions", fn=observations.last_action),
    )


def default_reward_terms(
    reward_cfg: ReachRewardCfg | None = None,
) -> tuple[RewardTermCfg, ...]:
    cfg = reward_cfg or ReachRewardCfg()

    terms: list[RewardTermCfg] = []
    reward_mode = str(cfg.reward_mode).strip().lower()

    if reward_mode == "dense_bounded":
        pos_w = max(0.0, float(cfg.dense_position_weight))
        ori_w = max(0.0, float(cfg.dense_orientation_weight)) if cfg.include_orientation else 0.0
        total_w = pos_w + ori_w
        if total_w <= 1e-12:
            pos_w, ori_w, total_w = 1.0, 0.0, 1.0
        pos_w /= total_w
        ori_w /= total_w

        # --- Two-scale position reward ---
        fine_std = float(getattr(cfg, "dense_position_fine_std", 0.0))
        fine_frac = float(getattr(cfg, "dense_position_fine_weight", 0.5))
        use_fine = fine_std > 1e-8 and fine_frac > 1e-8

        broad_fn = rewards.position_command_error_exp_with_std(cfg.dense_position_std)
        broad_w = pos_w * (1.0 - fine_frac) if use_fine else pos_w
        terms.append(
            RewardTermCfg(
                name="dense_position_proximity",
                fn=broad_fn,
                weight=broad_w,
            )
        )
        if use_fine:
            fine_fn = rewards.position_command_error_exp_with_std(fine_std)
            terms.append(
                RewardTermCfg(
                    name="dense_position_fine",
                    fn=fine_fn,
                    weight=pos_w * fine_frac,
                )
            )

        if ori_w > 1e-12:
            # --- Three-scale orientation reward ---
            ori_fine_std = float(getattr(cfg, "dense_orientation_fine_std", 0.0))
            ori_fine_frac = float(getattr(cfg, "dense_orientation_fine_weight", 0.4))
            ori_linear_frac = float(getattr(cfg, "dense_orientation_linear_weight", 0.0))
            use_ori_fine = ori_fine_std > 1e-8 and ori_fine_frac > 1e-8

            # Remaining budget for broad tanh after fine + linear are carved out.
            broad_frac = max(0.0, 1.0 - ori_fine_frac - ori_linear_frac) if (use_ori_fine or ori_linear_frac > 1e-8) else 1.0

            # Broad: tanh-based — provides gradient even at large errors (~π rad)
            broad_ori_fn = rewards.orientation_error_tanh_with_std(cfg.dense_orientation_std)
            terms.append(
                RewardTermCfg(
                    name="dense_orientation_proximity",
                    fn=broad_ori_fn,
                    weight=ori_w * broad_frac,
                )
            )
            if use_ori_fine:
                fine_ori_fn = rewards.orientation_command_error_exp_with_std(ori_fine_std)
                terms.append(
                    RewardTermCfg(
                        name="dense_orientation_fine",
                        fn=fine_ori_fn,
                        weight=ori_w * ori_fine_frac,
                    )
                )

            # --- Linear orientation term: constant gradient at ALL distances ---
            if ori_linear_frac > 1e-8:
                terms.append(
                    RewardTermCfg(
                        name="dense_orientation_linear",
                        fn=rewards.orientation_error_linear,
                        weight=ori_w * ori_linear_frac,
                    )
                )
    elif reward_mode == "legacy_isaac":
        if abs(float(cfg.position_tanh_std) - 0.1) < 1e-12:
            position_tanh_fn = rewards.position_command_error_tanh
        else:
            position_tanh_fn = rewards.position_command_error_tanh_with_std(cfg.position_tanh_std)

        terms.extend(
            (
                RewardTermCfg(
                    name="end_effector_position_tracking",
                    fn=rewards.position_command_error,
                    weight=float(cfg.position_error_weight),
                ),
                RewardTermCfg(
                    name="end_effector_position_tracking_fine_grained",
                    fn=position_tanh_fn,
                    weight=float(cfg.position_tanh_weight),
                ),
            )
        )

        orientation_weight = float(cfg.orientation_error_weight)
        if cfg.include_orientation or abs(orientation_weight) > 1e-12:
            terms.append(
                RewardTermCfg(
                    name="end_effector_orientation_tracking",
                    fn=rewards.orientation_command_error,
                    weight=orientation_weight,
                )
            )
    else:
        raise ValueError(
            "ReachRewardCfg.reward_mode must be 'dense_bounded' or 'legacy_isaac', "
            f"got '{cfg.reward_mode}'"
        )

    if cfg.include_action_rate:
        terms.append(
            RewardTermCfg(
                name="action_rate",
                fn=rewards.action_rate_l2,
                weight=cfg.action_rate_weight,
            )
        )
    if cfg.include_joint_vel:
        terms.append(
            RewardTermCfg(
                name="joint_vel",
                fn=rewards.joint_vel_l2,
                weight=cfg.joint_vel_weight,
            )
        )

    return tuple(terms)


def default_success_term() -> TerminationTermCfg:
    return TerminationTermCfg(name="success", fn=terminations.success)


def default_failure_term() -> TerminationTermCfg:
    return TerminationTermCfg(name="failure", fn=terminations.failure_self_collision)


def default_timeout_term() -> TerminationTermCfg:
    return TerminationTermCfg(name="time_out", fn=terminations.time_out)


def make_default_reach_mdp_cfg(
    action_term: ActionTermCfg | None = None,
    command_term: CommandTermCfg | None = None,
    reward_cfg: ReachRewardCfg | None = None,
) -> ReachMDPCfg:
    cfg = reward_cfg or ReachRewardCfg()
    return ReachMDPCfg(
        action_term=action_term
        or ActionTermCfg(name="variant_apply_action", fn=actions.call_variant_apply_action),
        command_term=command_term or default_command_term(),
        observation_terms=default_observation_terms(),
        reward_terms=default_reward_terms(reward_cfg=cfg),
        success_term=default_success_term(),
        failure_term=default_failure_term(),
        timeout_term=default_timeout_term(),
        include_reward_terms_in_info=False,
        reward_clip_to_unit_interval=bool(cfg.clip_to_unit_interval),
    )
