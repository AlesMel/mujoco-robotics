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
    """High-level reward configuration for reach default terms."""

    position_error_weight: float = -0.2
    position_tanh_weight: float = 0.1
    position_tanh_std: float = 0.1
    include_orientation: bool = True
    orientation_error_weight: float = -0.2
    orientation_tanh_weight: float = 0.05
    orientation_tanh_std: float = 0.2
    include_action_rate: bool = True
    action_rate_weight: float | WeightFn = -0.0001
    include_joint_vel: bool = True
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
        ObservationTermCfg(name="pose_command", fn=observations.generated_commands_ee_pose),
        ObservationTermCfg(name="actions", fn=observations.last_action),
    )


def default_reward_terms(
    reward_cfg: ReachRewardCfg | None = None,
) -> tuple[RewardTermCfg, ...]:
    cfg = reward_cfg or ReachRewardCfg()

    if abs(float(cfg.position_tanh_std) - 0.1) < 1e-12:
        position_tanh_fn = rewards.position_command_error_tanh
    else:
        position_tanh_fn = rewards.position_command_error_tanh_with_std(cfg.position_tanh_std)

    terms: list[RewardTermCfg] = [
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
    ]

    orientation_weight = float(cfg.orientation_error_weight)
    if cfg.include_orientation or abs(orientation_weight) > 1e-12:
        terms.append(
            RewardTermCfg(
                name="end_effector_orientation_tracking",
                fn=rewards.orientation_command_error,
                weight=orientation_weight,
            )
        )
        # ori_tanh_weight = float(cfg.orientation_tanh_weight)
        # if abs(ori_tanh_weight) > 1e-12:
        #     if abs(float(cfg.orientation_tanh_std) - 0.2) < 1e-12:
        #         ori_tanh_fn = rewards.orientation_error_tanh
        #     else:
        #         ori_tanh_fn = rewards.orientation_error_tanh_with_std(cfg.orientation_tanh_std)
        #     terms.append(
        #         RewardTermCfg(
        #             name="end_effector_orientation_tracking_fine_grained",
        #             fn=ori_tanh_fn,
        #             weight=ori_tanh_weight,
        #         )
        #     )

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
    return ReachMDPCfg(
        action_term=action_term
        or ActionTermCfg(name="variant_apply_action", fn=actions.call_variant_apply_action),
        command_term=command_term or default_command_term(),
        observation_terms=default_observation_terms(),
        reward_terms=default_reward_terms(reward_cfg=reward_cfg),
        success_term=default_success_term(),
        failure_term=default_failure_term(),
        timeout_term=default_timeout_term(),
        include_reward_terms_in_info=False,
    )
