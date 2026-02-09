"""Default MDP configs for manager-based reach environments."""
from __future__ import annotations

from . import actions, observations, rewards, terminations
from .terms import (
    ActionTermCfg,
    ObservationTermCfg,
    ReachMDPCfg,
    RewardTermCfg,
    TerminationTermCfg,
)


def default_observation_terms() -> tuple[ObservationTermCfg, ...]:
    return (
        ObservationTermCfg(name="joint_pos_rel", fn=observations.joint_pos_relative),
        ObservationTermCfg(name="joint_vel", fn=observations.joint_vel),
        ObservationTermCfg(name="pose_command", fn=observations.pose_command_base),
        ObservationTermCfg(name="last_action", fn=observations.last_action),
    )


def default_reward_terms() -> tuple[RewardTermCfg, ...]:
    return (
        RewardTermCfg(name="position_error_l2", fn=rewards.position_error_l2, weight=-0.2),
        RewardTermCfg(
            name="position_error_tanh",
            fn=rewards.position_error_tanh_std_01,
            weight=0.16,
        ),
        RewardTermCfg(name="orientation_error", fn=rewards.orientation_error, weight=-0.15),
        RewardTermCfg(
            name="orientation_error_tanh",
            fn=rewards.orientation_error_tanh_std_02,
            weight=0.1,
        ),
        RewardTermCfg(name="at_goal", fn=rewards.at_goal, weight=0.5),
        RewardTermCfg(
            name="action_rate_l2",
            fn=rewards.action_rate_l2,
            weight=rewards.action_rate_curriculum_weight,
        ),
        RewardTermCfg(
            name="joint_vel_l2",
            fn=rewards.joint_vel_l2,
            weight=rewards.joint_vel_curriculum_weight,
        ),
    )


def default_success_term() -> TerminationTermCfg:
    return TerminationTermCfg(name="success", fn=terminations.success)


def default_failure_term() -> TerminationTermCfg:
    return TerminationTermCfg(name="failure", fn=terminations.failure_self_collision)


def default_timeout_term() -> TerminationTermCfg:
    return TerminationTermCfg(name="time_out", fn=terminations.time_out)


def make_default_reach_mdp_cfg(
    action_term: ActionTermCfg | None = None,
) -> ReachMDPCfg:
    return ReachMDPCfg(
        action_term=action_term
        or ActionTermCfg(name="variant_apply_action", fn=actions.call_variant_apply_action),
        observation_terms=default_observation_terms(),
        reward_terms=default_reward_terms(),
        success_term=default_success_term(),
        failure_term=default_failure_term(),
        timeout_term=default_timeout_term(),
        include_reward_terms_in_info=False,
    )
