"""MuJoCo Robot — modular RL environments for Universal Robots arms.

Quick start::

    from mujoco_robot.envs import URReachEnv, ReachGymnasium

    # Raw environment (no Gymnasium dependency)
    env = URReachEnv(robot="ur5e")
    obs = env.reset()
    result = env.step(env.sample_action())

    # Gymnasium wrapper (for Stable-Baselines3 / VecEnv)
    gym_env = ReachGymnasium(robot="ur5e")
    obs, info = gym_env.reset()
    obs, reward, terminated, truncated, info = gym_env.step(gym_env.action_space.sample())

    # Via gymnasium.make (works with ANY Gymnasium-compatible library)
    import gymnasium
    env = gymnasium.make("MuJoCoRobot/Reach-v0", robot="ur3e")
    env_rel = gymnasium.make("MuJoCoRobot/Reach-IK-Rel-v0", robot="ur3e")
    env_abs = gymnasium.make("MuJoCoRobot/Reach-IK-Abs-v0", robot="ur3e")
    env_jpos = gymnasium.make("MuJoCoRobot/Reach-Joint-Pos-v0", robot="ur3e")
    env_jpos_isaac = gymnasium.make(
        "MuJoCoRobot/Reach-Joint-Pos-Isaac-Reward-v0", robot="ur3e"
    )

    # Or import the modular variant directly (Isaac Lab style):
    from mujoco_robot.envs.reach import ReachIKRelEnv, ReachIKRelGymnasium
"""

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Gymnasium environment registration
# ---------------------------------------------------------------------------
import gymnasium

# Default (backward-compatible) — dispatches via control_variant kwarg
gymnasium.register(
    id="MuJoCoRobot/Reach-v0",
    entry_point="mujoco_robot.envs.reach_env:ReachGymnasium",
    max_episode_steps=375,
)

# Modular variants — each points directly to its Gymnasium class
gymnasium.register(
    id="MuJoCoRobot/Reach-IK-Rel-v0",
    entry_point="mujoco_robot.envs.reach.reach_env_ik_rel:ReachIKRelGymnasium",
    max_episode_steps=375,
)

gymnasium.register(
    id="MuJoCoRobot/Reach-IK-Abs-v0",
    entry_point="mujoco_robot.envs.reach.reach_env_ik_abs:ReachIKAbsGymnasium",
    max_episode_steps=375,
)

gymnasium.register(
    id="MuJoCoRobot/Reach-Joint-Pos-v0",
    entry_point="mujoco_robot.envs.reach.reach_env_joint_pos:ReachJointPosGymnasium",
    max_episode_steps=375,
)

gymnasium.register(
    id="MuJoCoRobot/Reach-Joint-Pos-Isaac-Reward-v0",
    entry_point=(
        "mujoco_robot.envs.reach.reach_env_joint_pos_isaac_reward:"
        "ReachJointPosIsaacRewardGymnasium"
    ),
    # Isaac-inspired joint-pos reward setup uses ~3.0 s episodes by default.
    max_episode_steps=94,
)
