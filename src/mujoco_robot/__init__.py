"""MuJoCo Robot — modular RL environments for Universal Robots arms.

Quick start::

    from mujoco_robot.tasks.reach import ReachJointPosEnv, ReachJointPosGymnasium

    # Raw environment (no Gymnasium dependency)
    env = ReachJointPosEnv(robot="ur5e")
    obs = env.reset()
    result = env.step(env.sample_action())

    # Gymnasium wrapper (for Stable-Baselines3 / VecEnv)
    gym_env = ReachJointPosGymnasium(robot="ur5e")
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
    from mujoco_robot.tasks.reach import ReachIKRelEnv, ReachIKRelGymnasium
"""

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Gymnasium environment registration
# ---------------------------------------------------------------------------
import gymnasium

# Reach IDs now route through the manager-based runtime.
gymnasium.register(
    id="MuJoCoRobot/Reach-v0",
    entry_point=(
        "mujoco_robot.tasks.reach.reach_env:"
        "ReachManagerBasedRLEnv"
    ),
    kwargs={"cfg": "ur3e_joint_pos_dense_stable"},
    max_episode_steps=360,
)

# Reach control variants
gymnasium.register(
    id="MuJoCoRobot/Reach-IK-Rel-v0",
    entry_point=(
        "mujoco_robot.tasks.reach.reach_env:"
        "ReachManagerBasedRLEnv"
    ),
    kwargs={"cfg": "ur3e_ik_rel"},
    max_episode_steps=360,
)

gymnasium.register(
    id="MuJoCoRobot/Reach-IK-Abs-v0",
    entry_point=(
        "mujoco_robot.tasks.reach.reach_env:"
        "ReachManagerBasedRLEnv"
    ),
    kwargs={"cfg": "ur3e_ik_abs"},
    max_episode_steps=360,
)

gymnasium.register(
    id="MuJoCoRobot/Reach-Joint-Pos-v0",
    entry_point=(
        "mujoco_robot.tasks.reach.reach_env:"
        "ReachManagerBasedRLEnv"
    ),
    kwargs={"cfg": "ur3e_joint_pos_dense_stable"},
    max_episode_steps=360,
)

gymnasium.register(
    id="MuJoCoRobot/Reach-Joint-Pos-Isaac-Reward-v0",
    entry_point=(
        "mujoco_robot.tasks.reach.reach_env:"
        "ReachManagerBasedRLEnv"
    ),
    kwargs={
        "cfg": "ur3e_joint_pos_dense_stable",
        "control_variant": "joint_pos_isaac_reward",
    },
    max_episode_steps=360,
)

gymnasium.register(
    id="MuJoCoRobot/Slot-Sorter-v0",
    entry_point="mujoco_robot.tasks.slot_sorter.slot_sorter_env:SlotSorterGymnasium",
    max_episode_steps=400,
)

gymnasium.register(
    id="MuJoCoRobot/Lift-Suction-v0",
    entry_point="mujoco_robot.tasks.lift_suction.lift_suction_env:LiftSuctionGymnasium",
    max_episode_steps=300,
)

gymnasium.register(
    id="MuJoCoRobot/Suction-Contact-v0",
    entry_point=(
        "mujoco_robot.tasks.lift_suction.lift_suction_env:"
        "LiftSuctionContactGymnasium"
    ),
    max_episode_steps=200,
)
