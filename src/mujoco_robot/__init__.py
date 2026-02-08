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
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Gymnasium environment registration
# ---------------------------------------------------------------------------
import gymnasium

gymnasium.register(
    id="MuJoCoRobot/Reach-v0",
    entry_point="mujoco_robot.envs.reach_env:ReachGymnasium",
    max_episode_steps=375,
)
