from gym.envs.registration import register

register(
    id="libero-object-v0",
    entry_point="libero2gym.libero_env:LiberoObjectEnv",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="libero-goal-v0",
    entry_point="libero2gym.libero_env:LiberoGoalEnv",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="libero-spatial-v0",
    entry_point="libero2gym.libero_env:LiberoSpatialEnv",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="libero-10-v0",
    entry_point="libero2gym.libero_env:Libero10Env",
    max_episode_steps=600,
    reward_threshold=1.0,
)

register(
    id="libero-90-v0",
    entry_point="libero2gym.libero_env:Libero90Env",
    max_episode_steps=300,
    reward_threshold=1.0,
)
