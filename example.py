from functools import partial

import gym

import libero2gym  # noqa: F401
from libero2gym.vector_env import AsyncVectorEnv

if __name__ == "__main__":
    env_kwargs = {"task_id": 0}

    # single env example
    env = gym.make("libero-goal-v0", **env_kwargs)
    obs = env.reset()

    for k, v in obs.items():
        print(f"{k}: {v.shape}")

    # vectorized env example
    env_fn = partial(gym.make, id="libero-goal-v0", dummy=False, **env_kwargs)
    dummy_env_fn = partial(gym.make, id="libero-goal-v0", dummy=True, **env_kwargs)
    envs = AsyncVectorEnv(
        env_fns=[env_fn for _ in range(2)],
        dummy_env_fn=dummy_env_fn,
        shared_memory=False,
    )

    obs = envs.reset()
    for k, v in obs.items():
        print(f"{k}: {v.shape}")

    envs.close()
