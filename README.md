# libero2gym

A simple Gym wrapper for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark.

## Key Features

  - **Standard Gym Interface**: Simplifies environment setup by eliminating the complex configuration steps required by the original LIBERO library.
  - **Asynchronous Parallelization**: Supports efficient, large-scale rollouts and evaluations using an asynchronous vectorized environment.
  - **Enhanced Observations**: Provides properly oriented RGB, Depth, and Point Cloud observations out-of-the-box. Images are correctly flipped and ready to use.
  - **Ready-to-Use States**: Offers concatenated proprioceptive states (e.g., for models like OpenVLA and PI-0), removing the need for manual data processing and concatenation.

## Installation

```bash
pip install libero2gym
```

Or you can install from source. First, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/ZibinDong/libero2gym
cd libero2gym
pip install -e .
```

If you require point cloud observations, you must install additional dependencies for preprocessing:

```bash
pip install -e libero2gym[pc]
pip install -e .[pc]. # If installing from source
```

## Usage

### Single Environment

```python
import gym
import libero2gym

# Create the first task from the LIBERO-Goal suite
env = gym.make("libero-goal-v0", task_id=0) 

# Reset the environment to the first of LIBERO's 50 predefined initial states.
# If `init_state_id` is not specified, a random from the predefined states will be chosen.
obs = env.reset(init_state_id=0)  

# Sample a random action
act = env.action_space.sample()  
next_obs, reward, done, info = env.step(act)
```

### Multiple Environments (Vectorized)

```python
from functools import partial
import gym
import libero2gym
from libero2gym.vector_env import AsyncVectorEnv

# Define environment configuration
env_kwargs = {"task_id": 0}

# Create partial functions for the real and dummy environments
env_fn = partial(gym.make, id="libero-goal-v0", **env_kwargs)
dummy_env_fn = partial(gym.make, id="libero-goal-v0", dummy=True, **env_kwargs)

# Initialize 8 parallel environments
envs = AsyncVectorEnv(
    env_fns=[env_fn for _ in range(8)],
    dummy_env_fn=dummy_env_fn,
)

# Reset all 8 environments, each with a different initial state
obs = envs.reset(options=[{"init_state_id": i} for i in range(8)])

# Sample a single action to be applied to all environments
act = env.action_space.sample() 
next_obs, reward, done, info = envs.step(act)
```

### Environment Kwargs

You can customize the environment by passing keyword arguments to `gym.make()`:

```python
env_kwargs = {
    # Specify the task ID (from the task suite)
    "task_id": 0,    
    # Image resolution; returns a uint8 array of shape (H, W, 3) for RGB
    "image_size": 224,
    # If True, returns a float32 depth array of shape (H, W)
    "require_depth": False,
    # Maximum depth value; values beyond this are clipped
    "depth_clip": 2.0,
    # If True, returns a float32 point cloud array of shape (N, 3)
    "require_point_cloud": False,
    # Number of points (N) in the point cloud
    "num_points": 8192,
    # If True, returns detailed robot and object states. 
    # The default concatenated state of shape (9,) is usually sufficient.
    "require_detailed_states": False,
    # List of camera names to use; defaults to a dual-camera setup
    "camera_names": ["agentview", "robot0_eye_in_hand"],
    # Max steps per episode. Defaults to 600 for LIBERO-10 and 300 for others.
    "max_episode_steps": 300,
    # Random seed for the environment
    "seed": 0,
    # Device for point cloud preprocessing ("cpu" or "cuda:x")
    "pointcloud_process_device": "cpu",
}

env = gym.make("libero-goal-v0", **env_kwargs)
```
