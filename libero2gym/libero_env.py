import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Optional

import gym
import libero
import numpy as np
from gym.spaces import Box, Dict
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_real_depth_map
from termcolor import cprint

ROOT_PATH = Path(os.path.dirname(libero.libero.__file__))


def create_libero_env(
    task_suite_name: Literal[
        "libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90"
    ],
    task_id: int = 0,  # from 0 to 9
    image_size: int = 224,
    camera_names: List[str] = ["agentview", "robot0_eye_in_hand"],
    require_depth: bool = False,
    seed: int = 42,
):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = ROOT_PATH / "bddl_files" / task.problem_folder / task.bddl_file

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": image_size,
        "camera_widths": image_size,
        "camera_depths": require_depth,
        "camera_names": camera_names,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    init_states = task_suite.get_task_init_states(task_id)

    return env, init_states


class LiberoEnv(gym.Env):
    TASK_SUITE_NAME = ""  # can be "libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90"

    def __init__(
        self,
        dummy: bool = False,
        task_id: int = 0,  # from 0 to 9
        image_size: int = 224,
        require_depth: bool = False,
        require_point_cloud: bool = False,
        require_detailed_states: bool = False,
        num_points: int = 8192,
        camera_names: list = ["agentview", "robot0_eye_in_hand"],
        max_episode_steps: int = 600,
        depth_clip: float = 2.0,
        seed: int = 0,
        pointcloud_process_device: str = "cpu",
    ):
        super().__init__()
        self.dummy = dummy
        self.require_point_cloud = require_point_cloud
        self.require_depth = require_depth or require_point_cloud
        self.pointcloud_process_device = pointcloud_process_device
        self.num_points = num_points
        self.depth_clip = depth_clip

        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

        self.require_detailed_states = require_detailed_states
        self.camera_names = camera_names
        self.image_size = image_size

        if dummy:
            # self.env, self.init_states = None, None
            # self.language_instruction = None
            self.env, self.init_states = create_libero_env(
                task_suite_name=self.TASK_SUITE_NAME,
                task_id=task_id,
                image_size=image_size,
                camera_names=[],
                require_depth=require_depth,
                seed=seed,
            )
            self.language_instruction = self.env.language_instruction
        else:
            self.env, self.init_states = create_libero_env(
                task_suite_name=self.TASK_SUITE_NAME,
                task_id=task_id,
                image_size=image_size,
                camera_names=camera_names,
                require_depth=require_depth,
                seed=seed,
            )
            self.language_instruction = self.env.language_instruction
            self._build_3d_requirements()

        self._build_obs_act_spaces()

    def _build_obs_act_spaces(self):
        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        observation_space = dict()
        observation_space["state"] = Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        for cam in self.camera_names:
            observation_space[f"{cam}_image"] = Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8,
            )
            if self.require_depth:
                observation_space[f"{cam}_depth"] = Box(
                    low=0,
                    high=self.depth_clip,
                    shape=(self.image_size, self.image_size),
                    dtype=np.float32,
                )
            if self.require_point_cloud:
                observation_space[f"{cam}_pointcloud"] = Box(
                    low=-1.0,
                    high=1.6,
                    shape=(self.num_points, 3),
                    dtype=np.float32,
                )
        if self.require_detailed_states:
            example_obs = self.reset()
            for key, value in example_obs.items():
                if "image" in key:
                    observation_space[key] = Box(
                        low=0, high=255, shape=value.shape, dtype=np.uint8
                    )
                elif "depth" in key:
                    observation_space[key] = Box(
                        low=0, high=2.0, shape=value.shape, dtype=value.dtype
                    )
                elif "pointcloud" in key:
                    observation_space[key] = Box(
                        low=-1.0, high=1.6, shape=value.shape, dtype=value.dtype
                    )
                else:
                    observation_space[key] = Box(
                        low=-np.inf, high=np.inf, shape=value.shape, dtype=value.dtype
                    )

        self.observation_space = Dict(observation_space)

    def _build_3d_requirements(self):
        if self.require_point_cloud:
            import open3d as o3d

            def cammat2o3d(cam_mat, width, height):
                cx = cam_mat[0, 2]
                fx = cam_mat[0, 0]
                cy = cam_mat[1, 2]
                fy = cam_mat[1, 1]

                return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

            self.cam_intrinsics = dict()
            for camera_name in self.camera_names:
                cammat = get_camera_intrinsic_matrix(
                    self.sim, camera_name, self.image_size, self.image_size
                )
                self.cam_intrinsics[camera_name] = cammat2o3d(
                    cammat, self.image_size, self.image_size
                )

            self.pcd_bb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(-1.0, -1.0, 0.0), max_bound=(1.0, 1.0, 1.6)
            )

            try:
                from pytorch3d.ops import sample_farthest_points

                self.fps = sample_farthest_points
            except ImportError:
                cprint(
                    "Pytorch3d not installed. Please install pytorch3d.",
                    color="red",
                    attrs=["bold"],
                )
                self.fps = None

    @property
    def num_init_states(self):
        return len(self.init_states)

    def check_success(self):
        return self.env.env._check_success()

    @property
    def _visualizations(self):
        return self.env.env._visualizations

    @property
    def robots(self):
        return self.env.env.robots

    @property
    def sim(self):
        return self.env.env.sim

    def get_sim_state(self):
        return self.env.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.get_observations()

    def get_observations(self):
        raw_obs = self.env.env._get_observations()
        obs = raw_obs if self.require_detailed_states else dict()

        obs["state"] = np.concatenate(
            [
                raw_obs["robot0_gripper_qpos"],
                raw_obs["robot0_eef_pos"],
                raw_obs["robot0_eef_quat"],
            ]
        )

        if not self.dummy:
            for camera_name in self.camera_names:
                # flip to the correct orientation and transpose to (C, H, W)
                obs[f"{camera_name}_image"] = raw_obs[f"{camera_name}_image"][::-1]

            if self.require_depth:
                for camera_name in self.camera_names:
                    # convert to metric depth, flip to the correct orientation and reshape to (H, W)
                    depth = raw_obs[f"{camera_name}_depth"]
                    metric_depth = get_real_depth_map(self.sim, depth)
                    metric_depth = np.clip(metric_depth, 0.0, self.depth_clip)[::-1][
                        :, :, 0
                    ].astype(np.float32)
                    obs[f"{camera_name}_depth"] = metric_depth

            if self.require_point_cloud:
                import open3d as o3d
                import torch

                for camera_name in self.camera_names:
                    voxel_size = 0.003 if "eye_in_hand" in camera_name else 0.005

                    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d.geometry.Image(
                            np.copy(raw_obs[f"{camera_name}_depth"][:, :, None])
                        ),
                        self.cam_intrinsics[camera_name],
                    )
                    o3d_cloud = o3d_cloud.crop(self.pcd_bb)
                    o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size=voxel_size)

                    if self.fps is not None:
                        pointcloud = np.asarray(o3d_cloud.points)
                        pointcloud = torch.tensor(
                            pointcloud, device=self.pointcloud_process_device
                        )
                        pointcloud = self.fps(pointcloud[None], K=self.num_points)[0]
                        pointcloud = pointcloud[0].cpu().numpy()
                    else:
                        o3d_cloud = o3d_cloud.farthest_point_down_sample(
                            num_samples=self.num_points
                        )
                        pointcloud = np.asarray(o3d_cloud.points)

                    if pointcloud.shape[0] < self.num_points:
                        pointcloud = np.concatenate(
                            [
                                pointcloud,
                                pointcloud[
                                    np.random.choice(
                                        pointcloud.shape[0],
                                        self.num_points - pointcloud.shape[0],
                                        replace=False,
                                    )
                                ],
                            ],
                            axis=0,
                        )

                    obs[f"{camera_name}_pointcloud"] = pointcloud

        return OrderedDict(obs)

    def close(self):
        if self.env is not None:
            self.env.env.close()
            del self.env.env

    def reset(self, init_state_id: Optional[int] = None):
        if self.env is None:
            cprint(
                "Dummy Env: reset() does nothing. Please create a non-dummy env to use it.",
                color="red",
                attrs=["bold"],
            )
            return dict()

        if init_state_id is None:
            init_state_id = np.random.randint(self.num_init_states)
        self.env.reset()
        obs = self.set_init_state(self.init_states[init_state_id])
        self.episode_steps = 0
        return obs

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.get_observations()
        info["language_instruction"] = self.language_instruction
        self.episode_steps += 1
        done = done or self.episode_steps >= self.max_episode_steps
        return obs, reward, done, info


class LiberoObjectEnv(LiberoEnv):
    TASK_SUITE_NAME = "libero_object"


class LiberoGoalEnv(LiberoEnv):
    TASK_SUITE_NAME = "libero_goal"


class LiberoSpatialEnv(LiberoEnv):
    TASK_SUITE_NAME = "libero_spatial"


class Libero10Env(LiberoEnv):
    TASK_SUITE_NAME = "libero_10"


class Libero90Env(LiberoEnv):
    TASK_SUITE_NAME = "libero_90"


if __name__ == "__main__":
    from functools import partial

    from vector_env import AsyncVectorEnv

    env_fn = partial(LiberoGoalEnv, task_id=0, dummy=False)
    dummy_env_fn = partial(LiberoGoalEnv, task_id=0, dummy=True)

    envs = AsyncVectorEnv(
        env_fns=[env_fn for _ in range(2)],
        dummy_env_fn=dummy_env_fn,
        shared_memory=False,
    )

    a = envs.action_space.sample()

    o, r, d, info = envs.step(a)

    print(o["state"])

    envs.close()
