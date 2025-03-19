""" This module contains the dataset classes for ReplicaMultiagent and AriaMultiagent datasets.
    We preprocess the AriaMultiagent dataset to follow the Replica dataset format.
    This allows for using the same codebase for both datasets.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision

from src.utils.utils import get_render_settings, np2torch, rgbd2ptcloud


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.frame_ids = []
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []
        self.poses = []
        self.color_transform = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        """ Returns the number of frames in the dataset. """
        return len(self.color_paths) if self.frame_limit < 0 else min(int(self.frame_limit), len(self.color_paths))

    def get_point_cloud(self, frame_id: int, pose=None) -> o3d.geometry.PointCloud:
        """ Get the point cloud of a frame.
        Args:
            frame_id: The frame id.
            pose: The camera-to-world pose of shape (4, 4).
        Returns:
            Posed open3d point cloud
        """
        _, gt_color, gt_depth, gt_pose = self[frame_id]
        if pose is None:
            pose = gt_pose.copy()
        return rgbd2ptcloud(gt_color, gt_depth, self.intrinsics, np.linalg.inv(pose))

    def get_render_frame(self, frame_id: int, pose=None) -> dict:
        """ Get the render frame dictionary suitable for GS renderer
        Args:
            frame_id: The frame id.
            pose: The camera-to-world pose of shape (4, 4).
        Returns:
            The render frame dictionary
        """
        _, gt_color, gt_depth, gt_pose = self[frame_id]
        if pose is None:
            pose = np.linalg.inv(gt_pose)
        return {
            "gt_color": self.color_transform(gt_color).cuda(),
            "gt_depth": np2torch(gt_depth).cuda(),
            "render_settings": get_render_settings(self.width, self.height, self.intrinsics, pose)
        }


class Replica(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        # CP-SLAM dataset has a bug in terms of number of poses
        num_loaded_frames = len(self)
        self.frame_ids = list(range(num_loaded_frames))
        self.poses = self.poses[:num_loaded_frames]
        print(f"Loaded {num_loaded_frames} frames")

    def load_poses(self, path: str) -> None:
        """ Load the camera-to-world poses from a file.
        Args:
            path: The path to the txt pose file.
        """
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index: int) -> tuple:
        """ Returns the color, depth, pose and frame id of a specific frame.
        Args:
            index: The index of the frame.
        Returns:
            frame_id: The frame id.
            color_data: The color image of shape (H, W, 3) in [0, 255]
            depth_data: The depth image of shape (H, W).
            pose: The camera-to-world pose of shape (4, 4).
        """
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return self.frame_ids[index], color_data, depth_data, self.poses[index]


def get_dataset(dataset_name: str) -> BaseDataset:
    """ Get the dataset class by name.
    Args:
        dataset_name: The name of the dataset.
    """
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "aria":
        return Replica
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
