""" This module includes the Mapper class. The majority of this functionality is taken from:
    https://github.com/VladimirYugay/Gaussian-SLAM/blob/main/src/entities/mapper.py
"""
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

from src.entities.arguments import OptimizationParams
from src.entities.datasets import BaseDataset
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.utils import (get_render_settings, np2ptcloud, np2torch,
                             render_gaussian_model, torch2np)
from src.utils.mapping_eval import calc_psnr
from src.utils.mapping_utils import (create_point_cloud, geometric_edge_mask,
                                     sample_pixels_based_on_gradient)


class Mapper(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Sets up the mapper parameters
        Args:
            config: configuration of the mapper
            dataset: The dataset object used for extracting camera parameters and reading the data
            logger: The logger object used for logging the mapping process and saving visualizations
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset
        self.iterations = config["iterations"]
        self.new_submap_iterations = config["new_submap_iterations"]
        self.new_submap_points_num = config["new_submap_points_num"]
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.alpha_thre = config["alpha_thre"]
        self.pruning_thre = config["pruning_thre"]
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.keyframes = []

    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded
        based on alpha masks or color gradient
        Args:
            gaussian_model: The current submap
            keyframe (dict): Keyframe dict containing color, depth, and render settings
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        seeding_mask = None
        if new_submap:
            color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        else:
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)
            gt_depth_tensor = keyframe["depth"][None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch2np(seeding_mask[0])
        return seeding_mask

    def seed_new_gaussians(self, gt_color: np.ndarray, gt_depth: np.ndarray, intrinsics: np.ndarray,
                           estimate_c2w: np.ndarray, seeding_mask: np.ndarray, is_new_submap: bool) -> np.ndarray:
        """
        Seeds means for the new 3D Gaussian based on ground truth color and depth, camera intrinsics,
        estimated camera-to-world transformation, a seeding mask, and a flag indicating whether this is a new submap.
        Args:
            gt_color: The ground truth color image as a numpy array with shape (H, W, 3).
            gt_depth: The ground truth depth map as a numpy array with shape (H, W).
            intrinsics: The camera intrinsics matrix as a numpy array with shape (3, 3).
            estimate_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            is_new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns:
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 3)

        """
        pts = create_point_cloud(gt_color, 1.005 * gt_depth, intrinsics, estimate_c2w)
        flat_gt_depth = gt_depth.flatten()
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in gt_depth
        valid_ids = np.flatnonzero(seeding_mask)
        if is_new_submap:
            uniform_ids = np.random.choice(pts.shape[0], self.new_submap_points_num, replace=False)
            gradient_ids = sample_pixels_based_on_gradient(gt_color, self.new_submap_gradient_points_num)
            combined_ids = np.concatenate((uniform_ids, gradient_ids))
            combined_ids = np.concatenate((combined_ids, valid_ids))
            sample_ids = np.unique(combined_ids)
        else:
            sample_ids = valid_ids
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]
        return pts[sample_ids, :].astype(np.float32)

    def optimize_submap(self, keyframes: list, gaussian_model: GaussianModel, iterations: int = 100) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.
        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """

        losses_dict = {}
        start_time = time.time()
        for iteration in range(iterations):
            for frame_id, keyframe in keyframes:
                gaussian_model.optimizer.zero_grad(set_to_none=True)
                render_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"])

                image, depth = render_pkg["color"], render_pkg["depth"]
                gt_image = keyframe["color"]
                gt_depth = keyframe["depth"]

                mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
                color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                    image[:, mask], gt_image[:, mask]) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                depth_loss = l1_loss(depth[:, mask], gt_depth[mask])
                reg_loss = isotropic_loss(gaussian_model.get_scaling())
                total_loss = color_loss + depth_loss + reg_loss
                total_loss.backward()

                losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                         "depth_loss": depth_loss.item(),
                                         "total_loss": total_loss.item()}

                with torch.no_grad():

                    if iteration == iterations // 2 or iteration == iterations:
                        prune_mask = (gaussian_model.get_opacity() < self.pruning_thre).squeeze()
                        gaussian_model.prune_points(prune_mask)

                    # Optimizer step
                    if iteration < iterations:
                        gaussian_model.optimizer.step()
                    gaussian_model.optimizer.zero_grad(set_to_none=True)

        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict

    def grow_submap(self, pts: np.ndarray, gaussian_model: GaussianModel) -> int:
        """
        Expands the submap by integrating new points from the current keyframe
        Args:
            gt_depth: The ground truth depth map for the current keyframe, as a 2D numpy array.
            estimate_c2w: The estimated camera-to-world transformation matrix for the current keyframe of shape (4x4)
            gaussian_model (GaussianModel): The Gaussian model representing the current state of the submap.
            pts: The current set of 3D points in the keyframe of shape (N, 3)
            filter_cloud: A boolean flag indicating whether to apply filtering to the point cloud to remove
                outliers or noise before integrating it into the map.
        Returns:
            int: The number of points added to the submap
        """
        cloud_to_add = np2ptcloud(pts[:, :3], pts[:, 3:] / 255.0)
        gaussian_model.add_points(cloud_to_add)
        gaussian_model._features_dc.requires_grad = False
        gaussian_model._features_rest.requires_grad = False
        print("Gaussian model size", gaussian_model.get_size())
        return pts.shape[0]

    def map(self, frame_id: int, estimate_c2w: np.ndarray, gaussian_model: GaussianModel, is_new_submap: bool,
            max_iterations: int) -> dict:
        """ Calls out the mapping process described in paragraph 3.2
        The process goes as follows: seed new gaussians -> add to the submap -> optimize the submap
        Args:
            frame_id: current keyframe id
            estimate_c2w (np.ndarray): The estimated camera-to-world transformation matrix of shape (4x4)
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        Returns:
            opt_dict: Dictionary with statistics about the optimization process
        """

        _, gt_color, gt_depth, _ = self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimate_c2w)

        color_transform = torchvision.transforms.ToTensor()
        keyframe = {
            "color": color_transform(gt_color).cuda(),
            "depth": np2torch(gt_depth, device="cuda"),
            "render_settings": get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, estimate_w2c)}

        seeding_mask = self.compute_seeding_mask(gaussian_model, keyframe, is_new_submap)
        pts = self.seed_new_gaussians(
            gt_color, gt_depth, self.dataset.intrinsics, estimate_c2w, seeding_mask, is_new_submap)

        new_pts_num = self.grow_submap(pts, gaussian_model)

        start_time = time.time()
        opt_dict = self.optimize_submap([(frame_id, keyframe)] + self.keyframes, gaussian_model, max_iterations)
        optimization_time = time.time() - start_time
        print("Optimization time: ", optimization_time)

        self.keyframes.append((frame_id, keyframe))

        # Visualise the mapping for the current frame
        with torch.no_grad():
            render_pkg_vis = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]
            psnr_value = calc_psnr(image_vis, keyframe["color"]).mean().item()
            opt_dict["psnr_render"] = psnr_value
            print(f"PSNR this frame: {psnr_value}")
            if self.config["visualize"]:
                self.logger.vis_mapping_iteration(
                    frame_id, max_iterations,
                    image_vis.clone().detach().permute(1, 2, 0),
                    depth_vis.clone().detach().permute(1, 2, 0),
                    keyframe["color"].permute(1, 2, 0),
                    keyframe["depth"].unsqueeze(-1),
                    seeding_mask=seeding_mask)

        # Log the mapping numbers for the current frame
        self.logger.log_mapping_iteration(
            frame_id=frame_id, new_pts_num=new_pts_num,
            gaussian_model_size=gaussian_model.get_size(), optimization_time=opt_dict["optimization_time"],
            psnr=opt_dict["psnr_render"])
        return opt_dict
