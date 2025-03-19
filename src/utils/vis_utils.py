""" This module contains utility functions used in various parts of the pipeline for visualization. """
from collections import OrderedDict
from copy import deepcopy
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from matplotlib import colors
from PIL import Image
from tqdm import tqdm

from src.utils.utils import get_render_settings, render_gaussian_model, torch2np

COLORS_ANSI = OrderedDict({
    "blue": "\033[94m",
    "orange": "\033[93m",
    "green": "\033[92m",
    "red": "\033[91m",
    "purple": "\033[95m",
    "brown": "\033[93m",  # No exact match, using yellow
    "pink": "\033[95m",
    "gray": "\033[90m",
    "olive": "\033[93m",  # No exact match, using yellow
    "cyan": "\033[96m",
    "end": "\033[0m",  # Reset color
})


COLORS_MATPLOTLIB = OrderedDict({
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'yellow-green': '#bcbd22',
    'cyan': '#17becf'
})


COLORS_MATPLOTLIB_RGB = OrderedDict({
    'blue': np.array([31, 119, 180]) / 255.0,
    'orange': np.array([255, 127,  14]) / 255.0,
    'green': np.array([44, 160,  44]) / 255.0,
    'red': np.array([214,  39,  40]) / 255.0,
    'purple': np.array([148, 103, 189]) / 255.0,
    'brown': np.array([140,  86,  75]) / 255.0,
    'pink': np.array([227, 119, 194]) / 255.0,
    'gray': np.array([127, 127, 127]) / 255.0,
    'yellow-green': np.array([188, 189,  34]) / 255.0,
    'cyan': np.array([23, 190, 207]) / 255.0
})


def get_color(color_name: str):
    """ Returns the RGB values of a given color name as a normalized numpy array.
    Args:
        color_name: The name of the color. Can be any color name from CSS4_COLORS.
    Returns:
        A numpy array representing the RGB values of the specified color, normalized to the range [0, 1].
    """
    if color_name == "custom_yellow":
        return np.asarray([255.0, 204.0, 102.0]) / 255.0
    if color_name == "custom_blue":
        return np.asarray([102.0, 153.0, 255.0]) / 255.0
    assert color_name in colors.CSS4_COLORS
    return np.asarray(colors.to_rgb(colors.CSS4_COLORS[color_name]))


def plot_ptcloud(point_clouds: Union[List, o3d.geometry.PointCloud], show_frame: bool = True):
    """ Visualizes one or more point clouds, optionally showing the coordinate frame.
    Args:
        point_clouds: A single point cloud or a list of point clouds to be visualized.
        show_frame: If True, displays the coordinate frame in the visualization. Defaults to True.
    """
    # rotate down up
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]
    if show_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        point_clouds = point_clouds + [mesh_frame]
    o3d.visualization.draw_geometries(point_clouds)


def draw_registration_result_original_color(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                                            transformation: np.ndarray):
    """ Visualizes the result of a point cloud registration, keeping the original color of the source point cloud.
    Args:
        source: The source point cloud.
        target: The target point cloud.
        transformation: The transformation matrix applied to the source point cloud.
    """
    source_temp = deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                             transformation: np.ndarray, source_color: str = "blue", target_color: str = "orange"):
    """ Visualizes the result of a point cloud registration, coloring the source and target point clouds.
    Args:
        source: The source point cloud.
        target: The target point cloud.
        transformation: The transformation matrix applied to the source point cloud.
        source_color: The color to apply to the source point cloud. Defaults to "blue".
        target_color: The color to apply to the target point cloud. Defaults to "orange".
    """
    source_temp = deepcopy(source)
    source_temp.paint_uniform_color(COLORS_MATPLOTLIB_RGB[source_color])

    target_temp = deepcopy(target)
    target_temp.paint_uniform_color(COLORS_MATPLOTLIB_RGB[target_color])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def plot_trajectory(pts, ax=None, color="green", label="None", title="3D Trajectory in 2D",
                    x_lim=None, y_lim=None):
    """ Plots a 2D trajectory from a 3D trajectory.
    Args:
        pts: The 3D trajectory to plot.
        ax: The matplotlib axis to plot on. If None, a new axis is created.
        color: The color of the trajectory.
        label: The label of the trajectory.
        title: The title of the plot.
        x_lim: The limits of the x-axis.
        y_lim: The limits of the y-axis.
    Returns:
        The matplotlib axis containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1], color=color, s=0.7)
    ax.plot(pts[:, 0], pts[:, 1], color=color, label=label, linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return ax


def plot_agents_pose_graph(agents_poses: dict, gt_poses: list,
                           intra_loops: list, inter_loops: list, output_path: str):
    """ Visualizes the submaps and the pose graph constraints between them.
    Args:
        agents_poses: An dict of agent poses: agent_id -> (N, 4, 4) N is the number of poses
        loops: A list of Registration objects
    """
    agent_colors = ["orange", "yellow-green", "pink", "cyan", "red"]
    agent_gt_colors = ["blue", "purple", "gray", "brown", "green"]
    _, ax = plt.subplots()
    for agent_id in sorted(agents_poses.keys()):
        ax = plot_trajectory(agents_poses[agent_id][:, :3, 3], ax=ax,
                             color=COLORS_MATPLOTLIB[agent_colors[agent_id]], label=f"Agent {agent_id} estimated")

    for agent_id in range(len(gt_poses)):
        ax = plot_trajectory(gt_poses[agent_id][:, :3, 3], ax=ax,
                             color=COLORS_MATPLOTLIB[agent_gt_colors[agent_id]], label=f"Agent {agent_id} ground-truth")

    no_label = True
    for inter_loop in inter_loops:
        x1, y1 = agents_poses[inter_loop.source_agent_id][inter_loop.source_frame_id, :3, 3][:2]
        x2, y2 = agents_poses[inter_loop.target_agent_id][inter_loop.target_frame_id, :3, 3][:2]
        if no_label:
            ax.plot([x1, x2], [y1, y2], color=COLORS_MATPLOTLIB["red"], linewidth=1.0, label="Inter-loops")
            no_label = False
        else:
            ax.plot([x1, x2], [y1, y2], color=COLORS_MATPLOTLIB["red"], linewidth=1.0)

    no_label = True
    for intra_loop in intra_loops:
        x1, y1 = agents_poses[intra_loop.source_agent_id][intra_loop.source_frame_id, :3, 3][:2]
        x2, y2 = agents_poses[intra_loop.target_agent_id][intra_loop.target_frame_id, :3, 3][:2]
        if no_label:
            ax.plot([x1, x2], [y1, y2], color=COLORS_MATPLOTLIB["cyan"], linewidth=1.0, label="Intra-loops")
            no_label = False
        else:
            ax.plot([x1, x2], [y1, y2], color=COLORS_MATPLOTLIB["cyan"], linewidth=1.0)

    ax.legend()
    plt.savefig(output_path)


def tensor_to_image(image_tensor: torch.Tensor) -> Image:
    """ Converts a tensor to a PIL Image.
    Args:
        image_tensor: The tensor image to save.
    Returns:
        image: The PIL Image.
    """
    image_tensor = image_tensor.clone().detach().cpu()
    image_tensor = image_tensor.permute(1, 2, 0).numpy()
    image = Image.fromarray((image_tensor * 255).astype(np.uint8))
    return image


def gaussian_model_to_mesh(gaussian_model, c2ws: list, height: int, width: int, intrinsics: np.ndarray):
    """ Converts a Gaussian model to an Open3D mesh.
    Args:
        gaussian_model: The Gaussian model.
        c2ws: The camera-to-world camera poses
        height: The height of the image.
        width: The width of the image.
        intrinsics: The pinhole camera intrinsics.
    Returns:
        o3d_mesh: The Open3D mesh.
    """

    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for c2w in tqdm(c2ws):

        w2c = np.linalg.inv(c2w)

        render_dict = render_gaussian_model(gaussian_model, get_render_settings(width, height, intrinsics, w2c))
        rendered_color, rendered_depth = render_dict["color"].detach(), render_dict["depth"][0].detach()
        rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)

        rendered_color = (torch2np(rendered_color.permute(1, 2, 0)) * 255).astype(np.uint8)
        rendered_depth = torch2np(rendered_depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
            o3d.geometry.Image(rendered_depth),
            depth_scale=1.0,
            depth_trunc=30,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, o3d_intrinsics, w2c)
    o3d_mesh = volume.extract_triangle_mesh()
    return o3d_mesh
