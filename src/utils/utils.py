""" This module contains utility functions used in various parts of the pipeline. """
import copy
import os
import random

import numpy as np
import open3d as o3d
import torch
from gaussian_rasterizer import (GaussianRasterizationSettings,
                                 GaussianRasterizer)


def find_submap(frame_id: int, submaps: dict) -> dict:
    """ Finds the submap that starts with the given frame ID.
    Args:
        frame_id: The frame ID to search for.
        submaps: The dictionary of submaps to search in.
    Returns:
        The submap that contains the given frame ID.
    """
    for submap in submaps:
        if submap["submap_start_frame_id"] <= frame_id < submap["submap_end_frame_id"]:
            return submap
    return None


def setup_seed(seed: int) -> None:
    """ Sets the seed for generating random numbers to ensure reproducibility across multiple runs.
    Args:
        seed: The seed value to set for random number generators in torch, numpy, and random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.clone().detach().cpu().numpy()


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    """converts numpy array to point cloud
    Args:
        pts (ndarray): point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def get_render_settings(w, h, intrinsics, w2c, near=0.01, far=100, sh_degree=0):
    """
    Constructs and returns a GaussianRasterizationSettings object for rendering,
    configured with given camera parameters.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        intrinsic (array): 3*3, Intrinsic camera matrix.
        w2c (array): World to camera transformation matrix.
        near (float, optional): The near plane for the camera. Defaults to 0.01.
        far (float, optional): The far plane for the camera. Defaults to 100.

    Returns:
        GaussianRasterizationSettings: Configured settings for Gaussian rasterization.
    """
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1,
                                                  1], intrinsics[0, 2], intrinsics[1, 2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    viewmatrix = w2c.transpose(0, 1)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far /
                                    (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
    full_proj_matrix = viewmatrix.unsqueeze(
        0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
    return GaussianRasterizationSettings(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], device='cuda').float(),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_matrix,
        sh_degree=sh_degree,
        campos=cam_center,
        prefiltered=False,
        debug=False)


def render_gaussian_model(gaussian_model, render_settings,
                          override_means_3d=None, override_means_2d=None,
                          override_scales=None, override_rotations=None,
                          override_opacities=None, override_colors=None):
    """
    Renders a Gaussian model with specified rendering settings, allowing for
    optional overrides of various model parameters.

    Args:
        gaussian_model: A Gaussian model object that provides methods to get
            various properties like xyz coordinates, opacity, features, etc.
        render_settings: Configuration settings for the GaussianRasterizer.
        override_means_3d (Optional): If provided, these values will override
            the 3D mean values from the Gaussian model.
        override_means_2d (Optional): If provided, these values will override
            the 2D mean values. Defaults to zeros if not provided.
        override_scales (Optional): If provided, these values will override the
            scale values from the Gaussian model.
        override_rotations (Optional): If provided, these values will override
            the rotation values from the Gaussian model.
        override_opacities (Optional): If provided, these values will override
            the opacity values from the Gaussian model.
        override_colors (Optional): If provided, these values will override the
            color values from the Gaussian model.
    Returns:
        A dictionary containing the rendered color, depth, radii, and 2D means
        of the Gaussian model. The keys of this dictionary are 'color', 'depth',
        'radii', and 'means2D', each mapping to their respective rendered values.
    """
    renderer = GaussianRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    if override_opacities is None:
        opacities = gaussian_model.get_opacity()
    else:
        opacities = override_opacities

    shs, colors_precomp = None, None
    if override_colors is not None:
        colors_precomp = override_colors
    else:
        shs = gaussian_model.get_features()

    render_args = {
        "means3D": means3D,
        "means2D": means2D,
        "opacities": opacities,
        "colors_precomp": colors_precomp,
        "shs": shs,
        "scales": gaussian_model.get_scaling() if override_scales is None else override_scales,
        "rotations": gaussian_model.get_rotation() if override_rotations is None else override_rotations,
        "cov3D_precomp": None
    }
    color, depth, alpha, radii = renderer(**render_args)

    return {"color": color, "depth": depth, "radii": radii, "means2D": means2D, "alpha": alpha}


def rgbd2ptcloud(img, depth, intrinsics, pose=np.eye(4)):
    """converts rgbd image to point cloud
    Args:
        img (ndarray): rgb image
        depth (fcndarray): depth map
        intrinsics (ndarray): intrinsics matrix
    Returns:
        (PointCloud): resulting point cloud
    """
    height, width, _ = img.shape
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(img)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=100,
    )
    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=intrinsics[0][0],
        fy=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_img, intrinsics, extrinsic=pose, project_valid_depth_only=True)


def ptcloud2numpy(ptcloud: o3d.geometry.PointCloud) -> np.ndarray:
    """converts point cloud to numpy array
    Args:
        ptcloud (PointCloud): point cloud
    Returns:
        (ndarray): resulting numpy array
    """
    if ptcloud.has_colors():
        return np.hstack((np.asarray(ptcloud.points), np.asarray(ptcloud.colors)))
    return np.asarray(ptcloud.points)


def clone_obj(obj):
    """Deep copy an object, while detaching and cloning all tensors.
    Args:
        obj: The object to clone.
    Returns:
        clone_obj: The cloned object
    """
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        if hasattr(clone_obj.__class__, attr) and isinstance(getattr(clone_obj.__class__, attr), property):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj


def torch2np_decorator(func):
    """A decorator that creates the directory specified in the function's 'directory' keyword
       argument before calling the function.
    Args:
        func: The function to be decorated.
    Returns:
        The wrapper function.
    """
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(torch2np(arg))
            elif isinstance(arg, dict):
                new_arg = {}
                for k, v in arg.items():
                    new_arg[k] = torch2np(v) if isinstance(v, torch.Tensor) else v
                new_args.append(new_arg)
            else:
                new_args.append(arg)

        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                new_kwargs[k] = torch2np(v)
            elif isinstance(v, dict):
                new_kwargs[k] = {}
                for k1, v1 in v.items():
                    new_kwargs[k][k1] = torch2np(v1) if isinstance(v1, torch.Tensor) else v1
            else:
                new_kwargs[k] = v
        return func(*new_args, *new_kwargs)
    return wrapper
