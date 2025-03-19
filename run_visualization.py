""" This script is used to visualize the results of the MAGiC-SLAM pipeline.
    It takes the output of the pipeline and visualizes the submaps, the loops, and the optimized trajectory.
"""
import argparse
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import trimesh

from src.entities.agent import Agent
from src.entities.arguments import OptimizationParams
from src.entities.gaussian_model import GaussianModel
from src.utils.io_utils import load_agents_submaps_ckpt, load_config
from src.utils.utils import get_render_settings, render_gaussian_model
from src.utils.vis_utils import COLORS_MATPLOTLIB_RGB, gaussian_model_to_mesh

AGENT_COLORS = [
    np.array([0, 114, 178]),   # Bright Blue
    np.array([240, 228, 66]),  # Bright Yellow
    np.array([0, 158, 115])    # Bright Green
]


def submap_to_mesh(submap: dict, height: int, width: int, intrinsics: np.ndarray) -> trimesh.Trimesh:
    """ Convert a submap to a mesh.
    Args:
        submap: The submap dictionary.
        height: The height of the images.
        width: The width of the images.
        intrinsics: The camera intrinsics.
    Returns:
        submap_trimesh: The submap as a trimesh object.
    """
    opt_args = OptimizationParams(ArgumentParser(description="Training script parameters"))
    gaussian_map = GaussianModel(0)
    gaussian_map.training_setup(opt_args)
    gaussian_map.restore_from_params(submap["gaussian_model_params"], opt_args)
    kf_c2ws = submap["submap_c2ws"][submap["keyframe_ids"] - submap["submap_start_frame_id"]]
    o3d_mesh = gaussian_model_to_mesh(gaussian_map, kf_c2ws, height, width, intrinsics)
    submap_trimesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices),
                                     vertex_colors=np.asarray(o3d_mesh.vertex_colors),
                                     faces=np.asarray(o3d_mesh.triangles))
    return submap_trimesh


def merged_map_to_mesh(merged_map_path: Path, c2ws: np.ndarray, height: int, width: int,
                       intrinsics: np.ndarray) -> trimesh.Trimesh:
    """ Convert a merged map to a mesh.
    Args:
        merged_map_path: The path to the merged map.
        c2ws: The camera-to-world matrices.
        height: The height of the images.
        width: The width of the images.
        intrinsics: The camera intrinsics.
    Returns:
        map_trimesh: The map as a trimesh object.
    """
    opt_args = OptimizationParams(ArgumentParser(description="Training script parameters"))
    gaussian_map = GaussianModel(0)
    gaussian_map.training_setup(opt_args)
    gaussian_map.load_ply(merged_map_path)
    o3d_mesh = gaussian_model_to_mesh(gaussian_map, c2ws, height, width, intrinsics)
    map_trimesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices),
                                  vertex_colors=np.asarray(o3d_mesh.vertex_colors),
                                  faces=np.asarray(o3d_mesh.triangles))
    return map_trimesh


def compute_submap_meshes(agents_submaps: dict, agents_datasets: dict, output_path: Path) -> None:
    """ Compute the meshes of the submaps and save them to the output path.
        The meshes are needed to visualize the submaps in the rerun viewer.
    Args:
        agents_submaps: A dictionary of agent submaps (agent_id, submaps list).
        agents_datasets: A dictionary of agent datasets (agent_id, dataset).
        output_path: The output path to save the submaps meshes.
    """
    for agent_id, submaps in agents_submaps.items():
        (output_path / f"agent_{agent_id}").mkdir(parents=True, exist_ok=True)
        dataset = agents_datasets[agent_id]
        for submap in submaps:
            file_path = str(output_path / f"agent_{agent_id}" / f"submap_{submap['submap_id']}.ply")
            if Path(file_path).exists():  # Load the mesh if it already exists e.g. from the previous run
                submap_trimesh = trimesh.load(file_path, force="mesh")
            else:
                submap_trimesh = submap_to_mesh(submap, dataset.height, dataset.width, dataset.intrinsics)
                submap_trimesh.export(str(output_path / f"agent_{agent_id}" / f"submap_{submap['submap_id']}.ply"))
            submap["mesh"] = submap_trimesh


def compute_map_mesh(merged_map_path: Path, agents_poses: dict, height: int, width: int,
                     intrinsics: np.ndarray, output_file_path: Path):
    """ Compute the mesh of the merged map and save it to the output path.
        The mesh is needed to visualize the map in the rerun viewer.
    Args:
        merged_map_path: The path to the merged map.
        agents_poses: A dictionary of agents poses.
        height: The height of the images.
        width: The width of the images.
        intrinsics: The camera intrinsics.
        output_file_path: The output path to save the map mesh.
    Returns:
        map_trimesh: The map as a trimesh object.
    """

    if Path(output_file_path).exists():
        map_trimesh = trimesh.load(output_file_path, force="mesh")
        return map_trimesh

    gaussian_model = GaussianModel(0)
    gaussian_model.training_setup(OptimizationParams(ArgumentParser(description="Training script parameters")))
    gaussian_model.load_ply(merged_map_path)

    c2ws = []
    for agent_id in sorted(agents_poses.keys()):
        for c2w in agents_poses[agent_id]["optimized_kf_c2w"]:
            c2ws.append(c2w.cpu().numpy())

    o3d_mesh = gaussian_model_to_mesh(gaussian_model, c2ws, height, width, intrinsics)
    map_trimesh = trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices),
                                  vertex_colors=np.asarray(o3d_mesh.vertex_colors),
                                  faces=np.asarray(o3d_mesh.triangles))
    map_trimesh.export(str(output_file_path))
    return map_trimesh


def run_recon_visualization(agents_datasets: dict, agents_submaps: dict, agents_poses: dict) -> None:
    """ Run the reconstruction visualization.
        Plots a camera with image inside it for every keyframe together with the trajectory line.
        For every submap step adds a submap mesh to the canvas.
    Args:
        agents_datasets: A dictionary of agent datasets (agent_id, dataset).
        agents_submaps: A dictionary of agent submaps (agent_id, submaps list).
        agents_poses: A dictionary of agents poses.
    """

    kf_ids = agents_poses[0]["kf_ids"].int().tolist()  # kf ids are the same for all agents

    for i, kf_id in enumerate(kf_ids):
        for agent_id in sorted(agents_datasets.keys()):
            submaps = agents_submaps[agent_id]
            dataset = agents_datasets[agent_id]

            new_submap_every = len(dataset) // len(submaps)
            c2w = agents_poses[agent_id]["estimated_kf_c2w"][i].cpu().numpy()

            _, image, depth, _ = dataset[kf_id]

            rr.log(f"color_path_{agent_id}/rgb", rr.Image(image).compress(jpeg_quality=95))
            rr.log(f"depth_path_{agent_id}/depth", rr.DepthImage(depth, meter=1))

            rr.log(f"map/camera/{agent_id}",
                   rr.Pinhole(image_from_camera=dataset.intrinsics, resolution=[dataset.width, dataset.height]))
            rr.log(f"map/camera/{agent_id}", rr.Transform3D(mat3x3=c2w[:3, :3], translation=c2w[:3, 3]))
            rr.log(f"map/camera/{agent_id}", rr.Image(image))

            if kf_id > 0:
                rr.log(f"map/camera/{agent_id}_{kf_id}", rr.LineStrips3D(
                    [c2w[:3, 3], agents_poses[agent_id]["estimated_kf_c2w"][i - 1].cpu().numpy()[:3, 3]],
                    colors=AGENT_COLORS[agent_id]
                ))

            if kf_id % new_submap_every == 0:
                submap_trimesh = submaps[kf_id // new_submap_every]["mesh"]
                rr.log(
                    f"map/submaps/{agent_id}_{kf_id // new_submap_every}",
                    rr.Mesh3D(
                        vertex_positions=submap_trimesh.vertices,
                        vertex_colors=submap_trimesh.visual.vertex_colors,
                        triangle_indices=submap_trimesh.faces))


def run_lc_visualization(agents_datasets: dict, agents_submaps: dict, agents_poses: dict,
                         agents_loops: list, merged_map_mesh):
    """ Run the loop closure visualization.
    Args:
        agents_datasets: A dictionary of agent datasets (agent_id, dataset).
        agents_submaps: A dictionary of agent submaps (agent_id, submaps list).
        agents_poses: A dictionary of agents poses.
        agents_loops: A list of filtered loop closures.
        merged_map_mesh: The merged map as a trimesh object
    """

    kf_ids = agents_poses[0]["kf_ids"].int().tolist()

    for agent_id in sorted(agents_datasets.keys()):
        rr.log(f"map/camera/{agent_id}", rr.Clear(recursive=True))

    # Draw the loops between the trajectories
    loop_poses = []  # (n, 2, 3)
    for loop in agents_loops:
        start = agents_poses[loop.source_agent_id]["estimated_c2w"][loop.source_frame_id].cpu().numpy()
        end = agents_poses[loop.target_agent_id]["estimated_c2w"][loop.target_frame_id].cpu().numpy()
        loop_poses.append(np.array([start[:3, 3], end[:3, 3]]))
    loop_poses = np.array(loop_poses)
    color = (COLORS_MATPLOTLIB_RGB["red"] * 255.0).astype(np.uint8)
    rr.log("map/loops", rr.LineStrips3D(loop_poses, colors=color))
    time.sleep(3)

    # Apply corrections to the submaps
    for i, kf_id in enumerate(kf_ids):
        for agent_id in sorted(agents_datasets.keys()):
            submaps = agents_submaps[agent_id]
            dataset = agents_datasets[agent_id]
            new_submap_every = len(dataset) // len(submaps)

            if kf_id % new_submap_every == 0:
                c2w = agents_poses[agent_id]["estimated_kf_c2w"][i].cpu().numpy()
                opt_c2w = agents_poses[agent_id]["optimized_kf_c2w"][i].cpu().numpy()
                delta = opt_c2w @ np.linalg.inv(c2w)
                time.sleep(0.05)
                rr.log(f"map/submaps/{agent_id}_{kf_id // new_submap_every}",
                       rr.Transform3D(mat3x3=delta[:3, :3], translation=delta[:3, 3]))

    # Remove the submaps and the loops
    rr.log("map/submaps", rr.Clear(recursive=True))
    rr.log("map/loops", rr.Clear(recursive=True))
    rr.log("map/camera", rr.Clear(recursive=True))

    for agent_id in sorted(agents_poses.keys()):
        dataset = agents_datasets[agent_id]
        opt_c2ws = agents_poses[agent_id]["optimized_kf_c2w"].detach().cpu().numpy()  # (n, 4, 4)
        start, end = opt_c2ws[:-1, :3, 3], opt_c2ws[1:, :3, 3]  # (n - 1, 3)
        rr.log(f"map/camera/trajectory_{agent_id}", rr.LineStrips3D(np.concatenate((end[:, None, :], start[:, None, :]),
                                                                                   axis=1),
                                                                    colors=AGENT_COLORS[agent_id]))

    # Add the global map
    rr.log("map", rr.Mesh3D(vertex_positions=merged_map_mesh.vertices,
                            vertex_colors=merged_map_mesh.visual.vertex_colors,
                            triangle_indices=merged_map_mesh.faces))


def visualize_trajectory(gaussian_model_path: Path, dataset, name="Novel Views") -> None:
    """ Visualize the novel view rendering over the reconstructed map.
        Creates a new separate layout and logs the renders next to the ground truth images.
    Args:
        gaussian_model_path: The path to the Gaussian model.
        dataset: The dataset to render the novel views from.
        name: The name of the visualization.
    """

    gaussian_model = GaussianModel(0)
    gaussian_model.training_setup(OptimizationParams(ArgumentParser(description="Training script parameters")))
    gaussian_model.load_ply(gaussian_model_path)

    rr.init(name, spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Ground Truth RGB", origin="gt_rgb"),
                    rrb.Spatial2DView(name="Ground Truth Depth", origin="gt_depth")
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Rendered RGB", origin="rendered_rgb"),
                    rrb.Spatial2DView(name="Rendered Depth", origin="rendered_depth")
                )]))
    rr.send_blueprint(blueprint)
    height, width, intrinsics = dataset.height, dataset.width, dataset.intrinsics

    for i in range(len(dataset)):
        _, image, depth, c2w = dataset[i]
        w2c = np.linalg.inv(c2w)
        render_dict = render_gaussian_model(gaussian_model, get_render_settings(width, height, intrinsics, w2c))
        rendered_color, rendered_depth = render_dict["color"].detach(), render_dict["depth"][0].detach()
        rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
        rendered_color = rendered_color.permute(1, 2, 0).cpu().numpy()

        rr.log("gt_rgb", rr.Image(image).compress(jpeg_quality=95))
        rr.log("gt_depth", rr.DepthImage(depth, meter=1))
        rr.log("rendered_rgb", rr.Image(rendered_color))
        rr.log("rendered_depth", rr.DepthImage(rendered_depth, meter=1))


def get_args():
    parser = argparse.ArgumentParser(description='Arguments to run visualization')
    parser.add_argument('--checkpoint_path', type=str, help='Path to scene output artifacts')
    parser.add_argument('--artifacts_path', type=str, help='Path to the resulting visualization artifacts')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_path = Path(args.checkpoint_path)

    artifacts_path = Path(args.artifacts_path)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    config = load_config(data_path / "config.yaml")
    # If the experiment was run on a remote cluster and the visualization happens locally
    # We have to update the input path to the local path
    config["data"]["input_path"] = "ReplicaMultiagent/Apart-0/"

    # Load the agents's sub-maps and datasets
    agents_ids = config["data"]["agent_ids"]
    agents = [Agent(int(agent_id), "", config) for agent_id in agents_ids]
    agents_datsets = {agent_id: agents[agent_id].dataset for agent_id in agents_ids}
    agents_paths = {agent_id: path for agent_id, path in enumerate(sorted(data_path.glob('*agent_*')))}
    agents_submaps, agents_kf_ids, agents_opt_c2ws = load_agents_submaps_ckpt(agents_paths, device="cuda")

    # Load the filtered pose graph loops
    with open(data_path / "filtered_loops.pkl", "rb") as f:
        agents_loops = pickle.load(f)

    agents_poses = {}
    for agent in agents:
        agent_path = agents_paths[agent.agent_id]
        agents_poses[agent.agent_id] = {
            "estimated_c2w": torch.load(agent_path / "estimated_c2w.ckpt", weights_only=False),
            "estimated_kf_c2w": torch.load(agent_path / "estimated_kf_c2w.ckpt", weights_only=False),
            "optimized_kf_c2w": torch.load(agent_path / "optimized_kf_c2w.ckpt", weights_only=False),
            "kf_ids": torch.load(agent_path / "kf_ids.ckpt", weights_only=False)
        }

    compute_submap_meshes(agents_submaps, agents_datsets, artifacts_path)

    merged_map_path = data_path / "merged_refined.ply"
    dataset = agents_datsets[0]
    merged_map_mesh = compute_map_mesh(
        merged_map_path, agents_poses, dataset.height, dataset.width, dataset.intrinsics,
        artifacts_path / "merged_map_mesh.ply")

    # The general layout of the canvas
    rr.init("MAGiC-SLAM", spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    rrb.Spatial2DView(name=f"Agent {agent_id} RGB", origin=f"color_path_{agent_id}"),
                    rrb.Spatial2DView(name=f"Agent {agent_id} Depth", origin=f"depth_path_{agent_id}")
                ) for agent_id in agents_ids]
        ),
        rrb.Spatial3DView(name="Map", origin="map", background=[200, 200, 200]),
        column_shares=[0.4, 0.6]
    )
    rr.send_blueprint(blueprint)

    # Visualization of the mapping process of individual agents
    run_recon_visualization(agents_datsets, agents_submaps, agents_poses)

    # Visualization of the loop closure process
    run_lc_visualization(agents_datsets, agents_submaps, agents_poses, agents_loops, merged_map_mesh)

    # Visualizes the NVS
    # visualize_trajectory(merged_map_path, dataset, name="Novel Views")
