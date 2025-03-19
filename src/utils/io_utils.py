""" This module contains utility functions for input/output file operations. """
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Union

import open3d as o3d
import torch
import yaml


def mkdir_decorator(func):
    """A decorator that creates the directory specified in the function's 'directory' keyword
       argument before calling the function.
    Args:
        func: The function to be decorated.
    Returns:
        The wrapper function.
    """
    def wrapper(*args, **kwargs):
        output_path = Path(kwargs["directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper


@mkdir_decorator
def save_clouds(clouds: list, cloud_names: list, *, directory: Union[str, Path]) -> None:
    """ Saves a list of point clouds to the specified directory, creating the directory if it does not exist.
    Args:
        clouds: A list of point cloud objects to be saved.
        cloud_names: A list of filenames for the point clouds, corresponding by index to the clouds.
        directory: The directory where the point clouds will be saved.
    """
    for cld_name, cloud in zip(cloud_names, clouds):
        o3d.io.write_point_cloud(str(directory / cld_name), cloud)


@mkdir_decorator
def save_dict_to_ckpt(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a checkpoint file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the checkpoint file.
        directory: The directory where the checkpoint file will be saved.
    """
    torch.save(dictionary, directory / file_name,
               _use_new_zipfile_serialization=False)


@mkdir_decorator
def save_dict_to_yaml(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a YAML file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the YAML file.
        directory: The directory where the YAML file will be saved.
    """
    with open(directory / file_name, "w") as f:
        yaml.dump(dictionary, f)


@mkdir_decorator
def save_dict_to_json(dictionary, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a JSON file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the JSON file.
        directory: The directory where the JSON file will be saved.
    """
    with open(directory / file_name, "w") as f:
        json.dump(dictionary, f)


def merge_dicts(base_dict, additional_dict):
    """
    Recursively merges additional_dict into base_dict.
    If there's a conflict, values from additional_dict take precedence.
    """
    for key, value in additional_dict.items():
        if isinstance(value, dict) and key in base_dict:
            # Recursively merge dictionaries
            base_dict[key] = merge_dicts(base_dict.get(key, {}), value)
        else:
            # Override/add the new value
            base_dict[key] = value
    return base_dict


def load_config(path: str) -> dict:
    """
    Loads a configuration file from the given path.
    Args:
        path: The path to the specific configuration file.
    """
    with open(path, 'r') as f:
        config = yaml.full_load(f)
    if config.get('inherit_from'):
        return merge_dicts(load_config(config['inherit_from']), config)
    return config


def get_gpu_usage_by_id(gpu_id: int) -> tuple:
    """ Get the memory usage of a GPU by its ID.
    Args:
        gpu_id: The ID of the GPU to get the memory usage of.
    Returns:
        memory_used: The amount of memory used on the GPU.
        memory_total: The total amount of memory on the GPU.
    """
    memory_used, memory_total = 0, 0
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
        text=True
    )
    memory_used, memory_total = map(int, result.stdout.strip().split(','))
    return memory_used, memory_total


def load_agents_submaps_ckpt(agents_paths: dict, device: str = "cpu"):
    """ Load agents submaps from checkpoints.
    Args:
        agents_paths: A dictionary of agent paths.
        device: The device to load the checkpoints on.
    Returns:
        agents_submaps: A dictionary of agent submaps.
        agents_kf_ids: A dictionary of agent keyframe IDs.
        agents_opt_c2ws: A dictionary of agent optimized camera-to-world matrices.
    """
    agents_kf_ids, agents_opt_c2ws = {}, {}
    agents_submaps = defaultdict(list)
    for agent_id, agent_path in agents_paths.items():
        agents_kf_ids[agent_id] = torch.load(
            agent_path / "kf_ids.ckpt", weights_only=False).detach().cpu().numpy().astype(int)
        agents_opt_c2ws[agent_id] = torch.load(
            agent_path / "optimized_kf_c2w.ckpt", weights_only=False).detach().cpu().numpy()
        for submap_path in sorted((agent_path / "submaps").glob("*")):
            agent_submap = torch.load(submap_path, weights_only=False, map_location=device)
            agent_submap["submap_c2ws"] = agent_submap["submap_c2ws"]
            agents_submaps[agent_id].append(agent_submap)
    return agents_submaps, agents_kf_ids, agents_opt_c2ws
