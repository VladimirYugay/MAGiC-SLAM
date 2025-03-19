""" This module contains utility functions for rendering performance evaluation. """
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from src.utils import utils, vis_utils


def calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """ Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1: The first image.
        img2: The second image.
    Returns:
        The PSNR value.
    """
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def evaluate_rendering(gaussian_model, dataset,
                       keyframe_ids: np.ndarray, keyframe_c2ws: np.ndarray, output_path: Path = None) -> tuple:
    """ Evaluate the rendering quality of a Gaussian model against a dataset
    Args:
        gaussian_model: The Gaussian model.
        dataset: The dataset to evaluate against.
        keyframe_ids: The keyframe ids
        keyframe_c2ws: The keyframe camera-to-world matrices.
        output_path: The output path to save the rendered images.
    Returns:
        psnr_values: The PSNR values for all the keyframes.
        lpips: The LPIPS values for all the keyframes.
        ssim: The SSIM values for all the keyframes.
        depth_l1s: The depth L1 values for all the keyframes.
        """
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

    psnr_values, lpips, ssim, depth_l1s = [], [], [], []
    for i, (keyframe_id, c2w) in enumerate(zip(keyframe_ids, keyframe_c2ws)):
        render_frame = dataset.get_render_frame(keyframe_id, np.linalg.inv(c2w))
        with torch.no_grad():
            render_dict = utils.render_gaussian_model(gaussian_model, render_frame["render_settings"])
            if output_path is not None:
                image = vis_utils.tensor_to_image(render_dict["color"])
                image.save(output_path / f"frame_{i}.png")

            rendered_color = torch.clamp(render_dict["color"], 0.0, 1.0)
            psnr_values.append(calc_psnr(rendered_color, render_frame["gt_color"]).mean().item())
            lpips.append(lpips_model(rendered_color[None], render_frame["gt_color"][None]).mean().item())
            ssim.append(ms_ssim(rendered_color[None], render_frame["gt_color"][None], data_range=1.0).item())
            depth_l1s.append(torch.mean(torch.abs(render_dict["depth"] - render_frame["gt_depth"])).item())
    return psnr_values, lpips, ssim, depth_l1s


def eval_agents_rendering(gaussian_model, agents_datasets: dict, kf_ids: dict,
                          kf_c2ws: dict, print_eval: bool = True) -> tuple:
    """ Evaluate the rendering quality against agents' datasets.
    Args:
        gaussian_model: The Gaussian model.
        agents_datasets: The agents' datasets (agent_id: str -> dataset : Dataset)
        kf_ids: The keyframe ids (agent_id: str -> keyframe_ids : np.ndarray)
        kf_c2ws: The keyframe camera-to-world matrices (agent_id: str -> c2ws : np.ndarray)
        print_eval: Whether to print the evaluation results.
    Returns:
        psnr: The average PSNR value.
        lpips: The average LPIPS value.
        ssim: The average SSIM value.
        depth_l1: The average depth L1 value.
    """
    merged_psnrs, merged_lpips, merged_ssim, merged_depth_l1s = [], [], [], []
    for agent_id in tqdm(sorted(kf_ids.keys())):
        mpsnrs, mlpips, mssim, mdepthl1s = evaluate_rendering(
            gaussian_model, agents_datasets[agent_id], kf_ids[agent_id], kf_c2ws[agent_id])
        merged_psnrs.extend(mpsnrs)
        merged_lpips.extend(mlpips)
        merged_ssim.extend(mssim)
        merged_depth_l1s.extend(mdepthl1s)

    psnr, depth_l1 = 0, 0
    if merged_psnrs and merged_depth_l1s:
        psnr = sum(merged_psnrs) / len(merged_psnrs)
        ssim = sum(merged_ssim) / len(merged_ssim)
        lpips = sum(merged_lpips) / len(merged_lpips)
        depth_l1 = sum(merged_depth_l1s) / len(merged_depth_l1s)
    if print_eval:
        print(f"{psnr:.3f}, {ssim:.3f}, {lpips:.3f}, {depth_l1:.3f} \n")
    return psnr, lpips, ssim, depth_l1
