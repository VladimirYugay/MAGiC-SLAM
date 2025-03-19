""" This module includes the Logger class, which is responsible for logging for both Mapper and the Tracker """
import pickle
from io import BytesIO
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image

from src.utils.io_utils import get_gpu_usage_by_id, save_dict_to_json
from src.utils.tracking_eval import align_trajectories, compute_ate, pose_error
from src.utils.utils import find_submap, torch2np_decorator
from src.utils.vis_utils import plot_trajectory


class Logger(object):

    def __init__(self, output_path: Union[Path, str], agent_id=0, use_wandb=False) -> None:
        self.output_path = Path(output_path)
        self.agent_id = agent_id
        self.use_wandb = use_wandb

    def log_tracking_iteration(self, frame_id, cur_pose, gt_quat, gt_trans, total_loss,
                               color_loss, depth_loss, iter, num_iters,
                               print_output=False, print_wandb=False) -> None:
        """ Logs tracking iteration metrics including pose error, losses, and optionally reports to Weights & Biases.
        Logs the error between the current pose estimate and ground truth quaternion and translation,
        as well as various loss metrics. Can output to wandb if enabled and specified, and print to console.
        Args:
            frame_id: Identifier for the current frame.
            cur_pose: The current estimated pose as a tensor (quaternion + translation).
            gt_quat: Ground truth quaternion.
            gt_trans: Ground truth translation.
            total_loss: Total computed loss for the current iteration.
            color_loss: Computed color loss for the current iteration.
            depth_loss: Computed depth loss for the current iteration.
            iter: The current iteration number.
            num_iters: The total number of iterations planned.
            wandb_output: Whether to output the log to wandb.
            print_output: Whether to print the log output.
        """
        prefix = f"Agent_{self.agent_id}_tracking"
        quad_err = torch.abs(cur_pose[:4] - gt_quat).mean().item() * 100  # in degrees
        trans_err = torch.abs(cur_pose[4:] - gt_trans).mean().item() * 100  # in cm
        if self.use_wandb and print_wandb:
            wandb.log(
                {
                    f"{prefix}/idx": frame_id,
                    f"{prefix}/cam_quad_err": quad_err,
                    f"{prefix}/cam_position_err": trans_err,
                    f"{prefix}/total_loss": total_loss.item(),
                    f"{prefix}/color_loss": color_loss.item(),
                    f"{prefix}/depth_loss": depth_loss.item(),
                    f"{prefix}/num_iters": num_iters,
                })
        if iter == num_iters - 1:
            msg = f"frame_id: {frame_id}, cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f} "
        else:
            msg = f"iter: {iter}, total_loss: {total_loss.item():.5f}, color_loss: {color_loss.item():.5f}, depth_loss: {depth_loss.item():.5f} "
        msg = msg + f", cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f}"
        if print_output:
            print(msg, flush=True)

    def log_mapping_iteration(self, **kwargs) -> None:
        """ Logs mapping iteration metrics including the number of new points, model size, and optimization times,
        and optionally reports to Weights & Biases (wandb).
        Args:
            frame_id: Identifier for the current frame.
            new_pts_num: The number of new points added in the current mapping iteration.
            model_size: The total size of the model after the current mapping iteration.
            iter_opt_time: Time taken per optimization iteration.
            opt_dict: A dictionary containing optimization metrics such as PSNR, color loss, and depth loss.
        """
        prefix = f"Agent_{self.agent_id}_mapping"
        if self.use_wandb:
            wandb_log_data = {}
            for key, value in kwargs.items():
                wandb_log_data[f"{prefix}/{key}"] = value
            wandb.log(wandb_log_data)

    @torch2np_decorator
    def log_tracking_results(self, est_c2w, gt_c2w, artifact_name: str):
        (self.output_path / "traj_eval").mkdir(exist_ok=True, parents=True)

        est_t, gt_t = est_c2w[:, :3, 3], gt_c2w[:, :3, 3]
        est_t_aligned = align_trajectories(est_t, gt_t)
        ate = compute_ate(est_t, gt_t)
        ate_aligned = compute_ate(est_t_aligned, gt_t)
        ate_rmse, ate_rmse_aligned = ate["rmse"] * 100, ate_aligned["rmse"] * 100

        save_dict_to_json({
            "ate_rmse": ate_rmse,
            "ate_rmse_aligned": ate_rmse_aligned
        }, "ate.json", directory=self.output_path / "traj_eval")

        ax = plot_trajectory(
            est_t, label=f"ate-rmse: {round(ate_rmse, 3)} cm", color="orange")
        ax = plot_trajectory(
            est_t_aligned, ax, label=f"ate-rsme (aligned): {round(ate_rmse_aligned, 3)} cm", color="lightskyblue")
        ax = plot_trajectory(
            gt_t, ax, label="GT", color="green")

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        plt.close()
        buf.seek(0)
        plot_image = Image.open(buf)
        plot_image.save(str(self.output_path / "traj_eval" / f"{artifact_name}"))
        if self.use_wandb:
            prefix = f"Agent_{self.agent_id}_trajectory/all_eval"
            wandb.log({prefix: [wandb.Image(plot_image)]})

    def log_loops(self, loops, artifact_name: str):
        (self.output_path).mkdir(exist_ok=True, parents=True)
        with open(self.output_path / artifact_name, "wb") as f:
            pickle.dump(loops, f)

    def log_loops_quality(self, agents_submaps: dict, agents_datasets: dict, loops: list):
        output_file = open(str(self.output_path / "loops_quality.txt"), "w")
        good_loops = []
        errors_t, errors_q = [], []
        for i, loop in enumerate(loops):
            sa, sf, ta, tf = loop.source_agent_id, loop.source_frame_id, loop.target_agent_id, loop.target_frame_id
            source_sub, target_sub = find_submap(sf, agents_submaps[sa]), find_submap(tf, agents_submaps[ta])

            rel_pose = loop.transformation
            rel_pose_est = np.linalg.inv(target_sub["submap_c2ws"][0]) @ source_sub["submap_c2ws"][0]
            gt_rel_pose = np.linalg.inv(agents_datasets[ta][tf][-1]) @ agents_datasets[sa][sf][-1]

            t_reg_err, q_reg_err = pose_error(rel_pose, gt_rel_pose)
            t_est_err, q_est_err = pose_error(rel_pose_est, gt_rel_pose)

            if t_reg_err > t_est_err or q_reg_err > q_est_err:
                errors_t.append(t_reg_err)
                errors_q.append(q_reg_err)
                txt = f"{i} {loop.inlier_rmse:.3f} t_reg_err: {t_reg_err:.3f}, t_est_err: {t_est_err:.3f}" + \
                    f"q_reg_err: {q_reg_err: .3f}, q_est_err: {q_est_err:.3f} \n"
                output_file.write(txt)
                continue
            good_loops.append(loop)
        if len(errors_t) > 0:
            errors_t, errors_q = np.array(errors_t), np.array(errors_q)
            txt = f"Mean t_reg_err: {np.mean(errors_t):.3f}, Mean q_reg_err: {np.mean(errors_q):.3f}"
            output_file.write(txt)
        txt = f"Number of good loops: {len(good_loops)} out of {len(loops)}"
        output_file.write(txt)
        output_file.close()

    def vis_mapping_iteration(self, frame_id, iter, color, depth, gt_color, gt_depth, seeding_mask=None) -> None:
        """
        Visualization of depth, color images and save to file.

        Args:
            frame_id (int): current frame index.
            iter (int): the iteration number.
            save_rendered_image (bool): whether to save the rgb image in separate folder
            img_dir (str): the directory to save the visualization.
            seeding_mask: used in mapper when adding gaussians, if not none.
        """
        (self.output_path / "mapping_vis").mkdir(exist_ok=True, parents=True)
        gt_depth_np = gt_depth.cpu().numpy()
        gt_color_np = gt_color.cpu().numpy()

        depth_np = depth.detach().cpu().numpy()
        color = torch.round(color * 255.0) / 255.0
        color_np = color.detach().cpu().numpy()
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        # make errors >=5cm noticeable
        depth_residual = np.clip(depth_residual, 0.0, 0.05)

        color_residual = np.abs(gt_color_np - color_np)
        color_residual[np.squeeze(gt_depth_np == 0.0)] = 0.0

        # Determine Aspect Ratio and Figure Size
        aspect_ratio = color.shape[1] / color.shape[0]
        fig_height = 8
        # Adjust the multiplier as needed for better spacing
        fig_width = fig_height * aspect_ratio * 1.2

        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 0].set_title('Input Depth', fontsize=16)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 1].set_title('Rendered Depth', fontsize=16)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma")
        axs[0, 2].set_title('Depth Residual', fontsize=16)
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB', fontsize=16)
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Rendered RGB', fontsize=16)
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        if seeding_mask is not None:
            axs[1, 2].imshow(seeding_mask, cmap="gray")
            axs[1, 2].set_title('Densification Mask', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
        else:
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

        for ax in axs.flatten():
            ax.axis('off')
        fig.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust top margin
        fig_name = str(self.output_path / "mapping_vis" / f'{frame_id:04d}_{iter:04d}.jpg')
        fig_title = f"Mapper Color/Depth at frame {frame_id:04d} iters {iter:04d}"
        plt.suptitle(fig_title, y=0.98, fontsize=20)
        plt.savefig(fig_name, dpi=250, bbox_inches='tight')
        plt.clf()
        plt.close()
        if self.use_wandb:
            prefix = f"Agent_{self.agent_id}_mapping_vis/"
            log_title = prefix + f'{frame_id:04d}_{iter:04d}'
            wandb.log({log_title: [wandb.Image(fig_name)]})
        print(f"Saved rendering vis of color/depth at {frame_id:04d}_{iter:04d}.jpg")

    def log_gpu_usage(self, gpu_id: int) -> None:
        """ Logs the GPU memory usage to a JSON file.
        Args:
            gpu_id: The GPU ID to log the memory usage for.
        """
        memory_used, memory_total = get_gpu_usage_by_id(gpu_id)
        prefix = f"GPU_{gpu_id}"
        gpu_dict = {
            f"{prefix}/memory_used": memory_used,
            f"{prefix}/memory_total": memory_total
        }
        save_dict_to_json(gpu_dict, "system_stats.json", directory=self.output_path)
        if self.use_wandb:
            wandb.log(gpu_dict)
