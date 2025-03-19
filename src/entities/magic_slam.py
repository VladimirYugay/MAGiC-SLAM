""" The MAGiCSLAM class that orchestrates the pipeline."""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from src.entities.agent import Agent
from src.entities.datasets import get_dataset
from src.entities.logger import Logger
from src.entities.loop_detection.loop_detector import LoopDetector
from src.entities.pose_graph_adapter import PoseGraphAdapter
from src.utils import vis_utils
from src.utils.io_utils import save_dict_to_json, save_dict_to_yaml
from src.utils.magic_slam_utils import (apply_pose_correction, merge_submaps,
                                        refine_map, register_agents_submaps,
                                        register_submaps)
from src.utils.mapping_eval import eval_agents_rendering, evaluate_rendering


class MAGiCSLAM(object):

    def __init__(self, config: dict) -> None:
        self.config = config
        if "inherit_from" in self.config:
            del self.config["inherit_from"]
        self.multi_gpu = self.config["multi_gpu"]
        self.scene_name = self.config["data"]["scene_name"]
        self.agent_ids = np.array(self.config["data"]["agent_ids"])
        self.agents_submaps = {agent_id: [] for agent_id in self.agent_ids}
        self.agents_messages = {agent_id: {} for agent_id in self.agent_ids}
        self.run_info = self._init_wandb()
        self.agents = {agent_id: Agent(int(agent_id), self.run_info, self.config) for agent_id in self.agent_ids}

        self.input_path = Path(self.config["data"]["input_path"])
        self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        save_dict_to_yaml(self.config, "config.yaml", directory=self.output_path)
        self.logger = Logger(self.output_path, -1, self.config["use_wandb"])
        mp.set_start_method('spawn', force=True)  # Necessary for GaussianModel initialization

    def _init_wandb(self) -> str:
        """ Initialize the Weights and Biases logging.
        Returns:
            The run information. This information is necessary for connecting the agents
            operating in differnt processes to the same WanDB run.
        """
        if not self.config.get("use_wandb", False):
            return {}
        run_name = self.config.get("experiment_name", "") + "_" + datetime.now().strftime("%m_%d_%H_%M")
        entity = self.config.get("wandb_entity", "xokage")
        wandb.init(
            project=self.config["project_name"],
            entity=entity,
            mode="offline" if self.config.get("wandb_offline", False) else "online",
            config=self.config,
            settings=wandb.Settings(_disable_stats=True),
            name=run_name,
            group=self.config.get("group_name", ""))
        return {
            "run_id": wandb.run.id,
            "run_name": run_name,
            "entity": entity,
            "project_name": wandb.run.project_name(),
            "group": self.config.get("group_name", "")
        }

    def start_agent(self, agent_id: int, pipe) -> None:
        """ Start the agent process.
        Args:
            agent_id: The id of the agent.
            pipe: The pipe to communicate with the agent.
        """
        if self.multi_gpu:
            torch.cuda.set_device(int(agent_id) + 1)  # keep 0 for the main process
        self.agents[agent_id].set_pipe(pipe)
        self.agents[agent_id].run()

    def optimize_poses(self, agents_submaps: dict) -> dict:
        """ Optimize the poses of the agents using the loop closure.
            Detects all the loops between the submaps, registers the loops, and optimizes the poses.
        Args:
            agents_submaps: The submaps of the agents.
        Returns:
            The optimized keyframe poses of the agents.
        """

        loop_detector = LoopDetector(self.config["loop_detection"])

        agents_c2ws = {}
        for agent_id, agent_submaps in agents_submaps.items():
            agents_c2ws[agent_id] = np.vstack([submap["submap_c2ws"] for submap in agent_submaps])

        intra_loops, inter_loops = loop_detector.detect_loops(agents_submaps)
        vis_utils.plot_agents_pose_graph(agents_c2ws, {}, intra_loops, inter_loops,
                                         output_path=str(self.output_path / "all_loops.png"))

        intra_loops = register_agents_submaps(
            agents_submaps, intra_loops, register_submaps, max_threads=20)
        inter_loops = register_agents_submaps(
            agents_submaps, inter_loops, register_submaps, max_threads=20)

        self.logger.log_loops(intra_loops + inter_loops, "loops.pkl")
        intra_loops = loop_detector.filter_loops(intra_loops)
        inter_loops = loop_detector.filter_loops(inter_loops)
        self.logger.log_loops(intra_loops + inter_loops, "filtered_loops.pkl")

        graph_wrapper = PoseGraphAdapter(agents_submaps, intra_loops + inter_loops)
        if len(intra_loops + inter_loops) > 0:
            graph_wrapper.optimize()
        agents_optimized_kf_poses = apply_pose_correction(graph_wrapper.get_poses(), agents_submaps)
        return agents_optimized_kf_poses

    def run(self) -> None:
        """ Run the MAGiCSLAM pipeline.
            Creates a set of pipes to communicate with the agents and starts the agents in separate processes.
            Collects the submaps from the agents, optimizes the poses and sends them back.
            Merges the submaps into a single map, then refines and evaluates it.
        """
        pipes = {agent_id: mp.Pipe() for agent_id in self.agent_ids}
        self.pipes = {agent_id: pipe[0] for agent_id, pipe in pipes.items()}
        processes = [mp.Process(
            target=self.start_agent, args=(agent_id, pipes[agent_id][1])) for agent_id in self.agent_ids]
        for p in processes:
            p.start()

        while True:
            for agent_id in self.agent_ids:
                msg, agent_submaps = self.pipes[agent_id].recv()
                self.agents_messages[agent_id] = msg
                self.agents_submaps[agent_id] = agent_submaps

            registration_time = time.time()
            agents_opt_kf_c2ws = self.optimize_poses(self.agents_submaps)
            registration_time = time.time() - registration_time

            for agent_id in self.agent_ids:
                self.pipes[agent_id].send(agents_opt_kf_c2ws[agent_id])
            break

        for p in processes:
            p.join()

        agents_kf_ids = {}
        for agent_id, agent_submaps in self.agents_submaps.items():
            agents_kf_ids[agent_id] = np.empty((0))
            for submap in agent_submaps:
                agents_kf_ids[agent_id] = np.concatenate([agents_kf_ids[agent_id], submap["keyframe_ids"]])
            agents_kf_ids[agent_id] = agents_kf_ids[agent_id].astype(int)
        opt_args = self.agents[0].mapper.opt

        total_merge_time = 0
        start = time.time()
        merged_map = merge_submaps(
            self.agents_submaps, agents_kf_ids, agents_opt_kf_c2ws, opt_args)
        total_merge_time += time.time() - start

        merged_map.save_ply(str(self.output_path / "merged_coarse.ply"))

        # Load agents datasets for the evaluation
        agents_datasets = {agent.agent_id: agent.dataset for agent in self.agents.values()}

        print("Eval coarse merged map")
        coarse_psnr, coarse_lpips, coarse_ssim, coarse_dl1 = eval_agents_rendering(
            merged_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws)
        save_dict_to_json({"psnr": coarse_psnr,
                           "lpips": coarse_lpips,
                           "ssim": coarse_ssim,
                           "depth_l1": coarse_dl1}, "coarse_render_metrics.json",
                          directory=self.output_path)

        start = time.time()
        refined_map = refine_map(
            merged_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws, iterations=3000)
        total_merge_time += time.time() - start

        print(f"Total merge time: {total_merge_time}")

        if self.config["multi_gpu"]:  # Logging GPU usage makes sense only in multi-GPU mode
            # GPU:0 is used by the server
            self.logger.log_gpu_usage(0)

        save_dict_to_json({
            "registration_time": registration_time,
            "total_merge_time": total_merge_time,
            },
            "walltime_stats.json",
            directory=self.output_path)

        refined_map.save_ply(str(self.output_path / "merged_refined.ply"))
        print("Eval coarse merged map")
        fine_psnr, fine_lpips, fine_ssim, fine_dl1 = eval_agents_rendering(
            refined_map, agents_datasets, agents_kf_ids, agents_opt_kf_c2ws)
        save_dict_to_json({"psnr": fine_psnr,
                           "lpips": fine_lpips,
                           "ssim": fine_ssim,
                           "depth_l1": fine_dl1}, "fine_render_metrics.json",
                          directory=self.output_path)

        if self.config["dataset_name"] == "aria":  # we only have hold-out views in Aria
            # Temporal fix, test data lies in the parent folder
            input_path = Path(self.config["data"]["input_path"]).parent / "test" / self.config["data"]["scene_name"]
            dataset_config = {**{"input_path": str(input_path), "dataset_name": self.config["dataset_name"]},
                              **self.config["cam"]}
            nvs_test_dataset = get_dataset(self.config["dataset_name"])(dataset_config)

            keyframe_ids = list(range(len(nvs_test_dataset)))
            keyframe_c2w = np.array(nvs_test_dataset.poses)
            novel_psnr, novel_lpips, novel_ssim, novel_dl1 = evaluate_rendering(
                refined_map, nvs_test_dataset, keyframe_ids, keyframe_c2w)
            save_dict_to_json({"psnr": np.mean(np.array(novel_psnr)),
                               "lpips": np.mean(np.array(novel_lpips)),
                               "ssim": np.mean(np.array(novel_ssim)),
                               "depth_l1": np.mean(np.array(novel_dl1))},
                              "novel_render_metrics.json", directory=self.output_path)

        if self.config["use_wandb"]:
            wandb.finish()
