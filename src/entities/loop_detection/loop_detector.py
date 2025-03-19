""" This module contains the LoopDetector class, which is responsible for detecting loops between submaps. """
import warnings

import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch

from src.entities.loop_detection.feature_extractors import get_feature_extractor
from src.utils.magic_slam_utils import Registration
from src.utils.utils import find_submap

# Mentioned here: https://github.com/pytorch/pytorch/issues/97207
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


class LoopDetector(object):

    def __init__(self, config: dict) -> None:
        self.feature_extractor_name = config["feature_extractor_name"]
        self.device = config["device"]
        self.weights_path = config["weights_path"]
        self.feature_dist_threshold = config["feature_dist_threshold"]
        self.time_threshold = config["time_threshold"]
        self.embed_size = config["embed_size"]
        self.max_loops_per_frame = config["max_loops_per_frame"]
        self.fitness_threshold = config["fitness_threshold"]
        self.inlier_rmse_threshold = config["inlier_rmse_threshold"]

        self.agents_ids = torch.empty(0).int().to(self.device)
        self.frame_ids = torch.empty(0).int().to(self.device)

        self._feature_extractor = get_feature_extractor(config)
        self._setup_feature_db()

    def _setup_feature_db(self) -> None:
        """ Sets up the feature database for storing and searching image features. """
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            self.features_db = faiss.GpuIndexFlatL2(res, self.embed_size, flat_config)
        else:
            self.features_db = faiss.IndexFlatL2(self.embed_size)

    def get_db_size(self) -> int:
        """ Returns the number of features stored in the database. """
        return self.features_db.ntotal

    def add_features(self, features: torch.Tensor, frame_ids: torch.Tensor, agent_id: int) -> None:
        """ Adds the features to the database.
        Args:
            features: The features to be added of shape (bs, embed_size).
            frame_ids: The frame ids of the features of shape (bs)
            agent_id: The agent id of the features added
        """
        self.frame_ids = torch.cat([self.frame_ids, frame_ids]).int().to(self.device)
        agent_ids = torch.full((frame_ids.shape[0],), agent_id).int().to(self.device)
        self.agents_ids = torch.cat([self.agents_ids, agent_ids]).to(self.device)
        self.features_db.add(features)

    def clean_db(self) -> None:
        """ Cleans the database by removing all the features and cleaning the frame and agent ids. """
        self.frame_ids = torch.empty(0).int().to(self.device)
        self.agents_ids = torch.empty(0).int().to(self.device)
        self.features_db.reset()

    def search_closest_frames(self, agent_id: int, frame_ids: torch.Tensor, features: torch.Tensor,
                              use_time_threshold=True) -> list:
        """ Searches for the closest frames in the database for the given features.
        Args:
            agent_id: The agent id of the features.
            frame_ids: The frame ids of the features.
            features: The features to search for of shape (bs, embed_size).
            use_time_threshold: Whether to use the time threshold for searching.
        Returns:
            closest_frames: A list of Registration objects representing the closest frames in the feature space.
        """

        closest_distances, closest_db_ids = self.features_db.search(features, k=self.max_loops_per_frame)
        mask = closest_distances < self.feature_dist_threshold
        mask = mask * (closest_db_ids >= 0)
        if use_time_threshold:
            frame_db_ids = torch.arange(frame_ids.shape[0]).to(features.device) + self.get_db_size()
            mask = mask * (closest_db_ids + self.time_threshold < frame_db_ids[:, None])

        closest_frames = []
        closest_db_ids[mask == 0] = -1
        for i in range(frame_ids.shape[0]):
            for j, loop_db_id in enumerate(closest_db_ids[i]):
                if loop_db_id == -1:
                    continue
                loop_agent_id = self.agents_ids[loop_db_id].item()
                loop_frame_id = self.frame_ids[loop_db_id].item()
                registration = Registration(agent_id, frame_ids[i].item(), loop_agent_id, loop_frame_id)
                closest_frames.append(registration)
        return closest_frames

    def detect_intra_loops(self, agents_submaps: dict) -> list:
        """ Detects loops between the submaps of separate agents
        Args:
            agents_submaps: A dictionary containing the submaps of the agents
        Returns:
            intra_loops: A list of Registration objects representing loops of agents with themselves
        """
        intra_loops = []
        for agent_id, agents_submaps in agents_submaps.items():
            for submap in agents_submaps:
                frame_ids = torch.tensor(submap["keyframe_ids"][:1]).to(self.device)
                features = torch.from_numpy(submap["submap_features"]).to(self.device)
                if self.get_db_size() > 0:
                    intra_loop = self.search_closest_frames(
                        agent_id, frame_ids, features, use_time_threshold=True)
                    intra_loops.extend(intra_loop)
                self.add_features(features, frame_ids, agent_id)
            self.clean_db()
        return intra_loops

    def detect_inter_loops(self, agents_submaps: dict) -> list:
        """ Detects loops between the submaps of different agents.
        Args:
            agents_submaps: A dictionary containing the submaps of the agents.
        Returns:
            inter_loops: A list of Registration objects representing loops of agents with other agents.
        """
        inter_loops = []
        for agent_id, agent_submaps in agents_submaps.items():

            if self.get_db_size() > 0:
                for submap in agent_submaps:
                    frame_ids = torch.tensor(submap["keyframe_ids"][:1]).to(self.device)
                    features = torch.from_numpy(submap["submap_features"]).to(self.device)
                    inter_loop = self.search_closest_frames(
                        agent_id, frame_ids, features, use_time_threshold=False)
                    inter_loops.extend(inter_loop)

            for submap in agent_submaps:
                frame_ids = torch.tensor(submap["keyframe_ids"][:1]).to(self.device)
                features = torch.from_numpy(submap["submap_features"]).to(self.device)
                self.add_features(features, frame_ids, agent_id)
        return inter_loops

    def detect_loops(self, agents_submaps) -> tuple:
        """ Detects loops between the submaps of the agents.
            You can find more details in Section 3.3 of the paper.
        Args:
            agents_submaps: A dictionary containing the submaps of the agents.
            loop_detector: The loop detector object.
        Returns:
            intra_loops: A list of Registration objects representing loops of agents with itself.
            inter_loops: A list of Registration objects representing loops of agents with other agents.
        """
        intra_loops = self.detect_intra_loops(agents_submaps)
        inter_loops = self.detect_inter_loops(agents_submaps)
        print(f"Detected {len(intra_loops)} intra-loop(s) and {len(inter_loops)} inter-loop(s)")

        for loop in intra_loops + inter_loops:
            source_sub = find_submap(loop.source_frame_id, agents_submaps[loop.source_agent_id])
            target_sub = find_submap(loop.target_frame_id, agents_submaps[loop.target_agent_id])
            rel_pose = np.linalg.inv(target_sub["submap_c2ws"][0]) @ source_sub["submap_c2ws"][0]
            loop.init_transformation = rel_pose
        return intra_loops, inter_loops

    def filter_loops(self, loops: list) -> list:
        """ Filters the loops based on the fitness and inlier RMSE thresholds.
            Poorly registered loops can severely degrade the PGO accuracy.
        Args:
            loops: A list of Registration objects.
        Returns:
            filtered_loops: A list of Registration objects.
        """
        filtered_loops = []
        for loop in loops:
            if loop.fitness > self.fitness_threshold and loop.inlier_rmse < self.inlier_rmse_threshold:
                filtered_loops.append(loop)
        return filtered_loops
