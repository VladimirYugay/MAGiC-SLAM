""" This module contains a class that wraps a pose graph from Graphslam library.
    The API for GraphSLAM expects data in g2o format and operates on it internally.
    To avoid possible inconsistencies, we convert our data to g2o format and then
    pass it to the GraphSLAM API. The class module is responsible for it.

    You can get more details in Section 3.3 of the paper under "Pose Graph Optimization".

    While there are several PGO libraries, we chose GraphSLAM for several reasons.
    1. GTSAM is harder to grasp and its internals are not easy to understand.
       GraphSLAM is fully pythonic and easy to understand.
       It was coded live on youtube by the author: https://www.youtube.com/@jeffirion57
       I strongly believe in simplicity over complexity.
    2. PyPose, Open3d do not allow for freezing multiple vertices. In multi-agent SLAM, this is crucial
       as we want to freeze the first vertex of each agent since they are ground-truth.
"""
import tempfile

import numpy as np
from graphslam.graph import Graph
from scipy.spatial.transform import Rotation as R


def pose2g2o_vertex(id: int, T: np.ndarray) -> str:
    """ Converts a pose to a g2o vertex string.
    Args:
        id: The id of the vertex.
        T: The pose matrix of shape (4, 4).
    Returns:
        The g2o vertex string.
    """
    quat = R.from_matrix(T[:3, :3].copy()).as_quat()
    t = T[:3, 3].copy()
    return f"VERTEX_SE3:QUAT {id} {t[0]} {t[1]} {t[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}"


def pose2g2o_edge(start: int, end: int, T, info=1) -> str:
    """ Converts a pose to a g2o edge string.
    Args:
        start: The id of the start vertex.
        end: The id of the end vertex.
        T: The relative pose matrix of shape (4, 4).
        info: The information matrix value. Since we do not have access to information evaluation
              we set it to 1 in all our experiments.
    Returns:
        The g2o edge string.
    """
    quat = R.from_matrix(T[:3, :3].copy()).as_quat()
    t = T[:3, 3].copy()
    edge_str = (
        f"EDGE_SE3:QUAT {start} {end} "
        f"{t[0]} {t[1]} {t[2]} "
        f"{quat[0]} {quat[1]} {quat[2]} {quat[3]} "
        f"{info} 0 0 0 0 0 {info} 0 0 0 0 {info} 0 0 0 {info} 0 0 {info} 0 {info}")
    return edge_str


class PoseGraphAdapter(object):

    def __init__(self, agents_submaps: dict, loops: list) -> None:
        self.agents_submaps = agents_submaps
        self.loops = loops
        self.graph_info = {}
        self.edges = []
        self.nodes = []
        self.graph = None
        self.setup_odometry()
        self.setup_loops()
        self.setup_graph()

    def setup_odometry(self) -> None:
        """ Sets up the odometry edges in the pose graph. """
        agent_ids = sorted(self.agents_submaps.keys())
        for agent_id in agent_ids:
            nodes_count = len(self.nodes)
            for i in range(len(self.agents_submaps[agent_id])):
                if i == 0:
                    source_sub = self.agents_submaps[agent_id][i]
                    self.graph_info[(agent_id, source_sub["submap_start_frame_id"])] = len(self.nodes)
                    node = pose2g2o_vertex(len(self.nodes), source_sub["submap_c2ws"][0])
                    self.nodes.append(node)
                    continue

                source_sub = self.agents_submaps[agent_id][i - 1]
                target_sub = self.agents_submaps[agent_id][i]

                rel_pose = np.linalg.inv(source_sub["submap_c2ws"][0]) @ target_sub["submap_c2ws"][0]
                node = pose2g2o_vertex(len(self.nodes), target_sub["submap_c2ws"][0])
                edge = pose2g2o_edge(nodes_count + i - 1, nodes_count + i, rel_pose)

                self.graph_info[(agent_id, target_sub["submap_start_frame_id"])] = len(self.nodes)
                self.nodes.append(node)
                self.edges.append(edge)

    def setup_loops(self) -> None:
        """ Sets up the loop closure edges in the pose graph. """
        for loop in self.loops:
            source_node_id = self.graph_info[(loop.source_agent_id, loop.source_frame_id)]
            target_node_id = self.graph_info[(loop.target_agent_id, loop.target_frame_id)]
            edge = pose2g2o_edge(source_node_id, target_node_id, np.linalg.inv(loop.transformation))
            self.edges.append(edge)

    def setup_graph(self) -> None:
        """ Sets up the pose graph from the nodes and edges.
            Since GraphSLAM operates on g2o format, we write the nodes and edges to a temporary file
            and then pass it to the GraphSLAM Graph constructor.
            This ensures the correct work of the GraphSLAM internals.
        """
        with tempfile.NamedTemporaryFile("w+", suffix=".g2o", delete=True) as tmp_file:
            for node in self.nodes:
                tmp_file.write(node + "\n")
            for edge in self.edges:
                tmp_file.write(edge + "\n")
            tmp_file.flush()  # Ensure all data is written to disk
            self.graph = Graph.from_g2o(tmp_file.name)

    def optimize(self, max_iter: int = 20) -> None:
        """ Optimizes the pose graph.
        Args:
            max_iter: The maximum number of iterations for optimization.
        """
        for agent_id in sorted(self.agents_submaps.keys()):
            node_id = self.graph_info[(agent_id, self.agents_submaps[agent_id][0]["submap_start_frame_id"])]
            self.graph._vertices[node_id].fixed = True
        self.graph.optimize(max_iter=max_iter)

    def get_poses(self) -> dict:
        """ Returns the optimized poses for all agents.
        Returns:
            optimized_poses: The optimized poses for all agents (agent_id: str, poses: np.ndarray of shape (n, 4, 4)).
        """
        optimized_poses = {}
        for agent_id in sorted(self.agents_submaps.keys()):
            agent_poses = []
            for submap in self.agents_submaps[agent_id]:
                vertex_id = self.graph_info[(agent_id, submap["submap_start_frame_id"])]
                agent_poses.append(self.graph._vertices[vertex_id].pose.to_matrix())
            optimized_poses[agent_id] = np.array(agent_poses)
        return optimized_poses
