""" This module contains utility functions for tracking performance evaluation. """
import numpy as np
from scipy.spatial.transform import Rotation as R


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(
        np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def align_trajectories(t_pred: np.ndarray, t_gt: np.ndarray):
    """
    Args:
        t_pred: (n, 3) translations
        t_gt: (n, 3) translations
    Returns:
        t_align: (n, 3) aligned translations
    """
    t_align = np.matrix(t_pred).transpose()
    R, t, _ = align(t_align, np.matrix(t_gt).transpose())
    t_align = R * t_align + t
    t_align = np.asarray(t_align).T
    return t_align


def compute_ate(t_pred: np.ndarray, t_gt: np.ndarray):
    """
    Args:
        t_pred: (n, 3) translations
        t_gt: (n, 3) translations
    Returns:
        dict: error dict
    """
    n = t_pred.shape[0]
    trans_error = np.linalg.norm(t_pred - t_gt, axis=1)
    return {
        "compared_pose_pairs": n,
        "rmse": np.sqrt(np.dot(trans_error, trans_error) / n),
        "mean": np.mean(trans_error),
        "median": np.median(trans_error),
        "std": np.std(trans_error),
        "min": np.min(trans_error),
        "max": np.max(trans_error)
    }


def pose_error(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple:
    """ Computes the translation and rotation errors between two poses.
    Args:
        pose_a: The first pose as a 4x4 matrix.
        pose_b: The second pose as a 4x4 matrix.
    Returns:
        A tuple containing the translation and rotation errors.
    """
    t_reg_err = np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3])
    q_reg_err = np.linalg.norm(R.from_matrix(pose_a[:3, :3].copy()).as_quat() -
                               R.from_matrix(pose_b[:3, :3].copy()).as_quat())
    return t_reg_err, q_reg_err
