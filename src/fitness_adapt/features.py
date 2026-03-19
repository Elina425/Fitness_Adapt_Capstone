from __future__ import annotations

import numpy as np

from .geometry import COCO17, angle_3p


def compute_joint_angles(keypoints_xy: np.ndarray) -> np.ndarray:
    """
    Computes biomechanical angle features per frame.
    Output shape: (T, 6)
      [left_knee, right_knee, left_hip, right_hip, left_elbow, right_elbow]
    """
    t_len = keypoints_xy.shape[0]
    feats = np.zeros((t_len, 6), dtype=np.float32)

    lk, rk = COCO17["left_knee"], COCO17["right_knee"]
    lh, rh = COCO17["left_hip"], COCO17["right_hip"]
    la, ra = COCO17["left_ankle"], COCO17["right_ankle"]
    ls, rs = COCO17["left_shoulder"], COCO17["right_shoulder"]
    le, re = COCO17["left_elbow"], COCO17["right_elbow"]
    lw, rw = COCO17["left_wrist"], COCO17["right_wrist"]

    for t in range(t_len):
        p = keypoints_xy[t]
        feats[t, 0] = angle_3p(p[lh], p[lk], p[la])  # left knee
        feats[t, 1] = angle_3p(p[rh], p[rk], p[ra])  # right knee
        feats[t, 2] = angle_3p(p[ls], p[lh], p[lk])  # left hip
        feats[t, 3] = angle_3p(p[rs], p[rh], p[rk])  # right hip
        feats[t, 4] = angle_3p(p[ls], p[le], p[lw])  # left elbow
        feats[t, 5] = angle_3p(p[rs], p[re], p[rw])  # right elbow
    return feats


def flatten_raw_coordinates(keypoints_xy: np.ndarray) -> np.ndarray:
    """
    Baseline raw-coordinate features for ablation.
    Output shape: (T, 34)
    """
    t_len = keypoints_xy.shape[0]
    return keypoints_xy.reshape(t_len, -1).astype(np.float32)


def window_sequence(features: np.ndarray, window_size: int = 30, stride: int = 15) -> np.ndarray:
    """
    Convert frame-level features (T, D) into windows (N, window_size, D).
    """
    t_len, feat_dim = features.shape
    if t_len < window_size:
        pad = np.repeat(features[-1:, :], window_size - t_len, axis=0)
        features = np.concatenate([features, pad], axis=0)
        t_len = features.shape[0]
    windows = []
    for start in range(0, t_len - window_size + 1, stride):
        windows.append(features[start : start + window_size])
    if not windows:
        windows = [features[:window_size]]
    return np.stack(windows, axis=0).astype(np.float32)

