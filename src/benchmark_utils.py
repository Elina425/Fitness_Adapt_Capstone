from typing import List

import numpy as np

from .pose_estimators import knee_angle_from_coco17


def sample_frames_uniform(num_frames: int, stride: int = 2, seed: int = 0) -> np.ndarray:
    """
    Deterministic uniform sub-sampling by stride.
    """
    if num_frames <= 0:
        return np.asarray([], dtype=np.int32)
    start = seed % max(1, stride)
    idx = np.arange(start, num_frames, step=max(1, stride), dtype=np.int32)
    return idx


def compute_knee_angle_series(kp_xy: np.ndarray) -> np.ndarray:
    """
    kp_xy: (T,17,2)
    returns (T,) knee angle in degrees, or NaN if can't compute.
    """
    if kp_xy.ndim != 3 or kp_xy.shape[1:] != (17, 2):
        raise ValueError(f"Expected (T,17,2), got {kp_xy.shape}")
    T = kp_xy.shape[0]
    out = np.full((T,), np.nan, dtype=np.float32)
    for t in range(T):
        # Only knee/hip/ankle joints are needed; other joints may be missing.
        out[t] = knee_angle_from_coco17(kp_xy[t])
    return out


def valid_frame_mask(kp_xy: np.ndarray) -> np.ndarray:
    """
    Frame is valid if all 17 keypoints are present (non-NaN).
    """
    return ~np.isnan(kp_xy).any(axis=(1, 2))

