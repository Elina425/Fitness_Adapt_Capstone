from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .geometry import compute_knee_angles_deg

Interval = Tuple[float, float]


def intervals_to_mask(times_sec: np.ndarray, intervals: List[Interval]) -> np.ndarray:
    mask = np.zeros((times_sec.shape[0],), dtype=bool)
    if not intervals:
        return mask
    for start, end in intervals:
        mask |= (times_sec >= start) & (times_sec <= end)
    return mask


def knee_angle_separation_score(
    keypoints_xy: np.ndarray,
    times_sec: np.ndarray,
    error_intervals: List[Interval],
) -> float:
    if keypoints_xy.shape[0] == 0:
        return 0.0
    knee = compute_knee_angles_deg(keypoints_xy)
    valid = ~np.isnan(knee)
    if valid.sum() < 5:
        return 0.0
    err_mask = intervals_to_mask(times_sec, error_intervals) & valid
    non_err_mask = (~intervals_to_mask(times_sec, error_intervals)) & valid
    if err_mask.sum() < 3 or non_err_mask.sum() < 3:
        return 0.0
    mean_err = float(np.mean(knee[err_mask]))
    mean_non = float(np.mean(knee[non_err_mask]))
    std_all = float(np.std(knee[valid]) + 1e-6)
    return float(abs(mean_err - mean_non) / std_all)


def extraction_speed_fps(extraction_time_sec: float, num_frames: int) -> float:
    if extraction_time_sec <= 0 or num_frames <= 0:
        return 0.0
    return float(num_frames / extraction_time_sec)

