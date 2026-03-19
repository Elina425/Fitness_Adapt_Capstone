from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import COCO17


@dataclass(frozen=True)
class PreprocessConfig:
    target_fps: float = 30.0
    min_confidence: float = 0.2


def _interp_1d(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Linear interpolation for missing values.
    Falls back to nearest valid value at sequence boundaries.
    """
    out = values.copy()
    idx = np.arange(values.shape[0])
    if valid_mask.sum() == 0:
        return np.zeros_like(values)
    valid_idx = idx[valid_mask]
    valid_vals = values[valid_mask]
    out = np.interp(idx, valid_idx, valid_vals)
    return out.astype(values.dtype, copy=False)


def impute_missing_keypoints(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, min_confidence: float) -> np.ndarray:
    """
    Impute occluded/missing joints by temporal interpolation.
    """
    t_len, n_joints, _ = keypoints_xy.shape
    out = keypoints_xy.copy()
    valid = keypoints_conf >= min_confidence

    for j in range(n_joints):
        for c in range(2):
            series = keypoints_xy[:, j, c]
            mask = valid[:, j] & (~np.isclose(series, 0.0))
            out[:, j, c] = _interp_1d(series, mask)
    return out


def normalize_keypoints(keypoints_xy: np.ndarray) -> np.ndarray:
    """
    Normalize for camera distance and translation:
    - center around hip midpoint
    - scale by shoulder width (fallback to hip width)
    """
    out = keypoints_xy.copy()
    l_hip = COCO17["left_hip"]
    r_hip = COCO17["right_hip"]
    l_sh = COCO17["left_shoulder"]
    r_sh = COCO17["right_shoulder"]

    hip_center = 0.5 * (out[:, l_hip, :] + out[:, r_hip, :])
    out = out - hip_center[:, None, :]

    shoulder_width = np.linalg.norm(out[:, l_sh, :] - out[:, r_sh, :], axis=1)
    hip_width = np.linalg.norm(out[:, l_hip, :] - out[:, r_hip, :], axis=1)
    scale = np.where(shoulder_width > 1e-6, shoulder_width, hip_width)
    scale = np.where(scale > 1e-6, scale, 1.0)
    out = out / scale[:, None, None]
    return out.astype(np.float32)


def resample_sequence(keypoints_xy: np.ndarray, src_times_sec: np.ndarray, target_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample sequence to a fixed FPS using linear interpolation.
    """
    if keypoints_xy.shape[0] == 0:
        return keypoints_xy, src_times_sec
    if src_times_sec.shape[0] != keypoints_xy.shape[0]:
        raise ValueError("src_times_sec and keypoints length mismatch")

    t_start = float(src_times_sec[0])
    t_end = float(src_times_sec[-1])
    if t_end <= t_start:
        return keypoints_xy, src_times_sec

    num = int(np.floor((t_end - t_start) * target_fps)) + 1
    tgt_times = np.linspace(t_start, t_end, num=num, dtype=np.float32)

    t_len, n_joints, _ = keypoints_xy.shape
    out = np.zeros((num, n_joints, 2), dtype=np.float32)
    for j in range(n_joints):
        for c in range(2):
            out[:, j, c] = np.interp(tgt_times, src_times_sec, keypoints_xy[:, j, c])
    return out, tgt_times


def preprocess_keypoints(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    times_sec: np.ndarray,
    config: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full task-3 pipeline:
    1) impute missing joints
    2) normalize coordinates
    3) synchronize timing via fixed FPS resampling
    """
    imputed = impute_missing_keypoints(keypoints_xy, keypoints_conf, min_confidence=config.min_confidence)
    normalized = normalize_keypoints(imputed)
    synced_xy, synced_t = resample_sequence(normalized, times_sec, target_fps=config.target_fps)
    return synced_xy, synced_t

