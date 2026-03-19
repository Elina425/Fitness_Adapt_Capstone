from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .features import compute_joint_angles, flatten_raw_coordinates, window_sequence
from .labels import compute_quality_labels_for_key, load_error_intervals
from .pose import extract_yolo11_pose_coco17
from .preprocess import PreprocessConfig, preprocess_keypoints
from .project import ProjectPaths
from .io_utils import load_json


@dataclass(frozen=True)
class DatasetConfig:
    target_fps: float = 30.0
    window_size: int = 30
    window_stride: int = 15
    frame_stride: int = 3
    max_frames_per_video: int = 180
    conf_threshold: float = 0.35
    yolo_weights: str = "yolo11n-pose.pt"
    device: str = "cpu"


def _video_duration_sec(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frames / fps if fps > 0 else 0.0


def build_split_windows(
    split_name: str,
    *,
    paths: ProjectPaths,
    config: DatasetConfig,
    keys_override: List[str] | None = None,
) -> Dict[str, np.ndarray]:
    keys = keys_override if keys_override is not None else load_json(paths.split_path(split_name))
    error_fwd, error_inward = load_error_intervals(paths)

    x_angles, x_raw = [], []
    y_exercise, y_quality = [], []
    key_ids = []
    exercise_to_idx = {"squat": 0}

    for key in keys:
        video_path = str(paths.squat_video_dir / f"{key}.mp4")
        if not cv2.VideoCapture(video_path).isOpened():
            continue
        ext = extract_yolo11_pose_coco17(
            video_path,
            frame_stride=config.frame_stride,
            max_frames=config.max_frames_per_video,
            conf_threshold=config.conf_threshold,
            yolo_weights=config.yolo_weights,
            device=config.device,
        )
        if ext.keypoints_xy.shape[0] < 2:
            continue

        proc_xy, _ = preprocess_keypoints(
            ext.keypoints_xy,
            ext.keypoints_conf,
            ext.times_sec,
            PreprocessConfig(target_fps=config.target_fps, min_confidence=config.conf_threshold),
        )
        ang = compute_joint_angles(proc_xy)
        raw = flatten_raw_coordinates(proc_xy)
        win_ang = window_sequence(ang, window_size=config.window_size, stride=config.window_stride)
        win_raw = window_sequence(raw, window_size=config.window_size, stride=config.window_stride)

        duration = _video_duration_sec(video_path)
        lbl = compute_quality_labels_for_key(
            key,
            exercise_type="squat",
            duration_sec=duration,
            error_fwd=error_fwd,
            error_inward=error_inward,
        )
        ex_idx = exercise_to_idx[lbl.exercise_type]
        q_score = float(lbl.quality_score)

        n = win_ang.shape[0]
        x_angles.append(win_ang)
        x_raw.append(win_raw)
        y_exercise.append(np.full((n,), ex_idx, dtype=np.int64))
        y_quality.append(np.full((n,), q_score, dtype=np.float32))
        key_ids.extend([key] * n)

    if not x_angles:
        raise RuntimeError(f"No usable windows built for split={split_name}")

    return {
        "x_angles": np.concatenate(x_angles, axis=0),
        "x_raw": np.concatenate(x_raw, axis=0),
        "y_exercise": np.concatenate(y_exercise, axis=0),
        "y_quality": np.concatenate(y_quality, axis=0),
        "key_ids": np.asarray(key_ids, dtype=object),
    }

