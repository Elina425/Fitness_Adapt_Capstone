import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .benchmark_utils import sample_frames_uniform, valid_frame_mask, compute_knee_angle_series
from .dataset_labels import VideoSample, build_frame_level_quality_mask


@dataclass
class ModelBenchmarkMetrics:
    model_name: str
    fps_processing: float  # frames/sec processed by estimator
    frames_total: int
    frames_used: int
    roc_auc: float
    best_f1: float


def _collect_binary_labels(
    video_sample: VideoSample,
    fps: float,
    num_frames: int,
    use_forward: bool,
    use_inward: bool,
) -> List[int]:
    forward = video_sample.knee_forward_intervals if use_forward else []
    inward = video_sample.knee_inward_intervals if use_inward else []
    # Combine to one error mask (bad if any error).
    all_intervals = forward + inward
    return build_frame_level_quality_mask(fps=fps, num_frames=num_frames, intervals_sec=all_intervals)


def benchmark_pose_estimator_on_dataset(
    *,
    model_name: str,
    estimator,
    dataset: List[VideoSample],
    max_videos: int = 5,
    max_frames_per_video: int = 180,
    use_forward: bool = True,
    use_inward: bool = True,
    frame_sample_stride: int = 2,
    random_seed: int = 0,
    feature_fn: Optional[Callable[[np.ndarray], float]] = None,
    verbose: bool = True,
) -> ModelBenchmarkMetrics:
    """
    Accuracy proxy:
      - Ground truth: knee-error intervals -> frame-level binary "bad" label.
      - Prediction score: knee angle computed from extracted keypoints.
    """
    if feature_fn is None:
        feature_fn = compute_knee_angle_series

    # Speed benchmark: measure end-to-end extraction time across selected videos.
    total_frames = 0
    total_time_s = 0.0

    y_all: List[int] = []
    score_all: List[float] = []

    selected = dataset[:max_videos]
    for i, sample in enumerate(selected):
        if verbose:
            print(f"[{model_name}] video {i+1}/{len(selected)}: {sample.video_id}")
        t0 = time.perf_counter()
        res = estimator.extract_from_video(
            str(sample.video_path),
            max_frames=max_frames_per_video,
        )
        total_time_s += time.perf_counter() - t0
        num_frames = int(res.keypoints_xy.shape[0])
        total_frames += num_frames

        if num_frames == 0:
            continue

        y = _collect_binary_labels(
            sample,
            fps=res.fps,
            num_frames=num_frames,
            use_forward=use_forward,
            use_inward=use_inward,
        )
        y = np.asarray(y, dtype=np.int32)

        # Compute knee-angle score per frame.
        knee_angle = feature_fn(res.keypoints_xy)  # (T,)
        knee_angle = np.asarray(knee_angle, dtype=np.float32)

        # Sample frames to limit work.
        idx = sample_frames_uniform(num_frames=num_frames, stride=frame_sample_stride, seed=random_seed)
        knee_angle_s = knee_angle[idx]
        y_s = y[idx]

        # Mask NaNs.
        mask = ~np.isnan(knee_angle_s)
        knee_angle_s = knee_angle_s[mask]
        y_s = y_s[mask]
        y_all.extend(y_s.tolist())
        score_all.extend(knee_angle_s.tolist())

    if len(y_all) == 0:
        return ModelBenchmarkMetrics(
            model_name=model_name,
            fps_processing=0.0,
            frames_total=total_frames,
            frames_used=0,
            roc_auc=float("nan"),
            best_f1=float("nan"),
        )

    y_arr = np.asarray(y_all, dtype=np.int32)
    score_arr = np.asarray(score_all, dtype=np.float32)

    # ROC-AUC: higher score should mean "bad".
    # knee angle can correlate either direction; we take absolute deviation from 90 degrees.
    # Re-scale to ensure positive class isn't always "lower".
    score_bad = np.abs(score_arr - 90.0)

    try:
        roc = float(roc_auc_score(y_arr, score_bad))
    except Exception:
        roc = float("nan")

    # Best F1 by thresholding score_bad across quantiles.
    thresholds = np.quantile(score_bad, q=np.linspace(0.05, 0.95, 25)).tolist()
    best_f1 = 0.0
    for thr in thresholds:
        y_pred = (score_bad >= thr).astype(np.int32)
        f1 = f1_score(y_arr, y_pred, zero_division=0)
        best_f1 = max(best_f1, float(f1))

    fps_processing = float(total_frames / max(total_time_s, 1e-9))

    return ModelBenchmarkMetrics(
        model_name=model_name,
        fps_processing=fps_processing,
        frames_total=total_frames,
        frames_used=len(y_arr),
        roc_auc=roc,
        best_f1=best_f1,
    )

