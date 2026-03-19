from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._deps import raise_missing_dependency_error

try:
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise_missing_dependency_error(exc, script_name=Path(__file__).name)


VIDEO_DIR = PROJECT_ROOT / "videos_squat"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = PROJECT_ROOT / ".cache"

SPLIT_FILES = {
    "train": PROJECT_ROOT / "train_keys.json",
    "val": PROJECT_ROOT / "val_keys.json",
    "test": PROJECT_ROOT / "test_keys.json",
}

ERROR_FILES = {
    "knees_forward": PROJECT_ROOT / "error_knees_forward.json",
    "knees_inward": PROJECT_ROOT / "error_knees_inward.json",
}

MEDIAPIPE_TO_COCO_17 = [
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]

COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

EXTERNAL_DATASET_OPTIONS = [
    {
        "dataset_name": "Fitness-AQA",
        "source": "Parmar et al. (arXiv:2202.14019)",
        "exercise_types": "BackSquat, BarbellRow, OverheadPress",
        "quality_supervision": "Expert-annotated workout errors and form assessment labels",
        "fit_for_capstone": "Best paper-aligned public option, but not included in this repository.",
    },
    {
        "dataset_name": "UI-PRMD",
        "source": "University of Idaho Physical Rehabilitation Movement Dataset",
        "exercise_types": "10 rehab-style movements including deep squat and inline lunge",
        "quality_supervision": "Correct vs incorrect execution examples",
        "fit_for_capstone": "Useful backup for form-quality experiments, but less gym-specific than Fitness-AQA.",
    },
    {
        "dataset_name": "Repo squat videos",
        "source": "This project",
        "exercise_types": "Squat only",
        "quality_supervision": "Timestamped knees_forward / knees_inward annotations",
        "fit_for_capstone": "Best real data available locally right now; enough for squat-quality prototyping, not enough for multi-exercise classification.",
    },
]

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_mediapipe_model() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CACHE_DIR / "pose_landmarker_full.task"
    if not model_path.exists():
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, model_path)
    return model_path


def load_split_lookup() -> dict[str, str]:
    split_lookup: dict[str, str] = {}
    for split_name, path in SPLIT_FILES.items():
        for video_id in _read_json(path):
            split_lookup[video_id] = split_name
    return split_lookup


def load_error_maps() -> dict[str, dict[str, list[list[float]]]]:
    return {name: _read_json(path) for name, path in ERROR_FILES.items()}


def video_metadata(video_path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    duration_seconds = frame_count / fps if fps else 0.0
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def quality_label_from_errors(has_forward: bool, has_inward: bool) -> str:
    if has_forward and has_inward:
        return "multiple_annotated_errors"
    if has_forward:
        return "knees_forward"
    if has_inward:
        return "knees_inward"
    return "no_annotated_error"


def build_dataset_manifest() -> pd.DataFrame:
    split_lookup = load_split_lookup()
    error_maps = load_error_maps()

    rows = []
    for video_path in sorted(VIDEO_DIR.glob("*.mp4")):
        video_id = video_path.stem
        metadata = video_metadata(video_path)
        knees_forward_segments = error_maps["knees_forward"].get(video_id, [])
        knees_inward_segments = error_maps["knees_inward"].get(video_id, [])
        has_forward = len(knees_forward_segments) > 0
        has_inward = len(knees_inward_segments) > 0

        rows.append(
            {
                "video_id": video_id,
                "video_path": str(video_path.relative_to(PROJECT_ROOT)),
                "split": split_lookup.get(video_id, "unassigned"),
                "exercise_type": "squat",
                "quality_label": quality_label_from_errors(has_forward, has_inward),
                "has_knees_forward": has_forward,
                "has_knees_inward": has_inward,
                "knees_forward_segments": json.dumps(knees_forward_segments),
                "knees_inward_segments": json.dumps(knees_inward_segments),
                "annotated_error_segments": len(knees_forward_segments) + len(knees_inward_segments),
                **metadata,
            }
        )

    manifest = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)
    return manifest


def dataset_summary(manifest: pd.DataFrame) -> dict:
    split_counts = manifest["split"].value_counts().sort_index().to_dict()
    quality_counts = manifest["quality_label"].value_counts().sort_index().to_dict()

    return {
        "total_videos": int(len(manifest)),
        "exercise_types_present": sorted(manifest["exercise_type"].unique().tolist()),
        "split_counts": split_counts,
        "quality_label_counts": quality_counts,
        "mean_duration_seconds": round(float(manifest["duration_seconds"].mean()), 3),
        "median_duration_seconds": round(float(manifest["duration_seconds"].median()), 3),
        "task_1_assessment": {
            "supports_quality_labels": True,
            "supports_multiple_exercise_types": manifest["exercise_type"].nunique() > 1,
            "conclusion": (
                "The repository already contains real squat videos with timestamped quality-error annotations, "
                "so it is usable for squat-quality prototyping. It does not yet contain multiple exercise classes, "
                "so later exercise-recognition stages will still need more labeled exercise types."
            ),
        },
    }


def external_dataset_options() -> pd.DataFrame:
    return pd.DataFrame(EXTERNAL_DATASET_OPTIONS)


def select_benchmark_videos(
    manifest: pd.DataFrame,
    total_videos: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    available_categories = manifest["quality_label"].value_counts().index.tolist()
    per_category = max(1, total_videos // max(len(available_categories), 1))
    sampled_frames = []

    for category in available_categories:
        candidates = manifest[manifest["quality_label"] == category]
        take = min(per_category, len(candidates))
        sampled_frames.append(candidates.sample(n=take, random_state=random_state))

    benchmark = pd.concat(sampled_frames, ignore_index=True).drop_duplicates("video_id")

    if len(benchmark) < total_videos:
        remainder = manifest[~manifest["video_id"].isin(benchmark["video_id"])]
        fill_count = min(total_videos - len(benchmark), len(remainder))
        if fill_count:
            benchmark = pd.concat(
                [benchmark, remainder.sample(n=fill_count, random_state=random_state)],
                ignore_index=True,
            )

    benchmark = (
        benchmark.sort_values(["quality_label", "split", "video_id"])
        .head(total_videos)
        .reset_index(drop=True)
    )
    return benchmark


def torso_scale(keypoints_xy: np.ndarray, confidences: np.ndarray, threshold: float) -> float:
    candidate_pairs = [(5, 6), (11, 12), (5, 11), (6, 12)]
    distances: list[float] = []
    for first, second in candidate_pairs:
        if confidences[first] >= threshold and confidences[second] >= threshold:
            distances.append(float(np.linalg.norm(keypoints_xy[first] - keypoints_xy[second])))
    return float(np.mean(distances)) if distances else np.nan


def compute_pose_metrics(
    keypoints: np.ndarray,
    confidence_threshold: float,
) -> dict[str, float]:
    confidences = keypoints[:, :, 2]
    visible = confidences >= confidence_threshold
    per_frame_coverage = visible.mean(axis=1)

    visible_confidences = confidences[visible]
    mean_confidence = float(visible_confidences.mean()) if visible_confidences.size else 0.0

    jitter_values: list[float] = []
    for index in range(1, len(keypoints)):
        prev_conf = confidences[index - 1]
        curr_conf = confidences[index]
        valid = (prev_conf >= confidence_threshold) & (curr_conf >= confidence_threshold)
        if not valid.any():
            continue

        prev_xy = keypoints[index - 1, :, :2]
        curr_xy = keypoints[index, :, :2]
        scale = np.nanmean(
            [
                torso_scale(prev_xy, prev_conf, confidence_threshold),
                torso_scale(curr_xy, curr_conf, confidence_threshold),
            ]
        )
        if np.isnan(scale) or scale <= 1e-6:
            continue

        displacement = np.linalg.norm(curr_xy[valid] - prev_xy[valid], axis=1).mean()
        jitter_values.append(float(displacement / scale))

    return {
        "mean_visible_keypoint_ratio": float(per_frame_coverage.mean()),
        "fully_tracked_frame_ratio": float((per_frame_coverage == 1.0).mean()),
        "mean_confidence": mean_confidence,
        "temporal_jitter_proxy": float(np.mean(jitter_values)) if jitter_values else np.nan,
    }


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    confidences: np.ndarray,
    threshold: float,
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = frame_bgr.copy()
    for start, end in COCO_EDGES:
        if confidences[start] >= threshold and confidences[end] >= threshold:
            start_point = tuple(np.round(keypoints_xy[start]).astype(int))
            end_point = tuple(np.round(keypoints_xy[end]).astype(int))
            cv2.line(canvas, start_point, end_point, color, 2, cv2.LINE_AA)

    for point, confidence in zip(keypoints_xy, confidences):
        if confidence < threshold:
            continue
        center = tuple(np.round(point).astype(int))
        cv2.circle(canvas, center, 3, color, -1, cv2.LINE_AA)
    return canvas


@dataclass
class PoseInferenceOutput:
    keypoints_xy: np.ndarray
    confidences: np.ndarray


class BlazePoseExtractor:
    model_name = "MediaPipe BlazePose"

    def __init__(self):
        import mediapipe as mp

        self.mp = mp
        model_path = ensure_mediapipe_model()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def infer(self, frame_bgr: np.ndarray) -> PoseInferenceOutput:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.pose.detect(mp_image)
        height, width = frame_bgr.shape[:2]

        keypoints_xy = np.zeros((17, 2), dtype=np.float32)
        confidences = np.zeros(17, dtype=np.float32)

        if not result.pose_landmarks:
            return PoseInferenceOutput(keypoints_xy=keypoints_xy, confidences=confidences)

        landmarks = result.pose_landmarks[0]
        for target_index, source_index in enumerate(MEDIAPIPE_TO_COCO_17):
            landmark = landmarks[source_index]
            keypoints_xy[target_index] = [landmark.x * width, landmark.y * height]
            confidences[target_index] = getattr(landmark, "visibility", 0.0)

        return PoseInferenceOutput(keypoints_xy=keypoints_xy, confidences=confidences)

    def close(self):
        self.pose.close()


class YoloPoseExtractor:
    model_name = "YOLO11n-pose"

    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO("yolo11n-pose.pt")

    def infer(self, frame_bgr: np.ndarray) -> PoseInferenceOutput:
        result = self.model.predict(
            source=frame_bgr,
            verbose=False,
            conf=0.25,
            imgsz=640,
            device="cpu",
        )[0]

        keypoints_xy = np.zeros((17, 2), dtype=np.float32)
        confidences = np.zeros(17, dtype=np.float32)

        if result.keypoints is None or result.keypoints.data is None:
            return PoseInferenceOutput(keypoints_xy=keypoints_xy, confidences=confidences)

        data = result.keypoints.data.cpu().numpy()
        if len(data) == 0:
            return PoseInferenceOutput(keypoints_xy=keypoints_xy, confidences=confidences)

        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_index = int(np.argmax(areas))
        selected = data[best_index]
        keypoints_xy = selected[:, :2].astype(np.float32)
        confidences = selected[:, 2].astype(np.float32)

        return PoseInferenceOutput(keypoints_xy=keypoints_xy, confidences=confidences)

    def close(self):
        return None


def process_video_with_model(
    model,
    video_path: Path,
    max_frames: int | None,
    confidence_threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_outputs: list[np.ndarray] = []
    processed_frames = 0
    start = time.perf_counter()

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        output = model.infer(frame_bgr)
        frame_outputs.append(
            np.concatenate(
                [output.keypoints_xy, output.confidences[:, None]],
                axis=1,
            )
        )
        processed_frames += 1

        if max_frames and processed_frames >= max_frames:
            break

    elapsed = time.perf_counter() - start
    capture.release()

    if not frame_outputs:
        raise RuntimeError(f"No frames processed for {video_path}")

    keypoints = np.stack(frame_outputs).astype(np.float32)
    metrics = compute_pose_metrics(keypoints, confidence_threshold=confidence_threshold)
    metrics.update(
        {
            "processed_frames": processed_frames,
            "inference_seconds": elapsed,
            "effective_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        }
    )
    return metrics, keypoints


def save_overlay_comparison(
    video_path: Path,
    output_path: Path,
    confidence_threshold: float,
    frame_index: int = 30,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for overlay: {video_path}")

    current_index = 0
    frame_bgr = None
    while current_index <= frame_index:
        ok, next_frame = capture.read()
        if not ok:
            break
        frame_bgr = next_frame
        current_index += 1
    capture.release()

    if frame_bgr is None:
        raise RuntimeError(f"Could not read overlay frame from {video_path}")

    models = [BlazePoseExtractor(), YoloPoseExtractor()]
    try:
        rendered_frames = []
        titles = ["Original"]
        rendered_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        palette = {
            "MediaPipe BlazePose": (80, 220, 100),
            "YOLO11n-pose": (70, 140, 255),
        }

        for model in models:
            output = model.infer(frame_bgr)
            overlay = draw_pose_overlay(
                frame_bgr,
                output.keypoints_xy,
                output.confidences,
                threshold=confidence_threshold,
                color=palette[model.model_name],
            )
            rendered_frames.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            titles.append(model.model_name)

        figure, axes = plt.subplots(1, len(rendered_frames), figsize=(15, 5))
        for axis, image, title in zip(axes, rendered_frames, titles):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")

        figure.suptitle(f"Pose overlay comparison for {video_path.stem}")
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
    finally:
        for model in models:
            model.close()


def benchmark_pose_models(
    manifest: pd.DataFrame,
    benchmark_videos: pd.DataFrame,
    output_dir: Path,
    max_frames: int,
    confidence_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [BlazePoseExtractor(), YoloPoseExtractor()]
    detail_rows: list[dict] = []

    try:
        for model in models:
            for row in benchmark_videos.itertuples(index=False):
                video_path = PROJECT_ROOT / row.video_path
                metrics, _ = process_video_with_model(
                    model=model,
                    video_path=video_path,
                    max_frames=max_frames,
                    confidence_threshold=confidence_threshold,
                )
                detail_rows.append(
                    {
                        "model_name": model.model_name,
                        "video_id": row.video_id,
                        "split": row.split,
                        "quality_label": row.quality_label,
                        **metrics,
                    }
                )
    finally:
        for model in models:
            model.close()

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby("model_name", as_index=False)
        .agg(
            benchmark_videos=("video_id", "nunique"),
            processed_frames=("processed_frames", "sum"),
            mean_effective_fps=("effective_fps", "mean"),
            mean_visible_keypoint_ratio=("mean_visible_keypoint_ratio", "mean"),
            mean_fully_tracked_frame_ratio=("fully_tracked_frame_ratio", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_temporal_jitter_proxy=("temporal_jitter_proxy", "mean"),
        )
        .sort_values(
            ["mean_visible_keypoint_ratio", "mean_effective_fps"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    max_fps = max(float(summary_df["mean_effective_fps"].max()), 1e-6)
    summary_df["selection_score"] = (
        0.5 * summary_df["mean_visible_keypoint_ratio"]
        + 0.2 * summary_df["mean_fully_tracked_frame_ratio"]
        + 0.2 * (summary_df["mean_effective_fps"] / max_fps)
        + 0.1 * summary_df["mean_confidence"]
    )
    summary_df = summary_df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    summary_df["recommended_for_project"] = False
    summary_df.loc[0, "recommended_for_project"] = True

    detail_df.to_csv(output_dir / "task2_pose_benchmark_detail.csv", index=False)
    summary_df.to_csv(output_dir / "task2_pose_benchmark_summary.csv", index=False)
    benchmark_videos.to_csv(output_dir / "task2_benchmark_video_sample.csv", index=False)

    return detail_df, summary_df


def run_pipeline(
    total_benchmark_videos: int = 8,
    max_frames: int = 75,
    confidence_threshold: float = 0.5,
) -> dict[str, object]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_dataset_manifest()
    summary = dataset_summary(manifest)
    dataset_options_df = external_dataset_options()
    benchmark_videos = select_benchmark_videos(manifest, total_videos=total_benchmark_videos)

    manifest.to_csv(REPORTS_DIR / "task1_dataset_manifest.csv", index=False)
    dataset_options_df.to_csv(REPORTS_DIR / "task1_candidate_datasets.csv", index=False)
    with (REPORTS_DIR / "task1_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    detail_df, summary_df = benchmark_pose_models(
        manifest=manifest,
        benchmark_videos=benchmark_videos,
        output_dir=REPORTS_DIR,
        max_frames=max_frames,
        confidence_threshold=confidence_threshold,
    )

    overlay_video = PROJECT_ROOT / benchmark_videos.iloc[0]["video_path"]
    overlay_path = FIGURES_DIR / "task2_pose_overlay_comparison.png"
    save_overlay_comparison(
        video_path=overlay_video,
        output_path=overlay_path,
        confidence_threshold=confidence_threshold,
    )

    return {
        "manifest": manifest,
        "summary": summary,
        "dataset_options": dataset_options_df,
        "benchmark_videos": benchmark_videos,
        "benchmark_detail": detail_df,
        "benchmark_summary": summary_df,
        "overlay_path": overlay_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 and 2 dataset audit + pose benchmark")
    parser.add_argument("--benchmark-videos", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=75)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        total_benchmark_videos=args.benchmark_videos,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
    )

    recommended_model = outputs["benchmark_summary"].loc[0, "model_name"]
    print("Dataset manifest rows:", len(outputs["manifest"]))
    print("Benchmark videos:", len(outputs["benchmark_videos"]))
    print("Recommended pose model:", recommended_model)
    print("Reports written to:", REPORTS_DIR)
    print("Overlay figure:", outputs["overlay_path"])


if __name__ == "__main__":
    main()
