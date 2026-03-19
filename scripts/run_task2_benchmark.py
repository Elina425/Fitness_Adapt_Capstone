from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np

from fitness_adapt.evaluation import extraction_speed_fps, knee_angle_separation_score
from fitness_adapt.io_utils import load_json
from fitness_adapt.pose import extract_torchvision_keypointrcnn_coco17, extract_yolo11_pose_coco17
from fitness_adapt.project import ProjectPaths

Interval = Tuple[float, float]


def union_intervals(raw: List[List[float]]) -> List[Interval]:
    if not raw:
        return []
    intervals = [(float(a), float(b)) for a, b in raw]
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Interval] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def get_union_error_intervals(video_key: str, error_fwd, error_inward) -> List[Interval]:
    fwd = union_intervals(error_fwd.get(video_key, []))
    inward = union_intervals(error_inward.get(video_key, []))
    return union_intervals([[s, e] for s, e in (fwd + inward)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="Optional project root. Auto-discovered if omitted.")
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.root if args.root else None)
    train_keys = load_json(paths.split_path("train"))
    error_fwd = load_json(paths.error_knees_forward_path)
    error_inward = load_json(paths.error_knees_inward_path)

    sample_keys = train_keys[:2]
    frame_stride = 6
    max_frames = 80
    conf_threshold = 0.35

    models = [
        (
            "yolo11n_pose",
            lambda vp: extract_yolo11_pose_coco17(
                vp,
                frame_stride=frame_stride,
                max_frames=max_frames,
                conf_threshold=conf_threshold,
                yolo_weights="yolo11n-pose.pt",
                device="cpu",
            ),
        ),
        (
            "keypointrcnn_pose",
            lambda vp: extract_torchvision_keypointrcnn_coco17(
                vp,
                frame_stride=frame_stride,
                max_frames=max_frames,
                conf_threshold=conf_threshold,
                device="cpu",
            ),
        ),
    ]

    for model_name, extractor in models:
        fps_vals = []
        sep_vals = []
        for k in sample_keys:
            video_path = str(paths.squat_video_dir / f"{k}.mp4")
            err_intervals = get_union_error_intervals(k, error_fwd, error_inward)
            out = extractor(video_path)
            fps_vals.append(extraction_speed_fps(out.extraction_time_sec, out.keypoints_xy.shape[0]))
            sep_vals.append(knee_angle_separation_score(out.keypoints_xy, out.times_sec, err_intervals))
        print(model_name, "mean_fps", round(float(np.mean(fps_vals)), 2), "mean_sep", round(float(np.mean(sep_vals)), 3))


if __name__ == "__main__":
    main()

