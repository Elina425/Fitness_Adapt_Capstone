from __future__ import annotations

import cv2

from fitness_adapt.io_utils import load_json
from fitness_adapt.labels import compute_quality_labels_for_key, load_error_intervals
from fitness_adapt.project import ProjectPaths


def get_duration_sec(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frames / fps if fps > 0 else 0.0


def main():
    paths = ProjectPaths.from_root()
    train_keys = load_json(paths.split_path("train"))
    error_fwd, error_inward = load_error_intervals(paths)

    print("sample quality labels")
    for key in train_keys[:8]:
        duration = get_duration_sec(paths.squat_video_dir / f"{key}.mp4")
        labels = compute_quality_labels_for_key(
            key,
            exercise_type="squat",
            duration_sec=duration,
            error_fwd=error_fwd,
            error_inward=error_inward,
        )
        print(key, labels.quality_binary, round(labels.quality_score, 4))


if __name__ == "__main__":
    main()

