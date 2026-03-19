import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class VideoSample:
    video_id: str
    split: str  # train/val/test
    video_path: Path
    # Knee-error intervals in seconds; used as a quality proxy.
    knee_forward_intervals: List[Tuple[float, float]]
    knee_inward_intervals: List[Tuple[float, float]]

    @property
    def has_any_knee_error(self) -> bool:
        return len(self.knee_forward_intervals) + len(self.knee_inward_intervals) > 0


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _intervals_from_json(raw: dict, video_id: str) -> List[Tuple[float, float]]:
    intervals = raw.get(video_id, [])
    # JSON format: list[list[float,float]].
    out: List[Tuple[float, float]] = []
    for item in intervals:
        if not item:
            continue
        if isinstance(item, list) and len(item) == 2:
            out.append((float(item[0]), float(item[1])))
    return out


def load_dataset_manifest(
    root_dir: Path,
    videos_dir_name: str = "videos_squat",
) -> List[VideoSample]:
    """
    Builds a manifest from the real data in this repository:
    - videos_squat/<video_id>.mp4
    - train_keys.json / val_keys.json / test_keys.json
    - error_knees_forward.json / error_knees_inward.json (intervals in seconds)
    """
    root_dir = root_dir.resolve()
    videos_dir = root_dir / videos_dir_name

    train_ids = _load_json(root_dir / "train_keys.json")
    val_ids = _load_json(root_dir / "val_keys.json")
    test_ids = _load_json(root_dir / "test_keys.json")

    forward_raw = _load_json(root_dir / "error_knees_forward.json")
    inward_raw = _load_json(root_dir / "error_knees_inward.json")

    manifest: List[VideoSample] = []
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for video_id in ids:
            video_path = videos_dir / f"{video_id}.mp4"
            if not video_path.exists():
                # Keep manifest generation robust; missing files are better surfaced here.
                continue
            manifest.append(
                VideoSample(
                    video_id=video_id,
                    split=split_name,
                    video_path=video_path,
                    knee_forward_intervals=_intervals_from_json(forward_raw, video_id),
                    knee_inward_intervals=_intervals_from_json(inward_raw, video_id),
                )
            )

    return manifest


def build_frame_level_quality_mask(
    fps: float,
    num_frames: int,
    intervals_sec: List[Tuple[float, float]],
) -> List[int]:
    """
    Returns y[t] in {0,1} where 1 indicates "bad" frame.
    Uses time intervals in seconds by converting to frame indices.
    """
    y = [0 for _ in range(num_frames)]
    if fps <= 0:
        return y
    for start_sec, end_sec in intervals_sec:
        start_f = max(0, int(start_sec * fps))
        end_f = min(num_frames - 1, int(end_sec * fps))
        for t in range(start_f, end_f + 1):
            y[t] = 1
    return y

