import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

Interval = Tuple[float, float]


@dataclass(frozen=True)
class QualityLabels:
    """
    Derived labels from this project's annotations.

    - exercise_type: currently always "squat" (your project ships squat videos).
    - quality_binary: 1 if no annotated knee-error intervals exist.
    - quality_score: continuous [0, 1] score derived from error coverage.
    """

    exercise_type: str
    quality_binary: int
    quality_score: float


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_error_intervals(project_root: Path) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
    """
    Loads knee forward + knee inward annotations.

    Each json maps "<video_key>" -> list of [start_sec, end_sec] intervals.
    An empty list indicates "no annotated error".
    """
    fwd = _load_json(project_root / "error_knees_forward.json")
    inward = _load_json(project_root / "error_knees_inward.json")
    return fwd, inward


def _intervals_union(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Interval] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _clamp_interval(interval: Interval, lo: float, hi: float) -> Interval | None:
    start, end = interval
    start = max(start, lo)
    end = min(end, hi)
    if end <= start:
        return None
    return (start, end)


def intervals_to_coverage_sec(intervals: List[Interval], duration_sec: float) -> float:
    """
    Computes total union coverage in seconds, clamped to [0, duration_sec].
    """
    clamped: List[Interval] = []
    for it in intervals:
        maybe = _clamp_interval(it, 0.0, duration_sec)
        if maybe is not None:
            clamped.append(maybe)
    merged = _intervals_union(clamped)
    total = sum(end - start for start, end in merged)
    if duration_sec <= 0:
        return 0.0
    return max(0.0, min(1.0, total / duration_sec))


def compute_quality_labels_for_key(
    video_key: str,
    *,
    exercise_type: str,
    duration_sec: float,
    error_fwd: Dict[str, List[List[float]]],
    error_inward: Dict[str, List[List[float]]],
    quality_missing_default: float = 0.0,
) -> QualityLabels:
    """
    Derives labels from the project's two knee-error annotations.

    quality_score = 1 - union_error_coverage.
    quality_binary = 1 iff union_error_coverage == 0.

    If a key is missing from both jsons, we treat it as missing annotations
    and return quality_missing_default.
    """
    fwd_intervals_raw = error_fwd.get(video_key, [])
    inward_intervals_raw = error_inward.get(video_key, [])

    if video_key not in error_fwd and video_key not in error_inward:
        # Defensive fallback; should not happen for keys coming from train/val/test splits.
        return QualityLabels(
            exercise_type=exercise_type,
            quality_binary=int(quality_missing_default > 0.5),
            quality_score=float(quality_missing_default),
        )

    intervals: List[Interval] = []
    for start, end in fwd_intervals_raw:
        intervals.append((float(start), float(end)))
    for start, end in inward_intervals_raw:
        intervals.append((float(start), float(end)))

    coverage = intervals_to_coverage_sec(intervals, duration_sec)
    quality_score = 1.0 - coverage
    quality_binary = int(coverage == 0.0)
    return QualityLabels(
        exercise_type=exercise_type,
        quality_binary=quality_binary,
        quality_score=quality_score,
    )

