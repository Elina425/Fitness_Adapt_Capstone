from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .io_utils import load_json
from .project import ProjectPaths

Interval = Tuple[float, float]


@dataclass(frozen=True)
class QualityLabels:
    exercise_type: str
    quality_binary: int
    quality_score: float


def load_error_intervals(paths: ProjectPaths) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
    fwd = load_json(paths.error_knees_forward_path)
    inward = load_json(paths.error_knees_inward_path)
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
    fwd_intervals_raw = error_fwd.get(video_key, [])
    inward_intervals_raw = error_inward.get(video_key, [])

    if video_key not in error_fwd and video_key not in error_inward:
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
    return QualityLabels(exercise_type=exercise_type, quality_binary=quality_binary, quality_score=quality_score)

