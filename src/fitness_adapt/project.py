from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized path resolution for consistent project structure."""

    root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path
    models_dir: Path
    notebooks_dir: Path

    @classmethod
    def from_root(cls, root: str | Path = "/workspace") -> "ProjectPaths":
        root_path = Path(root).resolve()
        data_dir = root_path / "data"
        return cls(
            root=root_path,
            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            interim_dir=data_dir / "interim",
            processed_dir=data_dir / "processed",
            outputs_dir=root_path / "outputs",
            models_dir=root_path / ".models",
            notebooks_dir=root_path / "notebooks",
        )

    @property
    def squat_video_dir(self) -> Path:
        """Supports both legacy root layout and structured data/raw layout."""
        candidate_new = self.raw_dir / "videos_squat"
        candidate_legacy = self.root / "videos_squat"
        return candidate_new if candidate_new.exists() else candidate_legacy

    def split_path(self, split_name: str) -> Path:
        candidate_new = self.raw_dir / f"{split_name}_keys.json"
        candidate_legacy = self.root / f"{split_name}_keys.json"
        return candidate_new if candidate_new.exists() else candidate_legacy

    @property
    def error_knees_forward_path(self) -> Path:
        candidate_new = self.raw_dir / "error_knees_forward.json"
        candidate_legacy = self.root / "error_knees_forward.json"
        return candidate_new if candidate_new.exists() else candidate_legacy

    @property
    def error_knees_inward_path(self) -> Path:
        candidate_new = self.raw_dir / "error_knees_inward.json"
        candidate_legacy = self.root / "error_knees_inward.json"
        return candidate_new if candidate_new.exists() else candidate_legacy

