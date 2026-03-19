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

    @staticmethod
    def discover_root(start: str | Path | None = None) -> Path:
        """
        Discover repository root by searching upward for `pyproject.toml` or `.git`.
        Falls back to current working directory.
        """
        if start is None:
            cur = Path.cwd().resolve()
        else:
            cur = Path(start).resolve()
        for candidate in [cur, *cur.parents]:
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                return candidate
        return cur

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        root_path = cls.discover_root(root)
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

