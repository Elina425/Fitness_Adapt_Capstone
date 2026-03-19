"""Fitness Adapt Capstone package."""

from .project import ProjectPaths
from .preprocess import PreprocessConfig, preprocess_keypoints
from .features import compute_joint_angles

__all__ = ["ProjectPaths", "PreprocessConfig", "preprocess_keypoints", "compute_joint_angles"]

