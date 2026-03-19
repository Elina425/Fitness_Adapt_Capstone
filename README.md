# Fitness Adapt Capstone

Presentation-ready structure for exercise recognition + form quality assessment using real project data.

## Project Structure

- `src/fitness_adapt/` — core package (paths, labels, pose extraction, evaluation).
- `notebooks/` — numbered exploratory and milestone notebooks.
- `scripts/` — lightweight CLI entry points for reproducible runs.
- `data/raw/` — expected location for raw data (current dataset is still supported in legacy root paths).
- `data/interim/` — temporary intermediate artifacts.
- `data/processed/` — model-ready outputs.
- `docs/` — documentation.
- `tests/` — test modules.
- `outputs/` — generated charts/results (created as needed).

## Current Implemented Milestones

- **Task 1**: Dataset/label definition from real provided files.
  - Notebook: `notebooks/01_dataset_and_quality_labels.ipynb`
- **Task 2**: Pose extraction + benchmark.
  - Notebook: `notebooks/02_pose_estimation_benchmark.ipynb`
- **Tasks 3–8**: Preprocessing, biomechanical features, BiLSTM multi-task training, evaluation, ablation, and personalization adapter.
  - Notebook: `notebooks/03_preprocess_features_train_personalize.ipynb`
  - Script: `python3 scripts/run_task3_8_pipeline.py`
- **Task 9**: Real-time integration demo scaffold with live overlay + quality feedback.
  - Script: `python3 scripts/run_task9_realtime_demo.py --video /workspace/videos_squat/32903_8.mp4`

## Data Compatibility

Path resolution supports both:

1. **Legacy layout (currently in this repo)**:
   - `videos_squat/`
   - `train_keys.json`, `val_keys.json`, `test_keys.json`
   - `error_knees_forward.json`, `error_knees_inward.json`
2. **Structured layout**:
   - `data/raw/videos_squat/`
   - `data/raw/*_keys.json`
   - `data/raw/error_knees_*.json`

You can migrate to `data/raw/` later without code changes.

## Setup

Install editable package:

- `pip3 install -e .`

Or install from requirements:

- `pip3 install -r requirements.txt`

## Notes on MediaPipe

In this cloud runtime, MediaPipe tasks may fail with missing system libs (e.g. `libEGL.so.1`).  
YOLOv11 and torchvision Keypoint R-CNN extractors are fully runnable in current environment.

## Important dataset constraint

Current provided data is squat-only, so exercise classification currently has one class (`squat`).
The classification head and metrics are still implemented and runnable for multi-class extension when new exercise videos are added.

