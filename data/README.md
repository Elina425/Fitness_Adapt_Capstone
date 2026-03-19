# Data Layout

Recommended structure:

- `data/raw/videos_squat/*.mp4`
- `data/raw/train_keys.json`
- `data/raw/val_keys.json`
- `data/raw/test_keys.json`
- `data/raw/error_knees_forward.json`
- `data/raw/error_knees_inward.json`

Current repository still uses legacy root-level files.  
`fitness_adapt.project.ProjectPaths` supports both layouts automatically.

