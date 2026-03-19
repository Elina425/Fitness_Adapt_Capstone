# AGENTS.md

## Cursor Cloud specific instructions

### Repository Overview

This is a **data-only repository** containing a labeled video dataset for squat exercise form error detection. There is no application source code, no build system, no test framework, and no services to run.

**Contents:**
- `videos_squat/` — 1,739 MP4 video clips of people performing squats (480x600, 30 fps, ~930 MB total)
- `error_knees_forward.json` — Temporal annotations for "knees going forward" errors (1,623 entries; value is array of `[start_sec, end_sec]` intervals, empty array = no error)
- `error_knees_inward.json` — Temporal annotations for "knees caving inward" errors (1,623 entries, same format)
- `train_keys.json` / `val_keys.json` / `test_keys.json` — Train/val/test split (1,136 / 243 / 244 video IDs)
- `traj_nan.json` — 19 filenames flagged for NaN trajectory data (exclusion list)

### Development Environment

- **Python 3.12** is available system-wide.
- Key packages for working with this dataset: `opencv-python-headless`, `matplotlib`, `numpy`. These are installed via `pip3 install --user`.
- There are no lint checks, automated tests, or build steps — this is a pure data repo.
- To verify the dataset is intact, load the JSON files with `json.load()` and open videos with `cv2.VideoCapture()`.

### Key Gotchas

- Video IDs in JSON annotations do NOT include the `.mp4` extension; append `.mp4` when constructing file paths.
- `traj_nan.json` entries have `.json` extensions (e.g. `"37165_2.json"`) — strip the extension to get the video ID.
- There are 1,739 video files on disk but only 1,623 annotated/split IDs. The extra 116 videos are not part of the annotated dataset.
- `ReadMe.md.docx` is a Word document, not Markdown, despite the `.md` in its name.
