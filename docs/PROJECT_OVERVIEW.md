# Project Overview

This repository is structured to support a staged capstone workflow:

1. Dataset + quality label construction.
2. Pose extraction and model benchmarking.
3. Keypoint preprocessing.
4. Biomechanical feature extraction.
5. Sequence model training (BiLSTM / CNN variants / transformer-based variants).
6. Multi-head quality scoring.
7. Evaluation + ablations.
8. Personalization adapters.
9. Real-time app integration.

The first two stages are already implemented in `notebooks/01_*` and `notebooks/02_*`.

