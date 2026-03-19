from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from pathlib import Path


@dataclass(frozen=True)
class PoseExtractionResult:
    keypoints_xy: np.ndarray  # (T, 17, 2)
    keypoints_conf: np.ndarray  # (T, 17)
    times_sec: np.ndarray  # (T,)
    fps: float
    extraction_time_sec: float


def _open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, float(fps), total


def extract_mediapipe_blazepose_coco17(
    video_path: str,
    *,
    frame_stride: int = 2,
    max_frames: Optional[int] = 200,
    conf_threshold: float = 0.3,
    model_complexity: int = 1,
) -> PoseExtractionResult:
    """
    Extracts COCO-17 keypoints from MediaPipe BlazePose.
    Output coordinates are in pixels.

    Note on this environment:
    The classic `mediapipe.solutions.pose` API is not available in this runtime,
    so this implementation uses the newer MediaPipe **Tasks** API
    (`mediapipe.tasks.python.vision.PoseLandmarker`).
    """
    cap, fps, total = _open_video(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MediaPipe Tasks outputs 33 pose landmarks (PoseLandmarker.PoseLandmark enum).
    # Landmark indices (0..32):
    # nose=0, left_eye=2, right_eye=5, left_ear=7, right_ear=8,
    # left_shoulder=11, right_shoulder=12, left_elbow=13, right_elbow=14,
    # left_wrist=15, right_wrist=16,
    # left_hip=23, right_hip=24, left_knee=25, right_knee=26, left_ankle=27, right_ankle=28
    landmark_map = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    # Download a pose landmarker model bundle if missing.
    # (The package doesn't ship model assets in this runtime.)
    import os
    import urllib.request

    from mediapipe.tasks.python.vision import PoseLandmarker

    model_dir = Path("/workspace/.models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pose_landmarker_heavy.task"
    if not model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        print(f"[mediapipe] Downloading pose landmarker model to {model_path} ...")
        urllib.request.urlretrieve(url, str(model_path))

    from mediapipe.tasks.python.vision.core import image as mp_image
    from mediapipe.tasks.python.vision.core.image import ImageFormat

    keypoints_xy = []
    keypoints_conf = []
    times = []

    start = time.perf_counter()
    with PoseLandmarker.create_from_model_path(str(model_path)) as landmarker:
        frame_idx = 0
        extracted = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            if max_frames is not None and extracted >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp_image.Image(ImageFormat.SRGB, frame_rgb)
            res = landmarker.detect(mp_img)

            kp = np.zeros((17, 2), dtype=np.float32)
            kc = np.zeros((17,), dtype=np.float32)

            # res.pose_landmarks is a list[list[NormalizedLandmark]] per detected pose.
            if res.pose_landmarks:
                # Select the detected pose with the highest avg visibility.
                poses = res.pose_landmarks
                best_i = 0
                best_score = -1.0
                for i, pose_lms in enumerate(poses):
                    vis = [float(lm.visibility) for lm in pose_lms if hasattr(lm, "visibility")]
                    score = float(np.mean(vis)) if vis else 0.0
                    if score > best_score:
                        best_score = score
                        best_i = i
                landmarks = poses[best_i]

                for j, lm_idx in enumerate(landmark_map):
                    lm = landmarks[lm_idx]
                    conf = float(getattr(lm, "visibility", 1.0))
                    if conf >= conf_threshold:
                        kp[j, 0] = float(lm.x * w)
                        kp[j, 1] = float(lm.y * h)
                        kc[j] = conf

            keypoints_xy.append(kp)
            keypoints_conf.append(kc)
            times.append(frame_idx / fps)

            extracted += 1
            frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - start

    kps = np.stack(keypoints_xy, axis=0) if keypoints_xy else np.zeros((0, 17, 2), dtype=np.float32)
    conf = np.stack(keypoints_conf, axis=0) if keypoints_conf else np.zeros((0, 17), dtype=np.float32)
    t = np.asarray(times, dtype=np.float32)
    return PoseExtractionResult(
        keypoints_xy=kps,
        keypoints_conf=conf,
        times_sec=t,
        fps=fps,
        extraction_time_sec=float(elapsed),
    )


def extract_yolo11_pose_coco17(
    video_path: str,
    *,
    frame_stride: int = 2,
    max_frames: Optional[int] = 200,
    conf_threshold: float = 0.3,
    yolo_weights: str = "yolo11n-pose.pt",
    device: str = "cpu",
) -> PoseExtractionResult:
    """
    Extracts COCO-17 keypoints from YOLOv11 pose (Ultralytics).

    Notes:
    - This runs YOLO on sampled frames for speed benchmarking (not full extraction).
    - Coordinates are in pixels (as returned by Ultralytics).
    """
    from ultralytics import YOLO

    cap, fps, _ = _open_video(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(yolo_weights)

    keypoints_xy = []
    keypoints_conf = []
    times = []

    start = time.perf_counter()
    frame_idx = 0
    extracted = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if max_frames is not None and extracted >= max_frames:
            break

        # Ultralytics expects RGB/BGR is usually ok; pass frame directly.
        results = model.predict(
            source=frame_bgr,
            verbose=False,
            conf=conf_threshold,
            device=device,
        )
        r0 = results[0]

        kp = np.zeros((17, 2), dtype=np.float32)
        kc = np.zeros((17,), dtype=np.float32)

        if r0.keypoints is not None and len(r0.keypoints) > 0:
            # r0.keypoints.xy: (num_instances, 17, 2)
            xy = r0.keypoints.xy.cpu().numpy()  # type: ignore[union-attr]
            confs = r0.keypoints.conf.cpu().numpy()  # (num_instances, 17)

            # Choose the instance with the best overall keypoint confidence.
            best = int(np.argmax(confs.sum(axis=1)))
            kp_det = xy[best]
            kc_det = confs[best]

            for j in range(17):
                if kc_det[j] >= conf_threshold:
                    kp[j] = kp_det[j]
                    kc[j] = kc_det[j]

        keypoints_xy.append(kp)
        keypoints_conf.append(kc)
        times.append(frame_idx / fps)

        extracted += 1
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - start

    kps = np.stack(keypoints_xy, axis=0) if keypoints_xy else np.zeros((0, 17, 2), dtype=np.float32)
    conf = np.stack(keypoints_conf, axis=0) if keypoints_conf else np.zeros((0, 17), dtype=np.float32)
    t = np.asarray(times, dtype=np.float32)
    return PoseExtractionResult(
        keypoints_xy=kps,
        keypoints_conf=conf,
        times_sec=t,
        fps=fps,
        extraction_time_sec=float(elapsed),
    )


def extract_torchvision_keypointrcnn_coco17(
    video_path: str,
    *,
    frame_stride: int = 2,
    max_frames: Optional[int] = 200,
    conf_threshold: float = 0.5,
    device: str = "cpu",
) -> PoseExtractionResult:
    """
    Extract COCO-17 keypoints using torchvision's Keypoint R-CNN.

    Notes:
    - Output coordinates are in pixels.
    - This is used as a CPU-friendly alternative for environments where
      MediaPipe can't run due to missing system libraries.
    """
    import torch
    import torchvision
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNN_ResNet50_FPN_Weights
    from PIL import Image

    cap, fps, _ = _open_video(video_path)
    keypoints_xy = []
    keypoints_conf = []
    times = []

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    preprocess = weights.transforms()

    start = time.perf_counter()
    frame_idx = 0
    extracted = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if max_frames is not None and extracted >= max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        image_tensor = preprocess(pil).to(device)

        with torch.no_grad():
            outputs = model([image_tensor])[0]

        kp = np.zeros((17, 2), dtype=np.float32)
        kc = np.zeros((17,), dtype=np.float32)

        if len(outputs.get("keypoints", [])) > 0:
            keypoints = outputs["keypoints"].detach().cpu().numpy()  # (N, 17, 3)
            scores_kp = keypoints[:, :, 2]  # per-keypoint score
            best = int(np.argmax(scores_kp.sum(axis=1)))
            kp_det = keypoints[best, :, :2]
            kc_det = scores_kp[best]

            for j in range(17):
                if kc_det[j] >= conf_threshold:
                    kp[j] = kp_det[j]
                    kc[j] = kc_det[j]

        keypoints_xy.append(kp)
        keypoints_conf.append(kc)
        times.append(frame_idx / fps)

        extracted += 1
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - start

    kps = np.stack(keypoints_xy, axis=0) if keypoints_xy else np.zeros((0, 17, 2), dtype=np.float32)
    conf = np.stack(keypoints_conf, axis=0) if keypoints_conf else np.zeros((0, 17), dtype=np.float32)
    t = np.asarray(times, dtype=np.float32)
    return PoseExtractionResult(
        keypoints_xy=kps,
        keypoints_conf=conf,
        times_sec=t,
        fps=fps,
        extraction_time_sec=float(elapsed),
    )

