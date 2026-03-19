from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .project import ProjectPaths


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


def extract_yolo11_pose_coco17(
    video_path: str,
    *,
    frame_stride: int = 2,
    max_frames: Optional[int] = 200,
    conf_threshold: float = 0.3,
    yolo_weights: str = "yolo11n-pose.pt",
    device: str = "cpu",
) -> PoseExtractionResult:
    from ultralytics import YOLO

    cap, fps, _ = _open_video(video_path)
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

        results = model.predict(source=frame_bgr, verbose=False, conf=conf_threshold, device=device)
        r0 = results[0]
        kp = np.zeros((17, 2), dtype=np.float32)
        kc = np.zeros((17,), dtype=np.float32)

        if r0.keypoints is not None and len(r0.keypoints) > 0:
            xy = r0.keypoints.xy.cpu().numpy()
            confs = r0.keypoints.conf.cpu().numpy()
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
    return PoseExtractionResult(kps, conf, t, fps, float(elapsed))


def extract_torchvision_keypointrcnn_coco17(
    video_path: str,
    *,
    frame_stride: int = 2,
    max_frames: Optional[int] = 200,
    conf_threshold: float = 0.5,
    device: str = "cpu",
) -> PoseExtractionResult:
    import torch
    from PIL import Image
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNN_ResNet50_FPN_Weights

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
        image_tensor = preprocess(Image.fromarray(frame_rgb)).to(device)
        with torch.no_grad():
            outputs = model([image_tensor])[0]

        kp = np.zeros((17, 2), dtype=np.float32)
        kc = np.zeros((17,), dtype=np.float32)
        if len(outputs.get("keypoints", [])) > 0:
            keypoints = outputs["keypoints"].detach().cpu().numpy()
            scores_kp = keypoints[:, :, 2]
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
    return PoseExtractionResult(kps, conf, t, fps, float(elapsed))


def extract_mediapipe_blazepose_coco17(
    video_path: str,
    *,
    frame_stride: int = 2,
    max_frames: Optional[int] = 200,
    conf_threshold: float = 0.3,
    model_bundle_name: str = "pose_landmarker_heavy.task",
    paths: Optional[ProjectPaths] = None,
) -> PoseExtractionResult:
    """
    Optional MediaPipe extractor.
    This can fail in minimal cloud containers missing `libEGL.so.1`.
    """
    from mediapipe.tasks.python.vision import PoseLandmarker
    from mediapipe.tasks.python.vision.core import image as mp_image
    from mediapipe.tasks.python.vision.core.image import ImageFormat

    if paths is None:
        paths = ProjectPaths.from_root()

    cap, fps, _ = _open_video(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    landmark_map = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    model_path = paths.models_dir / model_bundle_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model bundle at {model_path}. "
            "Download it manually if you want to benchmark MediaPipe in this environment."
        )

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
            res = landmarker.detect(mp_image.Image(ImageFormat.SRGB, frame_rgb))
            kp = np.zeros((17, 2), dtype=np.float32)
            kc = np.zeros((17,), dtype=np.float32)
            if res.pose_landmarks:
                poses = res.pose_landmarks
                best_i = int(np.argmax([np.mean([float(lm.visibility) for lm in pose]) for pose in poses]))
                landmarks = poses[best_i]
                for j, lm_idx in enumerate(landmark_map):
                    lm = landmarks[lm_idx]
                    conf = float(getattr(lm, "visibility", 1.0))
                    if conf >= conf_threshold:
                        kp[j] = [float(lm.x * w), float(lm.y * h)]
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
    return PoseExtractionResult(kps, conf, t, fps, float(elapsed))

