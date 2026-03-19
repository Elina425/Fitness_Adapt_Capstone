import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


COCO_17_JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-6) -> float:
    """
    Angle ABC (degrees) at point B, using vectors BA and BC.
    a,b,c shape: (2,) or (3,)
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return float("nan")
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def knee_angle_from_coco17(kp17_xy: np.ndarray) -> float:
    """
    Compute average knee angle (degrees) from COCO17 joints.
    kp17_xy shape: (17,2); assumes non-missing.
    Uses:
      left_knee angle = angle(hip, knee, ankle) for left
      right_knee angle = angle(hip, knee, ankle) for right
    """
    if kp17_xy.shape != (17, 2):
        raise ValueError(f"Expected (17,2), got {kp17_xy.shape}")
    lh, rh = 11, 12
    lk, rk = 13, 14
    la, ra = 15, 16
    left = angle_at_joint(kp17_xy[lh], kp17_xy[lk], kp17_xy[la])
    right = angle_at_joint(kp17_xy[rh], kp17_xy[rk], kp17_xy[ra])
    if np.isnan(left) and np.isnan(right):
        return float("nan")
    if np.isnan(left):
        return float(right)
    if np.isnan(right):
        return float(left)
    return float((left + right) / 2.0)


@dataclass
class PoseExtractionResult:
    # Keypoints: (T,17,2) in pixels (or normalized-to-image space for MediaPipe).
    keypoints_xy: np.ndarray
    # Confidence: (T,17) where available.
    keypoints_conf: np.ndarray
    fps: float
    timings_s: Dict[str, float]


class MediaPipePoseLandmarker17:
    """
    MediaPipe PoseLandmarker wrapper that outputs COCO-17 keypoints
    (nose/eyes/ears/shoulders/elbows/wrists/hips/knees/ankles).
    """

    def __init__(self, task_model_path: str, score_threshold: float = 0.3):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        self._PoseLandmarker = PoseLandmarker
        self._task_model_path = task_model_path
        self._score_threshold = score_threshold
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            result_callback=None,
        )
        try:
            self._landmarker = PoseLandmarker.create_from_options(options)
        except OSError as e:
            # MediaPipe "tasks" runtime needs system EGL/OpenGL libs.
            raise RuntimeError(
                "MediaPipe PoseLandmarker failed to initialize (likely missing libEGL/OpenGL). "
                "This environment likely can't run it."
            ) from e

        # MediaPipe -> COCO-17 indices mapping (PoseLandmark enum values).
        # COCO order indices: 0 nose,1 left_eye,2 right_eye,3 left_ear,4 right_ear,5 left_shoulder,...
        self._mp_to_coco = {
            # eyes/ears
            0: 0,  # NOSE
            3: 1,  # LEFT_EYE_OUTER -> left_eye
            6: 2,  # RIGHT_EYE_OUTER -> right_eye
            7: 3,  # LEFT_EAR
            8: 4,  # RIGHT_EAR
            # shoulders
            11: 5,  # LEFT_SHOULDER
            12: 6,  # RIGHT_SHOULDER
            # elbows
            13: 7,
            14: 8,
            # wrists
            15: 9,
            16: 10,
            # hips
            23: 11,
            24: 12,
            # knees
            25: 13,
            26: 14,
            # ankles
            27: 15,
            28: 16,
        }

    def _extract_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Tuple[np.ndarray, np.ndarray]:
        # MediaPipe expects RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_result = self._landmarker.detect_for_video(frame_rgb, timestamp_ms)

        # Initialize outputs with NaNs so downstream can mask.
        kp_xy = np.full((17, 2), np.nan, dtype=np.float32)
        kp_conf = np.zeros((17,), dtype=np.float32)

        if not mp_result or not mp_result.pose_landmarks:
            return kp_xy, kp_conf

        # PoseLandmarksConnections supports multiple people, but usually 1.
        # We'll take the first pose.
        pose = mp_result.pose_landmarks[0]
        h, w = frame_bgr.shape[:2]

        for mp_idx, coco_idx in self._mp_to_coco.items():
            lm = pose[mp_idx]
            # lm is NormalizedLandmark.
            x_px = float(lm.x * w)
            y_px = float(lm.y * h)
            kp_xy[coco_idx] = (x_px, y_px)
            # MediaPipe PoseLandmarker returns "visibility" as z? In tasks API landmark has visibility attr.
            # We'll use "visibility" if present; else treat as 1.0.
            conf = getattr(lm, "visibility", 1.0)
            kp_conf[coco_idx] = float(conf)

        # Optionally zero out low confidence.
        low = kp_conf < self._score_threshold
        kp_conf[low] = 0.0
        kp_xy[low] = np.nan
        return kp_xy, kp_conf

    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> PoseExtractionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        keypoints_xy_list = []
        keypoints_conf_list = []

        t0 = time.perf_counter()
        frames_processed = 0
        timestamp_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frames_processed >= max_frames:
                break

            # Approx timestamp increment if fps is known; otherwise just increment by 33ms.
            if fps > 0:
                timestamp_ms = int(frames_processed * (1000.0 / fps))
            else:
                timestamp_ms = int(frames_processed * 33.3)

            kp_xy, kp_conf = self._extract_frame(frame, timestamp_ms)
            keypoints_xy_list.append(kp_xy)
            keypoints_conf_list.append(kp_conf)
            frames_processed += 1

        cap.release()
        total_s = time.perf_counter() - t0

        keypoints_xy = np.stack(keypoints_xy_list, axis=0) if keypoints_xy_list else np.empty((0, 17, 2), dtype=np.float32)
        keypoints_conf = np.stack(keypoints_conf_list, axis=0) if keypoints_conf_list else np.empty((0, 17), dtype=np.float32)

        return PoseExtractionResult(
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
            fps=fps,
            timings_s={"total": total_s},
        )


class YoloPose17:
    """
    Ultralytics YOLO pose wrapper that outputs COCO-17 keypoints in pixel space.
    """

    def __init__(
        self,
        model_name_or_path: str,
        conf_threshold: float = 0.25,
        max_det: int = 1,
    ):
        from ultralytics import YOLO

        self._model = YOLO(model_name_or_path)
        self._conf_threshold = conf_threshold
        self._max_det = max_det

        # COCO keypoint indices within YOLO's 17-length keypoints output are consistent with COCO-17 order.

    def extract_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        image_size: int = 640,
    ) -> PoseExtractionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        keypoints_xy_list = []
        keypoints_conf_list = []
        frames_processed = 0

        t0 = time.perf_counter()
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames is not None and frames_processed >= max_frames:
                break

            # Ultralytics handles BGR/RGB internally; pass numpy frame.
            results = self._model.predict(
                source=frame_bgr,
                imgsz=image_size,
                conf=self._conf_threshold,
                verbose=False,
                max_det=self._max_det,
            )
            # One result for one frame.
            res = results[0]

            kp_xy = np.full((17, 2), np.nan, dtype=np.float32)
            kp_conf = np.zeros((17,), dtype=np.float32)

            if res.keypoints is not None and len(res.keypoints) > 0:
                # Take first person detection.
                kps = res.keypoints
                # kps.xy shape: (det, 17, 2); kps.conf shape: (det, 17)
                if hasattr(kps, "xy") and kps.xy is not None:
                    xy = kps.xy.detach().cpu().numpy().astype(np.float32)
                    conf = (
                        kps.conf.detach().cpu().numpy().astype(np.float32)
                        if hasattr(kps, "conf") and kps.conf is not None
                        else np.ones((xy.shape[1],), dtype=np.float32)
                    )
                    kp_xy = xy[0]
                    kp_conf = conf[0]
                else:
                    # fallback: xyn (normalized)
                    if hasattr(kps, "xyn") and kps.xyn is not None:
                        h, w = frame_bgr.shape[:2]
                        xyn = kps.xyn.detach().cpu().numpy().astype(np.float32)  # (det,17,2)
                        xy = np.stack([xyn[:, :, 0] * w, xyn[:, :, 1] * h], axis=-1)  # (det,17,2)
                        conf = (
                            kps.conf.detach().cpu().numpy().astype(np.float32)
                            if hasattr(kps, "conf") and kps.conf is not None
                            else np.ones((xy.shape[1],), dtype=np.float32)
                        )
                        kp_xy = xy[0]
                        kp_conf = conf[0]

            # Apply confidence threshold.
            low = kp_conf < self._conf_threshold
            kp_xy[low] = np.nan
            kp_conf[low] = 0.0

            keypoints_xy_list.append(kp_xy)
            keypoints_conf_list.append(kp_conf)
            frames_processed += 1

        cap.release()
        total_s = time.perf_counter() - t0

        keypoints_xy = np.stack(keypoints_xy_list, axis=0) if keypoints_xy_list else np.empty((0, 17, 2), dtype=np.float32)
        keypoints_conf = np.stack(keypoints_conf_list, axis=0) if keypoints_conf_list else np.empty((0, 17), dtype=np.float32)

        return PoseExtractionResult(
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
            fps=fps,
            timings_s={"total": total_s},
        )


class OpenPoseDnn17:
    """
    OpenPose-style keypoint extraction using OpenCV DNN (Caffe).

    Outputs COCO-17 order by mapping from the OpenPose COCO18 keypoints.
    """

    # OpenPose COCO18 indices:
    # 0 nose, 1 neck,
    # 2 r shoulder, 3 r elbow, 4 r wrist,
    # 5 l shoulder, 6 l elbow, 7 l wrist,
    # 8 r hip, 9 r knee, 10 r ankle,
    # 11 l hip, 12 l knee, 13 l ankle,
    # 14 r eye, 15 l eye, 16 r ear, 17 l ear
    _op_to_coco17 = {
        0: 0,   # nose
        15: 1,  # left eye
        14: 2,  # right eye
        17: 3,  # left ear
        16: 4,  # right ear
        5: 5,   # left shoulder
        2: 6,   # right shoulder
        6: 7,   # left elbow
        3: 8,   # right elbow
        7: 9,   # left wrist
        4: 10,  # right wrist
        11: 11, # left hip
        8: 12,  # right hip
        12: 13, # left knee
        9: 14,   # right knee
        13: 15, # left ankle
        10: 16, # right ankle
    }

    def __init__(
        self,
        prototxt_path: str,
        caffemodel_path: str,
        conf_threshold: float = 0.2,
        in_width: int = 368,
        in_height: int = 368,
    ):
        self.conf_threshold = conf_threshold
        self.in_width = in_width
        self.in_height = in_height

        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        # Improve performance.
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass

    def _extract_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1.0 / 255.0,
            size=(self.in_width, self.in_height),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )

        self.net.setInput(blob)
        output = self.net.forward()
        # output shape: (1, 18, out_h, out_w)
        n_points = output.shape[1]
        out_h = output.shape[2]
        out_w = output.shape[3]

        kp_xy = np.full((17, 2), np.nan, dtype=np.float32)
        kp_conf = np.zeros((17,), dtype=np.float32)

        for op_idx in range(min(n_points, 18)):
            coco_idx = self._op_to_coco17.get(op_idx, None)
            if coco_idx is None:
                continue
            heatmap = output[0, op_idx, :, :]
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(heatmap)
            if maxVal < self.conf_threshold:
                continue

            x = float(w * maxLoc[0] / out_w)
            y = float(h * maxLoc[1] / out_h)
            kp_xy[coco_idx] = (x, y)
            kp_conf[coco_idx] = float(maxVal)

        return kp_xy, kp_conf

    def extract_from_video(self, video_path: str, max_frames: Optional[int] = None) -> PoseExtractionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        keypoints_xy_list = []
        keypoints_conf_list = []
        frames_processed = 0

        t0 = time.perf_counter()
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames is not None and frames_processed >= max_frames:
                break

            kp_xy, kp_conf = self._extract_frame(frame_bgr)
            keypoints_xy_list.append(kp_xy)
            keypoints_conf_list.append(kp_conf)
            frames_processed += 1

        cap.release()
        total_s = time.perf_counter() - t0

        keypoints_xy = np.stack(keypoints_xy_list, axis=0) if keypoints_xy_list else np.empty((0, 17, 2), dtype=np.float32)
        keypoints_conf = np.stack(keypoints_conf_list, axis=0) if keypoints_conf_list else np.empty((0, 17), dtype=np.float32)

        return PoseExtractionResult(
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
            fps=fps,
            timings_s={"total": total_s},
        )

