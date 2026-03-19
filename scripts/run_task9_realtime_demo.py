from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from fitness_adapt.features import compute_joint_angles
from fitness_adapt.models import BiLSTMMultiHead


def _draw_skeleton(frame: np.ndarray, points: np.ndarray):
    # Minimal overlay for key joints.
    for x, y in points:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)


def _quality_feedback(score: float) -> str:
    if score >= 0.85:
        return "Great form"
    if score >= 0.65:
        return "Good, minor adjustment needed"
    if score >= 0.45:
        return "Focus on knee tracking and depth"
    return "Significant form correction needed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="", help="Optional path to video file. If empty, webcam is used.")
    parser.add_argument("--model-path", type=str, default="/workspace/outputs/bilstm_multitask.pt")
    parser.add_argument("--window-size", type=int, default=30)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video/webcam.")

    # Use angle features (6 dims) for primary model.
    model = BiLSTMMultiHead(input_dim=6, num_classes=1)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        model.eval()
    except FileNotFoundError:
        print(f"Model not found at {args.model_path}; running overlay-only mode.")
        model = None

    pose_model = YOLO("yolo11n-pose.pt")
    angle_window: list[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pred = pose_model.predict(source=frame, verbose=False, conf=0.35, device="cpu")[0]
        if pred.keypoints is not None and len(pred.keypoints) > 0:
            xy = pred.keypoints.xy.cpu().numpy()
            conf = pred.keypoints.conf.cpu().numpy()
            best = int(np.argmax(conf.sum(axis=1)))
            pts = xy[best]
            _draw_skeleton(frame, pts)

            angle_feat = compute_joint_angles(pts[None, :, :])[0]  # (6,)
            angle_window.append(angle_feat.astype(np.float32))
            if len(angle_window) > args.window_size:
                angle_window = angle_window[-args.window_size :]

            if model is not None and len(angle_window) == args.window_size:
                x = torch.from_numpy(np.asarray(angle_window, dtype=np.float32)[None, :, :])
                with torch.no_grad():
                    logits, q = model(x, use_adapter=False)
                    exercise_idx = int(torch.argmax(logits, dim=1).item())
                    quality = float(torch.clamp(q, 0.0, 1.0).item())
                exercise_name = "squat" if exercise_idx == 0 else f"class_{exercise_idx}"
                feedback = _quality_feedback(quality)
                cv2.putText(
                    frame,
                    f"Exercise: {exercise_name}  Quality: {quality:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (40, 220, 40),
                    2,
                )
                cv2.putText(frame, feedback, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(
                    frame,
                    f"Collecting frames: {len(angle_window)}/{args.window_size}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (40, 220, 40),
                    2,
                )
        else:
            cv2.putText(frame, "No person detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 255), 2)

        cv2.imshow("Fitness Adapt Realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

