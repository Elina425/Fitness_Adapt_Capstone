import numpy as np


COCO_17_ORDER = [
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


COCO17 = {name: idx for idx, name in enumerate(COCO_17_ORDER)}


def angle_3p(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at point b for points (a, b, c), in degrees.
    """
    ba = a - b
    bc = c - b
    # Numerical stability: normalize and clamp.
    ba_norm = np.linalg.norm(ba) + 1e-9
    bc_norm = np.linalg.norm(bc) + 1e-9
    cosang = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def compute_knee_angles_deg(keypoints_xy: np.ndarray) -> np.ndarray:
    """
    keypoints_xy: (T, 17, 2) in pixel coords.
    Returns knee angles: (T,) averaged left+right (ignores None joints by masking).
    """
    if keypoints_xy.ndim != 3 or keypoints_xy.shape[1] != 17 or keypoints_xy.shape[2] != 2:
        raise ValueError(f"Expected (T, 17, 2), got {keypoints_xy.shape}")

    T = keypoints_xy.shape[0]
    out = np.full((T,), np.nan, dtype=np.float32)

    lk = COCO17["left_knee"]
    la = COCO17["left_ankle"]
    lh = COCO17["left_hip"]
    rk = COCO17["right_knee"]
    ra = COCO17["right_ankle"]
    rh = COCO17["right_hip"]

    left_angles = []
    right_angles = []
    for t in range(T):
        # Simple missing-joint heuristic: if a point is all-zeros, treat as missing.
        left_missing = np.allclose(keypoints_xy[t, lh], 0.0) or np.allclose(keypoints_xy[t, lk], 0.0) or np.allclose(
            keypoints_xy[t, la], 0.0
        )
        right_missing = np.allclose(keypoints_xy[t, rh], 0.0) or np.allclose(keypoints_xy[t, rk], 0.0) or np.allclose(
            keypoints_xy[t, ra], 0.0
        )

        vals = []
        if not left_missing:
            vals.append(angle_3p(keypoints_xy[t, lh], keypoints_xy[t, lk], keypoints_xy[t, la]))
        if not right_missing:
            vals.append(angle_3p(keypoints_xy[t, rh], keypoints_xy[t, rk], keypoints_xy[t, ra]))
        if vals:
            out[t] = float(np.mean(vals))
    return out

