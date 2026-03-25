"""
Exercise classification on Kaggle workout videos (t bar row vs squat).

Pipeline:
  - Download dataset via kagglehub (hasyimabdillah/workoutfitness-video).
  - Benchmark pose models (MediaPipe BlazePose, YOLO11-pose, MoveNet Thunder) on a
    sample of frames for speed vs tracking-quality metrics.
  - Per video: extract 17 COCO keypoints per frame, normalize (scale-invariant),
    impute low-confidence joints, resample to a target FPS for temporal alignment.
  - Biomechanical angles (knee, hip, elbow, shoulder) per frame.
  - BiLSTM on angle-only sequences (30-frame windows).
  - ST-GCN on normalized 2D joint coordinates (same windows).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._deps import raise_missing_dependency_error

try:
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise_missing_dependency_error(exc, script_name=Path(__file__).name)

from scripts.task1_task2_pose_benchmark import (
    BlazePoseExtractor,
    CACHE_DIR,
    FIGURES_DIR,
    MOVENET_MODELS,
    REPORTS_DIR,
    YoloPoseExtractor,
    process_video_with_model,
)
from scripts.task3_sequence_model_comparison import (
    ANGLE_SPECS,
    interpolate_missing,
    torso_center_and_scale,
    compute_angle,
)

try:
    import kagglehub
except ModuleNotFoundError as exc:
    raise_missing_dependency_error(exc, script_name=Path(__file__).name)

# COCO 17 skeleton edges (same as task1_task2_pose_benchmark.COCO_EDGES)
COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

EXERCISE_FOLDERS = {
    "t_bar_row": "t bar row",
    "squat": "squat",
}
LABEL_NAMES = ["t_bar_row", "squat"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABEL_NAMES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_kaggle_root() -> Path:
    path = kagglehub.dataset_download("hasyimabdillah/workoutfitness-video")
    return Path(path)


def list_videos_for_exercises(root: Path) -> pd.DataFrame:
    rows = []
    for label, folder_name in EXERCISE_FOLDERS.items():
        folder = root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing exercise folder: {folder}")
        for video_path in sorted(folder.glob("*.mp4")):
            rel = video_path.relative_to(root)
            rows.append(
                {
                    "exercise_label": label,
                    "video_relpath": str(rel),
                    "video_id": video_path.stem,
                }
            )
    return pd.DataFrame(rows)


def video_metadata(video_path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.release()
    return {"frame_count": max(frame_count, 1), "fps": fps}


def uniform_time_indices(frame_count: int, target_frames: int) -> np.ndarray:
    if frame_count <= 1:
        return np.zeros(target_frames, dtype=np.int64)
    return np.linspace(0, frame_count - 1, num=target_frames).round().astype(np.int64)


def read_frame_at(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {index}")
    return frame


class MoveNetExtractorSimple:
    """Thin wrapper matching YOLO/BlazePose interface for benchmarking."""

    def __init__(self, name: str, model_url: str, cache_filename: str, input_size: int):
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise_missing_dependency_error(exc, script_name=Path(__file__).name)

        from scripts.task1_task2_pose_benchmark import ensure_cached_download

        self.model_name = name
        self.tf = tf
        self.input_size = input_size
        self.model_path = ensure_cached_download(model_url, cache_filename)
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def infer(self, frame_bgr: np.ndarray):
        from scripts.task1_task2_pose_benchmark import PoseInferenceOutput

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_bgr.shape[:2]
        resized = self.tf.image.resize_with_pad(
            self.tf.expand_dims(frame_rgb, axis=0),
            self.input_size,
            self.input_size,
        )
        input_tensor = self.tf.cast(resized, dtype=self.input_details[0]["dtype"]).numpy()
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(self.output_details[0]["index"])[0, 0]

        keypoints_xy = np.zeros((17, 2), dtype=np.float32)
        confidences = np.zeros(17, dtype=np.float32)
        keypoints_xy[:, 0] = keypoints[:, 1] * width
        keypoints_xy[:, 1] = keypoints[:, 0] * height
        confidences[:] = keypoints[:, 2]
        return PoseInferenceOutput(keypoints_xy=keypoints_xy, confidences=confidences)

    def close(self) -> None:
        return None


def build_pose_models_for_task4(include_mediapipe: bool) -> list:
    """YOLO + MoveNet are reliable headless; MediaPipe Tasks may require EGL/GL."""
    models: list = [YoloPoseExtractor()]
    _, thunder_url, thunder_file, thunder_size = MOVENET_MODELS[1]
    models.append(MoveNetExtractorSimple("MoveNet Thunder", thunder_url, thunder_file, thunder_size))
    if include_mediapipe:
        models.insert(0, BlazePoseExtractor())
    return models


def build_pose_extractor(name: str):
    name = name.lower().strip()
    if name in ("yolo", "yolo11", "yolo11n-pose"):
        return YoloPoseExtractor()
    if name in ("mediapipe", "blazepose", "mp"):
        return BlazePoseExtractor()
    if name in ("movenet", "movenet_thunder", "thunder"):
        _, thunder_url, thunder_file, thunder_size = MOVENET_MODELS[1]
        return MoveNetExtractorSimple("MoveNet Thunder", thunder_url, thunder_file, thunder_size)
    raise ValueError(f"Unknown pose backbone: {name}. Use yolo, mediapipe, or movenet_thunder.")


def benchmark_pose_on_sample(
    video_paths: list[Path],
    max_frames_per_video: int,
    confidence_threshold: float,
    include_mediapipe: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = build_pose_models_for_task4(include_mediapipe=include_mediapipe)
    rows: list[dict] = []
    try:
        for model in models:
            for video_path in video_paths:
                metrics, _ = process_video_with_model(
                    model=model,
                    video_path=video_path,
                    max_frames=max_frames_per_video,
                    confidence_threshold=confidence_threshold,
                )
                rows.append({"model_name": model.model_name, "video_path": str(video_path), **metrics})
    finally:
        for model in models:
            model.close()

    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df.groupby("model_name", as_index=False)
        .agg(
            videos=("video_path", "nunique"),
            processed_frames=("processed_frames", "sum"),
            mean_effective_fps=("effective_fps", "mean"),
            mean_visible_keypoint_ratio=("mean_visible_keypoint_ratio", "mean"),
            mean_fully_tracked_frame_ratio=("fully_tracked_frame_ratio", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            mean_temporal_jitter_proxy=("temporal_jitter_proxy", "mean"),
        )
        .sort_values(["mean_visible_keypoint_ratio", "mean_effective_fps"], ascending=[False, False])
        .reset_index(drop=True)
    )
    max_fps = max(float(summary_df["mean_effective_fps"].max()), 1e-6)
    summary_df["selection_score"] = (
        0.5 * summary_df["mean_visible_keypoint_ratio"]
        + 0.2 * summary_df["mean_fully_tracked_frame_ratio"]
        + 0.2 * (summary_df["mean_effective_fps"] / max_fps)
        + 0.1 * summary_df["mean_confidence"]
    )
    summary_df = summary_df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    summary_df["recommended_for_project"] = False
    summary_df.loc[0, "recommended_for_project"] = True
    return detail_df, summary_df


def extract_keypoint_sequence_resampled(
    video_path: Path,
    sequence_length: int,
    target_fps: float,
    detector,
) -> tuple[np.ndarray, float]:
    """Returns keypoints (T, 17, 3) and source FPS used for metadata."""
    meta = video_metadata(video_path)
    source_fps = float(meta["fps"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(meta["frame_count"])
    duration = frame_count / max(source_fps, 1e-6)
    ideal_span_frames = max(int(round(duration * target_fps)), 1)
    span = min(ideal_span_frames, frame_count)
    indices = uniform_time_indices(span, sequence_length)

    frames: list[np.ndarray] = []
    try:
        for local_idx in indices:
            idx = int(min(local_idx, frame_count - 1))
            frame = read_frame_at(capture, idx)
            output = detector.infer(frame)
            frames.append(
                np.concatenate([output.keypoints_xy, output.confidences[:, None]], axis=1).astype(np.float32)
            )
    finally:
        capture.release()

    return np.stack(frames, axis=0), source_fps


def sequence_to_angle_features(keypoint_sequence: np.ndarray, confidence_threshold: float) -> np.ndarray:
    """8 joint angles per frame, normalized to [0,1] by dividing by 180."""
    xy = keypoint_sequence[:, :, :2]
    confidence = keypoint_sequence[:, :, 2]
    valid_mask = confidence >= confidence_threshold
    imputed_xy = interpolate_missing(xy, valid_mask)

    angles: list[np.ndarray] = []
    for frame_xy in imputed_xy:
        vals = [compute_angle(frame_xy[a], frame_xy[b], frame_xy[c]) / 180.0 for _, a, b, c in ANGLE_SPECS]
        angles.append(np.array(vals, dtype=np.float32))
    return np.stack(angles, axis=0)


def sequence_to_stgcn_features(keypoint_sequence: np.ndarray, confidence_threshold: float) -> np.ndarray:
    """Root-centered, scale-normalized xy per joint (17*2 per frame)."""
    xy = keypoint_sequence[:, :, :2]
    confidence = keypoint_sequence[:, :, 2]
    valid_mask = confidence >= confidence_threshold
    imputed_xy = interpolate_missing(xy, valid_mask)

    out: list[np.ndarray] = []
    last_scale = 1.0
    for frame_xy in imputed_xy:
        _, scale = torso_center_and_scale(frame_xy)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = last_scale
        last_scale = scale
        hip_center = (frame_xy[11] + frame_xy[12]) / 2.0
        normalized = (frame_xy - hip_center) / scale
        out.append(normalized.reshape(-1).astype(np.float32))
    return np.stack(out, axis=0)


def build_normalized_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    a = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    d = a.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    a_norm = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    return torch.from_numpy(a_norm).float()


class StgcnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor, temporal_kernel: int = 9):
        super().__init__()
        self.register_buffer("adj", adj)
        pad = temporal_kernel // 2
        self.gcn = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tcn = nn.Conv2d(out_channels, out_channels, (temporal_kernel, 1), padding=(pad, 0))
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, T, V
        b, c, t, v = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        gcn_in = x_perm.view(b * t, v, c)
        adj = self.adj.to(dtype=gcn_in.dtype, device=gcn_in.device)
        gcn_out = torch.matmul(adj.unsqueeze(0), gcn_in)
        gcn_out = self.gcn(gcn_out)
        gcn_out = gcn_out.view(b, t, v, -1).permute(0, 3, 1, 2).contiguous()
        gcn_out = self.bn(gcn_out)
        out = self.relu(self.tcn(gcn_out))
        res_in = x.permute(0, 2, 3, 1).reshape(b * t, v, c)
        res = self.residual(res_in)
        res = res.view(b, t, v, -1).permute(0, 3, 1, 2)
        return out + res


class STGCNClassifier(nn.Module):
    """ST-GCN-style skeleton classifier (2 input channels = xy per joint)."""

    def __init__(self, num_nodes: int, num_classes: int, adj: torch.Tensor):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed = nn.Linear(2, 64)
        self.stgcn1 = StgcnBlock(64, 128, adj)
        self.stgcn2 = StgcnBlock(128, 128, adj)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, V*2  -> B, T, V, 2
        b, t, feat = x.shape
        v = self.num_nodes
        x = x.view(b, t, v, 2)
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        x = self.stgcn1(x)
        x = self.stgcn2(x)
        x = self.pool(x)
        return self.head(x)


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


def standardize_features_train(
    sequences: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_data = sequences[train_mask]
    mean = train_data.mean(axis=(0, 1), keepdims=True)
    std = train_data.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((sequences - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        labels.append(batch_y.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds), np.concatenate(labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    best_state = None
    best_f1 = -1.0
    patience, bad = 8, 0

    for epoch in range(1, epochs + 1):
        _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        _, val_pred, val_y = run_epoch(model, val_loader, criterion, None, device)
        val_f1 = f1_score(val_y, val_pred, average="macro")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    _, test_pred, test_y = run_epoch(model, test_loader, criterion, None, device)
    return {
        "test_accuracy": float(accuracy_score(test_y, test_pred)),
        "test_macro_f1": float(f1_score(test_y, test_pred, average="macro")),
        "confusion": confusion_matrix(test_y, test_pred, labels=[0, 1]),
    }


def build_feature_cache(
    manifest: pd.DataFrame,
    sequence_length: int,
    target_fps: float,
    confidence_threshold: float,
    seed: int,
    force_rebuild: bool,
    cache_suffix: str,
    pose_backbone: str,
) -> dict[str, np.ndarray]:
    cache_key = f"task4_kaggle_len{sequence_length}_fps{int(target_fps)}_seed{seed}{cache_suffix}.npz"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists() and not force_rebuild:
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    detector = build_pose_extractor(pose_backbone)
    angle_list: list[np.ndarray] = []
    stgcn_list: list[np.ndarray] = []
    labels: list[int] = []
    meta_fps: list[float] = []
    video_ids: list[str] = []

    try:
        for i, row in enumerate(manifest.itertuples(index=False), start=1):
            path = Path(row.dataset_root) / row.video_relpath
            raw, src_fps = extract_keypoint_sequence_resampled(
                path, sequence_length=sequence_length, target_fps=target_fps, detector=detector
            )
            angle_list.append(sequence_to_angle_features(raw, confidence_threshold))
            stgcn_list.append(sequence_to_stgcn_features(raw, confidence_threshold))
            labels.append(LABEL_TO_INDEX[row.exercise_label])
            meta_fps.append(src_fps)
            video_ids.append(row.video_id)
            if i % 10 == 0 or i == len(manifest):
                print(f"Extracted {i}/{len(manifest)} videos")
    finally:
        detector.close()

    payload = {
        "angle_sequences": np.stack(angle_list).astype(np.float32),
        "stgcn_sequences": np.stack(stgcn_list).astype(np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "source_fps": np.array(meta_fps, dtype=np.float32),
        "video_ids": np.array(video_ids, dtype=object),
    }
    np.savez_compressed(cache_path, **payload)
    return payload


def run_pipeline(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    root = discover_kaggle_root()
    manifest = list_videos_for_exercises(root)
    manifest["dataset_root"] = str(root)
    cache_suffix = ""
    if args.max_videos is not None:
        per_class = max(1, args.max_videos // 2)
        sampled_parts = []
        for label in sorted(manifest["exercise_label"].unique()):
            group = manifest[manifest["exercise_label"] == label]
            take = min(per_class, len(group))
            sampled_parts.append(group.sample(n=take, random_state=args.seed))
        manifest = pd.concat(sampled_parts, ignore_index=True).sort_values("exercise_label").reset_index(drop=True)
        cache_suffix = f"_max{len(manifest)}"
    manifest.to_csv(REPORTS_DIR / "task4_kaggle_manifest.csv", index=False)

    sample_videos = [root / p for p in manifest["video_relpath"].head(args.benchmark_video_sample).tolist()]
    bench_detail, bench_summary = benchmark_pose_on_sample(
        sample_videos,
        max_frames_per_video=args.benchmark_max_frames,
        confidence_threshold=args.confidence_threshold,
        include_mediapipe=args.benchmark_include_mediapipe,
    )
    bench_detail.to_csv(REPORTS_DIR / "task4_pose_benchmark_detail.csv", index=False)
    bench_summary.to_csv(REPORTS_DIR / "task4_pose_benchmark_summary.csv", index=False)
    recommended = bench_summary.loc[0, "model_name"]

    train_df, temp_df = train_test_split(
        manifest,
        test_size=0.30,
        random_state=args.seed,
        stratify=manifest["exercise_label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df["exercise_label"],
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
    splits.to_csv(REPORTS_DIR / "task4_train_val_test_split.csv", index=False)

    cache = build_feature_cache(
        manifest=splits,
        sequence_length=args.sequence_length,
        target_fps=args.target_fps,
        confidence_threshold=args.confidence_threshold,
        seed=args.seed,
        force_rebuild=args.force_rebuild_cache,
        cache_suffix=cache_suffix + f"_{args.pose_backbone.replace(' ', '_')}",
        pose_backbone=args.pose_backbone,
    )

    split_arr = splits["split"].to_numpy()
    train_mask = split_arr == "train"
    val_mask = split_arr == "val"
    test_mask = split_arr == "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BiLSTM on angles only
    angle_x, _, _ = standardize_features_train(cache["angle_sequences"], train_mask)
    angle_loaders = (
        DataLoader(SequenceDataset(angle_x[train_mask], cache["labels"][train_mask]), batch_size=args.batch_size, shuffle=True),
        DataLoader(SequenceDataset(angle_x[val_mask], cache["labels"][val_mask]), batch_size=args.batch_size, shuffle=False),
        DataLoader(SequenceDataset(angle_x[test_mask], cache["labels"][test_mask]), batch_size=args.batch_size, shuffle=False),
    )
    bilstm = BiLSTMClassifier(input_dim=angle_x.shape[-1], num_classes=2)
    bilstm_metrics = train_model(bilstm, *angle_loaders, epochs=args.epochs, lr=args.learning_rate, device=device)

    # ST-GCN on normalized joints
    st_x, _, _ = standardize_features_train(cache["stgcn_sequences"], train_mask)
    st_loaders = (
        DataLoader(SequenceDataset(st_x[train_mask], cache["labels"][train_mask]), batch_size=args.batch_size, shuffle=True),
        DataLoader(SequenceDataset(st_x[val_mask], cache["labels"][val_mask]), batch_size=args.batch_size, shuffle=False),
        DataLoader(SequenceDataset(st_x[test_mask], cache["labels"][test_mask]), batch_size=args.batch_size, shuffle=False),
    )
    adj = build_normalized_adjacency(17, COCO17_EDGES)
    stgcn = STGCNClassifier(num_nodes=17, num_classes=2, adj=adj)
    stgcn_metrics = train_model(stgcn, *st_loaders, epochs=args.epochs, lr=args.learning_rate, device=device)

    summary = pd.DataFrame(
        [
            {"model": "BiLSTM (angle features)", **{k: v for k, v in bilstm_metrics.items() if k != "confusion"}},
            {"model": "ST-GCN (normalized xy)", **{k: v for k, v in stgcn_metrics.items() if k != "confusion"}},
        ]
    )
    summary.to_csv(REPORTS_DIR / "task4_exercise_classification_summary.csv", index=False)

    notes = {
        "dataset": "hasyimabdillah/workoutfitness-video (t bar row, squat)",
        "pose_backbone_for_features": args.pose_backbone,
        "recommended_pose_model_from_benchmark": recommended,
        "preprocessing": {
            "normalization": "Hip-centered xy scaled by torso size (shoulder/hip width/height)",
            "imputation": "Linear interpolation across time for joints below confidence threshold",
            "temporal_sync": f"Uniform resampling to {args.sequence_length} frames (target_fps={args.target_fps} Hz metadata)",
        },
        "bilstm_input": "8 biomechanical angles per frame (elbow, shoulder, hip, knee)",
        "stgcn_input": "17 joints x 2 normalized coordinates per frame",
        "window_size_frames": args.sequence_length,
    }
    with (REPORTS_DIR / "task4_pipeline_notes.json").open("w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)

    # Simple figure: confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, cm in zip(
        axes,
        ["BiLSTM", "ST-GCN"],
        [bilstm_metrics["confusion"], stgcn_metrics["confusion"]],
    ):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(LABEL_NAMES)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "task4_confusion_matrices.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "bench_summary": bench_summary,
        "bilstm_metrics": bilstm_metrics,
        "stgcn_metrics": stgcn_metrics,
        "figure_path": fig_path,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kaggle t bar vs squat exercise classification (BiLSTM + ST-GCN)")
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--target-fps", type=float, default=30.0)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--benchmark-video-sample", type=int, default=4)
    p.add_argument("--benchmark-max-frames", type=int, default=24)
    p.add_argument("--force-rebuild-cache", action="store_true")
    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on videos per class for faster experiments (default: use all)",
    )
    p.add_argument(
        "--pose-backbone",
        type=str,
        default="yolo",
        help="Pose model for keypoint extraction: yolo (default), mediapipe, movenet_thunder",
    )
    p.add_argument(
        "--benchmark-include-mediapipe",
        action="store_true",
        help="Include MediaPipe BlazePose in the speed/accuracy benchmark (may need EGL/GL)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_pipeline(args)
    print("Recommended pose model (benchmark):", out["bench_summary"].iloc[0]["model_name"])
    print("BiLSTM test accuracy:", out["bilstm_metrics"]["test_accuracy"])
    print("ST-GCN test accuracy:", out["stgcn_metrics"]["test_accuracy"])
    print("Reports:", REPORTS_DIR)
    print("Figure:", out["figure_path"])


if __name__ == "__main__":
    main()
