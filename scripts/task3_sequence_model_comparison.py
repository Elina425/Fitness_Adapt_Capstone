from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
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
    REPORTS_DIR,
    FIGURES_DIR,
    build_dataset_manifest,
)


QUALITY_LABELS = [
    "no_annotated_error",
    "knees_forward",
    "knees_inward",
    "multiple_annotated_errors",
]
LABEL_TO_INDEX = {label: index for index, label in enumerate(QUALITY_LABELS)}

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

ANGLE_SPECS = [
    ("left_elbow_angle", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    ("right_elbow_angle", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ("left_shoulder_angle", LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    ("right_shoulder_angle", RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    ("left_hip_angle", LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    ("right_hip_angle", RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    ("left_knee_angle", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    ("right_knee_angle", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def sample_balanced_subset(
    manifest: pd.DataFrame,
    sample_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for label in QUALITY_LABELS:
        candidates = manifest[manifest["quality_label"] == label]
        if len(candidates) < sample_per_class:
            raise ValueError(
                f"Class '{label}' only has {len(candidates)} videos, "
                f"but sample_per_class={sample_per_class} was requested."
            )
        rows.append(candidates.sample(n=sample_per_class, random_state=seed))

    subset = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["quality_label", "video_id"])
        .reset_index(drop=True)
    )
    return subset


def split_balanced_subset(
    subset: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        subset,
        test_size=0.30,
        random_state=seed,
        stratify=subset["quality_label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["quality_label"],
    )

    train_df = train_df.assign(benchmark_split="train")
    val_df = val_df.assign(benchmark_split="val")
    test_df = test_df.assign(benchmark_split="test")

    return (
        pd.concat([train_df, val_df, test_df], ignore_index=True)
        .sort_values(["benchmark_split", "quality_label", "video_id"])
        .reset_index(drop=True)
    )


def sample_frame_indices(frame_count: int, sequence_length: int) -> np.ndarray:
    if frame_count <= 1:
        return np.zeros(sequence_length, dtype=int)
    return np.linspace(0, frame_count - 1, num=sequence_length).round().astype(int)


def read_specific_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def extract_keypoint_sequence(
    video_path: Path,
    sequence_length: int,
    detector: BlazePoseExtractor,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(frame_count, sequence_length)
    frames = []

    try:
        for frame_index in frame_indices:
            frame = read_specific_frame(capture, int(frame_index))
            output = detector.infer(frame)
            frames.append(
                np.concatenate(
                    [output.keypoints_xy, output.confidences[:, None]],
                    axis=1,
                )
            )
    finally:
        capture.release()

    return np.stack(frames).astype(np.float32)


def interpolate_missing(sequence_xy: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    imputed = sequence_xy.copy()
    time_axis = np.arange(sequence_xy.shape[0])

    for joint_index in range(sequence_xy.shape[1]):
        valid_time = time_axis[valid_mask[:, joint_index]]
        if len(valid_time) == 0:
            imputed[:, joint_index, :] = 0.0
            continue

        for dim in range(sequence_xy.shape[2]):
            valid_values = sequence_xy[valid_mask[:, joint_index], joint_index, dim]
            imputed[:, joint_index, dim] = np.interp(time_axis, valid_time, valid_values)

    return imputed


def torso_center_and_scale(keypoints_xy: np.ndarray) -> tuple[np.ndarray, float]:
    hip_center = (keypoints_xy[LEFT_HIP] + keypoints_xy[RIGHT_HIP]) / 2.0
    shoulder_center = (keypoints_xy[LEFT_SHOULDER] + keypoints_xy[RIGHT_SHOULDER]) / 2.0
    hip_width = np.linalg.norm(keypoints_xy[LEFT_HIP] - keypoints_xy[RIGHT_HIP])
    shoulder_width = np.linalg.norm(keypoints_xy[LEFT_SHOULDER] - keypoints_xy[RIGHT_SHOULDER])
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    scale = float(np.mean([max(hip_width, 1e-6), max(shoulder_width, 1e-6), max(torso_height, 1e-6)]))
    return hip_center, max(scale, 1e-6)


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm <= 1e-6 or bc_norm <= 1e-6:
        return 180.0
    cosine = np.clip(np.dot(ba, bc) / (ba_norm * bc_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def sequence_to_features(
    keypoint_sequence: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    xy = keypoint_sequence[:, :, :2]
    confidence = keypoint_sequence[:, :, 2]
    valid_mask = confidence >= confidence_threshold
    imputed_xy = interpolate_missing(xy, valid_mask)

    features = []
    last_scale = 1.0
    for frame_xy, frame_conf in zip(imputed_xy, confidence):
        center, scale = torso_center_and_scale(frame_xy)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = last_scale
        last_scale = scale

        normalized_xy = (frame_xy - center) / scale
        angle_values = [
            compute_angle(frame_xy[a], frame_xy[b], frame_xy[c]) / 180.0
            for _, a, b, c in ANGLE_SPECS
        ]
        visible_ratio = float((frame_conf >= confidence_threshold).mean())
        mean_confidence = float(frame_conf.mean())

        frame_features = np.concatenate(
            [
                normalized_xy.reshape(-1),
                np.array(angle_values, dtype=np.float32),
                np.array([visible_ratio, mean_confidence], dtype=np.float32),
            ]
        )
        features.append(frame_features)

    return np.stack(features).astype(np.float32)


def cache_path(sequence_length: int, sample_per_class: int, seed: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"task3_seq_features_len{sequence_length}_perclass{sample_per_class}_seed{seed}.npz"


def build_feature_cache(
    subset_with_splits: pd.DataFrame,
    sequence_length: int,
    sample_per_class: int,
    seed: int,
    confidence_threshold: float,
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    output_path = cache_path(sequence_length, sample_per_class, seed)
    if output_path.exists() and not force_rebuild:
        data = np.load(output_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    detector = BlazePoseExtractor()
    sequences = []
    labels = []
    benchmark_splits = []
    video_ids = []
    original_splits = []

    try:
        for index, row in enumerate(subset_with_splits.itertuples(index=False), start=1):
            video_path = PROJECT_ROOT / row.video_path
            raw_sequence = extract_keypoint_sequence(video_path, sequence_length, detector)
            feature_sequence = sequence_to_features(raw_sequence, confidence_threshold=confidence_threshold)

            sequences.append(feature_sequence)
            labels.append(LABEL_TO_INDEX[row.quality_label])
            benchmark_splits.append(row.benchmark_split)
            video_ids.append(row.video_id)
            original_splits.append(row.split)

            if index % 20 == 0 or index == len(subset_with_splits):
                print(f"Extracted features for {index}/{len(subset_with_splits)} videos")
    finally:
        detector.close()

    cache_data = {
        "sequences": np.stack(sequences).astype(np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "benchmark_splits": np.array(benchmark_splits, dtype=object),
        "video_ids": np.array(video_ids, dtype=object),
        "original_splits": np.array(original_splits, dtype=object),
    }
    np.savez_compressed(output_path, **cache_data)
    return cache_data


def standardize_features(
    sequences: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_data = sequences[train_mask]
    mean = train_data.mean(axis=(0, 1), keepdims=True)
    std = train_data.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (sequences - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
        outputs, _ = self.lstm(x)
        pooled = outputs.mean(dim=1)
        return self.head(pooled)


class TemporalCNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        return self.head(encoded)


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length: int, d_model: int):
        super().__init__()
        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(sequence_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, num_classes: int):
        super().__init__()
        d_model = 64
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(sequence_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


class AdapterTransformerBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_layer(x)
        return x + 0.25 * self.adapter(x)


class AdapterTransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, num_classes: int):
        super().__init__()
        d_model = 64
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(sequence_length, d_model)
        self.blocks = nn.ModuleList([AdapterTransformerBlock(d_model) for _ in range(2)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


@dataclass
class TrainingResult:
    model_name: str
    parameter_count: int
    best_epoch: int
    val_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
    confusion_matrix: np.ndarray
    history: list[dict[str, float]]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(is_training)

    losses = []
    predictions = []
    labels = []

    for batch_sequences, batch_labels in loader:
        batch_sequences = batch_sequences.to(device)
        batch_labels = batch_labels.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(batch_sequences)
        loss = criterion(logits, batch_labels)

        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        labels.append(batch_labels.detach().cpu().numpy())

    all_predictions = np.concatenate(predictions)
    all_labels = np.concatenate(labels)
    return float(np.mean(losses)), all_predictions, all_labels


def evaluate_predictions(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> TrainingResult:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)

    best_state = None
    best_epoch = 0
    best_val_f1 = -1.0
    patience = 8
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_predictions, train_labels = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_predictions, val_labels = run_epoch(
            model, val_loader, criterion, None, device
        )

        train_metrics = evaluate_predictions(train_predictions, train_labels)
        val_metrics = evaluate_predictions(val_predictions, val_labels)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint saved for {model_name}")

    model.load_state_dict(best_state)
    test_loss, test_predictions, test_labels = run_epoch(model, test_loader, criterion, None, device)
    del test_loss
    test_metrics = evaluate_predictions(test_predictions, test_labels)
    cm = confusion_matrix(test_labels, test_predictions, labels=list(range(len(QUALITY_LABELS))))

    return TrainingResult(
        model_name=model_name,
        parameter_count=count_parameters(model),
        best_epoch=best_epoch,
        val_macro_f1=best_val_f1,
        test_accuracy=test_metrics["accuracy"],
        test_macro_f1=test_metrics["macro_f1"],
        confusion_matrix=cm,
        history=history,
    )


def build_dataloaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    benchmark_splits: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_mask = benchmark_splits == "train"
    val_mask = benchmark_splits == "val"
    test_mask = benchmark_splits == "test"

    standardized_sequences, _, _ = standardize_features(sequences, train_mask)

    train_loader = DataLoader(
        SequenceDataset(standardized_sequences[train_mask], labels[train_mask]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(standardized_sequences[val_mask], labels[val_mask]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        SequenceDataset(standardized_sequences[test_mask], labels[test_mask]),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def save_confusion_matrices(results: list[TrainingResult], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, result in zip(axes.flat, results):
        matrix = result.confusion_matrix
        im = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{result.model_name}\nmacro-F1={result.test_macro_f1:.3f}")
        axis.set_xticks(range(len(QUALITY_LABELS)))
        axis.set_xticklabels(QUALITY_LABELS, rotation=45, ha="right")
        axis.set_yticks(range(len(QUALITY_LABELS)))
        axis.set_yticklabels(QUALITY_LABELS)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
    figure.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_metric_bar_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    summary_df.plot.bar(x="model_name", y="test_macro_f1", ax=axes[0], legend=False, color="#5b8ff9")
    summary_df.plot.bar(x="model_name", y="test_accuracy", ax=axes[1], legend=False, color="#5ad8a6")
    axes[0].set_ylabel("macro F1")
    axes[1].set_ylabel("accuracy")
    axes[0].set_title("Quality classification macro F1")
    axes[1].set_title("Quality classification accuracy")
    for axis in axes:
        axis.set_xlabel("")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_model_comparison(
    sample_per_class: int = 40,
    sequence_length: int = 30,
    epochs: int = 35,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    confidence_threshold: float = 0.5,
    seed: int = 42,
    force_rebuild_cache: bool = False,
) -> dict[str, object]:
    set_seed(seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_dataset_manifest()
    subset = sample_balanced_subset(manifest, sample_per_class=sample_per_class, seed=seed)
    subset_with_splits = split_balanced_subset(subset, seed=seed)
    subset_with_splits.to_csv(REPORTS_DIR / "task3_balanced_quality_subset.csv", index=False)

    cache = build_feature_cache(
        subset_with_splits=subset_with_splits,
        sequence_length=sequence_length,
        sample_per_class=sample_per_class,
        seed=seed,
        confidence_threshold=confidence_threshold,
        force_rebuild=force_rebuild_cache,
    )

    sequences = cache["sequences"]
    labels = cache["labels"]
    benchmark_splits = cache["benchmark_splits"]

    train_loader, val_loader, test_loader = build_dataloaders(
        sequences=sequences,
        labels=labels,
        benchmark_splits=benchmark_splits,
        batch_size=batch_size,
    )

    input_dim = sequences.shape[-1]
    num_classes = len(QUALITY_LABELS)
    device = torch.device("cpu")

    model_builders = [
        ("BiLSTM", lambda: BiLSTMClassifier(input_dim, num_classes)),
        ("Temporal CNN", lambda: TemporalCNNClassifier(input_dim, num_classes)),
        (
            "Transformer Encoder",
            lambda: TransformerEncoderClassifier(input_dim, sequence_length, num_classes),
        ),
        (
            "Transformer + Adapters",
            lambda: AdapterTransformerClassifier(input_dim, sequence_length, num_classes),
        ),
    ]

    results = []
    history_payload = {}
    for model_name, builder in model_builders:
        print(f"Training {model_name}")
        result = train_single_model(
            model_name=model_name,
            model=builder(),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=learning_rate,
            device=device,
        )
        results.append(result)
        history_payload[model_name] = result.history

    summary_df = pd.DataFrame(
        [
            {
                "model_name": result.model_name,
                "parameter_count": result.parameter_count,
                "best_epoch": result.best_epoch,
                "val_macro_f1": result.val_macro_f1,
                "test_accuracy": result.test_accuracy,
                "test_macro_f1": result.test_macro_f1,
            }
            for result in results
        ]
    ).sort_values(["test_macro_f1", "test_accuracy"], ascending=False).reset_index(drop=True)
    summary_df["recommended_for_project"] = False
    summary_df.loc[0, "recommended_for_project"] = True

    summary_df.to_csv(REPORTS_DIR / "task3_model_comparison_summary.csv", index=False)
    with (REPORTS_DIR / "task3_model_histories.json").open("w", encoding="utf-8") as handle:
        json.dump(history_payload, handle, indent=2)

    confusion_path = FIGURES_DIR / "task3_confusion_matrices.png"
    metrics_path = FIGURES_DIR / "task3_model_metrics.png"
    save_confusion_matrices(results, confusion_path)
    save_metric_bar_chart(summary_df, metrics_path)

    dataset_note = {
        "task_scope": "4-class squat-quality classification on a balanced subset built from the repository's real videos",
        "why_balanced_subset": (
            "The rarest label, knees_inward, has only 40 videos in the repository. "
            "Using 40 videos per class creates a fair architecture comparison without overwhelming the rare class."
        ),
        "sequence_length": sequence_length,
        "feature_dimension": int(sequences.shape[-1]),
        "selected_pose_backbone": "MediaPipe BlazePose",
        "transformer_adapter_note": (
            "A bottleneck adapter transformer is used instead of DoRA because this project uses a small custom sequence encoder, "
            "not a large pretrained foundation model where DoRA is typically most useful."
        ),
    }
    with (REPORTS_DIR / "task3_benchmark_notes.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_note, handle, indent=2)

    return {
        "subset_with_splits": subset_with_splits,
        "summary_df": summary_df,
        "history_payload": history_payload,
        "confusion_path": confusion_path,
        "metrics_path": metrics_path,
        "dataset_note": dataset_note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sequence models on real squat-quality data")
    parser.add_argument("--samples-per-class", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_model_comparison(
        sample_per_class=args.samples_per_class,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        confidence_threshold=args.confidence_threshold,
        seed=args.seed,
        force_rebuild_cache=args.force_rebuild_cache,
    )
    best_model = outputs["summary_df"].iloc[0]["model_name"]
    print("Best model:", best_model)
    print("Summary report:", REPORTS_DIR / "task3_model_comparison_summary.csv")
    print("Confusion figure:", outputs["confusion_path"])


if __name__ == "__main__":
    main()
