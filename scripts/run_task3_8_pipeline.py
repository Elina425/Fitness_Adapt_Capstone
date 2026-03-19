from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from fitness_adapt.dataset_builder import DatasetConfig, build_split_windows
from fitness_adapt.io_utils import load_json
from fitness_adapt.project import ProjectPaths
from fitness_adapt.train_eval import TrainConfig, evaluate_model, personalize_on_user_windows, run_ablation, train_multitask_bilstm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="Optional project root. Auto-discovered if omitted.")
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.root if args.root else None)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    # Keep runtime practical in cloud: use subset for demonstration.
    train_keys = load_json(paths.split_path("train"))[:20]
    val_keys = load_json(paths.split_path("val"))[:8]

    dcfg = DatasetConfig(
        target_fps=30.0,
        window_size=30,
        window_stride=15,
        frame_stride=6,
        max_frames_per_video=120,
        conf_threshold=0.35,
        yolo_weights="yolo11n-pose.pt",
        device="cpu",
    )

    train_data = build_split_windows("train", paths=paths, config=dcfg, keys_override=train_keys)
    val_data = build_split_windows("val", paths=paths, config=dcfg, keys_override=val_keys)

    np.savez_compressed(paths.processed_dir / "train_windows.npz", **train_data)
    np.savez_compressed(paths.processed_dir / "val_windows.npz", **val_data)

    tcfg = TrainConfig(batch_size=32, lr=1e-3, epochs=6, device="cpu")
    num_classes = int(np.max(train_data["y_exercise"])) + 1

    # Task 7 ablation: raw coordinates vs joint angles
    ablation = run_ablation(train_data, val_data, cfg=tcfg, num_classes=num_classes)
    print("ablation_results", ablation)

    # Train the angle-feature model as primary
    model, val_metrics = train_multitask_bilstm(
        train_data["x_angles"],
        train_data["y_exercise"],
        train_data["y_quality"],
        val_data["x_angles"],
        val_data["y_exercise"],
        val_data["y_quality"],
        input_dim=train_data["x_angles"].shape[-1],
        num_classes=num_classes,
        cfg=tcfg,
    )
    print("val_metrics_angle_model", val_metrics)

    torch.save(model.state_dict(), paths.outputs_dir / "bilstm_multitask.pt")
    (paths.outputs_dir / "ablation_results.json").write_text(json.dumps(ablation, indent=2))

    # Task 8 personalization demo using one user's windows (video key as user_id proxy)
    key_ids = train_data["key_ids"]
    first_user = key_ids[0]
    mask = key_ids == first_user
    user_metrics = personalize_on_user_windows(
        model,
        train_data["x_angles"][mask],
        train_data["y_exercise"][mask],
        train_data["y_quality"][mask],
        cfg=tcfg,
        adapter_steps=20,
    )
    print("personalization_user_key", first_user)
    print("personalization_metrics", user_metrics)
    (paths.outputs_dir / "personalization_metrics.json").write_text(json.dumps(user_metrics, indent=2))


if __name__ == "__main__":
    main()

