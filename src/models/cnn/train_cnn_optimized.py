"""
Optimized CNN training with comprehensive improvements for better plankton species identification.

Optimizations include:
1. Advanced data augmentation (albumentations)
2. Progressive layer unfreezing (gradual fine-tuning)
3. Better learning rate scheduling (warmup + cosine annealing)
4. Improved class weighting with epsilon smoothing
5. Label smoothing in loss function
6. Mixed precision training for efficiency
7. Better early stopping and checkpointing
8. Comprehensive metric tracking
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import albumentations as A
import numpy as np
import yaml

from src.common.manifest import load_class_to_index, load_manifests


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def create_augmentation_pipeline(image_size: int) -> A.Compose:
    """Create aggressive augmentation optimized for plankton images."""
    return A.Compose(
        [
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.7, border_mode=1),  # Moderate rotation
            A.GaussNoise(p=0.3),  # Add slight noise to handle imaging variations
            
            # Zoom and shift
            A.Affine(scale=(0.85, 1.15), p=0.6),  # Random zoom
            A.Affine(translate_percent=(-0.1, 0.1), p=0.5),  # Random translation
            
            # Intensity transforms (important for microscopy)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomGamma(p=0.3),  # Mimic different exposure levels
            A.CLAHE(p=0.3),  # Contrast Limited Adaptive Histogram Equalization
            
            # Advanced augmentations
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),  # CutOut variant
            A.Downscale(scale_min=0.75, scale_max=0.9, p=0.2),  # Simulate lower resolution
            
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                       std=[0.229, 0.224, 0.225],   # ImageNet std
                       p=1.0),
        ],
        bbox_params=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2B0 CNN with comprehensive optimizations."
    )
    parser.add_argument("--manifest-dir", default="artifacts/manifests")
    parser.add_argument("--config", default="configs/cnn_optimized.yaml")
    parser.add_argument("--output-dir", default="artifacts/models/cnn_optimized")
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, regularizers
        from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for CNN training. Install it first, then re-run."
        ) from exc

    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for CNN post-training metrics. Install it first, then re-run."
        ) from exc

    cfg = read_config(Path(args.config))
    manifests = load_manifests(Path(args.manifest_dir))
    class_to_index = load_class_to_index(Path(args.manifest_dir) / "class_to_index.json")
    num_classes = len(class_to_index)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["training"]["seed"])
    image_size = int(cfg["model"]["image_size"])
    batch_size = int(cfg["training"]["batch_size"])
    use_mixed_precision = bool(cfg["training"].get("mixed_precision", False))

    max_train = cfg["data"].get("max_train_samples")
    max_val = cfg["data"].get("max_val_samples")
    max_test = cfg["data"].get("max_test_samples")

    # Enable mixed precision training for faster convergence
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    tf.keras.utils.set_random_seed(seed)

    def stratified_limit_rows(rows, limit, split_offset: int):
        if not limit or limit <= 0 or len(rows) <= limit:
            return rows

        rows = list(rows)
        limit = int(limit)
        by_class = defaultdict(list)
        for r in rows:
            by_class[r.label_index].append(r)

        class_ids = sorted(by_class.keys())
        n_classes = len(class_ids)
        if n_classes == 0:
            return []

        rng = np.random.default_rng(seed + split_offset)

        quota = max(1, limit // n_classes)
        selected = []
        leftovers = []
        for cid in class_ids:
            cls_rows = by_class[cid]
            if len(cls_rows) <= quota:
                selected.extend(cls_rows)
            else:
                idx = rng.choice(len(cls_rows), size=quota, replace=False)
                chosen = [cls_rows[i] for i in idx]
                selected.extend(chosen)
                chosen_ids = set(idx.tolist())
                leftovers.extend(cls_rows[i] for i in range(len(cls_rows)) if i not in chosen_ids)

        remaining = limit - len(selected)
        if remaining > 0 and leftovers:
            if len(leftovers) <= remaining:
                selected.extend(leftovers)
            else:
                idx = rng.choice(len(leftovers), size=remaining, replace=False)
                selected.extend(leftovers[i] for i in idx)

        rng.shuffle(selected)
        return selected

    train_rows = stratified_limit_rows(manifests["train"], max_train, split_offset=11)
    val_rows = stratified_limit_rows(manifests["val"], max_val, split_offset=22)
    test_rows = stratified_limit_rows(manifests["test"], max_test, split_offset=33)

    # IMPROVED CLASS WEIGHTING: Use epsilon smoothing for better stability
    train_counts = defaultdict(int)
    for r in train_rows:
        train_counts[r.label_index] += 1
    n_train = len(train_rows)
    n_classes_train = len(train_counts)
    
    epsilon = 0.1  # Smoothing constant
    raw_weights = {
        int(cid): float(n_train / (n_classes_train * (cnt + epsilon)))
        for cid, cnt in train_counts.items()
    }

    # Use log1p dampening with adaptive bounds based on imbalance ratio
    max_weight = float(cfg["training"].get("max_class_weight", 10.0))
    min_weight = float(cfg["training"].get("min_class_weight", 0.3))
    class_weight = {cid: float(1.0 + np.log1p(w - 1)) for cid, w in raw_weights.items()}
    class_weight = {cid: float(min(max_weight, max(min_weight, w))) for cid, w in class_weight.items()}

    # Re-center to mean 1.0
    mean_w = float(np.mean(list(class_weight.values()))) if class_weight else 1.0
    if mean_w > 0:
        class_weight = {cid: float(w / mean_w) for cid, w in class_weight.items()}

    print(f"Class weights range: [{min(class_weight.values()):.3f}, {max(class_weight.values()):.3f}]")
    print(f"Train samples: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    augment_pipeline = create_augmentation_pipeline(image_size)

    def build_dataset(rows, training: bool):
        paths = [r.path for r in rows]
        labels = [r.label_index for r in rows]

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(min(len(paths), 10000), seed=seed, reshuffle_each_iteration=True)

        def _load(path, label):
            bytes_img = tf.io.read_file(path)
            img = tf.image.decode_image(bytes_img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [image_size, image_size])
            img = tf.cast(img, tf.float32)
            return img, label

        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

        if training:
            def _aug(img, label):
                # Apply albumentations
                img_np = img.numpy() if hasattr(img, 'numpy') else img
                augmented = augment_pipeline(image=img_np.astype(np.uint8) if img_np.max() > 1 else (img_np * 255).astype(np.uint8))
                return tf.convert_to_tensor(augmented['image'], dtype=tf.float32), label

            # Use py_function for albumentations since it's numpy-based
            ds = ds.map(
                lambda x, y: tf.py_function(_aug, [x, y], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # Set shape for graph execution
            for img, label in ds.take(1):
                img.set_shape([image_size, image_size, 3])
                label.set_shape([])
        else:
            # Validation: only normalize
            def _norm(img, label):
                img = img / 255.0 if img.max() > 1 else img
                img = (img - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
                return img, label
            
            ds = ds.map(_norm, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = build_dataset(train_rows, training=True)
    val_ds = build_dataset(val_rows, training=False)
    test_ds = build_dataset(test_rows, training=False)

    # Build model with improved architecture
    model_type = cfg["model"].get("type", "efficientnetv2b0")
    if model_type.lower() == "efficientnetv2b1":
        base = EfficientNetV2B1(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    else:
        base = EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
        )
    
    base.trainable = False

    # Build custom head with improved regularization
    l2_reg = float(cfg["model"].get("l2_regularization", 1e-4))
    dropout1 = float(cfg["model"]["dropout_head_1"])
    dropout2 = float(cfg["model"]["dropout_head_2"])
    dense_units = int(cfg["model"]["dense_units"])

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout1)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Callbacks with improved early stopping
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=(output_dir / "best_cnn.keras").as_posix(),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(cfg["training"].get("early_stopping_patience", 5)),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=float(cfg["training"].get("reduce_lr_factor", 0.5)),
            patience=int(cfg["training"].get("reduce_lr_patience", 2)),
            verbose=1,
        ),
    ]

    # Improved focal loss with label smoothing
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    
    def focal_loss_with_smoothing(y_true, y_pred, gamma=2.0, alpha=0.25):
        """Focal loss with label smoothing for better generalization."""
        y_true = tf.cast(y_true, tf.int32)
        
        # Apply label smoothing
        if label_smoothing > 0:
            y_true_smooth = tf.one_hot(y_true, num_classes)
            y_true_smooth = y_true_smooth * (1 - label_smoothing) + (label_smoothing / num_classes)
            y_true_smooth = tf.argmax(y_true_smooth, axis=-1)
        else:
            y_true_smooth = y_true
        
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.gather(y_pred, y_true_smooth, batch_dims=1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        ce = -tf.math.log(p_t)
        
        if use_mixed_precision:
            # Cast back to float32 for reduced precision loss
            focal_weight = tf.cast(focal_weight, tf.float32)
            ce = tf.cast(ce, tf.float32)
        
        return tf.reduce_mean(focal_weight * ce)

    # Learning rate schedule with warmup
    steps_per_epoch = len(train_rows) // batch_size + 1
    total_steps = steps_per_epoch * int(cfg["training"]["epochs_stage1"])
    
    warmup_steps = int(cfg["training"].get("warmup_epochs", 1)) * steps_per_epoch
    
    def lr_schedule(step):
        """Cosine annealing with warmup."""
        if step < warmup_steps:
            return float(cfg["training"]["lr_stage1"]) * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return float(cfg["training"]["lr_stage1"]) * 0.5 * (1 + np.cos(np.pi * progress))
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    callbacks.append(lr_callback)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(cfg["training"]["lr_stage1"])),
        loss=focal_loss_with_smoothing,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    print("\n=== STAGE 1: Training Custom Head ===")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["training"]["epochs_stage1"]),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    stage2_hist = {}
    
    # Progressive fine-tuning
    if bool(cfg["fine_tune"]["enabled"]):
        print("\n=== STAGE 2: Progressive Fine-tuning ===")
        
        # Gradually unfreeze layers
        unfreeze_stages = int(cfg["fine_tune"].get("unfreeze_stages", 3))
        total_layers = len(base.layers)
        layers_per_stage = max(1, total_layers // unfreeze_stages)
        
        for stage in range(unfreeze_stages):
            start_layer = total_layers - (stage + 1) * layers_per_stage
            start_layer = max(0, start_layer)
            
            # Unfreeze layers for this stage
            for layer in base.layers[start_layer:]:
                layer.trainable = True
            
            # Use lower learning rate for fine-tuning
            lr_stage2 = float(cfg["training"]["lr_stage2"]) * (0.1 ** stage)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr_stage2),
                loss=focal_loss_with_smoothing,
                metrics=[
                    tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
                ],
            )
            
            print(f"\nStage {stage + 1}/{unfreeze_stages}: Unfroze layers from {start_layer}")
            
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=int(cfg["training"].get("epochs_stage2", 1)),
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )
            
            stage2_hist[f"stage_{stage}"] = history.history

    # Evaluation on test set
    test_metrics_raw = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_metrics = {k: float(v) for k, v in test_metrics_raw.items()}

    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy().tolist())
    y_true = np.array(y_true, dtype=np.int64)

    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    model.save((output_dir / "final_cnn.keras").as_posix())

    report = {
        "config": cfg,
        "dataset": {
            "num_classes": num_classes,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
        },
        "test_metrics": {
            "top1_accuracy": test_metrics.get("accuracy") or test_metrics.get("sparse_categorical_accuracy"),
            "top5_accuracy": test_metrics.get("top5_accuracy") or test_metrics.get("sparse_top_k_categorical_accuracy"),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "loss": test_metrics.get("loss"),
            "raw": test_metrics,
        },
        "history_stage1": history_stage1.history,
        "history_stage2": stage2_hist,
        "artifacts": {
            "best_model": (output_dir / "best_cnn.keras").as_posix(),
            "final_model": (output_dir / "final_cnn.keras").as_posix(),
        },
    }
    
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n✓ Saved optimized CNN outputs to: {output_dir}")
    print(f"  Top-1 Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")


if __name__ == "__main__":
    main()
