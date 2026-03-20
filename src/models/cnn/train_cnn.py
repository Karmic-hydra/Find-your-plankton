from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.common.manifest import load_class_to_index, load_manifests


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNetV2B0 CNN on manifest splits.")
    parser.add_argument("--manifest-dir", default="artifacts/manifests")
    parser.add_argument("--config", default="configs/cnn.yaml")
    parser.add_argument("--output-dir", default="artifacts/models/cnn")
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow.keras.applications import EfficientNetV2B0
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for CNN training. Install it first, then re-run."
        ) from exc

    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score
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

    max_train = cfg["data"].get("max_train_samples")
    max_val = cfg["data"].get("max_val_samples")
    max_test = cfg["data"].get("max_test_samples")

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

        # First pass: equal quota per class.
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

        # Second pass: fill remaining budget from leftovers.
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

    # Aggressive class weights to combat bias towards majority classes.
    train_counts = defaultdict(int)
    for r in train_rows:
        train_counts[r.label_index] += 1
    n_train = len(train_rows)
    n_classes_train = len(train_counts)
    raw_weights = {
        int(cid): float(n_train / (n_classes_train * cnt))
        for cid, cnt in train_counts.items()
        if cnt > 0
    }

    # Use log1p dampening (gentler than sqrt) with wider bounds [0.3, 10.0].
    # This gives minority classes much stronger weight without destabilizing.
    class_weight = {cid: float(1.0 + np.log1p(w - 1)) for cid, w in raw_weights.items()}
    class_weight = {cid: float(min(10.0, max(0.3, w))) for cid, w in class_weight.items()}

    # Re-center around mean 1.0 so learning-rate behavior stays comparable.
    mean_w = float(np.mean(list(class_weight.values()))) if class_weight else 1.0
    if mean_w > 0:
        class_weight = {cid: float(w / mean_w) for cid, w in class_weight.items()}

    def build_dataset(rows, training: bool):
        paths = [r.path for r in rows]
        labels = [r.label_index for r in rows]

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(min(len(paths), 10000), seed=seed, reshuffle_each_iteration=True)

        def _load(path, label):
            bytes_img = tf.io.read_file(path)
            # Decode to RGB even if source image is grayscale.
            img = tf.image.decode_image(bytes_img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [image_size, image_size])
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
            return img, label

        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

        if training:
            augment = tf.keras.Sequential(
                [
                    layers.RandomFlip(mode="horizontal_and_vertical"),
                    layers.RandomRotation(0.10),
                    layers.RandomZoom(0.20),
                    layers.RandomTranslation(0.10, 0.10),
                ],
                name="augment",
            )

            def _aug(x, y):
                return augment(x, training=True), y

            ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = build_dataset(train_rows, training=True)
    val_ds = build_dataset(val_rows, training=False)
    test_ds = build_dataset(test_rows, training=False)

    base = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(float(cfg["model"]["dropout_head_1"]))(x)
    x = layers.Dense(int(cfg["model"]["dense_units"]), activation="relu")(x)
    x = layers.Dropout(float(cfg["model"]["dropout_head_2"]))(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Monitor val_loss instead of val_accuracy - loss is less biased towards majority classes
    # when using class weights and focal loss.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=(output_dir / "best_cnn.keras").as_posix(),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(cfg["training"]["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=float(cfg["training"]["reduce_lr_factor"]),
            patience=int(cfg["training"]["reduce_lr_patience"]),
        ),
    ]

    # Focal loss: down-weights easy (majority class) examples, focuses on hard ones.
    # gamma=2.0 is standard; alpha weighting is handled by class_weight.
    def focal_loss(y_true, y_pred, gamma=2.0):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.gather(y_pred, y_true, batch_dims=1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        ce = -tf.math.log(p_t)
        return tf.reduce_mean(focal_weight * ce)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(cfg["training"]["lr_stage1"])),
        loss=focal_loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["training"]["epochs_stage1"]),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    if bool(cfg["fine_tune"]["enabled"]):
        base.trainable = True
        unfreeze_last_n = int(cfg["fine_tune"]["unfreeze_last_n_layers"])
        for layer in base.layers[:-unfreeze_last_n]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(float(cfg["training"]["lr_stage2"])),
            loss=focal_loss,
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
            ],
        )

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(cfg["training"]["epochs_stage2"]),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
        stage2_hist = history_stage2.history
    else:
        stage2_hist = {}

    test_metrics_raw = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_metrics = {k: float(v) for k, v in test_metrics_raw.items()}

    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy().tolist())
    y_true = tf.constant(y_true, dtype=tf.int64).numpy()

    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

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
            "top1_accuracy": test_metrics.get("accuracy")
            or test_metrics.get("sparse_categorical_accuracy"),
            "top5_accuracy": test_metrics.get("top5_accuracy")
            or test_metrics.get("sparse_top_k_categorical_accuracy"),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
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
    print(f"Saved CNN outputs to: {output_dir}")


if __name__ == "__main__":
    main()
