from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import yaml
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from src.common.manifest import ManifestRow, load_manifests

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler

    HAVE_IMBLEARN = True
except Exception:
    HAVE_IMBLEARN = False

try:
    from skimage.feature import hog, local_binary_pattern

    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sample_rows_per_class(rows: list[ManifestRow], max_per_class: int | None, seed: int) -> list[ManifestRow]:
    if not max_per_class or max_per_class <= 0:
        return rows
    by_class: dict[int, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_class[row.label_index].append(row)

    rng = np.random.default_rng(seed)
    sampled: list[ManifestRow] = []
    for label in sorted(by_class.keys()):
        cls_rows = by_class[label]
        if len(cls_rows) <= max_per_class:
            sampled.extend(cls_rows)
            continue
        indices = rng.choice(len(cls_rows), size=max_per_class, replace=False)
        sampled.extend(cls_rows[i] for i in indices)
    return sampled


def _safe_open_grayscale(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)
    return arr


def _fallback_features(gray: np.ndarray) -> np.ndarray:
    # Lightweight fallback if skimage is unavailable: gradients + intensity stats.
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    mag = np.sqrt(gx * gx + gy * gy)
    grad_hist, _ = np.histogram(mag, bins=16, range=(0, 255), density=True)
    intensity_hist, _ = np.histogram(gray, bins=16, range=(0, 255), density=True)
    stats = np.array([gray.mean(), gray.std(), gray.min(), gray.max()], dtype=np.float32)
    return np.concatenate([grad_hist.astype(np.float32), intensity_hist.astype(np.float32), stats])


def extract_features(img_path: str, image_size: int) -> np.ndarray:
    gray = _safe_open_grayscale(Path(img_path), image_size)

    if HAVE_SKIMAGE:
        h = hog(
            gray,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    else:
        h = _fallback_features(gray)
        lbp_hist = np.array([], dtype=np.float32)

    thresh = gray > np.percentile(gray, 65)
    area = float(np.sum(thresh))
    perimeter = float(np.sum(np.abs(np.diff(thresh.astype(np.int32), axis=0))) + np.sum(np.abs(np.diff(thresh.astype(np.int32), axis=1))))
    shape_feats = np.array([area, perimeter], dtype=np.float32)

    return np.concatenate([h.astype(np.float32), lbp_hist.astype(np.float32), shape_feats])


def build_matrix(rows: list[ManifestRow], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        feats.append(extract_features(row.path, image_size=image_size))
        labels.append(row.label_index)
    x = np.stack(feats, axis=0)
    y = np.array(labels, dtype=np.int64)
    return x, y


def balance_dataset(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes using SMOTE or RandomOverSampler as fallback."""
    if not HAVE_IMBLEARN:
        print("Warning: imbalanced-learn not installed. Using manual oversampling.")
        return _manual_oversample(x, y, seed)

    # Use SMOTE if classes have enough samples (k_neighbors + 1 = 6 minimum).
    # Otherwise fall back to RandomOverSampler which just duplicates samples.
    class_counts = np.bincount(y)
    min_samples = class_counts[class_counts > 0].min()

    if min_samples >= 6:
        sampler = SMOTE(random_state=seed, k_neighbors=5)
    else:
        # RandomOverSampler duplicates existing samples - works for any class size.
        sampler = RandomOverSampler(random_state=seed)

    x_balanced, y_balanced = sampler.fit_resample(x, y)
    print(f"Balanced dataset: {len(y)} -> {len(y_balanced)} samples")
    return x_balanced, y_balanced


def _manual_oversample(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simple oversampling by duplicating minority class samples."""
    rng = np.random.default_rng(seed)
    class_counts = np.bincount(y)
    max_count = class_counts.max()

    new_x, new_y = [x], [y]
    for class_id in range(len(class_counts)):
        if class_counts[class_id] == 0:
            continue
        count = class_counts[class_id]
        if count < max_count:
            # Oversample to match majority class
            indices = np.where(y == class_id)[0]
            n_to_add = max_count - count
            sampled_indices = rng.choice(indices, size=n_to_add, replace=True)
            new_x.append(x[sampled_indices])
            new_y.append(np.full(n_to_add, class_id, dtype=np.int64))

    return np.vstack(new_x), np.concatenate(new_y)


def top5_from_scores(scores: np.ndarray, y_true: np.ndarray) -> float:
    top5 = np.argsort(scores, axis=1)[:, -5:]
    hits = sum(1 for i, y in enumerate(y_true) if y in top5[i])
    return float(hits / len(y_true))


def evaluate_model(model: Pipeline, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(x)
    out = {
        "top1_accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }

    if hasattr(model[-1], "predict_proba"):
        proba = model.predict_proba(x)
        out["top5_accuracy"] = top5_from_scores(proba, y)
    elif hasattr(model[-1], "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        out["top5_accuracy"] = top5_from_scores(scores, y)
    else:
        out["top5_accuracy"] = float("nan")

    return out


def train_and_eval(
    model_name: str,
    estimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict:
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    pipeline.fit(x_train, y_train)

    val_metrics = evaluate_model(pipeline, x_val, y_val)
    test_metrics = evaluate_model(pipeline, x_test, y_test)

    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    return {
        "model": model_name,
        "model_path": model_path.as_posix(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train traditional ML models on manifest splits.")
    parser.add_argument("--manifest-dir", default="artifacts/manifests")
    parser.add_argument("--config", default="configs/traditional_ml.yaml")
    parser.add_argument("--output-dir", default="artifacts/models/traditional")
    args = parser.parse_args()

    cfg = read_config(Path(args.config))
    manifests = load_manifests(Path(args.manifest_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["training"]["seed"])
    image_size = int(cfg["features"]["image_size"])

    max_train = cfg["data"].get("max_train_samples_per_class")
    max_val = cfg["data"].get("max_val_samples_per_class")
    max_test = cfg["data"].get("max_test_samples_per_class")

    train_rows = sample_rows_per_class(manifests["train"], max_train, seed)
    val_rows = sample_rows_per_class(manifests["val"], max_val, seed)
    test_rows = sample_rows_per_class(manifests["test"], max_test, seed)

    x_train, y_train = build_matrix(train_rows, image_size=image_size)
    x_val, y_val = build_matrix(val_rows, image_size=image_size)
    x_test, y_test = build_matrix(test_rows, image_size=image_size)

    # Balance training data to prevent bias towards majority classes.
    x_train_balanced, y_train_balanced = balance_dataset(x_train, y_train, seed)

    # Compute aggressive class weights (useful even after balancing for borderline cases).
    unique_classes = np.unique(y_train_balanced)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train_balanced)
    class_weight_dict = {c: w for c, w in zip(unique_classes, weights)}

    svm_cfg = cfg["models"]["svm"]
    rf_cfg = cfg["models"]["random_forest"]

    svm = SVC(
        kernel=str(svm_cfg.get("kernel", "rbf")),
        C=float(svm_cfg.get("C", 10.0)),
        gamma=str(svm_cfg.get("gamma", "scale")),
        class_weight=class_weight_dict,
        probability=bool(svm_cfg.get("probability", False)),
        random_state=seed,
    )
    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 300)),
        max_depth=rf_cfg.get("max_depth", None),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        class_weight=class_weight_dict,
        random_state=seed,
    )

    results = {
        "config": cfg,
        "dataset": {
            "train_samples_original": len(train_rows),
            "train_samples_balanced": len(y_train_balanced),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
            "have_skimage": HAVE_SKIMAGE,
            "have_imblearn": HAVE_IMBLEARN,
        },
        "models": [],
    }

    results["models"].append(
        train_and_eval("svm", svm, x_train_balanced, y_train_balanced, x_val, y_val, x_test, y_test, output_dir)
    )
    results["models"].append(
        train_and_eval("random_forest", rf, x_train_balanced, y_train_balanced, x_val, y_val, x_test, y_test, output_dir)
    )

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved traditional model outputs to: {output_dir}")


if __name__ == "__main__":
    main()
