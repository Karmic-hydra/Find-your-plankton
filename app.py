from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    from skimage.feature import hog, local_binary_pattern
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_MAP_PATH = PROJECT_ROOT / "artifacts" / "manifests" / "class_to_index.json"
SVM_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "traditional" / "svm.joblib"
CNN_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "cnn_continue" / "final_cnn.keras"
CNN_BEST_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "cnn_continue" / "best_cnn.keras"
TRADITIONAL_METRICS_PATH = PROJECT_ROOT / "artifacts" / "models" / "traditional" / "metrics.json"
CNN_METRICS_PATH = PROJECT_ROOT / "artifacts" / "models" / "cnn_continue" / "metrics.json"
DATA_DIR = PROJECT_ROOT / "data" / "2014"

TRADITIONAL_IMAGE_SIZE = 64
CNN_IMAGE_SIZE = 160


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def _fallback_features(gray: np.ndarray) -> np.ndarray:
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    mag = np.sqrt(gx * gx + gy * gy)
    grad_hist, _ = np.histogram(mag, bins=16, range=(0, 255), density=True)
    intensity_hist, _ = np.histogram(gray, bins=16, range=(0, 255), density=True)
    stats = np.array([gray.mean(), gray.std(), gray.min(), gray.max()], dtype=np.float32)
    return np.concatenate([grad_hist.astype(np.float32), intensity_hist.astype(np.float32), stats])


def extract_traditional_features_from_pil(img: Image.Image, image_size: int = TRADITIONAL_IMAGE_SIZE) -> np.ndarray:
    """Extract HOG + LBP + shape features matching training-time settings."""
    gray_img = img.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    gray = np.asarray(gray_img, dtype=np.float32)
    gray_u8 = np.asarray(gray_img, dtype=np.uint8)

    if HAVE_SKIMAGE:
        h = hog(
            gray,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    else:
        h = _fallback_features(gray)
        lbp_hist = np.array([], dtype=np.float32)

    thresh = gray > np.percentile(gray, 65)
    area = float(np.sum(thresh))
    perimeter = float(
        np.sum(np.abs(np.diff(thresh.astype(np.int32), axis=0)))
        + np.sum(np.abs(np.diff(thresh.astype(np.int32), axis=1)))
    )
    shape_feats = np.array([area, perimeter], dtype=np.float32)

    return np.concatenate([h.astype(np.float32), lbp_hist.astype(np.float32), shape_feats])


def preprocess_for_cnn(img: Image.Image, image_size: int = CNN_IMAGE_SIZE) -> np.ndarray:
    """Preprocess image for CNN using the same TensorFlow path as training."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    x = tf.convert_to_tensor(rgb)
    x = tf.image.resize(x, [image_size, image_size])
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    x = tf.expand_dims(x, axis=0)
    return x.numpy()


# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load models and class mapping. Cached for efficiency."""
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing class map: {CLASS_MAP_PATH}")

    class_to_index = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    index_to_class = {idx: cls for cls, idx in class_to_index.items()}

    missing = []
    for p in [SVM_MODEL_PATH]:
        if not p.exists():
            missing.append(str(p))
    if not CNN_BEST_MODEL_PATH.exists() and not CNN_MODEL_PATH.exists():
        missing.append(f"{CNN_BEST_MODEL_PATH} or {CNN_MODEL_PATH}")
    if missing:
        raise FileNotFoundError("Missing model artifacts:\n" + "\n".join(missing))

    svm = joblib.load(SVM_MODEL_PATH)
    cnn_path = CNN_BEST_MODEL_PATH if CNN_BEST_MODEL_PATH.exists() else CNN_MODEL_PATH
    cnn = tf.keras.models.load_model(cnn_path)
    return svm, cnn, index_to_class


@st.cache_resource
def load_metrics():
    """Load precomputed model metrics."""
    metrics = {"traditional": {}, "cnn": {}}

    def _extract_traditional_test_metrics(raw: dict) -> dict[str, float]:
        # Traditional metrics are stored under models[] -> {model: "svm", test_metrics: {...}}
        for item in raw.get("models", []):
            if str(item.get("model", "")).lower() == "svm":
                out = dict(item.get("test_metrics", {}) or {})
                # Alias used by summary section
                if "top1_accuracy" in out:
                    out["test_accuracy"] = out["top1_accuracy"]
                return out
        return {}

    def _extract_cnn_test_metrics(raw: dict) -> dict[str, float]:
        # CNN metrics are stored under test_metrics
        out = dict(raw.get("test_metrics", {}) or {})
        if "top1_accuracy" in out:
            out["test_accuracy"] = out["top1_accuracy"]
        # Raw nested block is not directly displayable as metrics
        out.pop("raw", None)
        return out

    if TRADITIONAL_METRICS_PATH.exists():
        raw_traditional = json.loads(TRADITIONAL_METRICS_PATH.read_text())
        metrics["traditional"] = _extract_traditional_test_metrics(raw_traditional)

    if CNN_METRICS_PATH.exists():
        raw_cnn = json.loads(CNN_METRICS_PATH.read_text())
        metrics["cnn"] = _extract_cnn_test_metrics(raw_cnn)

    return metrics


@st.cache_resource
def get_test_samples(supported_species: tuple[str, ...]):
    """Collect verification samples for classes that exist in the trained model."""
    samples = {}

    if not DATA_DIR.exists():
        return samples

    allowed = set(supported_species)

    for species_dir in sorted(DATA_DIR.iterdir()):
        if species_dir.is_dir() and species_dir.name in allowed:
            images = list(species_dir.glob("*.png")) + list(species_dir.glob("*.jpg")) + list(species_dir.glob("*.jpeg"))
            if images:
                samples[species_dir.name] = images[:3]  # Cache up to 3 per species for speed

    return samples


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def _top_prediction_from_scores(scores: np.ndarray, index_to_class: dict[int, str]) -> tuple[str, float]:
    top_idx = int(np.argmax(scores))
    top_label = index_to_class.get(top_idx, f"class_{top_idx}")
    top_conf = float(scores[top_idx])
    return top_label, top_conf


def _top_k_rows(scores: np.ndarray, index_to_class: dict[int, str], k: int = 3) -> list[dict[str, str]]:
    top_idx = np.argsort(scores)[-k:][::-1]
    rows: list[dict[str, str]] = []
    for rank, idx in enumerate(top_idx, start=1):
        rows.append({
            "Rank": str(rank),
            "Species": index_to_class.get(int(idx), f"class_{int(idx)}"),
            "Confidence": f"{float(scores[int(idx)]):.2%}",
        })
    return rows


def traditional_model_scores(model, features: np.ndarray) -> np.ndarray:
    """Get confidence scores from SVM."""
    x = features.reshape(1, -1)
    expected = getattr(model, "n_features_in_", None)
    if expected is None and hasattr(model, "named_steps") and "scaler" in model.named_steps:
        expected = getattr(model.named_steps["scaler"], "n_features_in_", None)
    if expected is not None and x.shape[1] != int(expected):
        raise ValueError(
            f"Traditional feature dimension mismatch: got {x.shape[1]}, expected {int(expected)}. "
            "Feature extraction settings in app must match training settings."
        )

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[0]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return _softmax(scores[0])
    pred_idx = int(model.predict(x)[0])
    scores = np.zeros((max(pred_idx + 1, 1),), dtype=np.float64)
    scores[pred_idx] = 1.0
    return scores


def cnn_scores(model, image: Image.Image) -> np.ndarray:
    """Get confidence scores from CNN."""
    x = preprocess_for_cnn(image)
    return model.predict(x, verbose=0)[0]


def load_image_from_url(url: str) -> Image.Image:
    """Download an image from a public URL and return it as a PIL image."""
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    try:
        with urlopen(req, timeout=20) as resp:
            data = resp.read()
    except HTTPError as exc:
        raise ValueError(f"HTTP error while downloading image: {exc.code}") from exc
    except URLError as exc:
        raise ValueError("Unable to reach the provided URL.") from exc
    except Exception as exc:
        raise ValueError("Failed to download image from URL.") from exc

    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError("URL did not return a valid image file.") from exc


# ============================================================================
# MAIN APP
# ============================================================================

def main() -> None:
    st.set_page_config(page_title="Plankton Species Classifier", layout="wide")
    st.title("Plankton Classification Platform")
    st.caption("Fast predictions using Traditional ML (SVM) and Deep Learning (CNN) with verification tools")

    try:
        svm_model, cnn_model, index_to_class = load_artifacts()
    except Exception as exc:
        st.error(f"Error loading models: {exc}")
        st.stop()

    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Test Samples", "Performance", "Model Info"])

    # ====================
    # Tab 1: Prediction
    # ====================
    with tab1:
        st.header("Upload & Predict")
        uploaded = st.file_uploader("Drop a plankton image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
        image_url = st.text_input(
            "Or paste an image URL (works for website images)",
            placeholder="https://example.com/plankton.jpg",
        )

        image = None
        image_source = ""

        if uploaded is not None:
            image = Image.open(uploaded)
            image_source = uploaded.name
        elif image_url.strip():
            with st.spinner("Downloading image from URL..."):
                try:
                    image = load_image_from_url(image_url.strip())
                    image_source = image_url.strip()
                except ValueError as exc:
                    st.error(str(exc))

        if image is not None:
            col_img, col_pred = st.columns([1, 1.2])

            with col_img:
                st.image(image, caption=f"Input Image: {image_source}", use_container_width=True)

            with col_pred:
                with st.spinner("Running predictions..."):
                    features = extract_traditional_features_from_pil(image)
                    svm_probs = traditional_model_scores(svm_model, features)
                    svm_label, svm_conf = _top_prediction_from_scores(svm_probs, index_to_class)

                    cnn_probs = cnn_scores(cnn_model, image)
                    cnn_label, cnn_conf = _top_prediction_from_scores(cnn_probs, index_to_class)

                st.subheader("Top-1 Predictions")
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric("SVM", svm_label, f"{svm_conf:.1%}")
                with pred_col2:
                    st.metric("CNN", cnn_label, f"{cnn_conf:.1%}")

                st.divider()

                st.subheader("Top-3 Predictions")
                top3_col1, top3_col2 = st.columns(2)
                with top3_col1:
                    st.markdown("**Traditional (SVM)**")
                    st.table(_top_k_rows(svm_probs, index_to_class, k=3))
                with top3_col2:
                    st.markdown("**CNN (EfficientNetV2B0)**")
                    st.table(_top_k_rows(cnn_probs, index_to_class, k=3))

        else:
            st.info("Upload an image or paste an image URL to see predictions from both models")

    # ====================
    # Tab 2: Test Samples
    # ====================
    with tab2:
        st.header("Test with Dataset Samples")
        st.caption("Verify predictions on supported species only (classes used during training)")

        supported_species = tuple(sorted(index_to_class.values()))
        test_samples = get_test_samples(supported_species)

        if test_samples:
            col_species, col_sample = st.columns([1, 1])
            
            with col_species:
                selected_species = st.selectbox(
                    "Choose species to test:",
                    sorted(test_samples.keys()),
                    key="species_select"
                )

            with col_sample:
                if test_samples[selected_species]:
                    selected_image_path = st.selectbox(
                        "Choose sample image:",
                        test_samples[selected_species],
                        format_func=lambda x: x.name,
                        key="image_select"
                    )
                else:
                    st.warning("No samples found")
                    selected_image_path = None

            if selected_image_path:
                test_image = Image.open(selected_image_path)
                col_img, col_result = st.columns([1, 1.2])

                with col_img:
                    st.image(test_image, caption=f"Sample: {selected_image_path.name}", use_container_width=True)

                with col_result:
                    with st.spinner("Analyzing..."):
                        test_features = extract_traditional_features_from_pil(test_image)
                        test_svm_probs = traditional_model_scores(svm_model, test_features)
                        test_svm_label, _ = _top_prediction_from_scores(test_svm_probs, index_to_class)

                        test_cnn_probs = cnn_scores(cnn_model, test_image)
                        test_cnn_label, _ = _top_prediction_from_scores(test_cnn_probs, index_to_class)

                    ground_truth = selected_species
                    st.subheader("Ground Truth vs Predictions")
                    
                    gt_col, svm_col, cnn_col = st.columns(3)
                    with gt_col:
                        st.metric("Ground Truth", ground_truth)
                    with svm_col:
                        match_svm = "CORRECT" if test_svm_label == ground_truth else "WRONG"
                        st.metric("SVM", test_svm_label, match_svm)
                    with cnn_col:
                        match_cnn = "CORRECT" if test_cnn_label == ground_truth else "WRONG"
                        st.metric("CNN", test_cnn_label, match_cnn)

                    st.divider()
                    st.subheader("Top-3 Predictions Comparison")
                    top3_col1, top3_col2 = st.columns(2)
                    with top3_col1:
                        st.markdown("**Traditional (SVM)**")
                        st.table(_top_k_rows(test_svm_probs, index_to_class, k=3))
                    with top3_col2:
                        st.markdown("**CNN (EfficientNetV2B0)**")
                        st.table(_top_k_rows(test_cnn_probs, index_to_class, k=3))
        else:
            st.warning("Dataset not found. Make sure data/2014/ exists with species folders.")

    # ====================
    # Tab 3: Model Performance
    # ====================
    with tab3:
        st.header("Model Performance Metrics")
        metrics = load_metrics()

        if metrics["traditional"] or metrics["cnn"]:
            perf_col1, perf_col2 = st.columns(2)

            with perf_col1:
                st.subheader("Traditional ML (SVM)")
                if metrics["traditional"]:
                    for key, value in metrics["traditional"].items():
                        if isinstance(value, (int, float)):
                            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                            st.metric(key.replace("_", " ").title(), formatted)
                else:
                    st.info("No metrics found")

            with perf_col2:
                st.subheader("CNN (EfficientNetV2B0)")
                if metrics["cnn"]:
                    for key, value in metrics["cnn"].items():
                        if isinstance(value, (int, float)):
                            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                            st.metric(key.replace("_", " ").title(), formatted)
                else:
                    st.info("No metrics found")

            st.divider()

            # Summary comparison
            if metrics["traditional"].get("test_accuracy") and metrics["cnn"].get("test_accuracy"):
                st.subheader("Performance Summary")
                svm_acc = metrics["traditional"].get("test_accuracy", 0)
                cnn_acc = metrics["cnn"].get("test_accuracy", 0)

                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("SVM Test Accuracy", f"{svm_acc:.2%}")
                with summary_col2:
                    st.metric("CNN Test Accuracy", f"{cnn_acc:.2%}")
                with summary_col3:
                    improvement = ((cnn_acc - svm_acc) / svm_acc * 100) if svm_acc > 0 else 0
                    st.metric("CNN Improvement", f"{improvement:+.1f}%")
        else:
            st.warning("Metrics not available. Ensure models have been trained.")

    # ====================
    # Tab 4: Model Info
    # ====================
    with tab4:
        st.header("Model Architecture & Details")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.subheader("Traditional ML (SVM)")
            st.markdown("""
            **Model Type**: Support Vector Machine (SVM)
            
            **Feature Extraction**:
            - **HOG**: 16×16 pixel cells (optimized for speed)
            - **LBP**: 10 bins (Local Binary Patterns)
            - **Shape**: Area, perimeter
            
            **Preprocessing**:
            - Input size: 64×64 grayscale
            - Standardized features
            
            **Why SVM?**
            - Fast inference
            - Works well with handcrafted features
            - Traditional baseline
            """)

        with info_col2:
            st.subheader("CNN (EfficientNetV2B0)")
            st.markdown("""
            **Model Type**: Convolutional Neural Network (Transfer Learning)
            
            **Architecture**:
            - Base: EfficientNetV2B0 (ImageNet pretrained)
            - Custom head: 384-unit dense layer
            - Regularization: Dropout 0.5
            
            **Preprocessing**:
            - Input size: 160×160 RGB
            - EfficientNetV2 normalization
            
            **Why CNN?**
            - Learns hierarchical features
            - Better with large datasets
            - Excellent performance on image classification
            """)

        st.divider()

        col_count, col_species_list = st.columns([1, 3])
        
        with col_count:
            num_classes = len(index_to_class)
            st.metric("Total Species", num_classes)
            st.markdown("""
            ### Optimization Tips
            - HOG cells: 8x8 (matches training pipeline)
            - Image resampling: BILINEAR (faster than cubic)
            - Model caching: Loaded once per session
            - Batch processing: Can be added for multiple images
            """)

        with col_species_list:
            st.markdown("### Supported Species")
            species_list = sorted(index_to_class.values())
            
            # Display in 3 columns
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            for idx, species in enumerate(species_list):
                col = cols[idx % 3]
                with col:
                    st.caption(f"• {species}")


if __name__ == "__main__":
    main()
