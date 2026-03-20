# Plankton Species Identification ML Project - Complete Exploration

## 1. PROJECT OVERVIEW

This is a multi-model plankton species classification system built on the WHOI 2014 plankton dataset. The project implements both **traditional ML** (SVM, Random Forest) and **deep learning** (EfficientNetV2B0 CNN) approaches with careful handling of severe class imbalance.

**Current Status**: Two trained model pipelines deployed in a Streamlit app
- Traditional ML models: SVM + Random Forest
- CNN model: EfficientNetV2B0 with focal loss

---

## 2. DATA PIPELINE & CLASS DISTRIBUTION

### Data Source
- **Location**: `data/2014/` - 94 plankton species class folders
- **Total Images**: ~329,832 .png files (across 79 eligible classes)
- **Severe Class Imbalance**: Median class size ~27.5 images; largest class ("mix") has 266,156 images

### Class Eligibility & Filtering
- **Policy**: [configs/class_policy.yaml](configs/class_policy.yaml)
  - Minimum images per class: **5** (action: `exclude_and_log`)
  - **79 classes included**, 15 classes excluded
  - Excluded classes flagged as ambiguous: `bad`, `mix`, `detritus`, `pollen`, `other_interaction`, `mix_elongated`

### Data Split Strategy
- **Policy**: [configs/split_policy.yaml](configs/split_policy.yaml)
  - Strategy: **Stratified** (preserves class distribution across splits)
  - Seed: 42 (deterministic)
  - Ratios: **70% train / 15% val / 15% test**
  - Generated manifests: [artifacts/manifests/](artifacts/manifests/)
    - `train.csv` (230,848 samples)
    - `val.csv` (49,473 samples) 
    - `test.csv` (49,473 samples)
    - `class_to_index.json` (class name → label index mapping)

### Class Imbalance Handling
**Long-tail distribution buckets**:
- `lt_10`: Classes with < 10 images
- `lt_50`: Classes with 10-49 images  
- `lt_100`: Classes with 50-99 images
- `lt_500`: Classes with 100-499 images

---

## 3. MODEL ARCHITECTURES

### 3.1 CNN Model (Deep Learning)

**Base Architecture**: [src/models/cnn/train_cnn.py](src/models/cnn/train_cnn.py)

**Model Type**: EfficientNetV2B0 (transfer learning)
- Pre-trained on ImageNet weights
- Frozen base in stage 1, optional fine-tuning in stage 2
- Custom head for 79-class classification

**Architecture Details**:
```
EfficientNetV2B0 (frozen) 
    ↓ GlobalAveragePooling2D
    ↓ Dropout(0.35)
    ↓ Dense(384 units, ReLU)
    ↓ Dropout(0.25)
    ↓ Dense(79 units, Softmax)
```

**Current Config**: [configs/cnn_continue.yaml](configs/cnn_continue.yaml)
- Image size: **160×160** (preprocessed via EfficientNetV2 preprocess_input)
- Dense units: **384**
- Dropout rates: **0.35, 0.25**

### 3.2 Traditional ML Models

**Framework**: [src/models/traditional/train_traditional.py](src/models/traditional/train_traditional.py)

**Two Models Trained**:

#### 1. **Support Vector Machine (SVM)**
- Kernel: **RBF** (Radial Basis Function)
- C (regularization): **10.0**
- Gamma: **scale** (auto-computed as 1/(n_features))
- Class weights: **balanced** (computed from class frequencies)
- Probability: **False** (uses decision_function for scoring)

#### 2. **Random Forest**
- Estimators: **300 trees**
- Max depth: **None** (unrestricted)
- Parallelization: **-1 (all cores)**
- Class weights: **balanced_subsample** (per-split balancing)

**Feature Extraction**:
```python
# Extracted per image at 64×64 resolution:
1. HOG (Histogram of Oriented Gradients)
   - pixels_per_cell: (8, 8)
   - cells_per_block: (2, 2)
   
2. LBP (Local Binary Pattern)
   - P=8, R=1, method="uniform"
   - 10-bin histogram
   
3. Morphological features
   - Area (thresholded at 65th percentile intensity)
   - Perimeter
   
4. Fallback features (if skimage unavailable):
   - Gradient magnitude histogram (16 bins)
   - Intensity histogram (16 bins)
   - Statistics: mean, std, min, max intensity
```

---

## 4. TRAINING STRATEGY & HYPERPARAMETERS

### 4.1 CNN Training Loop (2-Stage)

**Stage 1: Feature Head Training**
- Epochs: **10** (in main config) / **2** (continue config)
- Batch size: **32** (main) / **16** (continue)
- Learning rate: **0.0001** (Adam optimizer)
- Callbacks:
  - `ModelCheckpoint`: monitors **val_loss** (not accuracy - more balanced for imbalanced data)
  - `EarlyStopping`: patience=**5** (main) / **2** (continue)
  - `ReduceLROnPlateau`: factor=**0.5**, patience=**3** (main) / **1** (continue)

**Stage 2: Fine-Tuning (Optional)**
- Epochs: **8** (main) / **1** (continue)
- Learning rate: **0.00001** (10× lower)
- Unfreeze last **20** layers of EfficientNetV2B0
- Fine-tuning is **disabled** in continue config

**Loss Function**: **Focal Loss**
```python
def focal_loss(y_true, y_pred, gamma=2.0):
    # Down-weights easy (majority class) examples
    # gamma=2.0 standard; higher γ = stronger focus on hard examples
    p_t = tf.gather(y_pred, y_true, batch_dims=1)
    focal_weight = (1.0 - p_t)^gamma
    ce = -log(p_t)
    return mean(focal_weight * ce)
```

### 4.2 Class Weight Strategy (CNN)

**Aggressive log1p dampening**:
```python
raw_weight[c] = n_train / (n_classes * count[c])  # Inverse frequency
weight[c] = 1.0 + log1p(raw_weight[c] - 1)       # Gentle log dampening
weight[c] = clip(weight[c], 0.3, 10.0)           # Bounds: [0.3, 10.0]
weight[c] = weight[c] / mean(all_weights)        # Re-center to ~1.0
```

### 4.3 Traditional ML Training

**Data Preprocessing**:
1. Sample per class: **300** max (for SVM) via `max_train_samples_per_class`
2. Balance training set: **SMOTE** (if ≥6 samples/class) or RandomOverSampler
   - SMOTE handles synthetic minority oversampling
   - Fallback: duplicate minority samples
3. StandardScaler normalization

**SVM-specific**: Class weights passed to `class_weight` parameter (used in hinge loss)

---

## 5. DATA AUGMENTATION & REGULARIZATION

### CNN Augmentation (Training Only)
```python
tf.keras.Sequential([
    RandomFlip(mode="horizontal_and_vertical"),
    RandomRotation(0.10),           # ±10% rotation
    RandomZoom(0.20),               # ±20% zoom
    RandomTranslation(0.10, 0.10),  # ±10% translation
])
```

### CNN Regularization
- **Dropout**: 0.35 (after GlobalAveragePooling) + 0.25 (after Dense)
- **Loss weighting**: Focal loss + aggressive class weights
- **Learning rate scheduling**: ReduceLROnPlateau

### Traditional ML Regularization
- **Class weights**: Inverse frequency scaled for SVM; balanced_subsample for RF
- **Standardization**: StandardScaler for feature normalization
- **Feature constraining**: SVM's RBF kernel naturally limits overfit; RF limited by tree depth

---

## 6. EVALUATION METRICS & CURRENT PERFORMANCE

### Metrics Tracked
| Metric | Purpose |
|--------|---------|
| **Top-1 Accuracy** | % of predictions with correct class as top prediction |
| **Top-5 Accuracy** | % of predictions with correct class in top 5 predictions |
| **Macro F1** | Unweighted F1, treats all classes equally (good for imbalanced) |
| **Weighted F1** | F1 weighted by class support |
| **Balanced Accuracy** | Mean per-class recall, class-agnostic to imbalance |
| **Loss** | Sparse categorical crossentropy (CNN) / decision function (SVM) |

### Current Performance (Latest Trained Models)

#### CNN Results (cnn_continue)
- **Test Data**: 2,000 samples (79 classes)
- **Top-1 Accuracy**: **0.7175** (71.75%)
- **Top-5 Accuracy**: **0.8995** (89.95%)
- **Macro F1**: **0.3032** (imbalance shows here)
- **Weighted F1**: **0.6825** (weighted by class support)
- **Balanced Accuracy**: **0.3028** (per-class recall average)
- **Loss**: 1.221

#### Traditional ML Results (traditional)
**Best Model: SVM**
- **Top-1 Accuracy**: **0.6557** (65.57%)
- **Top-5 Accuracy**: **0.8897** (88.97%)
- **Macro F1**: **0.3314**
- **Weighted F1**: **0.6343**
- **Balanced Accuracy**: **0.3187**

**Alternative: Random Forest**
- **Top-1 Accuracy**: **0.5740** (57.40%)
- **Top-5 Accuracy**: **0.8543** (85.43%)
- **Macro F1**: **0.2622**

**Summary**: CNN outperforms SVM by ~8% on top-1 accuracy; SVM > RF.

---

## 7. DATA LOADING & PREPROCESSING PIPELINE

### CNN Data Loading
[src/models/cnn/train_cnn.py::build_dataset()]
```python
# 1. Load image paths + labels from manifest
# 2. Shuffle training data (buffer=10,000, reshuffle each epoch)
# 3. Load image bytes → decode RGB → resize to 160×160
# 4. EfficientNetV2 preprocess_input (channel scaling)
# 5. Apply augmentation (training only)
# 6. Batch & prefetch (AUTOTUNE)
```

### Traditional ML Feature Extraction
[src/models/traditional/train_traditional.py::extract_features()]
```python
# 1. Load image → convert to grayscale → resize to 64×64
# 2. Compute HOG features (at 8×8 pixel cell, 2×2 blocks)
# 3. Compute LBP histogram (8-neighbor, uniform)
# 4. Extract morphological features (area, perimeter)
# 5. Standardize features (StandardScaler)
# 6. Stack into feature matrix
```

### Deployment Preprocessing
[app.py]
- **CNN**: `preprocess_for_cnn()` - resize to 160×160, EfficientNetV2 preprocess_input
- **Traditional**: `extract_traditional_features_from_pil()` - same HOG/LBP extraction at 64×64

---

## 8. DIRECTORY STRUCTURE & KEY FILES

```
ml project/
├── configs/
│   ├── cnn.yaml                    # CNN config (main)
│   ├── cnn_continue.yaml           # CNN config (reduced for continued runs)
│   ├── traditional_ml.yaml         # SVM + RF config
│   ├── class_policy.yaml           # Class filtering rules
│   ├── split_policy.yaml           # Train/val/test split rules
│   └── experiment_contract.yaml    # Experiment metadata
│
├── src/
│   ├── models/
│   │   ├── cnn/
│   │   │   └── train_cnn.py        # CNN training (EfficientNetV2B0 + focal loss)
│   │   └── traditional/
│   │       └── train_traditional.py # SVM + RF training (HOG+LBP features)
│   ├── data/
│   │   ├── generate_splits.py      # Stratified split generation
│   │   └── audit_dataset.py        # Dataset audit/validation
│   ├── eval/
│   │   └── compare_models.py       # CNN vs traditional comparison report
│   └── common/
│       └── manifest.py             # ManifestRow dataclass + loaders
│
├── artifacts/
│   ├── manifests/
│   │   ├── train.csv / val.csv / test.csv   # Split data with paths & labels
│   │   ├── class_to_index.json               # 79-class name→index mapping
│   │   └── excluded_classes.csv              # Classes filtered out
│   └── models/
│       ├── traditional/
│       │   ├── svm.joblib              # Trained SVM (with StandardScaler)
│       │   ├── random_forest.joblib    # Trained RF (with StandardScaler)
│       │   └── metrics.json            # SVM + RF test metrics
│       └── cnn_continue/
│           ├── best_cnn.keras         # Best checkpoint (early stopping)
│           ├── final_cnn.keras        # Final model
│           └── metrics.json           # CNN test metrics
│
├── data/
│   └── 2014/                          # Raw dataset: 94 class folders
│       └── [species_name]/            # 329,832 images across folders
│
├── app.py                             # Streamlit deployment UI
└── requirements.txt                   # Dependencies
```

---

## 9. DEPLOYMENT & INFERENCE

### Streamlit App ([app.py](app.py))
**Features**:
- Upload plankton image or paste URL
- Real-time prediction via SVM + CNN
- Top-3 predictions with confidence scores
- Test samples gallery for each species
- Performance metrics dashboard
- Model info tab

**UI Tabs**:
1. **Predict**: Single image classification
2. **Test Samples**: Browse verification samples by species
3. **Performance**: Metrics comparison table
4. **Model Info**: Architecture & config details

**Model Loading**:
- Cached with `@st.cache_resource`
- Models loaded at startup:
  - SVM: `artifacts/models/traditional/svm.joblib`
  - CNN (best): `artifacts/models/cnn_continue/best_cnn.keras`
  - Class map: `artifacts/manifests/class_to_index.json`

### Inference Flow
```
Image → Preprocessing → Feature extraction → Model → Confidences → Top-K → UI
```

---

## 10. DEPENDENCIES & ENVIRONMENT

### Python Packages
```
numpy                # Numerical ops
Pillow              # Image loading
PyYAML              # Config parsing
scikit-learn        # Traditional ML (SVM, RF, metrics)
joblib              # Model serialization
scikit-image        # HOG, LBP features
tensorflow          # CNN (EfficientNetV2B0)
streamlit           # Web UI
```

### Environment Notes
- **Python Version**: 3.12 required for TensorFlow compatibility (3.14 not supported)
- **Keras Backend**: TensorFlow 2.x (integrated)

---

## 11. CLASS IMBALANCE MITIGATION STRATEGIES

### CNN-Level
1. **Focal Loss**: Down-weights easy (majority) examples $L = -\alpha (1-p_t)^\gamma \log(p_t)$ with $\gamma=2.0$
2. **Class Weights**: Aggressive scaling with log dampening, bounded [0.3, 10.0]
3. **Early Stopping**: Monitors `val_loss` (not accuracy, which is biased toward majority)
4. **Monitoring**: Uses macro F1 + balanced accuracy for evaluation

### Traditional ML Level
1. **Feature Balancing**: SMOTE for synthetic oversampling (or RandomOverSampler fallback)
2. **SVM Class Weights**: Passed directly to loss (`class_weight` param)
3. **RF Class Weights**: `balanced_subsample` mode (per-split weighting)

### Data Level
1. **Stratified Split**: Preserves class distribution across train/val/test
2. **Long-tail Analysis**: Buckets for monitoring:low-sample classes
3. **Ambiguous Class Exclusion**: Classes like "mix", "detritus" pre-filtered

---

## 12. KEY OBSERVATIONS & INSIGHTS

### Strengths
✅ **Robust imbalance handling**: Focal loss + class weights + stratified splits  
✅ **Multi-model approach**: Combines interpretable ML (SVM) with high-capacity DL (CNN)  
✅ **Reproducibility**: Deterministic seeding (42), config-driven pipeline  
✅ **Comprehensive metrics**: Macro/weighted F1, balanced accuracy beyond top-1  
✅ **Interactive deployment**: Streamlit UI with upload + URL inference  

### Limitations & Improvement Opportunities
⚠️ **Metric interpretation**: Macro F1 ~0.30 reflects severe class imbalance (many rare classes)  
⚠️ **CNN config underutilized**: Continue config uses only 2-3 epochs; stage 2 fine-tuning disabled  
⚠️ **Traditional ML features**: Handcrafted (HOG/LBP) vs learned deep features  
⚠️ **Data scale**: Training on ~8,000 samples (continue config) is conservative; full dataset available  
⚠️ **No ensemble**: Could combine CNN + SVM predictions for robustness  

---

## 13. QUICK START COMMANDS

### Generate Data Splits (if needed)
```bash
python -m src.data.generate_splits --config configs/split_policy.yaml
```

### Train Traditional ML  
```bash
python -m src.models.traditional.train_traditional --config configs/traditional_ml.yaml
```

### Train CNN
```bash
python -m src.models.cnn.train_cnn --config configs/cnn_continue.yaml
```

### Compare Models
```bash
python -m src.eval.compare_models \
  --traditional-metrics artifacts/models/traditional/metrics.json \
  --cnn-metrics artifacts/models/cnn_continue/metrics.json
```

### Run Streamlit App
```bash
streamlit run app.py
```

---

## 14. CONFIGURATION PARAMETERS REFERENCE

| Component | Parameter | Values | Notes |
|-----------|-----------|--------|-------|
| **CNN** | Image size | 160 (continue) / 224 (main) | Affects resolution & memory |
| | Dense units | 384 (continue) / 512 (main) | Head capacity |
| | Dropout | 0.35, 0.25 | L2 regularization equivalent |
| | Batch size | 16 (continue) / 32 (main) | Training stability |
| | Focal loss γ | 2.0 | Standard; higher = harder focus |
| **SVM** | Kernel | RBF | Non-linear; works well for imbalanced |
| | C | 10.0 | Reg strength; lower = more reg |
| | Gamma | scale | Auto-computed |
| **RF** | n_estimators | 300 | Ensemble size |
| | max_depth | None | Unrestricted depth |
| **Data** | Min images/class | 5 | Filtering threshold |
| | Augmentation | Flip, Rotate, Zoom, Translate | Training only |
| | Feature size (Traditional) | 64×64 | For HOG/LBP |
| | Feature size (CNN) | 160×160 | For EfficientNetV2 |

