# ML Optimization Guide for Plankton Species Classification

## Overview

This guide explains the comprehensive optimizations implemented to improve plankton species identification accuracy from the baseline 71.75% to target >75-80%.

## Key Optimizations Implemented

### 1. **Enhanced Data Augmentation** ✓

**What was improved:**
- Upgraded from TensorFlow Keras augmentation layers to **albumentations** library
- Added microscopy-specific augmentation strategies
- Implemented multiple augmentation pipelines for different training stages

**Details:**
- **Geometric transforms**: Rotation (±15°), zoom (0.85-1.15), translation (±10%)
- **Intensity transforms**: Brightness/contrast variations, gamma correction, CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Noise/degradation**: Gaussian noise, downscaling (simulates different zoom levels)
- **Advanced**: CutOut (CoarseDropout), Downsampling

**Why it helps:**
- Microscopy images have natural variations in lighting, focus, and sensor noise
- Plankton species from different samples vary due to imaging conditions
- Better augmentation increases effective dataset size and robustness

**Location**: `src/models/cnn/preprocessing.py`

---

### 2. **Progressive Layer Unfreezing** ✓

**What was improved:**
- Changed from single fine-tuning stage to **multi-stage progressive unfreezing**
- Gradual layer unfreezing prevents catastrophic forgetting
- Stage-specific learning rate scheduling

**Strategy:**
```
Stage 1: Train only custom head (base frozen)
         └─ Learn class-specific features with ImageNet knowledge

Stage 2a: Unfreeze last 1/3 of base layers
          └─ Fine-tune high-level features (LR = 0.0001)

Stage 2b: Unfreeze last 2/3 of base layers
          └─ Fine-tune mid-level features (LR = 0.00001)

Stage 2c: Unfreeze all base layers
          └─ Full fine-tuning (LR = 0.000001)
```

**Why it helps:**
- Preserves learned ImageNet features while adapting to plankton domain
- Prevents overfitting to early-stage learning rate
- Better convergence on small datasets

**Location**: `src/models/cnn/train_cnn_optimized.py` (lines ~300-320)

---

### 3. **Better Learning Rate Scheduling** ✓

**What was improved:**
- Added **warmup phase** (1 epoch at the start)
- Implemented **cosine annealing** decay schedule
- Much larger learning rate range explored

**Schedule:**
```
Phase 1 (Warmup): LR gradually increases from 0 to 0.001
Phase 2 (Cosine): LR smoothly decays following cosine function
                  └─ Better convergence than sudden drops
```

**Why it helps:**
- Warmup prevents unstable training at the beginning
- Cosine annealing provides smooth optimization trajectory
- Combined with ReduceLROnPlateau for additional flexibility

**Location**: `src/models/cnn/train_cnn_optimized.py` (lines ~230-245)

---

### 4. **Improved Class Weighting** ✓

**What was improved:**
- Added **epsilon smoothing** to class weight calculation
- Made weight bounds adaptive and more aggressive
- Better handling of extreme class imbalance (266K vs 27 images)

**Formula:**
```
raw_weight = n_train / (n_classes * (count + epsilon))
adjusted_weight = log1p(raw_weight - 1)
final_weight = clip(adjusted_weight, min=0.2, max=15.0)
normalized = final_weight / mean(all_weights)
```

**Why it helps:**
- Epsilon prevents division by zero and extreme weights for very rare classes
- Log dampening is gentler than previous approach
- Normalized weights keep learning dynamics stable

**Bounds (configurable)**:
- Min: 0.2 (prevent majority classes from being ignored)
- Max: 15.0 (stronger emphasis on rare species)

---

### 5. **Label Smoothing** ✓

**What was improved:**
- Added label smoothing in loss function
- Prevents overconfident predictions
- Improves generalization

**Mechanism:**
```
Instead of hard labels [0, 1, 0, ...]
Use soft labels [0.05, 0.9, 0.05, ...] with smoothing_factor=0.1
```

**Configuration**: 
- `label_smoothing: 0.1` in config

**Why it helps:**
- Reduces overfitting to training labels
- Makes model more calibrated (better confidence scores)
- Helps with small datasets prone to memorization

---

### 6. **Focal Loss with Class Weighting** ✓

**What was improved:**
- Already using focal loss γ=2.0 (good!)
- Enhanced with better label smoothing integration
- Clearer implementation in loss function

**Formula:**
```
p_t = probability of true class
focal_weight = (1 - p_t)^γ  where γ=2.0
focal_loss = -focal_weight * log(p_t)
```

**Why it works:**
- Hard examples (low p_t) get higher weight
- Easy examples get downweighted
- Perfect for imbalanced datasets

**Location**: `src/models/cnn/train_cnn_optimized.py` (lines ~250-280)

---

### 7. **Increased Training Duration** ✓

**What was improved:**
- Extended from 2 epochs to 15 epochs for stage 1
- Better early stopping patient (8 epochs vs 2)
- More time to converge

**Current config**:
```yaml
epochs_stage1: 15    # Was: 2
epochs_stage2: 5     # Was: 1
early_stopping_patience: 8  # Was: 2
```

**Why it helps:**
- 2 epochs was barely one full pass through data with small batch size
- Modern models need more iterations to converge
- Larger datasets benefit from longer training

---

### 8. **Batch Size Optimization** ✓

**What was improved:**
- Increased from batch_size=16 to batch_size=32
- Better gradient estimates
- More stable training

**Trade-off:**
- Slightly more memory needed
- Better convergence on modern GPUs

---

### 9. **Mixed Precision Training** ✓

**What was improved:**
- Added optional mixed precision (float16 + float32)
- Faster training without accuracy loss
- Reduced memory usage

**Enable in config**:
```yaml
mixed_precision: true
```

**Why it helps:**
- 2-3x speedup on certain GPUs
- Reduces memory consumption
- TensorFlow handles stability automatically

---

### 10. **Test-Time Augmentation (TTA)** ✓

**What was improved:**
- Created robust TTA pipeline
- Averages predictions across 10 augmented versions
- Significant accuracy boost at inference time (typical: +2-4%)

**Usage**:
```python
from src.models.cnn.tta_ensemble import TTAPredictor

predictor = TTAPredictor("artifacts/models/cnn_optimized/best_cnn.keras")
result = predictor.predict_with_tta(image)
```

**What it does**:
1. Apply original image → get prediction
2. Apply 10 random augmentations → get 10 predictions
3. Average all 11 predictions
4. Return most confident class

**Why it helps:**
- Reduces sensitivity to augmentation variations
- Better robustness on edge cases
- Industry standard for competition

**Location**: `src/models/cnn/tta_ensemble.py`

---

### 11. **Model Ensemble Support** ✓

**What was improved:**
- Created ensemble prediction infrastructure
- Can combine multiple trained models
- Significantly boosts accuracy (typical: +1-3%)

**Usage**:
```python
predictor = TTAPredictor(
    model_paths=["model1.keras", "model2.keras", "model3.keras"],
    use_ensemble=True
)
result = predictor.predict_ensemble(image)
```

**Strategy**:
- Train models with different seeds (randomness)
- Average their predictions
- Each model captures different patterns

**Expected boost**: +1-3% additional accuracy

---

### 12. **Better Image Preprocessing** ✓

**What was improved:**
- Added specialized microscopy preprocessing
- CLAHE for contrast enhancement
- Adaptive gamma correction
- Better normalization strategies

**Available methods** (in `src/models/cnn/preprocessing.py`):
- `PlanktonImageProcessor.apply_clahe()` - Contrast enhancement
- `PlanktonImageProcessor.adaptive_gamma_correction()` - Exposure correction
- `PlanktonImageProcessor.extract_morphological_features()` - Complementary features

**Configuration options**:
- Light augmentation for validation
- Standard augmentation for training
- Aggressive augmentation for small datasets
- Microscopy-aware augmentation for domain adaptation

---

### 13. **Larger Model Input Size** ✓

**What was improved:**
- Increased image_size from 160 to 224 pixels
- Better detail capture
- Slight performance cost (~5-10% slower)

**Trade-off analysis**:
- 160×160: Faster, lower detail
- 224×224: Slower, captures more morphological details
- 256×256: Even more detail, significant slowdown

**Current recommendation**: 224×224 (good balance)

---

### 14. **Better Monitoring & Metrics** ✓

**What was improved:**
- Track more metrics: precision, recall, weighted F1
- Monitor balanced accuracy for imbalanced data
- Better visualization of performance

**Metrics tracked**:
```
- Top-1 Accuracy: Direct class match
- Top-5 Accuracy: Predicted class in top 5
- Macro F1: All classes weighted equally
- Weighted F1: F1 weighted by class frequency
- Balanced Accuracy: Average recall per class
- Precision & Recall: Per-class performance
```

**Why it matters:**
- Weighted accuracy can hide poor minority class performance
- Balanced accuracy is more honest for imbalanced data
- F1 score balances precision and recall

---

## Performance Expectations

### Baseline (Current)
- Top-1 Accuracy: **71.75%**
- Training time: ~15 minutes per stage
- No TTA, no ensemble

### With Optimized Training
- Expected: **74-76%** (+2-4%)
- With TTA: **76-78%** (+4-6% total)
- With 3-model ensemble + TTA: **77-80%** (+5-8% total)
- Training time: ~45 minutes (longer training)

### Expected Gains Per Optimization

| Optimization | Est. Gain |
|--------------|-----------|
| Better augmentation | +0.5-1.0% |
| Progressive unfreezing | +0.5-1.0% |
| Learning rate schedule | +0.5-1.0% |
| Better class weights | +0.5-1.0% |
| Label smoothing | +0.3-0.5% |
| Longer training | +1.0-2.0% |
| TTA at inference | +2.0-4.0% |
| 3-model ensemble | +1.0-2.0% |
| **Total expected** | **+6-13%** |

---

## How to Use

### 1. Install Dependencies

```powershell
pip install albumentations opencv-python
```

### 2. Train Optimized Model

```powershell
cd "D:\college notes\6th sem\ML\ml project"
.\.venv312\Scripts\Activate.ps1
python -m src.models.cnn.train_cnn_optimized --config configs/cnn_optimized.yaml
```

### 3. Use TTA at Inference

```python
from src.models.cnn.tta_ensemble import TTAPredictor
from src.common.manifest import load_class_to_index
import json

# Load model and class mapping
model_path = "artifacts/models/cnn_optimized/best_cnn.keras"
with open("artifacts/manifests/class_to_index.json") as f:
    class_to_index = json.load(f)
    index_to_class = {v: k for k, v in class_to_index.items()}

# Create predictor with TTA
predictor = TTAPredictor(model_path, image_size=224, num_augmentations=10)

# Make prediction
result = predictor.predict_with_tta(image_array)
print(f"Class: {index_to_class[result['class_idx']]}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 4. Create 3-Model Ensemble

```python
# Train 3 models with different random seeds
seeds = [42, 123, 456]
model_paths = []

for seed in seeds:
    # Modify config seed, train model
    model_paths.append(f"model_seed_{seed}.keras")

# Use ensemble
predictor = TTAPredictor(model_paths, use_ensemble=True)
result = predictor.predict_ensemble(image)
```

---

## Advanced Configuration

### For Maximum Accuracy (Slower Training)

```yaml
training:
  batch_size: 16  # Smaller batches
  epochs_stage1: 30
  epochs_stage2: 10
  lr_stage1: 0.0001  # Conservative
  warmup_epochs: 2
  label_smoothing: 0.15  # More smoothing

model:
  type: efficientnetv2b1  # Larger model
  image_size: 256  # More detail
  dense_units: 768
  l2_regularization: 5.0e-4  # More regularization
```

### For Fast Training (Still High Accuracy)

```yaml
training:
  batch_size: 64
  epochs_stage1: 8
  epochs_stage2: 2
  mixed_precision: true

model:
  type: efficientnetv2b0
  image_size: 160
  dense_units: 384
```

---

## Troubleshooting

### OutOfMemory Error
- Reduce `batch_size` from 32 to 16
- Reduce `image_size` from 224 to 160
- Disable `mixed_precision`

### Training Too Slow
- Increase `batch_size` to 64
- Enable `mixed_precision: true`
- Reduce `epochs_stage1` to 10

### Model Not Improving
- Increase `label_smoothing` (more regularization needed)
- Increase `max_class_weight` (worse imbalance than expected)
- Try different seed values (randomness matters)

### Low Validation Accuracy
- Check if class imbalance worsened
- Verify augmentation isn't too aggressive
- Ensure proper class weighting is applied

---

## Files Created/Modified

**New files:**
- `src/models/cnn/train_cnn_optimized.py` - Optimized training script
- `src/models/cnn/tta_ensemble.py` - TTA and ensemble utilities
- `src/models/cnn/preprocessing.py` - Advanced preprocessing
- `configs/cnn_optimized.yaml` - Optimized hyperparameters

**Modified files:**
- `requirements.txt` - Added albumentations, opencv-python

---

## Next Steps for Further Improvement

1. **Ensemble multiple model architectures**
   - Combine EfficientNet with ResNet, ViT, or ConvNeXt
   - Different architectures capture different patterns

2. **Knowledge distillation**
   - Train ensemble, then distill into single model
   - Gets ensemble benefits with single model speed

3. **Hyperparameter search**
   - Use Optuna or Ray Tune for automated search
   - Optimize learning rate, dropout, layer unfreezing parameters

4. **Custom loss functions**
   - Contrastive loss for similar species
   - Metric learning for fine-grained distinction

5. **Data augmentation search**
   - Use AutoAugment or RandAugment
   - Find optimal augmentation strategies automatically

6. **Multi-task learning**
   - Predict species AND morphological features
   - Better representation learning

7. **Semi-supervised learning**
   - Use unlabeled plankton images
   - Pseudo-labeling or consistency regularization

---

## Summary

These optimizations target the main challenge: **severe class imbalance with limited minority class samples**. 

The strategy combines:
- **Better data**: Advanced augmentation + preprocessing
- **Better training**: Progressive unfreezing + learning schedules
- **Better inference**: TTA + ensembles
- **Better regularization**: Class weights + label smoothing

**Expected improvement: 71.75% → 77-80%** with all optimizations.
