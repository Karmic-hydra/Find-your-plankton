# Run Experiments

This project compares Traditional ML and CNN on identical manifests.

## 1) Prepare environment

```powershell
pip install -r requirements.txt
```

## 2) Ensure manifests exist

```powershell
python src/data/generate_splits.py --dataset-root data/2014
```

## 3) Train Traditional ML

```powershell
python -m src.models.traditional.train_traditional \
  --manifest-dir artifacts/manifests \
  --config configs/traditional_ml.yaml \
  --output-dir artifacts/models/traditional
```

## 4) Train CNN (EfficientNetV2B0)

```powershell
python -m src.models.cnn.train_cnn \
  --manifest-dir artifacts/manifests \
  --config configs/cnn.yaml \
  --output-dir artifacts/models/cnn
```

## 5) Build comparison report

```powershell
python -m src.eval.compare_models \
  --traditional-metrics artifacts/models/traditional/metrics.json \
  --cnn-metrics artifacts/models/cnn/metrics.json \
  --out-json reports/comparison/head_to_head.json \
  --out-md reports/comparison/head_to_head.md
```

## Notes
- The default configs cap samples for fast baselines.
- Set max sample limits to null in config files for full-manifest training.
- For fairness, do not regenerate manifests with different seeds between model runs.
