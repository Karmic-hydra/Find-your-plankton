"""
Quick-start script to run optimized CNN training and inference.

Usage:
    # Train optimized model
    python run_optimization.py --mode train
    
    # Test optimized model with TTA
    python run_optimization.py --mode test
    
    # Create 3-model ensemble
    python run_optimization.py --mode ensemble --seeds 42 123 456
"""

import argparse
import json
from pathlib import Path
from typing import List
import subprocess
import sys


def train_optimized(**kwargs):
    """Run optimized CNN training."""
    print("=" * 70)
    print("PLANKTON CNN OPTIMIZATION - TRAINING")
    print("=" * 70)
    print("\nOptimizations enabled:")
    print("  ✓ Advanced data augmentation (albumentations)")
    print("  ✓ Progressive layer unfreezing")
    print("  ✓ Learning rate warmup + cosine annealing")
    print("  ✓ Improved class weighting with epsilon smoothing")
    print("  ✓ Label smoothing (0.1)")
    print("  ✓ Focal loss with class weighting")
    print("  ✓ Extended training (15 epochs)")
    print("  ✓ Batch size 32")
    print("  ✓ Mixed precision training")
    print("  ✓ Better early stopping and checkpointing")
    print("  ✓ Image size: 224×224 (more detail)")
    print("\nStarting training...\n")
    
    cmd = [
        sys.executable,
        "-m",
        "src.models.cnn.train_cnn_optimized",
        "--config", "configs/cnn_optimized.yaml",
        "--output-dir", "artifacts/models/cnn_optimized",
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE")
        print("=" * 70)
        print("\nModel saved to: artifacts/models/cnn_optimized/")
        print("  - best_cnn.keras (best validation loss)")
        print("  - final_cnn.keras (final trained model)")
        print("  - metrics.json (detailed metrics)")
        print("\nNext steps:")
        print("  1. Evaluate with TTA:")
        print("     python run_optimization.py --mode test")
        print("  2. Create ensemble of multiple models:")
        print("     python run_optimization.py --mode ensemble --seeds 42 123 456")
    else:
        print("\n✗ Training failed!")
        sys.exit(1)


def test_with_tta(**kwargs):
    """Test trained model with Test-Time Augmentation."""
    print("\n" + "=" * 70)
    print("TESTING WITH TEST-TIME AUGMENTATION (TTA)")
    print("=" * 70)
    
    model_path = "artifacts/models/cnn_optimized/best_cnn.keras"
    
    if not Path(model_path).exists():
        print(f"\n✗ Model not found: {model_path}")
        print("Please train the model first:")
        print("  python run_optimization.py --mode train")
        sys.exit(1)
    
    try:
        import tensorflow as tf
        from src.models.cnn.tta_ensemble import TTAPredictor
        from src.common.manifest import load_class_to_index
    except Exception as e:
        print(f"\n✗ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    print(f"\nLoading model: {model_path}")
    print("Creating TTA predictor with 10 augmentations...")
    
    try:
        predictor = TTAPredictor(
            model_paths=model_path,
            image_size=224,
            num_augmentations=10,
        )
        
        # Load class mapping
        class_to_index = load_class_to_index("artifacts/manifests/class_to_index.json")
        index_to_class = {v: k for k, v in class_to_index.items()}
        
        # Test on a sample image
        import numpy as np
        from PIL import Image
        
        # Find a test image
        test_images = list(Path("data/2014").rglob("*.png"))[:3]
        
        if test_images:
            print(f"\nTesting on {len(test_images)} sample images...")
            for img_path in test_images:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = image.resize((224, 224))
                    image_array = np.array(image, dtype=np.float32)
                    
                    result = predictor.predict_with_tta(image_array)
                    
                    predicted_class = index_to_class[result['class_idx']]
                    confidence = result['confidence']
                    
                    print(f"\n  Image: {img_path.name}")
                    print(f"  Predicted: {predicted_class}")
                    print(f"  Confidence: {confidence:.2%} (±{result['confidence_std']:.2%})")
                    print(f"  Top-5: {[index_to_class[i] for i in result['top5']]}")
                    
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
        
        print("\n" + "=" * 70)
        print("✓ TTA INFERENCE COMPLETE")
        print("=" * 70)
        print("\nKey insight: TTA typically improves accuracy by 2-4%")
        print("This average of 10 augmented predictions is more robust than single inference.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_ensemble(seeds: List[int] = None, **kwargs):
    """Create ensemble by training models with different seeds."""
    if seeds is None:
        seeds = [42, 123, 456]
    
    print("\n" + "=" * 70)
    print("CREATING 3-MODEL ENSEMBLE")
    print("=" * 70)
    print(f"\nTraining {len(seeds)} models with different random seeds...")
    print(f"Seeds: {seeds}\n")
    
    model_paths = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL {i}/{len(seeds)} (seed={seed})")
        print(f"{'='*70}")
        
        # Create modified config with different seed
        config_path = Path("configs/cnn_optimized.yaml")
        config_content = config_path.read_text()
        
        # Modify seed
        config_content_modified = config_content.replace(
            "seed: 42",
            f"seed: {seed}"
        )
        
        # Write temp config
        temp_config = Path(f"configs/cnn_optimized_seed_{seed}.yaml")
        temp_config.write_text(config_content_modified)
        
        # Train
        output_dir = f"artifacts/models/cnn_ensemble_seed_{seed}"
        cmd = [
            sys.executable,
            "-m",
            "src.models.cnn.train_cnn_optimized",
            "--config", str(temp_config),
            "--output-dir", output_dir,
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            model_path = f"{output_dir}/best_cnn.keras"
            model_paths.append(model_path)
            print(f"\n✓ Model {i} saved to: {model_path}")
        else:
            print(f"\n✗ Model {i} training failed!")
            sys.exit(1)
        
        temp_config.unlink()  # Clean up temp config
    
    print("\n" + "=" * 70)
    print("✓ ENSEMBLE CREATION COMPLETE")
    print("=" * 70)
    print(f"\nEnsemble models:")
    for i, path in enumerate(model_paths, 1):
        print(f"  {i}. {path}")
    
    # Save ensemble config
    ensemble_config = {
        "model_paths": model_paths,
        "num_models": len(model_paths),
        "seeds": seeds,
        "tta_augmentations": 10,
    }
    
    ensemble_path = Path("artifacts/models/ensemble_config.json")
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_path.write_text(json.dumps(ensemble_config, indent=2))
    
    print(f"\nEnsemble config saved to: {ensemble_path}")
    print("\nUsage:")
    print("  from src.models.cnn.tta_ensemble import TTAPredictor")
    print(f"  predictor = TTAPredictor({model_paths})")
    print("  result = predictor.predict_ensemble(image)")


def main():
    parser = argparse.ArgumentParser(
        description="Plankton CNN Optimization Quick-Start"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "ensemble"],
        default="train",
        help="Mode to run: train, test with TTA, or ensemble",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Seeds for ensemble training",
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_optimized()
    elif args.mode == "test":
        test_with_tta()
    elif args.mode == "ensemble":
        create_ensemble(seeds=args.seeds)


if __name__ == "__main__":
    main()
