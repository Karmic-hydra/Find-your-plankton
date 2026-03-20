"""
Test-Time Augmentation (TTA) and Ensemble methods for improved plankton species prediction.

TTA applies random augmentations during inference and averages predictions
to get more robust and accurate results, especially for edge cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import albumentations as A


class TTAPredictor:
    """Test-Time Augmentation predictor with ensemble support."""

    def __init__(
        self,
        model_paths: list[str] | str,
        image_size: int = 224,
        num_augmentations: int = 10,
        use_ensemble: bool = False,
    ):
        """
        Initialize TTA predictor.

        Args:
            model_paths: Path(s) to saved model(s)
            image_size: Input image size
            num_augmentations: Number of augmented versions to generate per image
            use_ensemble: Use multiple models for ensemble
        """
        self.image_size = image_size
        self.num_augmentations = num_augmentations
        self.use_ensemble = use_ensemble

        if isinstance(model_paths, str):
            model_paths = [model_paths]

        self.models = []
        for model_path in model_paths:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'focal_loss_with_smoothing': self._focal_loss_with_smoothing}
            )
            self.models.append(model)

        self.augment_pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=10, p=0.7, border_mode=1),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.Affine(scale=(0.9, 1.1), p=0.5),
                A.GaussNoise(p=0.2),
                A.CLAHE(p=0.3),
            ],
            bbox_params=None,
        )

    @staticmethod
    def _focal_loss_with_smoothing(y_true, y_pred):
        """Placeholder for custom loss function."""
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    def predict_with_tta(
        self,
        image: np.ndarray,
        return_all: bool = False,
    ) -> dict:
        """
        Predict with Test-Time Augmentation.

        Args:
            image: Input image (H, W, 3) in [0, 255] or [0, 1]
            return_all: If True, return predictions from all augmentations

        Returns:
            Dictionary with:
                - 'class_idx': Most confident predicted class
                - 'confidence': Mean confidence across augmentations
                - 'top5': Top 5 class indices
                - 'all_probs': All augmentation predictions (if return_all=True)
        """
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        augmented_predictions = []

        # Original prediction
        img_normalized = self._preprocess(image)
        pred = self.models[0].predict(np.array([img_normalized]), verbose=0)[0]
        augmented_predictions.append(pred)

        # TTA predictions
        for _ in range(self.num_augmentations):
            img_uint8 = (image * 255).astype(np.uint8)
            augmented = self.augment_pipeline(image=img_uint8)['image']
            augmented_normalized = self._preprocess(augmented.astype(np.float32) / 255.0)
            
            pred = self.models[0].predict(np.array([augmented_normalized]), verbose=0)[0]
            augmented_predictions.append(pred)

        # Average predictions
        mean_pred = np.array(augmented_predictions).mean(axis=0)
        std_pred = np.array(augmented_predictions).std(axis=0)

        top1_idx = np.argmax(mean_pred)
        top5_idx = np.argsort(mean_pred)[-5:][::-1]
        confidence = float(mean_pred[top1_idx])

        result = {
            'class_idx': int(top1_idx),
            'confidence': confidence,
            'confidence_std': float(std_pred[top1_idx]),
            'top5': [int(i) for i in top5_idx],
            'top5_confidence': [float(mean_pred[i]) for i in top5_idx],
        }

        if return_all:
            result['all_predictions'] = augmented_predictions

        return result

    def predict_ensemble(self, image: np.ndarray) -> dict:
        """
        Predict using ensemble of multiple models with TTA.

        Args:
            image: Input image

        Returns:
            Ensemble prediction dictionary
        """
        if len(self.models) == 1:
            return self.predict_with_tta(image)

        all_preds = []
        for model in self.models:
            # Generate augmented versions
            predictions = []
            
            if image.max() > 1.0:
                image_norm = image / 255.0
            else:
                image_norm = image

            img_preprocessed = self._preprocess(image_norm)
            pred = model.predict(np.array([img_preprocessed]), verbose=0)[0]
            predictions.append(pred)

            for _ in range(self.num_augmentations // 2):
                img_uint8 = (image_norm * 255).astype(np.uint8)
                augmented = self.augment_pipeline(image=img_uint8)['image']
                augmented_norm = self._preprocess(augmented.astype(np.float32) / 255.0)
                pred = model.predict(np.array([augmented_norm]), verbose=0)[0]
                predictions.append(pred)

            all_preds.append(np.mean(predictions, axis=0))

        # Final ensemble average
        ensemble_pred = np.array(all_preds).mean(axis=0)
        
        top1_idx = np.argmax(ensemble_pred)
        top5_idx = np.argsort(ensemble_pred)[-5:][::-1]

        return {
            'class_idx': int(top1_idx),
            'confidence': float(ensemble_pred[top1_idx]),
            'top5': [int(i) for i in top5_idx],
            'top5_confidence': [float(ensemble_pred[i]) for i in top5_idx],
            'num_models': len(self.models),
        }

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image with ImageNet normalization."""
        # Ensure float32 and shape
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if image.shape != (self.image_size, self.image_size, 3):
            image = tf.image.resize(image, [self.image_size, self.image_size]).numpy()

        # ImageNet normalization
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        return image


def create_ensemble_predictions(
    image_paths: list[str],
    model_paths: list[str],
    class_names: list[str],
    image_size: int = 224,
    num_tta_augmentations: int = 10,
) -> list[dict]:
    """
    Create predictions for multiple images using ensemble with TTA.

    Args:
        image_paths: List of image file paths
        model_paths: List of model file paths for ensemble
        class_names: List of class names
        image_size: Input size
        num_tta_augmentations: Number of TTA augmentations

    Returns:
        List of prediction results
    """
    predictor = TTAPredictor(
        model_paths=model_paths,
        image_size=image_size,
        num_augmentations=num_tta_augmentations,
        use_ensemble=len(model_paths) > 1,
    )

    results = []
    for img_path in image_paths:
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_size, image_size))
        image_array = np.array(image, dtype=np.float32)

        if len(model_paths) > 1:
            pred = predictor.predict_ensemble(image_array)
        else:
            pred = predictor.predict_with_tta(image_array)

        pred['image_path'] = img_path
        pred['predicted_class'] = class_names[pred['class_idx']]
        pred['top5_classes'] = [class_names[i] for i in pred['top5']]

        results.append(pred)

    return results
