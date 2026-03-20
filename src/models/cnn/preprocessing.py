"""
Enhanced preprocessing and feature engineering for plankton classification.

Includes:
1. Better image normalization for microscopy images
2. Histogram equalization techniques
3. Advanced feature extraction
4. Data augmentation strategies
"""

from __future__ import annotations

import numpy as np
import cv2
import tensorflow as tf
from typing import Tuple


class PlanktonImageProcessor:
    """Specialized image processor for plankton microscopy images."""

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Improves contrast for better feature visibility in microscopy images.
        """
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            l_channel_clahe = clahe.apply(l_channel)
            
            lab[:, :, 0] = l_channel_clahe
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            result = clahe.apply(image)
        
        return result.astype(np.float32) / 255.0

    @staticmethod
    def adaptive_gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """Apply adaptive gamma correction for better visibility."""
        if image.max() > 1.0:
            image = image / 255.0
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)], dtype=np.uint8)
        
        image_uint8 = (image * 255).astype(np.uint8)
        corrected = cv2.LUT(image_uint8, table)
        
        return corrected.astype(np.float32) / 255.0

    @staticmethod
    def normalize_for_cnn(image: np.ndarray, mean: Tuple = (0.485, 0.456, 0.406),
                          std: Tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
        """Normalize image using ImageNet statistics."""
        if image.max() > 1.0:
            image = image / 255.0
        
        image = (image - np.array(mean)) / np.array(std)
        return image

    @staticmethod
    def extract_morphological_features(image: np.ndarray) -> dict:
        """
        Extract morphological features for complementary analysis.
        
        Features: contour area, perimeter, circularity, aspect ratio, etc.
        """
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / max(h, 1e-6)
        
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx, cy = 0, 0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'centroid_x': float(cx),
            'centroid_y': float(cy),
        }


class DataAugmentationStrategy:
    """Optimized augmentation strategies for plankton classification."""

    @staticmethod
    def get_light_augmentation():
        """Light augmentation for validation/test."""
        import albumentations as A
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def get_standard_augmentation():
        """Standard augmentation for training."""
        import albumentations as A
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.Affine(scale=(0.85, 1.15), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def get_aggressive_augmentation():
        """Aggressive augmentation for small datasets with high class imbalance."""
        import albumentations as A
        return A.Compose([
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.8),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.15, 0.15), p=0.7),
            
            # Intensity
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.RandomGamma(p=0.4),
            A.GaussNoise(p=0.3),
            A.CLAHE(p=0.4),
            
            # Advanced
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Downscale(scale_min=0.7, scale_max=0.95, p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def get_microscopy_aware_augmentation():
        """Augmentation aware of microscopy imaging characteristics."""
        import albumentations as A
        return A.Compose([
            # Preserve spatial relationships
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.6, border_mode=1),
            
            # Simulate imaging variations
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),  # Exposure variations
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(blur_limit=3, p=1),
            ], p=0.3),
            
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
