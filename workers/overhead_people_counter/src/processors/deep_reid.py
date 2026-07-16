"""
Deep Learning ReID Feature Extractor
Uses OSNet pretrained on person re-identification datasets for robust appearance matching.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
import logging

# Try to import torchreid
try:
    import torch
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeepReIDExtractor:
    """
    Deep learning-based ReID feature extractor using OSNet.

    Extracts 512-dimensional appearance features from person bounding boxes
    for robust re-identification across frames, especially during occlusion.
    """

    def __init__(
        self,
        model_name: str = 'osnet_x1_0',
        device: str = None,
        verbose: bool = True
    ):
        """
        Initialize the ReID model.

        Args:
            model_name: Model architecture. Options (ordered by accuracy):
                - 'osnet_x1_0'  (~6MB, best accuracy, ~8ms GPU / ~40ms CPU)
                - 'osnet_x0_75' (~3MB, good accuracy)
                - 'osnet_x0_5'  (~1.5MB, balanced)
                - 'osnet_x0_25' (~500KB, fastest, ~3ms GPU / ~15ms CPU)
            device: 'cuda' or 'cpu'. Auto-detects if None.
            verbose: Whether to print loading messages.
        """
        if not TORCHREID_AVAILABLE:
            raise ImportError(
                "torchreid is required for deep ReID. "
                "Install with: pip install torchreid"
            )

        self.model_name = model_name
        self.verbose = verbose

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if self.verbose:
            print(f"[DeepReID] Initializing {model_name} on {self.device}...", flush=True)

        # Build model pretrained on combined Market1501 + MSMT17 datasets
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,  # Doesn't matter for feature extraction
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Standard ReID preprocessing parameters
        self.input_size = (128, 256)  # Width x Height (standard ReID aspect ratio)
        self.pixel_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.pixel_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Feature dimension (OSNet outputs 512-dim features)
        self.feature_dim = 512

        if self.verbose:
            print(f"[DeepReID] Model loaded successfully. Feature dim: {self.feature_dim}", flush=True)

        logger.info(f"DeepReID initialized: model={model_name}, device={self.device}")

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess a person crop for the model.

        Args:
            crop: BGR image crop of a person (any size)

        Returns:
            Preprocessed tensor ready for model input
        """
        # Resize to model input size
        img = cv2.resize(crop, self.input_size)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        img = (img - self.pixel_mean) / self.pixel_std

        # HWC to CHW format
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img).float().to(self.device)

    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> Optional[np.ndarray]:
        """
        Extract ReID features from a bounding box in a frame.

        Args:
            frame: Full BGR frame
            x1, y1, x2, y2: Bounding box coordinates

        Returns:
            512-dimensional L2-normalized feature vector, or None if extraction fails.
        """
        try:
            # Ensure valid coordinates
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            # Validate box dimensions
            if x2 <= x1 or y2 <= y1:
                return None

            # Minimum box size check (too small = unreliable features)
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < 20 or box_height < 40:
                return None

            # Crop person region
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                return None

            # Preprocess and extract features
            input_tensor = self._preprocess(crop)
            features = self.model(input_tensor)

            # Convert to numpy and L2 normalize
            features = features.cpu().numpy().flatten()
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features

        except Exception as e:
            logger.debug(f"ReID feature extraction error: {e}")
            return None

    @torch.no_grad()
    def extract_batch(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Optional[np.ndarray]]:
        """
        Extract features for multiple bounding boxes in one forward pass.
        More efficient than calling extract() multiple times.

        Args:
            frame: Full BGR frame
            boxes: List of (x1, y1, x2, y2) tuples

        Returns:
            List of feature vectors (or None for failed extractions)
        """
        if not boxes:
            return []

        h, w = frame.shape[:2]
        crops = []
        valid_indices = []

        # Collect valid crops
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            box_width = x2 - x1
            box_height = y2 - y1

            if x2 > x1 and y2 > y1 and box_width >= 20 and box_height >= 40:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(self._preprocess(crop))
                    valid_indices.append(i)

        # Return all None if no valid crops
        if not crops:
            return [None] * len(boxes)

        # Batch forward pass
        batch = torch.cat(crops, dim=0)
        features = self.model(batch)
        features = features.cpu().numpy()

        # L2 normalize each feature vector
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        features = features / norms

        # Map back to original indices
        results = [None] * len(boxes)
        for idx, feat in zip(valid_indices, features):
            results[idx] = feat

        return results

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.

        Args:
            feat1, feat2: L2-normalized feature vectors

        Returns:
            Similarity score in range [0, 1] (higher = more similar)
        """
        if feat1 is None or feat2 is None:
            return 0.0

        # For L2-normalized vectors, cosine similarity = dot product
        similarity = np.dot(feat1, feat2)

        # Clamp to [0, 1] (can be slightly negative due to floating point)
        return max(0.0, min(1.0, similarity))


class FallbackReIDExtractor:
    """
    Fallback ReID extractor using HSV color histograms.
    Used when torchreid is not available.
    """

    def __init__(self):
        self.feature_dim = 96
        print("[FallbackReID] Using HSV histogram features (torchreid not available)", flush=True)
        logger.warning("Using fallback HSV ReID - install torchreid for better accuracy")

    def extract(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> Optional[np.ndarray]:
        """Extract HSV histogram features (fallback method)."""
        try:
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            # Resize to standard size
            target_size = (64, 128)
            person_resized = cv2.resize(crop, target_size)

            # HSV histogram features
            hsv = cv2.cvtColor(person_resized, cv2.COLOR_BGR2HSV)

            # Split into upper/lower body
            h_split = target_size[1] // 2
            upper = hsv[:h_split, :]
            lower = hsv[h_split:, :]

            features = []
            for region in [upper, lower]:
                h_hist = cv2.calcHist([region], [0], None, [16], [0, 180])
                s_hist = cv2.calcHist([region], [1], None, [8], [0, 256])
                v_hist = cv2.calcHist([region], [2], None, [8], [0, 256])

                h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
                s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
                v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)

                features.extend([h_hist, s_hist, v_hist])

            feature_vector = np.concatenate(features)
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm

            return feature_vector

        except Exception:
            return None

    def extract_batch(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Optional[np.ndarray]]:
        """Extract features for multiple boxes (sequential for fallback)."""
        return [self.extract(frame, *box) for box in boxes]

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        if feat1 is None or feat2 is None:
            return 0.0
        similarity = np.dot(feat1, feat2)
        return max(0.0, min(1.0, similarity))


def create_reid_extractor(
    model_name: str = 'osnet_x1_0',
    device: str = None,
    verbose: bool = True
):
    """
    Factory function to create the best available ReID extractor.

    Returns DeepReIDExtractor if torchreid is available, otherwise FallbackReIDExtractor.
    """
    if TORCHREID_AVAILABLE:
        try:
            return DeepReIDExtractor(model_name=model_name, device=device, verbose=verbose)
        except Exception as e:
            logger.error(f"Failed to initialize DeepReID: {e}")
            print(f"[ReID] Failed to load deep model: {e}. Using fallback.", flush=True)
            return FallbackReIDExtractor()
    else:
        return FallbackReIDExtractor()
