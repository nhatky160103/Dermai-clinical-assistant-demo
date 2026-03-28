"""
Task 2 — Dermoscopic Attribute Segmentation

Mock inference module that generates realistic segmentation outputs.
When real model weights are available, swap the MockSegmenter with RealSegmenter.

Interface:
    predict(image: PIL.Image) -> dict[str, SegmentationResult]
"""
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ATTRIBUTES, ATTRIBUTE_LABELS


@dataclass
class SegmentationResult:
    """Result for a single attribute segmentation."""
    attribute: str
    label: str
    mask: np.ndarray           # Binary mask (H, W), 0/1
    confidence: float          # Overall confidence [0, 1]
    coverage_pct: float        # % of image area covered
    present: bool              # Whether the structure is detected


class MockSegmenter:
    """
    Mock segmenter — generates realistic-looking masks using noise + morphology.
    Simulates output of a SegFormer / TransUNet model.
    """

    def __init__(self):
        self.model_name = "MockSegmenter-v1 (placeholder)"
        # Prevalence probabilities from ISIC 2018 EDA
        self._prevalence = {
            "pigment_network":  0.587,
            "negative_network": 0.073,
            "streaks":          0.039,
            "milia_like_cyst":  0.263,
            "globules":         0.232,
        }
        # Typical coverage ranges (min, max) as fraction
        self._coverage_range = {
            "pigment_network":  (0.02, 0.25),
            "negative_network": (0.01, 0.08),
            "streaks":          (0.005, 0.04),
            "milia_like_cyst":  (0.003, 0.02),
            "globules":         (0.005, 0.05),
        }

    def predict(self, image: Image.Image) -> dict:
        """
        Run mock segmentation on an image.

        Args:
            image: PIL Image (dermoscopy)

        Returns:
            dict[str, SegmentationResult] — one result per attribute
        """
        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]
        results = {}

        np.random.seed(hash(img_array.tobytes()[:1000]) % (2**31))

        for attr in ATTRIBUTES:
            # Decide if this structure is "detected"
            is_present = np.random.random() < self._prevalence[attr]

            if is_present:
                mask = self._generate_mask(img_array, attr, h, w)
                coverage = mask.sum() / (h * w)
                confidence = np.clip(np.random.beta(5, 2), 0.4, 0.95)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                coverage = 0.0
                confidence = np.clip(np.random.beta(2, 5), 0.05, 0.3)

            results[attr] = SegmentationResult(
                attribute=attr,
                label=ATTRIBUTE_LABELS[attr],
                mask=mask,
                confidence=round(float(confidence), 3),
                coverage_pct=round(float(coverage * 100), 2),
                present=is_present,
            )

        return results

    def _generate_mask(self, img: np.ndarray, attr: str, h: int, w: int) -> np.ndarray:
        """Generate a realistic-looking binary mask based on image content."""
        # Use grayscale intensity to guide mask placement
        gray = np.mean(img, axis=2)
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        # Create center-weighted region (lesions are often centered)
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_dist = np.sqrt(cy**2 + cx**2)
        center_weight = 1 - (dist / max_dist)

        # Combine intensity + center weighting + noise
        noise = np.random.random((h, w))
        cov_min, cov_max = self._coverage_range[attr]
        target_coverage = np.random.uniform(cov_min, cov_max)

        if attr == "pigment_network":
            # Grid-like pattern
            score = (1 - gray_norm) * 0.4 + center_weight * 0.4 + noise * 0.2
        elif attr == "streaks":
            # Peripheral, elongated — weight edges
            score = (1 - center_weight) * 0.3 + gray_norm * 0.3 + noise * 0.4
        elif attr == "milia_like_cyst":
            # Bright spots
            score = gray_norm * 0.5 + center_weight * 0.3 + noise * 0.2
        elif attr == "globules":
            # Dark spots
            score = (1 - gray_norm) * 0.5 + center_weight * 0.3 + noise * 0.2
        else:
            # negative_network
            score = gray_norm * 0.3 + center_weight * 0.4 + noise * 0.3

        # Threshold to achieve target coverage
        threshold = np.percentile(score, 100 * (1 - target_coverage))
        mask = (score >= threshold).astype(np.uint8)

        return mask


class RealSegmenter:
    """
    Real segmenter — loads trained SegFormer / TransUNet weights.
    TODO: Implement when model weights are available.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.input_size = int(os.environ.get("TASK2_INPUT_SIZE", "256"))
        self.threshold = float(os.environ.get("TASK2_THRESHOLD", "0.5"))
        self._load_model()

    def _load_model(self):
        """
        Load pretrained model for real inference.

        Supported artifacts:
        1) TorchScript file (recommended for easy deployment)
        2) Serialized nn.Module
        3) state_dict checkpoint + factory function from env:
           TASK2_MODEL_FACTORY="module.path:create_model"
        """
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("PyTorch is required for RealSegmenter.") from e

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # First try TorchScript
        try:
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.eval()
            self.model = model
            self.model_name = f"RealSegmenter[TorchScript:{os.path.basename(self.model_path)}]"
            return
        except Exception:
            pass

        ckpt = torch.load(self.model_path, map_location=self.device)

        if isinstance(ckpt, torch.nn.Module):
            self.model = ckpt.to(self.device).eval()
        elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), torch.nn.Module):
            self.model = ckpt["model"].to(self.device).eval()
        elif isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
            factory_ref = os.environ.get("TASK2_MODEL_FACTORY", "").strip()
            if not factory_ref or ":" not in factory_ref:
                raise RuntimeError(
                    "Checkpoint looks like state_dict, but TASK2_MODEL_FACTORY is missing. "
                    "Set TASK2_MODEL_FACTORY='your_module:create_model'."
                )
            module_path, fn_name = factory_ref.split(":", 1)
            module = __import__(module_path, fromlist=[fn_name])
            factory_fn = getattr(module, fn_name)
            model = factory_fn()
            state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
            model.load_state_dict(state_dict, strict=False)
            self.model = model.to(self.device).eval()
        else:
            raise RuntimeError(
                f"Unsupported checkpoint format at {self.model_path}. "
                "Use TorchScript, nn.Module, or state_dict + TASK2_MODEL_FACTORY."
            )

        self.model_name = f"RealSegmenter[{self.model.__class__.__name__}]"

    def predict(self, image: Image.Image) -> dict:
        """Run real model inference and return SegmentationResult dict."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for RealSegmenter prediction.")

        img_rgb = image.convert("RGB")
        orig_w, orig_h = img_rgb.size

        # Minimal preprocessing; override in your own model wrapper if needed.
        resized = img_rgb.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_out = self.model(x)

        out_tensor = self._extract_output_tensor(raw_out)
        if out_tensor.ndim == 3:
            out_tensor = out_tensor.unsqueeze(0)
        if out_tensor.ndim != 4:
            raise RuntimeError(f"Unexpected segmentation output shape: {tuple(out_tensor.shape)}")

        # Expect [B, C, H, W], use first sample.
        logits = out_tensor[0]
        probs = torch.sigmoid(logits) if (logits.min() < 0 or logits.max() > 1) else logits

        # Align channel count with number of target attributes.
        if probs.shape[0] < len(ATTRIBUTES):
            pad = torch.zeros(
                (len(ATTRIBUTES) - probs.shape[0], probs.shape[1], probs.shape[2]),
                dtype=probs.dtype,
                device=probs.device,
            )
            probs = torch.cat([probs, pad], dim=0)
        probs = probs[:len(ATTRIBUTES)]

        results = {}
        for i, attr in enumerate(ATTRIBUTES):
            prob_map = probs[i].detach().cpu().numpy().astype(np.float32)

            prob_img = Image.fromarray((np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8))
            prob_img = prob_img.resize((orig_w, orig_h), Image.BILINEAR)
            prob_map_resized = np.array(prob_img, dtype=np.float32) / 255.0

            mask = (prob_map_resized >= self.threshold).astype(np.uint8)
            coverage = float(mask.mean())
            present = coverage > 0.003

            if present:
                confidence = float(prob_map_resized[mask.astype(bool)].mean())
            else:
                confidence = float(prob_map_resized.mean() * 0.5)

            results[attr] = SegmentationResult(
                attribute=attr,
                label=ATTRIBUTE_LABELS[attr],
                mask=mask,
                confidence=round(float(np.clip(confidence, 0.0, 1.0)), 3),
                coverage_pct=round(float(coverage * 100.0), 2),
                present=present,
            )

        return results

    @staticmethod
    def _extract_output_tensor(model_output):
        """Normalize different model output formats into a tensor."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for RealSegmenter output parsing.")

        if isinstance(model_output, torch.Tensor):
            return model_output

        if isinstance(model_output, (list, tuple)) and model_output:
            first = model_output[0]
            if isinstance(first, torch.Tensor):
                return first

        if isinstance(model_output, dict):
            for key in ("logits", "masks", "mask", "out", "pred"):
                tensor = model_output.get(key)
                if isinstance(tensor, torch.Tensor):
                    return tensor

        raise RuntimeError("Could not extract tensor from model output.")


def get_segmenter(model_path: Optional[str] = None):
    """Factory: return real segmenter if weights exist, else mock."""
    if model_path and os.path.exists(model_path):
        try:
            return RealSegmenter(model_path)
        except NotImplementedError:
            pass
    return MockSegmenter()
