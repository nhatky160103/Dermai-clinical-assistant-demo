"""
Task 2 TransUNet checkpoint inference helper.

This helper adapts the notebook-style Task 2 code:
  - VisionTransformer (R50-ViT-B_16)
  - checkpoint with ckpt["state_dict"]
  - image preprocessing: resize + ImageNet normalization
"""
from __future__ import annotations

import os
import sys
import argparse
import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


DEFAULT_ATTRIBUTES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
]


@dataclass
class Task2PredictResult:
    """Single-image segmentation output."""
    prob_maps: Dict[str, np.ndarray]  # resized to original image size, float32 [0,1]
    masks: Dict[str, np.ndarray]      # resized to original image size, uint8 {0,1}
    confidence: Dict[str, float]
    coverage_pct: Dict[str, float]


class ISICTask2CheckpointPredictor:
    """
    Load TransUNet Task 2 model + checkpoint and run single-image inference.
    """

    def __init__(
        self,
        checkpoint_path: str,
        source_dir: Optional[str] = None,
        vit_name: str = "R50-ViT-B_16",
        img_size: int = 512,
        attributes: Optional[List[str]] = None,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.source_dir = source_dir or os.environ.get("TASK2_TRANSUNET_SOURCE_DIR", "")
        self.vit_name = vit_name
        self.img_size = int(img_size)
        self.attributes = attributes or list(DEFAULT_ATTRIBUTES)
        self.threshold = float(threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.model = None

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        vit_seg_cls, vit_configs = self._import_transunet_modules()
        if self.vit_name not in vit_configs:
            raise ValueError(f"Unknown vit_name: {self.vit_name}")

        config_vit = vit_configs[self.vit_name]
        config_vit = self._deepcopy_config(config_vit)
        config_vit.n_classes = len(self.attributes)
        config_vit.n_skip = 3
        if "R50" in self.vit_name:
            config_vit.patches.grid = (self.img_size // 16, self.img_size // 16)

        model = vit_seg_cls(config_vit, img_size=self.img_size, num_classes=len(self.attributes)).to(self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
                state_dict = ckpt["model_state"]
            else:
                state_dict = ckpt
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) >= len(model.state_dict()):
            raise RuntimeError(
                "Checkpoint could not be mapped to TransUNet model keys. "
                "Please verify checkpoint and source_dir."
            )
        if missing:
            print(f"⚠ Task2 missing keys: {len(missing)}")
        if unexpected:
            print(f"⚠ Task2 unexpected keys: {len(unexpected)}")

        self.model = model.eval()

    @staticmethod
    def _deepcopy_config(config_obj):
        # Avoid importing copy globally just for this operation.
        import copy
        return copy.deepcopy(config_obj)

    def _import_transunet_modules(self):
        """
        Import:
          from networks.vit_seg_modeling import VisionTransformer, CONFIGS
        """
        if self.source_dir:
            source_abs = os.path.abspath(self.source_dir)
            if source_abs not in sys.path:
                sys.path.insert(0, source_abs)

        try:
            module = importlib.import_module("networks.vit_seg_modeling")
            return module.VisionTransformer, module.CONFIGS
        except Exception as e:
            raise RuntimeError(
                "Cannot import TransUNet module 'networks.vit_seg_modeling'. "
                "Set TASK2_TRANSUNET_SOURCE_DIR (or pass source_dir) to your TransUNet root."
            ) from e

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr).float()

    def predict_pil(self, image: Image.Image) -> Task2PredictResult:
        img = image.convert("RGB")
        orig_w, orig_h = img.size
        x = self._preprocess(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = self.model(x)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()  # [C,H,W]

        prob_maps: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        confidence: Dict[str, float] = {}
        coverage_pct: Dict[str, float] = {}

        for i, attr in enumerate(self.attributes):
            p = np.clip(probs[i], 0.0, 1.0).astype(np.float32)
            p_img = Image.fromarray((p * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
            p_resized = (np.asarray(p_img, dtype=np.float32) / 255.0).astype(np.float32)
            m = (p_resized >= self.threshold).astype(np.uint8)

            prob_maps[attr] = p_resized
            masks[attr] = m
            coverage_pct[attr] = float(m.mean() * 100.0)
            if m.any():
                confidence[attr] = float(p_resized[m.astype(bool)].mean())
            else:
                confidence[attr] = float(p_resized.mean() * 0.5)

        return Task2PredictResult(
            prob_maps=prob_maps,
            masks=masks,
            confidence=confidence,
            coverage_pct=coverage_pct,
        )


def build_task2_predict_fn(
    checkpoint_path: str,
    source_dir: Optional[str] = None,
    vit_name: str = "R50-ViT-B_16",
    img_size: int = 512,
    attributes: Optional[List[str]] = None,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Callable[[Image.Image], Task2PredictResult]:
    """
    Factory returning callable for app integration.
    """
    predictor = ISICTask2CheckpointPredictor(
        checkpoint_path=checkpoint_path,
        source_dir=source_dir,
        vit_name=vit_name,
        img_size=img_size,
        attributes=attributes,
        threshold=threshold,
        device=device,
    )

    def _predict(image: Image.Image) -> Task2PredictResult:
        return predictor.predict_pil(image)

    return _predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-image Task2 TransUNet inference.")
    parser.add_argument("--checkpoint_path", type=str, default="models/task2_segmenter/weights/transunet_best.pth")
    parser.add_argument("--source_dir", type=str, default="")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predict_fn = build_task2_predict_fn(
        checkpoint_path=args.checkpoint_path,
        source_dir=args.source_dir or None,
        vit_name=args.vit_name,
        img_size=args.img_size,
        threshold=args.threshold,
    )

    image = Image.open(args.image_path).convert("RGB")
    result = predict_fn(image)
    print("Done. Attributes:")
    for attr in DEFAULT_ATTRIBUTES:
        print(
            f"  {attr:20s} coverage={result.coverage_pct[attr]:6.2f}% "
            f"confidence={result.confidence[attr]:.4f}"
        )

