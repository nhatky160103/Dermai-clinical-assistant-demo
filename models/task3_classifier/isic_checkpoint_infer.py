"""
ISIC checkpoint inference helper.

This module adapts the training/evaluation checkpoint format:
    ckpt["model_state"]
for runtime inference in an application.
"""
from __future__ import annotations

import os
import sys
import glob
import random
import argparse
import importlib.util
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


DEFAULT_CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


@dataclass
class ISICPredictResult:
    """Inference output for one image."""
    probabilities: Dict[str, float]
    top_k: List[Dict[str, float]]
    predicted_class: str
    confidence: float


class ISICCheckpointPredictor:
    """
    Load Network + checkpoint and run single-image inference.

    Expected source code (same style as your training repo):
      - model.py containing: Network, BACKBONE_CONFIGS
    """

    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "efficientnet_v2_s",
        num_classes: int = 7,
        class_names: Optional[List[str]] = None,
        source_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        training_model, BACKBONE_CONFIGS = self._import_training_model(source_dir)

        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone: {backbone}. Available: {list(BACKBONE_CONFIGS.keys())}")

        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.num_classes = num_classes
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        _, input_size, mean, std = BACKBONE_CONFIGS[backbone]
        self.input_size = int(input_size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        self.model = self._build_network(training_model).to(self.device)
        self._load_checkpoint_weights()
        self.model.eval()

    @staticmethod
    def _import_training_model(source_dir: Optional[str]):
        """
        Import training model module.

        Priority:
        1) source_dir/model.py (if source_dir is provided)
        2) local project module: models.task2_segmenter.model
        """
        if source_dir:
            source_abs = os.path.abspath(source_dir)
            if source_abs not in sys.path:
                sys.path.insert(0, source_abs)
            import model as training_model  # type: ignore
            from model import BACKBONE_CONFIGS  # type: ignore
            return training_model, BACKBONE_CONFIGS

        # Load sibling model.py directly so this file can be executed as a script.
        local_model_path = os.path.join(os.path.dirname(__file__), "model.py")
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model.py not found: {local_model_path}")

        spec = importlib.util.spec_from_file_location("isic_local_model", local_model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for: {local_model_path}")
        training_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training_model)

        if not hasattr(training_model, "BACKBONE_CONFIGS"):
            raise RuntimeError("BACKBONE_CONFIGS not found in local model.py")

        return training_model, training_model.BACKBONE_CONFIGS

    def _build_network(self, training_model):
        """
        Build training-repo Network with a safe inference default.
        Prefer pretrained=False to avoid external downloads at runtime.
        """
        try:
            return training_model.Network(
                backbone=self.backbone,
                num_classes=self.num_classes,
                pretrained=False,
            )
        except TypeError:
            # Fallback for repos that only accept string style flags.
            return training_model.Network(
                backbone=self.backbone,
                num_classes=self.num_classes,
                pretrained="default",
            )

    def _load_checkpoint_weights(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict):
            if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
                state_dict = ckpt["model_state"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            else:
                # fallback: maybe checkpoint itself is a raw state dict
                state_dict = ckpt
        else:
            raise RuntimeError("Unsupported checkpoint format. Expected dict-like checkpoint.")

        state_dict = self._normalize_state_dict_keys(state_dict, self.model.state_dict().keys())
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        # Keep strict=False for resilience in app runtime, but fail hard if everything mismatches.
        if len(missing) >= len(self.model.state_dict()):
            raise RuntimeError(
                "Checkpoint could not be mapped to model keys. "
                "Please verify backbone/model architecture and checkpoint source."
            )

        if missing:
            print(f"⚠ Missing keys while loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"⚠ Unexpected keys in checkpoint: {len(unexpected)}")

    @staticmethod
    def _normalize_state_dict_keys(state_dict: dict, net_keys) -> dict:
        out = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            out[nk] = v

        net_has_model_prefix = any(k.startswith("model.") for k in net_keys)
        ckpt_has_model_prefix = any(k.startswith("model.") for k in out.keys())

        if net_has_model_prefix and not ckpt_has_model_prefix:
            out = {f"model.{k}": v for k, v in out.items()}
        elif (not net_has_model_prefix) and ckpt_has_model_prefix:
            out = {k[len("model."):] if k.startswith("model.") else k: v for k, v in out.items()}

        return out

    def predict_pil(self, image: Image.Image, top_k: int = 3) -> ISICPredictResult:
        img = image.convert("RGB")
        x = self._preprocess(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        probs_dict = {
            cls_name: float(prob)
            for cls_name, prob in zip(self.class_names, probs)
        }

        sorted_idx = np.argsort(probs)[::-1]
        top_items = []
        for idx in sorted_idx[:top_k]:
            top_items.append({
                "class_name": self.class_names[idx],
                "confidence": float(probs[idx]),
            })

        best_idx = int(sorted_idx[0])
        return ISICPredictResult(
            probabilities=probs_dict,
            top_k=top_items,
            predicted_class=self.class_names[best_idx],
            confidence=float(probs[best_idx]),
        )

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Resize + normalize without torchvision dependency."""
        resized = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).float()


def build_isic_predict_fn(
    checkpoint_path: str,
    backbone: str = "efficientnet_v2_s",
    num_classes: int = 7,
    class_names: Optional[List[str]] = None,
    source_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Callable[[Image.Image, int], ISICPredictResult]:
    """
    Factory returning a callable for app integration.

    Example:
        predict_fn = build_isic_predict_fn(
            checkpoint_path="/path/best.pth",
            backbone="efficientnet_v2_s",
            source_dir="/path/Skin-Lesion-Recognition.Pytorch/src",
        )
        result = predict_fn(pil_image, top_k=3)
    """
    predictor = ISICCheckpointPredictor(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        num_classes=num_classes,
        class_names=class_names,
        source_dir=source_dir,
        device=device,
    )

    def _predict(image: Image.Image, top_k: int = 3) -> ISICPredictResult:
        return predictor.predict_pil(image=image, top_k=top_k)

    return _predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-image ISIC checkpoint inference.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/task2_segmenter/weights/exp_000/best.pth",
    )
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--source_dir", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    predict_fn = build_isic_predict_fn(
        checkpoint_path=args.checkpoint_path,
        backbone=args.backbone,
        num_classes=args.num_classes,
        source_dir=args.source_dir or None,
    )

    selected_image = args.image_path
    if not selected_image and args.image_dir:
        candidates = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            candidates.extend(glob.glob(os.path.join(args.image_dir, ext)))
        if not candidates:
            raise FileNotFoundError(f"No images found in directory: {args.image_dir}")
        selected_image = random.choice(candidates)

    if not selected_image:
        raise ValueError("Provide --image_path or --image_dir to run inference.")

    pil_image = Image.open(selected_image).convert("RGB")
    result = predict_fn(pil_image, top_k=args.top_k)

    print(f"Image: {selected_image}")
    print(f"Predicted: {result.predicted_class} ({result.confidence:.4f})")
    print(f"Top-{args.top_k}: {result.top_k}")
