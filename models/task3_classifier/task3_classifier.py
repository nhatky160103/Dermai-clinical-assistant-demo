"""
Task 3 — Skin Lesion Disease Classification (7 classes)

Mock inference module that generates realistic classification outputs.
When real model weights are available, swap with RealClassifier.

Interface:
    predict(image: PIL.Image) -> ClassificationResult
"""
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DISEASE_CLASSES, DISEASE_FULL_NAMES, DISEASE_RISK_LEVELS


@dataclass
class DiagnosisEntry:
    """Single diagnosis prediction."""
    code: str               # e.g. "MEL"
    name: str               # e.g. "Melanoma"
    confidence: float       # Calibrated probability [0, 1]
    risk_level: str         # "HIGH", "MODERATE", "LOW"


@dataclass
class ClassificationResult:
    """Full classification output."""
    top_k: List[DiagnosisEntry]    # Sorted by confidence (descending)
    all_probabilities: dict        # {class_code: probability}
    is_uncertain: bool             # True if top-1 confidence < 0.5
    primary_diagnosis: str         # Top-1 class code
    primary_confidence: float      # Top-1 confidence


class MockClassifier:
    """
    Mock classifier — generates realistic probability distributions.
    Simulates EfficientNet-B4 trained on ISIC 2018 Task 3.
    """

    def __init__(self):
        self.model_name = "MockClassifier-v1 (placeholder)"
        # Approximate class distribution in ISIC 2018
        self._class_priors = {
            "MEL":   0.11,
            "NV":    0.67,
            "BCC":   0.05,
            "AKIEC": 0.03,
            "BKL":   0.11,
            "DF":    0.01,
            "VASC":  0.01,
        }

    def predict(self, image: Image.Image, top_k: int = 3) -> ClassificationResult:
        """
        Run mock classification on an image.

        Args:
            image: PIL Image (dermoscopy)
            top_k: Number of top diagnoses to return

        Returns:
            ClassificationResult with calibrated probabilities
        """
        img_array = np.array(image.convert("RGB"))
        np.random.seed(hash(img_array.tobytes()[:1000]) % (2**31))

        # Generate raw scores biased by priors + image features
        raw_scores = self._generate_scores(img_array)

        # Temperature scaling for calibration (T > 1 = less confident)
        temperature = 1.5
        calibrated = np.exp(raw_scores / temperature)
        probabilities = calibrated / calibrated.sum()

        # Build result
        all_probs = {cls: float(prob) for cls, prob in zip(DISEASE_CLASSES, probabilities)}

        sorted_indices = np.argsort(probabilities)[::-1]
        top_entries = []
        for idx in sorted_indices[:top_k]:
            code = DISEASE_CLASSES[idx]
            top_entries.append(DiagnosisEntry(
                code=code,
                name=DISEASE_FULL_NAMES[code],
                confidence=round(float(probabilities[idx]), 4),
                risk_level=DISEASE_RISK_LEVELS[code],
            ))

        primary_code = DISEASE_CLASSES[sorted_indices[0]]
        primary_conf = float(probabilities[sorted_indices[0]])

        return ClassificationResult(
            top_k=top_entries,
            all_probabilities=all_probs,
            is_uncertain=primary_conf < 0.5,
            primary_diagnosis=primary_code,
            primary_confidence=round(primary_conf, 4),
        )

    def _generate_scores(self, img: np.ndarray) -> np.ndarray:
        """Generate realistic raw logits based on image features."""
        # Extract simple color features to bias predictions
        mean_color = img.mean(axis=(0, 1)) / 255.0
        brightness = mean_color.mean()
        redness = mean_color[0] - mean_color.mean()

        scores = np.zeros(len(DISEASE_CLASSES))
        for i, cls in enumerate(DISEASE_CLASSES):
            # Base score from prior
            prior_score = np.log(self._class_priors[cls] + 1e-8)

            # Image-based adjustments
            if cls == "MEL" and redness > 0.05:
                prior_score += 0.5   # Darker/redder → slightly more MEL
            elif cls == "NV" and brightness > 0.4:
                prior_score += 0.3   # Lighter → more NV
            elif cls == "VASC" and redness > 0.1:
                prior_score += 0.4   # Very red → vascular
            elif cls == "BKL" and brightness > 0.5:
                prior_score += 0.2

            # Add noise
            scores[i] = prior_score + np.random.normal(0, 0.5)

        return scores


class RealClassifier:
    """
    Real classifier — loads trained EfficientNet / other CNN weights.
    TODO: Implement when model weights are available.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.input_size = int(os.environ.get("TASK3_INPUT_SIZE", "224"))
        self._predict_fn = None
        self._load_model()

    def _load_model(self):
        """
        Load pretrained classifier.

        Supported artifacts:
        1) TorchScript file (recommended for easy deployment)
        2) Serialized nn.Module
        3) state_dict checkpoint + factory function from env:
           TASK3_MODEL_FACTORY="module.path:create_model"
        """
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("PyTorch is required for RealClassifier.") from e

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Preferred path: ISIC helper (supports model_state checkpoints directly).
        model_type = os.environ.get("TASK3_MODEL_TYPE", "auto").lower()
        if model_type in ("auto", "isic"):
            try:
                from models.task3_classifier.isic_checkpoint_infer import build_isic_predict_fn

                backbone = os.environ.get("TASK3_BACKBONE", "efficientnet_v2_s")
                source_dir = os.environ.get("TASK3_SOURCE_DIR", "")
                self._predict_fn = build_isic_predict_fn(
                    checkpoint_path=self.model_path,
                    backbone=backbone,
                    num_classes=len(DISEASE_CLASSES),
                    source_dir=source_dir or None,
                    device=str(self.device),
                )
                self.model_name = f"RealClassifier[ISIC:{backbone}:{os.path.basename(self.model_path)}]"
                return
            except Exception as e:
                if model_type == "isic":
                    raise
                print(f"⚠ Task3 ISIC helper unavailable ({e}). Falling back to generic loader.")

        # First try TorchScript
        try:
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.eval()
            self.model = model
            self.model_name = f"RealClassifier[TorchScript:{os.path.basename(self.model_path)}]"
            return
        except Exception:
            pass

        ckpt = torch.load(self.model_path, map_location=self.device)

        if isinstance(ckpt, torch.nn.Module):
            self.model = ckpt.to(self.device).eval()
        elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), torch.nn.Module):
            self.model = ckpt["model"].to(self.device).eval()
        elif isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
            factory_ref = os.environ.get("TASK3_MODEL_FACTORY", "").strip()
            if not factory_ref or ":" not in factory_ref:
                raise RuntimeError(
                    "Checkpoint looks like state_dict, but TASK3_MODEL_FACTORY is missing. "
                    "Set TASK3_MODEL_FACTORY='your_module:create_model'."
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
                "Use TorchScript, nn.Module, or state_dict + TASK3_MODEL_FACTORY."
            )

        self.model_name = f"RealClassifier[{self.model.__class__.__name__}]"

    def predict(self, image: Image.Image, top_k: int = 3) -> ClassificationResult:
        """Run real model inference and return calibrated top-k diagnoses."""
        if self._predict_fn is not None:
            pred = self._predict_fn(image, top_k=top_k)
            probs = pred.probabilities
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_entries = []
            for code, conf in sorted_items[:top_k]:
                top_entries.append(DiagnosisEntry(
                    code=code,
                    name=DISEASE_FULL_NAMES.get(code, code),
                    confidence=round(float(conf), 4),
                    risk_level=DISEASE_RISK_LEVELS.get(code, "LOW"),
                ))

            primary_code = sorted_items[0][0]
            primary_conf = float(sorted_items[0][1])
            return ClassificationResult(
                top_k=top_entries,
                all_probabilities={k: float(v) for k, v in probs.items()},
                is_uncertain=primary_conf < 0.5,
                primary_diagnosis=primary_code,
                primary_confidence=round(primary_conf, 4),
            )

        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for RealClassifier prediction.")

        img_rgb = image.convert("RGB")
        resized = img_rgb.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # ImageNet normalization by default (override in your custom model wrapper if needed).
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            raw_out = self.model(x)

        logits = self._extract_logits(raw_out).flatten()
        if logits.numel() < len(DISEASE_CLASSES):
            pad = torch.full(
                (len(DISEASE_CLASSES) - logits.numel(),),
                fill_value=-1e9,
                dtype=logits.dtype,
                device=logits.device,
            )
            logits = torch.cat([logits, pad], dim=0)
        logits = logits[:len(DISEASE_CLASSES)]

        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        all_probs = {cls: float(prob) for cls, prob in zip(DISEASE_CLASSES, probs)}

        sorted_indices = np.argsort(probs)[::-1]
        top_entries = []
        for idx in sorted_indices[:top_k]:
            code = DISEASE_CLASSES[idx]
            top_entries.append(DiagnosisEntry(
                code=code,
                name=DISEASE_FULL_NAMES[code],
                confidence=round(float(probs[idx]), 4),
                risk_level=DISEASE_RISK_LEVELS[code],
            ))

        primary_idx = int(sorted_indices[0])
        primary_code = DISEASE_CLASSES[primary_idx]
        primary_conf = float(probs[primary_idx])

        return ClassificationResult(
            top_k=top_entries,
            all_probabilities=all_probs,
            is_uncertain=primary_conf < 0.5,
            primary_diagnosis=primary_code,
            primary_confidence=round(primary_conf, 4),
        )

    @staticmethod
    def _extract_logits(model_output):
        """Normalize different model output formats into logits tensor."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for RealClassifier output parsing.")

        if isinstance(model_output, torch.Tensor):
            logits = model_output
        elif isinstance(model_output, (list, tuple)) and model_output:
            first = model_output[0]
            if not isinstance(first, torch.Tensor):
                raise RuntimeError("First tuple/list element is not a tensor.")
            logits = first
        elif isinstance(model_output, dict):
            logits = None
            for key in ("logits", "out", "pred"):
                val = model_output.get(key)
                if isinstance(val, torch.Tensor):
                    logits = val
                    break
            if logits is None:
                raise RuntimeError("Could not find logits tensor in model output dict.")
        else:
            raise RuntimeError("Unsupported model output type for classification.")

        if logits.ndim == 2:
            return logits[0]
        return logits


def get_classifier(model_path: Optional[str] = None):
    """Factory: return real classifier if weights exist, else mock."""
    if model_path and os.path.exists(model_path):
        try:
            return RealClassifier(model_path)
        except Exception as e:
            print(f"⚠ RealClassifier init failed ({e}). Falling back to MockClassifier.")
            pass
    return MockClassifier()
