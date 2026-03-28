"""
Report Generator — Generates the final clinical report combining all pipeline outputs.

Produces:
1. Annotated image with segmentation overlays
2. Classification confidence chart
3. Structured clinical report (HTML/Markdown)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import sys
import os
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ATTRIBUTE_COLORS, ATTRIBUTE_LABELS, DISEASE_COLORS, DISEASE_FULL_NAMES


def create_annotated_image(
    original_image: Image.Image,
    segmentation_results: dict,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Create an annotated image with segmentation mask overlays.

    Args:
        original_image: Original dermoscopy image
        segmentation_results: dict of SegmentationResult
        alpha: Overlay transparency (0 = transparent, 1 = opaque)

    Returns:
        PIL Image with color-coded mask overlays
    """
    img = original_image.convert("RGBA")
    img_array = np.array(img)

    for attr, result in segmentation_results.items():
        if not result.present or result.mask is None:
            continue

        color = ATTRIBUTE_COLORS.get(attr, (255, 255, 255))
        mask = result.mask

        # Resize mask if needed
        if mask.shape[:2] != (img_array.shape[0], img_array.shape[1]):
            from PIL import Image as PILImage
            mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), PILImage.NEAREST)
            mask = np.array(mask_img) > 127

        # Create colored overlay
        overlay = np.zeros_like(img_array)
        overlay[mask == 1] = [color[0], color[1], color[2], int(255 * alpha)]

        # Blend
        mask_bool = mask.astype(bool)
        for c in range(3):
            img_array[:, :, c] = np.where(
                mask_bool,
                np.clip(
                    img_array[:, :, c] * (1 - alpha) + color[c] * alpha,
                    0, 255
                ).astype(np.uint8),
                img_array[:, :, c],
            )

    return Image.fromarray(img_array)


def create_structure_heatmap(
    original_image: Image.Image,
    segmentation_results: dict,
) -> Image.Image:
    """Create a side-by-side visualization: original + individual structure masks."""
    present_attrs = [
        (attr, res) for attr, res in segmentation_results.items() if res.present
    ]

    n_panels = 1 + len(present_attrs)
    if n_panels == 1:
        n_panels = 2  # at least show "No structures"

    fig, axes = plt.subplots(1, min(n_panels, 6), figsize=(4 * min(n_panels, 6), 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    img_array = np.array(original_image.convert("RGB"))

    axes[0].imshow(img_array)
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    if not present_attrs:
        if len(axes) > 1:
            axes[1].imshow(img_array)
            axes[1].set_title("No structures\ndetected", fontsize=9, color="gray")
            axes[1].axis("off")
    else:
        for i, (attr, result) in enumerate(present_attrs[:5]):
            ax = axes[i + 1]
            ax.imshow(img_array)

            mask = result.mask
            if mask.shape[:2] != img_array.shape[:2]:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((img_array.shape[1], img_array.shape[0]))
                mask = np.array(mask_pil) > 127

            # Color overlay
            color = ATTRIBUTE_COLORS.get(attr, (255, 255, 255))
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask == 1] = [c / 255 for c in color] + [0.5]
            ax.imshow(overlay)

            ax.set_title(
                f"{ATTRIBUTE_LABELS.get(attr, attr)}\n{result.coverage_pct:.1f}%",
                fontsize=9,
                fontweight="bold",
            )
            ax.axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_diagnosis_chart(classification_result) -> Image.Image:
    """Create a horizontal bar chart of diagnosis probabilities."""
    fig, ax = plt.subplots(figsize=(8, 4))

    codes = list(classification_result.all_probabilities.keys())
    probs = [classification_result.all_probabilities[c] for c in codes]
    names = [f"{DISEASE_FULL_NAMES.get(c, c)}\n({c})" for c in codes]
    colors = [DISEASE_COLORS.get(c, "#95A5A6") for c in codes]

    # Sort by probability
    sorted_idx = np.argsort(probs)
    codes = [codes[i] for i in sorted_idx]
    probs = [probs[i] for i in sorted_idx]
    names = [names[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(range(len(codes)), probs, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_title("Disease Classification — AI Confidence", fontsize=12, fontweight="bold")

    # Add percentage labels
    for bar, prob in zip(bars, probs):
        if prob > 0.03:
            ax.text(
                bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=9, fontweight="bold",
            )

    # Add uncertainty notice if needed
    if classification_result.is_uncertain:
        ax.text(
            0.5, -0.15,
            "⚠️ UNCERTAIN — Top diagnosis confidence < 50%",
            transform=ax.transAxes, ha="center", fontsize=10,
            color="#E74C3C", fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def generate_report_markdown(
    clinical_context,
    llm_explanation: str,
    similar_cases: list,
) -> str:
    """
    Generate the full clinical report as Markdown.

    Args:
        clinical_context: ClinicalContext object
        llm_explanation: LLM-generated clinical explanation
        similar_cases: List of ReferenceCase

    Returns:
        Markdown string of the complete clinical report
    """
    ctx = clinical_context
    lines = []

    # Header
    lines.append("# 🔬 DermAI Clinical Report")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Risk banner
    if ctx.risk_level == "HIGH":
        lines.append("> 🔴 **HIGH RISK** — This lesion has features associated with malignancy. Immediate dermatological review recommended.")
    elif ctx.risk_level == "MODERATE":
        lines.append("> 🟡 **MODERATE RISK** — Some concerning features detected. Clinical correlation recommended.")
    else:
        lines.append("> 🟢 **LOW RISK** — Findings are consistent with a benign lesion. Routine monitoring advised.")
    lines.append("")

    # Section 1: Structure Analysis
    lines.append("## 📊 1. Dermoscopic Structure Analysis")
    lines.append("")
    if ctx.detected_structures:
        lines.append("| Structure | Coverage | Confidence | Clinical Significance |")
        lines.append("|-----------|----------|------------|----------------------|")
        for s in ctx.detected_structures:
            lines.append(f"| **{s.label}** | {s.coverage_pct:.1f}% | {s.confidence:.0%} | {s.clinical_description} |")
    else:
        lines.append("*No dermoscopic structures confidently detected.*")
    lines.append("")

    # Section 2: Classification
    lines.append("## 🎯 2. Disease Classification")
    lines.append("")
    lines.append(f"**Primary Diagnosis: {ctx.primary_diagnosis_name}** ({ctx.primary_diagnosis}) — {ctx.primary_confidence:.0%}")
    lines.append("")
    if ctx.is_uncertain:
        lines.append("> ⚠️ **Model uncertainty is HIGH.** Clinical correlation is strongly recommended.")
        lines.append("")
    lines.append("| Rank | Diagnosis | Confidence | Risk |")
    lines.append("|------|-----------|------------|------|")
    for i, d in enumerate(ctx.top_3_diagnoses, 1):
        risk_icon = "🔴" if d["risk_level"] == "HIGH" else "🟡" if d["risk_level"] == "MODERATE" else "🟢"
        lines.append(f"| {i} | {d['name']} ({d['code']}) | {d['confidence']:.1%} | {risk_icon} {d['risk_level']} |")
    lines.append("")

    # Section 3: Risk Indicators
    if ctx.risk_flags:
        lines.append("## ⚠️ 3. Risk Indicators")
        lines.append("")
        for flag in ctx.risk_flags:
            lines.append(f"- {flag}")
        lines.append("")

    # Section 4: Similar Cases
    lines.append("## 🔍 4. Similar Reference Cases")
    lines.append("")
    if similar_cases:
        for i, case in enumerate(similar_cases, 1):
            lines.append(f"### Case {i} — {case.diagnosis_name} (Similarity: {case.similarity_score:.0%})")
            lines.append(f"- **Diagnosis:** {case.diagnosis_name} ({case.diagnosis})")
            if case.structures:
                lines.append(f"- **Structures:** {', '.join(s.replace('_', ' ').title() for s in case.structures)}")
            lines.append(f"- **Description:** {case.description}")
            lines.append(f"- **Source:** {case.source}")
            lines.append("")
    else:
        lines.append("*No similar reference cases found in the knowledge base.*")
        lines.append("")

    # Section 5: AI Clinical Explanation
    lines.append("## 🤖 5. AI Clinical Interpretation")
    lines.append("")
    lines.append(llm_explanation)
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("> ⚕️ **Disclaimer:** This report is generated by an AI-assisted clinical decision support system. "
                 "It is intended to augment — not replace — professional medical judgment. All findings must be "
                 "correlated with clinical examination, patient history, and dermoscopic expertise. The treating "
                 "physician bears sole responsibility for diagnostic and therapeutic decisions.")

    return "\n".join(lines)
