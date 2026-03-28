"""
Context Builder — Aggregates Task 2 + Task 3 outputs into structured clinical context.

This module acts as the "brain" that combines all model outputs and prepares
the context for RAG search and LLM interpretation.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    ATTRIBUTE_DESCRIPTIONS, ATTRIBUTE_LABELS,
    DISEASE_RISK_LEVELS, DISEASE_FULL_NAMES,
)


@dataclass
class DetectedStructure:
    """A detected dermoscopic structure with clinical info."""
    attribute: str
    label: str
    confidence: float
    coverage_pct: float
    clinical_description: str


@dataclass
class ClinicalContext:
    """Aggregated clinical context for LLM and RAG."""
    # Structure analysis
    detected_structures: List[DetectedStructure]
    total_structures_detected: int
    structure_summary: str

    # Classification
    primary_diagnosis: str
    primary_diagnosis_name: str
    primary_confidence: float
    top_3_diagnoses: list
    is_uncertain: bool
    risk_level: str

    # Risk indicators
    risk_flags: List[str]

    # Combined context string for LLM
    context_text: str

    # Search query for RAG
    search_query: str

    # Optional patient metadata
    patient_metadata: Optional[dict] = None


def build_context(
    segmentation_results: dict,
    classification_result,
    patient_metadata: Optional[dict] = None,
) -> ClinicalContext:
    """
    Build clinical context from model outputs.

    Args:
        segmentation_results: dict from Task 2 segmenter
        classification_result: ClassificationResult from Task 3 classifier
        patient_metadata: Optional patient info (age, sex, location, etc.)

    Returns:
        ClinicalContext — structured context for downstream pipeline
    """
    # ─── Detected structures ───
    detected = []
    for attr, result in segmentation_results.items():
        if result.present:
            detected.append(DetectedStructure(
                attribute=attr,
                label=result.label,
                confidence=result.confidence,
                coverage_pct=result.coverage_pct,
                clinical_description=ATTRIBUTE_DESCRIPTIONS.get(attr, ""),
            ))

    # Sort by coverage (largest first)
    detected.sort(key=lambda x: x.coverage_pct, reverse=True)

    # Structure summary
    if detected:
        struct_parts = [
            f"{s.label} ({s.coverage_pct:.1f}% coverage, {s.confidence:.0%} conf)"
            for s in detected
        ]
        structure_summary = f"Detected {len(detected)} dermoscopic structures: " + ", ".join(struct_parts)
    else:
        structure_summary = "No dermoscopic structures confidently detected."

    # ─── Classification ───
    primary = classification_result.primary_diagnosis
    primary_name = DISEASE_FULL_NAMES.get(primary, primary)
    primary_conf = classification_result.primary_confidence
    risk_level = DISEASE_RISK_LEVELS.get(primary, "UNKNOWN")

    top_3 = [
        {
            "code": d.code,
            "name": d.name,
            "confidence": d.confidence,
            "risk_level": d.risk_level,
        }
        for d in classification_result.top_k[:3]
    ]

    # ─── Risk flags ───
    risk_flags = []
    if risk_level == "HIGH":
        risk_flags.append(f"⚠️ HIGH RISK: Primary diagnosis is {primary_name} — requires urgent dermatological review")
    if classification_result.is_uncertain:
        risk_flags.append("⚠️ UNCERTAIN: Top diagnosis confidence < 50% — clinical correlation strongly recommended")
    if any(d.risk_level == "HIGH" and d.confidence > 0.1 for d in classification_result.top_k):
        high_risk_in_top = [d for d in classification_result.top_k if d.risk_level == "HIGH" and d.confidence > 0.1]
        for d in high_risk_in_top:
            if d.code != primary:
                risk_flags.append(
                    f"⚠️ DIFFERENTIAL: {d.name} ({d.code}) at {d.confidence:.0%} — cannot be ruled out"
                )

    # Check structure-diagnosis correlation
    struct_names = [s.attribute for s in detected]
    if "streaks" in struct_names and primary != "MEL":
        risk_flags.append("⚠️ Streaks detected — associated with melanoma, but primary diagnosis is not MEL")
    if "pigment_network" in struct_names and primary not in ("MEL", "NV"):
        risk_flags.append("⚠️ Pigment network detected — typically melanocytic, but diagnosis is non-melanocytic")

    # ─── Context text for LLM ───
    ctx_parts = [
        "=== DERMOSCOPY IMAGE ANALYSIS ===",
        "",
        "--- Structure Analysis (Task 2) ---",
        structure_summary,
    ]

    for s in detected:
        ctx_parts.append(
            f"  • {s.label}: coverage {s.coverage_pct:.1f}%, confidence {s.confidence:.0%}")
        ctx_parts.append(f"    Clinical: {s.clinical_description}")

    ctx_parts += [
        "",
        "--- Disease Classification (Task 3) ---",
        f"Primary Diagnosis: {primary_name} ({primary}) — {primary_conf:.0%} confidence",
        f"Risk Level: {risk_level}",
        f"Uncertainty: {'YES — clinical review recommended' if classification_result.is_uncertain else 'No'}",
        "",
        "Top-3 Differential:",
    ]
    for d in top_3:
        ctx_parts.append(f"  {d['code']:6s} {d['name']:45s} {d['confidence']:.1%}  [{d['risk_level']}]")

    if risk_flags:
        ctx_parts += ["", "--- Risk Indicators ---"]
        for flag in risk_flags:
            ctx_parts.append(f"  {flag}")

    if patient_metadata:
        ctx_parts += ["", "--- Patient Information ---"]
        for key, value in patient_metadata.items():
            ctx_parts.append(f"  {key}: {value}")

    context_text = "\n".join(ctx_parts)

    # ─── Search query for RAG ───
    search_parts = []
    if detected:
        search_parts.append("dermoscopy image with " + ", ".join(s.attribute.replace("_", " ") for s in detected))
    search_parts.append(f"diagnosis {primary_name}")
    search_query = " ".join(search_parts)

    return ClinicalContext(
        detected_structures=detected,
        total_structures_detected=len(detected),
        structure_summary=structure_summary,
        primary_diagnosis=primary,
        primary_diagnosis_name=primary_name,
        primary_confidence=primary_conf,
        top_3_diagnoses=top_3,
        is_uncertain=classification_result.is_uncertain,
        risk_level=risk_level,
        risk_flags=risk_flags,
        context_text=context_text,
        search_query=search_query,
        patient_metadata=patient_metadata,
    )
