"""
Reference Data loader + enrich logic for dermoscopy knowledge base.

Primary storage is JSON in data/reference_cases/*.json (not hardcoded in Python).
"""
import os
import json
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


REFERENCE_CASES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "reference_cases",
    "isic_reference_cases.json",
)

TASK2_LABELS = [
    "pigment_network",
    "streaks",
    "negative_network",
    "milia_like_cyst",
    "globules",
]

# Evidence hierarchy for medical-label trustworthiness.
# Higher = more trustworthy label source.
CONFIRM_TYPE_WEIGHTS = {
    "histopathology": 1.0,
    "serial imaging showing no change": 0.8,
    "single image expert consensus": 0.6,
}


def _load_reference_cases() -> List[Dict[str, Any]]:
    if not os.path.exists(REFERENCE_CASES_PATH):
        print(f"⚠ Reference cases JSON not found: {REFERENCE_CASES_PATH}")
        return []

    try:
        with open(REFERENCE_CASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("⚠ Reference cases JSON must be a list. Using empty list.")
            return []
        return data
    except Exception as e:
        print(f"⚠ Failed to load reference cases JSON: {e}")
        return []


REFERENCE_CASES = _load_reference_cases()


def _default_task2_features(structures: List[str]) -> Dict[str, Dict[str, float]]:
    sset = set(structures or [])
    features: Dict[str, Dict[str, float]] = {}
    for label in TASK2_LABELS:
        present = label in sset
        features[label] = {
            "present": bool(present),
            "coverage_pct": 3.0 if present else 0.0,
            "confidence": 0.7 if present else 0.0,
        }
    return features


def _normalize_task2_features(task2_features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    normalized = _default_task2_features([])
    if not isinstance(task2_features, dict):
        return normalized

    for label in TASK2_LABELS:
        raw = task2_features.get(label, {})
        if not isinstance(raw, dict):
            continue
        present = bool(raw.get("present", False))
        coverage = float(raw.get("coverage_pct", 0.0) or 0.0)
        confidence = float(raw.get("confidence", 0.0) or 0.0)
        normalized[label] = {
            "present": present,
            "coverage_pct": max(0.0, coverage),
            "confidence": min(max(0.0, confidence), 1.0),
        }
    return normalized


def _structures_from_task2(task2_features: Dict[str, Dict[str, float]]) -> List[str]:
    out = []
    for label in TASK2_LABELS:
        feat = task2_features.get(label, {})
        if bool(feat.get("present", False)):
            out.append(label)
    return out


def _infer_rule_tags(diagnosis: str, structures: List[str]) -> List[str]:
    tags = []
    sset = set(structures or [])

    if diagnosis in {"MEL", "NV", "DF"} or "pigment_network" in sset:
        tags.append("melanocytic_pattern")
    if diagnosis in {"MEL", "BCC"}:
        tags.append("high_risk")
    if diagnosis == "AKIEC":
        tags.append("premalignant_pattern")
    if diagnosis == "BKL":
        tags.append("keratin_pattern")
    if diagnosis == "VASC":
        tags.append("vascular_pattern")
    if "streaks" in sset:
        tags.append("high_risk_growth")
    if "milia_like_cyst" in sset:
        tags.append("benign_keratosis_clue")
    if "negative_network" in sset:
        tags.append("atypical_network")
    if not sset:
        tags.append("no_specific_structure")
    return sorted(set(tags))


def _build_clinical_text(case: Dict[str, Any]) -> str:
    structures = case.get("structures", [])
    struct_text = ", ".join(s.replace("_", " ") for s in structures) if structures else "no specific dermoscopic structure"
    task2_features = case.get("task2_features", {})
    task2_parts = []
    for label in TASK2_LABELS:
        feat = task2_features.get(label, {})
        if feat.get("present", False):
            task2_parts.append(
                f"{label.replace('_', ' ')}(coverage={feat.get('coverage_pct', 0.0):.1f}%, conf={feat.get('confidence', 0.0):.2f})"
            )
    task2_text = ", ".join(task2_parts) if task2_parts else "no confident task2 feature"
    rule_text = ", ".join(case.get("rule_tags", [])) if case.get("rule_tags") else "no explicit rule tags"
    return (
        f"Diagnosis: {case.get('diagnosis_name', case.get('diagnosis', 'Unknown'))} ({case.get('diagnosis', 'UNK')}). "
        f"Structures: {struct_text}. "
        f"Task2 features: {task2_text}. "
        f"Description: {case.get('description', '')} "
        f"Clinical notes: {case.get('clinical_notes', '')} "
        f"Rule tags: {rule_text}. "
        f"Confirmation type: {case.get('diagnosis_confirm_type', 'single image expert consensus')} "
        f"(evidence weight={case.get('evidence_weight', 0.6):.2f})."
    ).strip()


def _enrich_case(case: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(case)
    task2_features = enriched.get("task2_features")
    if task2_features is None:
        task2_features = _default_task2_features(enriched.get("structures", []))
    task2_features = _normalize_task2_features(task2_features)
    enriched["task2_features"] = task2_features
    enriched["structures"] = _structures_from_task2(task2_features)

    confirm_type = enriched.get("diagnosis_confirm_type", "single image expert consensus")
    weight = CONFIRM_TYPE_WEIGHTS.get(confirm_type, 0.6)
    enriched["diagnosis_confirm_type"] = confirm_type
    enriched["evidence_weight"] = float(enriched.get("evidence_weight", weight))
    enriched["rule_tags"] = enriched.get("rule_tags") or _infer_rule_tags(
        enriched.get("diagnosis", ""),
        enriched.get("structures", []),
    )
    enriched["clinical_text"] = enriched.get("clinical_text") or _build_clinical_text(enriched)
    return enriched


def populate_store(store) -> int:
    """
    Populate the embedding store with reference cases.

    Args:
        store: EmbeddingStore instance

    Returns:
        Number of cases added
    """
    count = 0
    for raw_case in REFERENCE_CASES:
        case = _enrich_case(raw_case)
        store.add_reference_case(
            case_id=case["case_id"],
            description=case["clinical_text"],
            metadata={
                "diagnosis": case["diagnosis"],
                "diagnosis_name": case["diagnosis_name"],
                "structures": case["structures"],
                "task2_features": case["task2_features"],
                "rule_tags": case["rule_tags"],
                "diagnosis_confirm_type": case["diagnosis_confirm_type"],
                "evidence_weight": case["evidence_weight"],
                "clinical_notes": case.get("clinical_notes", ""),
                "source": case.get("source", "ISIC 2018"),
            },
        )
        count += 1
    return count


def get_fallback_cases(query_text: str, top_k: int = 3):
    """
    Fallback case retrieval when ChromaDB/CLIP not available.
    Uses simple keyword matching.
    """
    from rag.embedding_store import ReferenceCase

    query_lower = query_text.lower()
    scored_cases = []

    for raw_case in REFERENCE_CASES:
        case = _enrich_case(raw_case)
        score = 0

        # Match diagnosis
        if case["diagnosis"].lower() in query_lower:
            score += 3
        if case["diagnosis_name"].lower() in query_lower:
            score += 2

        # Match structures
        for struct in case["structures"]:
            if struct.replace("_", " ") in query_lower:
                score += 1

        # Match rule tags
        for tag in case.get("rule_tags", []):
            if tag.replace("_", " ") in query_lower:
                score += 0.8

        # Match clinical text keywords
        desc_words = case["clinical_text"].lower().split()
        query_words = set(query_lower.split())
        overlap = len(set(desc_words) & query_words)
        score += overlap * 0.1
        score += 0.5 * case.get("evidence_weight", 0.6)

        scored_cases.append((score, case))

    # Sort by score
    scored_cases.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, case in scored_cases[:top_k]:
        results.append(ReferenceCase(
            case_id=case["case_id"],
            diagnosis=case["diagnosis"],
            diagnosis_name=case["diagnosis_name"],
            structures=case["structures"],
            task2_features=case.get("task2_features", {}),
            description=case["clinical_text"],
            similarity_score=round(min(score / 5.0, 1.0), 3),
            source=case.get("source", "ISIC 2018"),
            rule_tags=case.get("rule_tags", []),
            diagnosis_confirm_type=case.get("diagnosis_confirm_type", "single image expert consensus"),
            evidence_weight=float(case.get("evidence_weight", 0.6)),
        ))

    return results
