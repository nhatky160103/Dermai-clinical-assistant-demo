"""
Image Search — RAG-based similar case retrieval for dermoscopy images.

Combines CLIP image and text embeddings for hybrid similarity matching
against the reference database.
"""
from typing import List, Optional
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.embedding_store import get_embedding_store, ReferenceCase


def _rerank_cases(cases: List[ReferenceCase], clinical_context) -> List[ReferenceCase]:
    """
    Re-rank retrieved cases with clinically informed priors:
    - diagnosis alignment
    - structure overlap
    - rule-tag overlap
    - evidence quality
    """
    if not cases:
        return cases

    query_diag = getattr(clinical_context, "primary_diagnosis", "")
    query_structs = {s.attribute for s in getattr(clinical_context, "detected_structures", [])}
    query_tags = set(getattr(clinical_context, "rule_tags", []))

    reranked = []
    for c in cases:
        base = float(getattr(c, "similarity_score", 0.0))
        diag_bonus = 0.15 if c.diagnosis == query_diag else 0.0

        case_structs = set(c.structures or [])
        struct_overlap = (len(case_structs & query_structs) / max(len(query_structs), 1)) if query_structs else 0.0
        struct_bonus = 0.10 * struct_overlap

        case_tags = set(getattr(c, "rule_tags", []) or [])
        tag_overlap = (len(case_tags & query_tags) / max(len(query_tags), 1)) if query_tags else 0.0
        tag_bonus = 0.07 * tag_overlap

        evidence_bonus = 0.08 * float(getattr(c, "evidence_weight", 0.6))
        new_score = min(max(0.65 * base + diag_bonus + struct_bonus + tag_bonus + evidence_bonus, 0.0), 1.0)
        c.similarity_score = round(new_score, 3)
        reranked.append(c)

    reranked.sort(key=lambda x: x.similarity_score, reverse=True)
    return reranked


def search_similar_cases(
    image: Image.Image,
    clinical_context,
    top_k: int = 3,
) -> List[ReferenceCase]:
    """
    Search for similar reference cases using:
    1. Hybrid CLIP query embedding from image + clinical context text
    2. Clinical metadata reranking after vector retrieval

    Args:
        image: Original dermoscopy image
        clinical_context: ClinicalContext object
        top_k: Number of results to return

    Returns:
        List of ReferenceCase sorted by similarity
    """
    store = get_embedding_store()

    # Use the context builder's search query for text matching
    query_text = clinical_context.search_query

    # Search with fused image + text query embedding
    results = store.search_similar(
        query_text=query_text,
        query_image=image,
        top_k=top_k,
    )

    return _rerank_cases(results, clinical_context)


def format_similar_cases_text(cases: List[ReferenceCase]) -> str:
    """Format reference cases as text for LLM context."""
    if not cases:
        return "No similar reference cases found."

    parts = [f"Found {len(cases)} similar reference cases:\n"]
    for i, case in enumerate(cases, 1):
        parts.append(f"--- Reference Case {i} (Similarity: {case.similarity_score:.0%}) ---")
        parts.append(f"Diagnosis: {case.diagnosis_name} ({case.diagnosis})")
        if case.structures:
            parts.append(f"Structures: {', '.join(s.replace('_', ' ') for s in case.structures)}")
        if getattr(case, "rule_tags", None):
            parts.append(f"Rule Tags: {', '.join(t.replace('_', ' ') for t in case.rule_tags)}")
        parts.append(f"Description: {case.description}")
        parts.append(
            f"Evidence: {getattr(case, 'diagnosis_confirm_type', 'single image expert consensus')} "
            f"(weight={getattr(case, 'evidence_weight', 0.6):.2f})"
        )
        parts.append(f"Source: {case.source}")
        parts.append("")
    return "\n".join(parts)


def initialize_reference_database():
    """
    Initialize the reference database with curated cases.
    Call this once at app startup.
    """
    store = get_embedding_store()
    from rag.reference_data import REFERENCE_CASES, populate_store

    count = store.get_collection_count()
    if count == 0:
        count = populate_store(store)
        print(f"✓ Populated reference database with {count} cases")
    else:
        # Auto-sync schema if existing collection was created by older code.
        sample_id = REFERENCE_CASES[0]["case_id"] if REFERENCE_CASES else None
        sample_meta = store.get_case_metadata(sample_id) if sample_id else {}
        has_new_schema = (
            bool(sample_meta.get("diagnosis_confirm_type"))
            and bool(sample_meta.get("evidence_weight"))
            and bool(sample_meta.get("task2_features"))
        )
        if not has_new_schema:
            synced = populate_store(store)
            print(f"✓ Synced reference database to latest schema ({synced} cases upserted)")
        else:
            print(f"✓ Reference database already has {count} cases")
