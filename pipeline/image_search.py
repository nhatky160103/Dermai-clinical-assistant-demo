"""
Image Search — RAG-based similar case retrieval for dermoscopy images.

Combines CLIP image embeddings with text-based context search
for hybrid similarity matching against the reference database.
"""
from typing import List, Optional
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.embedding_store import get_embedding_store, ReferenceCase


def search_similar_cases(
    image: Image.Image,
    clinical_context,
    top_k: int = 3,
) -> List[ReferenceCase]:
    """
    Search for similar reference cases using:
    1. CLIP image embedding (visual similarity)
    2. Text context (detected structures + diagnosis)

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

    # Search with both image and text
    results = store.search_similar(
        query_text=query_text,
        query_image=image,
        top_k=top_k,
    )

    return results


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
        parts.append(f"Description: {case.description}")
        parts.append(f"Source: {case.source}")
        parts.append("")
    return "\n".join(parts)


def initialize_reference_database():
    """
    Initialize the reference database with curated cases.
    Call this once at app startup.
    """
    store = get_embedding_store()
    if store.get_collection_count() == 0:
        from rag.reference_data import populate_store
        count = populate_store(store)
        print(f"✓ Populated reference database with {count} cases")
    else:
        print(f"✓ Reference database already has {store.get_collection_count()} cases")
