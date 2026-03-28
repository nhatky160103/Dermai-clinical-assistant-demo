"""
RAG Embedding Store — ChromaDB + CLIP for dermoscopy image/text similarity search.

Uses sentence-transformers CLIP model for multi-modal embeddings and
ChromaDB as the vector store for fast similarity search.
"""
import os
import json
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHROMA_DB_PATH, EMBEDDING_MODEL


@dataclass
class ReferenceCase:
    """A reference case from the knowledge base."""
    case_id: str
    diagnosis: str
    diagnosis_name: str
    structures: List[str]
    description: str
    similarity_score: float
    source: str
    image_path: Optional[str] = None


class EmbeddingStore:
    """
    ChromaDB-backed embedding store for dermoscopy reference cases.
    Uses CLIP for multi-modal (image + text) embeddings.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_DB_PATH
        self._clip_model = None
        self._chroma_client = None
        self._collection = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization — only load heavy models when first needed."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.persist_dir, exist_ok=True)

            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="dermoscopy_references",
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            print(f"✓ ChromaDB initialized at {self.persist_dir}")
            print(f"  Collection has {self._collection.count()} documents")

        except ImportError:
            print("⚠ ChromaDB not installed. RAG search will use fallback mode.")
            self._initialized = False
        except Exception as e:
            print(f"⚠ ChromaDB init error: {e}. Using fallback mode.")
            self._initialized = False

    def _get_clip_model(self):
        """Load CLIP model lazily."""
        if self._clip_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._clip_model = SentenceTransformer(EMBEDDING_MODEL)
                print(f"✓ CLIP model loaded: {EMBEDDING_MODEL}")
            except ImportError:
                print("⚠ sentence-transformers not installed. Using text-only embeddings.")
            except Exception as e:
                print(f"⚠ CLIP model error: {e}. Using text-only fallback.")
        return self._clip_model

    def encode_text(self, text: str) -> Optional[List[float]]:
        """Encode text into embedding vector using CLIP."""
        model = self._get_clip_model()
        if model is None:
            return None
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_image(self, image: Image.Image) -> Optional[List[float]]:
        """Encode image into embedding vector using CLIP."""
        model = self._get_clip_model()
        if model is None:
            return None
        embedding = model.encode(image, convert_to_numpy=True)
        return embedding.tolist()

    def add_reference_case(
        self,
        case_id: str,
        description: str,
        metadata: dict,
        embedding: Optional[List[float]] = None,
    ):
        """
        Add a reference case to the vector store.

        Args:
            case_id: Unique identifier
            description: Text description for embedding
            metadata: Case metadata (diagnosis, structures, source)
            embedding: Pre-computed embedding, or None to compute from description
        """
        self._lazy_init()
        if not self._initialized:
            return

        if embedding is None:
            embedding = self.encode_text(description)
            if embedding is None:
                return

        # ChromaDB requires string values in metadata
        safe_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (list, tuple)):
                safe_metadata[k] = json.dumps(v)
            else:
                safe_metadata[k] = str(v)

        self._collection.upsert(
            ids=[case_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[safe_metadata],
        )

    def search_similar(
        self,
        query_text: str,
        query_image: Optional[Image.Image] = None,
        top_k: int = 3,
    ) -> List[ReferenceCase]:
        """
        Search for similar reference cases using hybrid (image + text) search.

        Args:
            query_text: Text query (e.g., detected structures + diagnosis)
            query_image: Optional image for visual similarity
            top_k: Number of results to return

        Returns:
            List of ReferenceCase with similarity scores
        """
        self._lazy_init()

        # If ChromaDB not available, use fallback
        if not self._initialized or self._collection.count() == 0:
            return self._fallback_search(query_text, top_k)

        # Try image embedding first, fall back to text
        if query_image is not None:
            embedding = self.encode_image(query_image)
        else:
            embedding = None

        if embedding is None:
            embedding = self.encode_text(query_text)

        if embedding is None:
            return self._fallback_search(query_text, top_k)

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
        )

        cases = []
        if results and results["ids"] and results["ids"][0]:
            for i, case_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                doc = results["documents"][0][i] if results["documents"] else ""
                dist = results["distances"][0][i] if results["distances"] else 1.0

                structures = json.loads(meta.get("structures", "[]"))
                if isinstance(structures, str):
                    structures = [structures]

                cases.append(ReferenceCase(
                    case_id=case_id,
                    diagnosis=meta.get("diagnosis", "Unknown"),
                    diagnosis_name=meta.get("diagnosis_name", "Unknown"),
                    structures=structures,
                    description=doc,
                    similarity_score=round(1 - dist, 3),   # cosine distance → similarity
                    source=meta.get("source", "ISIC 2018"),
                    image_path=meta.get("image_path"),
                ))

        return cases

    def _fallback_search(self, query_text: str, top_k: int) -> List[ReferenceCase]:
        """Fallback: return pre-defined reference cases based on keyword matching."""
        from rag.reference_data import get_fallback_cases
        return get_fallback_cases(query_text, top_k)

    def get_collection_count(self) -> int:
        """Return number of documents in the collection."""
        self._lazy_init()
        if self._initialized:
            return self._collection.count()
        return 0


# Singleton instance
_store = None

def get_embedding_store() -> EmbeddingStore:
    """Get the global embedding store instance."""
    global _store
    if _store is None:
        _store = EmbeddingStore()
    return _store
