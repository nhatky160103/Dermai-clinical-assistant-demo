"""
Configuration and constants for the Clinical Decision Support Pipeline.
ISIC 2018 — Dermatology AI Assistant
"""
import os

# ──────────────────────────────────────────────────────────────── .env loader
def _load_local_env():
    """
    Lightweight .env loader (no extra dependency).
    Only sets variables that are not already present in process env.
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()

# ──────────────────────────────────────────────────────────────── API Keys
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
VERTEX_API_KEY = os.environ.get("VERTEX_API_KEY", "")
VERTEX_BASE_URL = os.environ.get(
    "VERTEX_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
VERTEX_REST_BASE_URL = os.environ.get(
    "VERTEX_REST_BASE_URL",
    "https://aiplatform.googleapis.com/v1",
)
# Gemini API-key mode (recommended for local demo)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", VERTEX_API_KEY)
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
GEMINI_REST_BASE_URL = os.environ.get(
    "GEMINI_REST_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
)
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.0-flash")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", VERTEX_MODEL)

# ──────────────────────────────────────────────────────────────── Task 2: Segmentation
ATTRIBUTES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
]

ATTRIBUTE_LABELS = {
    "pigment_network": "Pigment Network",
    "negative_network": "Negative Network",
    "streaks": "Streaks",
    "milia_like_cyst": "Milia-like Cyst",
    "globules": "Globules",
}

ATTRIBUTE_DESCRIPTIONS = {
    "pigment_network": "Pigmented reticular grid — primary indicator of melanocytic origin",
    "negative_network": "Hypopigmented inverse network — associated with specific melanoma subtypes",
    "streaks": "Radial projections at lesion periphery — may indicate aggressive growth",
    "milia_like_cyst": "Small white/yellow round globules — typically benign",
    "globules": "Small round brown structures — correspond to melanocyte nests",
}

ATTRIBUTE_COLORS = {
    "pigment_network":  (231,  76,  60),   # Red
    "negative_network": ( 52, 152, 219),   # Blue
    "streaks":          ( 46, 204, 113),   # Green
    "milia_like_cyst":  (243, 156,  18),   # Orange
    "globules":         (155,  89, 182),   # Purple
}

# ──────────────────────────────────────────────────────────────── Task 3: Classification
DISEASE_CLASSES = [
    "MEL",    # Melanoma
    "NV",     # Melanocytic Nevus
    "BCC",    # Basal Cell Carcinoma
    "AKIEC",  # Actinic Keratosis / Intraepithelial Carcinoma
    "BKL",    # Benign Keratosis
    "DF",     # Dermatofibroma
    "VASC",   # Vascular Lesion
]

DISEASE_FULL_NAMES = {
    "MEL":   "Melanoma",
    "NV":    "Melanocytic Nevus",
    "BCC":   "Basal Cell Carcinoma",
    "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma",
    "BKL":   "Benign Keratosis",
    "DF":    "Dermatofibroma",
    "VASC":  "Vascular Lesion",
}

DISEASE_RISK_LEVELS = {
    "MEL":   "HIGH",
    "BCC":   "HIGH",
    "AKIEC": "MODERATE",
    "NV":    "LOW",
    "BKL":   "LOW",
    "DF":    "LOW",
    "VASC":  "LOW",
}

DISEASE_COLORS = {
    "MEL":   "#E74C3C",
    "NV":    "#27AE60",
    "BCC":   "#E67E22",
    "AKIEC": "#F39C12",
    "BKL":   "#3498DB",
    "DF":    "#1ABC9C",
    "VASC":  "#9B59B6",
}

# ──────────────────────────────────────────────────────────────── RAG / Embeddings
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
EMBEDDING_MODEL = "clip-ViT-B-32"     # sentence-transformers CLIP model
REFERENCE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "reference_cases")

# ──────────────────────────────────────────────────────────────── Model Paths
TASK2_MODEL_PATH = os.environ.get("TASK2_MODEL_PATH", "")
TASK3_MODEL_PATH = os.environ.get("TASK3_MODEL_PATH", "")

# ──────────────────────────────────────────────────────────────── UI
APP_TITLE = "🔬 DermAI Clinical Assistant"
APP_DESCRIPTION = (
    "AI-powered clinical decision support for dermatology. "
    "Upload a dermoscopy image to receive structure analysis, "
    "disease classification, and AI-generated clinical interpretation."
)
