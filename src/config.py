"""Paramètres du projet, lus depuis le fichier .env (ou les variables d'environnement)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

# Dossiers
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data" / "bdpm"))  # fichiers BDPM bruts
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT_DIR / "index"))  # index FAISS sauvegardé

# Embeddings : modèle multilingue, vecteurs de 384 dimensions
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Découpage des fiches (en caractères). Le modèle d'embeddings tronque les textes
# au-delà de 128 tokens (~500 caractères) : des passages plus longs seraient mal représentés.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Recherche
TOP_K = int(os.getenv("TOP_K", 5))
# En dessous de ce score de similarité, un passage est jugé non pertinent
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.30))

# LLM : Claude
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")  # low | medium | high
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4000))
# Si Claude refuse une requête, l'API la relance automatiquement sur un autre modèle
USE_FALLBACK = os.getenv("USE_FALLBACK", "true").lower() == "true"
# Reformuler les questions de suivi (« et son prix ? ») avant la recherche
REFORMULATION = os.getenv("REFORMULATION", "true").lower() == "true"

# Nombre d'échanges (question + réponse) gardés en mémoire pour la conversation
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 3))
