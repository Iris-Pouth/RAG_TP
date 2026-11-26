"""Paramètres du projet, lus depuis le fichier .env (ou les variables d'environnement)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

# Dossiers
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data" / "bdpm"))  # fichiers BDPM bruts
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT_DIR / "index"))  # index FAISS sauvegardé

# Embeddings : modèle multilingue entraîné pour la recherche question -> document,
# vecteurs de 384 dimensions, lit jusqu'à 512 tokens (~2 000 caractères)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Découpage des fiches (en caractères) : la plupart des fiches tiennent en un passage
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Recherche
TOP_K = int(os.getenv("TOP_K", 5))
# Si la question ne cite aucun médicament, un passage doit dépasser ce score pour être gardé.
# Les scores du modèle e5 sont resserrés : ~0,75 pour un texte sans rapport, ~0,90 pour un très proche.
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.85))
# Bornes utilisées pour convertir un score de similarité en confiance de 0 à 100 %
SCORE_FLOOR = float(os.getenv("SCORE_FLOOR", 0.80))  # score -> 0 %
SCORE_CEIL = float(os.getenv("SCORE_CEIL", 0.90))  # score -> 100 %

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
