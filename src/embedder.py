"""Transformation de textes en vecteurs (embeddings)."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        # Le modèle est téléchargé une seule fois, puis mis en cache sur la machine
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Renvoie une matrice (nb_textes, dimension) de vecteurs normalisés.

        Vecteurs normalisés (longueur 1) => le produit scalaire entre deux
        vecteurs est égal à leur similarité cosinus (entre -1 et 1).
        """
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
        )
        return vectors.astype("float32")  # FAISS attend du float32
