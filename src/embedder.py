"""Transformation de textes en vecteurs (embeddings)."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Modèle d'embeddings pour la recherche.

    Le modèle e5 est entraîné pour la recherche « question -> document » : il attend
    le préfixe « query: » devant une question et « passage: » devant un document.
    """

    def __init__(self, model_name: str, query_prefix: str = "query: ",
                 passage_prefix: str = "passage: "):
        # Le modèle est téléchargé une seule fois, puis mis en cache sur la machine
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Renvoie une matrice (nb_textes, dimension) de vecteurs normalisés.

        Vecteurs normalisés (longueur 1) => le produit scalaire entre deux
        vecteurs est égal à leur similarité cosinus.
        """
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
        )
        return vectors.astype("float32")  # FAISS attend du float32

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_query(self, question: str) -> np.ndarray:
        return self._encode([self.query_prefix + question])[0]
