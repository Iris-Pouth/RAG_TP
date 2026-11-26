"""Base vectorielle FAISS : stocke les vecteurs et retrouve les passages les plus proches."""

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, dimension: int):
        # IndexFlatIP = recherche exacte par produit scalaire (Inner Product).
        # Avec des vecteurs normalisés, c'est une recherche par similarité cosinus.
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []  # chunks[i] correspond au vecteur i de l'index

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("Il faut autant de vecteurs que de passages")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int,
               among: list[int] | None = None) -> list[tuple[Chunk, float]]:
        """Renvoie les k passages les plus proches, avec leur score (du plus au moins proche).

        `among` : si fourni, on ne cherche que parmi ces numéros de passages
        (utilisé par la recherche hybride quand la question cite un médicament).
        """
        if len(self) == 0:
            return []
        if among:
            ids = np.array(among, dtype="int64")
            vectors = self.index.reconstruct_batch(ids)  # vecteurs de ces passages
            scores = vectors @ query_vector  # similarité cosinus avec la question
            best = np.argsort(-scores)[:k]
            return [(self.chunks[ids[i]], float(scores[i])) for i in best]

        k = min(k, len(self))
        scores, ids = self.index.search(query_vector.reshape(1, -1), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        data = [asdict(c) for c in self.chunks]
        (folder / "chunks.json").write_text(json.dumps(data, ensure_ascii=False),
                                            encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        index = faiss.read_index(str(folder / "index.faiss"))
        data = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        store = cls(index.d)
        store.index = index
        store.chunks = [Chunk(**c) for c in data]
        return store

    @staticmethod
    def exists(folder: Path) -> bool:
        return (folder / "index.faiss").exists() and (folder / "chunks.json").exists()
