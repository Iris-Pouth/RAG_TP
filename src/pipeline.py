"""Le pipeline RAG complet : indexation, recherche hybride, score de confiance,
génération et historique de conversation."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .bdpm import load_medicaments
from .chunker import Chunk, chunk_medicament
from .embedder import Embedder
from .keywords import KeywordIndex, build_keyword_index
from .llm import ClaudeLLM
from .vector_store import VectorStore

NO_ANSWER = (
    "Je n'ai trouvé aucune information suffisamment pertinente dans la base des "
    "médicaments pour répondre à cette question. Pour un conseil personnalisé, "
    "adressez-vous à un pharmacien ou à un médecin."
)


def compute_confidence(scores: list[float]) -> float:
    """Score de confiance entre 0 et 1, calculé à partir des similarités des passages retrouvés.

    Moyenne pondérée : le 1er passage compte pour 1, le 2e pour 1/2, le 3e pour 1/3...
    Le meilleur passage pèse donc le plus, mais avoir plusieurs passages
    pertinents augmente aussi la confiance.
    """
    if not scores:
        return 0.0
    weights = [1 / rank for rank in range(1, len(scores) + 1)]
    weighted = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
    return max(0.0, min(1.0, weighted))


def confidence_label(confidence: float) -> str:
    if confidence >= 0.60:
        return "élevée"
    if confidence >= 0.45:
        return "moyenne"
    return "faible"


@dataclass
class Retrieval:
    """Résultat de l'étape de recherche, avant l'appel au LLM."""
    question: str  # question posée par l'utilisateur
    search_query: str  # question utilisée pour la recherche (éventuellement reformulée)
    results: list[tuple[Chunk, float]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)  # médicaments / substances reconnus
    confidence: float = 0.0

    @property
    def chunks(self) -> list[Chunk]:
        return [chunk for chunk, _ in self.results]

    @property
    def label(self) -> str:
        return confidence_label(self.confidence)

    @property
    def is_relevant(self) -> bool:
        return bool(self.results)


class RAGPipeline:
    def __init__(self, embedder: Embedder | None = None, llm: ClaudeLLM | None = None,
                 reformulation: bool = config.REFORMULATION):
        self.embedder = embedder or Embedder(config.EMBEDDING_MODEL)
        self.llm = llm or ClaudeLLM(
            config.CLAUDE_MODEL, config.CLAUDE_EFFORT, config.MAX_TOKENS, config.USE_FALLBACK
        )
        self.reformulation = reformulation
        self.store = VectorStore(self.embedder.dimension)
        self.keywords = KeywordIndex()
        self.history: list[dict] = []  # messages {"role": ..., "content": ...}

    # ---------- 1. Indexation ----------

    def build_index(self, data_dir: Path = config.DATA_DIR,
                    index_dir: Path = config.INDEX_DIR,
                    only_commercialised: bool = True) -> int:
        """Lit la BDPM, découpe les fiches, calcule les embeddings et sauvegarde l'index.
        Renvoie le nombre de passages indexés."""
        meds = load_medicaments(data_dir, only_commercialised)

        chunks: list[Chunk] = []
        chunk_ids: list[list[int]] = []  # numéros des passages de chaque médicament
        for med in meds:
            med_chunks = chunk_medicament(med, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            chunk_ids.append(list(range(len(chunks), len(chunks) + len(med_chunks))))
            chunks.extend(med_chunks)

        self.store = VectorStore(self.embedder.dimension)
        if chunks:
            self.store.add(self.embedder.encode([c.text for c in chunks]), chunks)
        self.keywords = build_keyword_index(meds, chunk_ids)

        self.store.save(index_dir)
        self.keywords.save(index_dir / "keywords.json")
        return len(chunks)

    def load_index(self, index_dir: Path = config.INDEX_DIR) -> bool:
        """Recharge un index sauvegardé. Renvoie False s'il n'existe pas encore."""
        if not VectorStore.exists(index_dir):
            return False
        self.store = VectorStore.load(index_dir)
        self.keywords = KeywordIndex.load(index_dir / "keywords.json")
        return True

    # ---------- 2. Recherche ----------

    def retrieve(self, question: str, top_k: int = config.TOP_K,
                 min_score: float = config.MIN_SCORE) -> Retrieval:
        """Trouve les passages les plus proches de la question et calcule la confiance."""
        # Question de suivi (« et son prix ? ») : on la rend autonome grâce à l'historique
        search_query = question
        if self.reformulation and self.history:
            search_query = self.llm.rewrite_question(question, self.history)

        # Recherche hybride : si un médicament est cité, on cherche parmi ses passages
        keywords, candidate_ids = self.keywords.match(search_query)
        query_vector = self.embedder.encode([search_query])[0]
        results = self.store.search(query_vector, top_k, among=candidate_ids or None)

        relevant = [(chunk, score) for chunk, score in results if score >= min_score]
        return Retrieval(
            question=question,
            search_query=search_query,
            results=relevant,
            keywords=keywords,
            confidence=compute_confidence([score for _, score in relevant]),
        )

    # ---------- 3. Génération ----------

    def stream_answer(self, retrieval: Retrieval) -> Iterator[str]:
        """Génère la réponse en streaming, puis l'ajoute à l'historique."""
        if not retrieval.is_relevant:
            # Pas de passage pertinent : on n'appelle pas le LLM (évite les hallucinations)
            answer = NO_ANSWER
            yield answer
        else:
            parts = []
            for text in self.llm.stream_answer(retrieval.question, retrieval.chunks,
                                               self.history):
                parts.append(text)
                yield text
            answer = "".join(parts)
        self._remember(retrieval.question, answer)

    def ask(self, question: str) -> tuple[str, Retrieval]:
        """Version sans streaming : renvoie la réponse complète et les infos de recherche."""
        retrieval = self.retrieve(question)
        answer = "".join(self.stream_answer(retrieval))
        return answer, retrieval

    # ---------- 4. Historique ----------

    def _remember(self, question: str, answer: str) -> None:
        # On garde la question seule (sans les passages) pour ne pas alourdir les requêtes
        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self.history = self.history[-2 * config.HISTORY_TURNS:]

    def reset_history(self) -> None:
        self.history = []
