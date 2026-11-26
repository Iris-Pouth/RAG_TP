"""Tests sans réseau : le modèle d'embeddings et Claude sont remplacés par des faux objets."""

import numpy as np

from src.chunker import Chunk
from src.keywords import KeywordIndex
from src.pipeline import NO_ANSWER, RAGPipeline, compute_confidence, confidence_label
from src.vector_store import VectorStore


def unit(v):
    v = np.array(v, dtype="float32")
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Associe chaque texte connu à un vecteur fixe en 3 dimensions."""
    dimension = 3
    vectors = {
        "fiche alpha": unit([1, 0, 0]),
        "fiche beta": unit([0.8, 0.6, 0]),
        "fiche gamma": unit([0, 0, 1]),
        "prix du médicament": unit([1, 0.1, 0]),
        "prix du BETAMED": unit([1, 0.1, 0]),
        "question hors sujet": unit([0, -1, 0]),
    }

    def encode(self, texts):
        return np.stack([self.vectors[t] for t in texts])


class FakeLLM:
    def __init__(self):
        self.calls = []

    def rewrite_question(self, question, history):
        return "prix du BETAMED"

    def stream_answer(self, question, chunks, history):
        self.calls.append((question, [c.text for c in chunks], list(history)))
        yield "Réponse "
        yield "[1]"


def make_pipeline(reformulation=False):
    pipeline = RAGPipeline(embedder=FakeEmbedder(), llm=FakeLLM(), reformulation=reformulation)
    texts = ["fiche alpha", "fiche beta", "fiche gamma"]
    pipeline.store.add(pipeline.embedder.encode(texts),
                       [Chunk(t, t.upper(), cis=str(i)) for i, t in enumerate(texts)])
    pipeline.keywords = KeywordIndex({"BETAMED": [1]})
    return pipeline


def test_confidence():
    assert compute_confidence([]) == 0.0
    assert abs(compute_confidence([0.8]) - 0.8) < 1e-9
    # le 1er passage pèse plus que le 2e
    assert compute_confidence([0.9, 0.3]) > compute_confidence([0.3, 0.9])
    assert confidence_label(0.7) == "élevée"
    assert confidence_label(0.5) == "moyenne"
    assert confidence_label(0.2) == "faible"


def test_recherche_vectorielle_ordonnee():
    retrieval = make_pipeline().retrieve("prix du médicament", top_k=3, min_score=0.3)
    assert [c.text for c in retrieval.chunks] == ["fiche alpha", "fiche beta"]  # gamma trop loin
    assert retrieval.keywords == []
    assert retrieval.label == "élevée"


def test_recherche_hybride_sur_le_medicament_cite():
    retrieval = make_pipeline().retrieve("prix du BETAMED", top_k=3, min_score=0.3)
    # alpha est plus proche en vecteur, mais seul BETAMED est cité dans la question
    assert retrieval.keywords == ["BETAMED"]
    assert [c.text for c in retrieval.chunks] == ["fiche beta"]


def test_pas_d_appel_llm_si_rien_de_pertinent():
    pipeline = make_pipeline()
    answer, retrieval = pipeline.ask("question hors sujet")
    assert answer == NO_ANSWER
    assert retrieval.confidence == 0.0
    assert pipeline.llm.calls == []


def test_historique_et_reformulation():
    pipeline = make_pipeline(reformulation=True)
    _, first = pipeline.ask("prix du médicament")
    assert first.search_query == "prix du médicament"  # pas d'historique : pas de reformulation

    _, second = pipeline.ask("prix du médicament")
    assert second.search_query == "prix du BETAMED"  # question de suivi reformulée
    assert second.question == "prix du médicament"
    # au 2e appel, Claude reçoit le 1er échange
    assert pipeline.llm.calls[1][2] == [
        {"role": "user", "content": "prix du médicament"},
        {"role": "assistant", "content": "Réponse [1]"},
    ]
    pipeline.reset_history()
    assert pipeline.history == []


def test_sauvegarde_et_rechargement(tmp_path):
    pipeline = make_pipeline()
    pipeline.store.save(tmp_path)
    store = VectorStore.load(tmp_path)
    assert len(store) == 3
    chunk, score = store.search(unit([1, 0, 0]), k=1)[0]
    assert chunk.text == "fiche alpha" and chunk.cis == "0" and abs(score - 1.0) < 1e-5
