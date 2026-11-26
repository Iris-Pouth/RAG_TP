import pytest

from src.bdpm import Medicament
from src.chunker import chunk_medicament, split_text


def test_texte_court_un_seul_morceau():
    assert split_text("Bonjour   le\n monde.", chunk_size=100, overlap=10) == ["Bonjour le monde."]


def test_texte_vide():
    assert split_text("   ", chunk_size=100, overlap=10) == []


def test_morceaux_respectent_la_taille():
    text = " ".join(f"mot{i}" for i in range(500))
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chevauchement_entre_morceaux():
    text = " ".join(f"mot{i}" for i in range(300))
    chunks = split_text(text, chunk_size=200, overlap=50)
    # le début du 2e morceau se trouve dans la fin du 1er
    assert chunks[1].split()[0] in chunks[0]


def test_aucun_mot_coupe():
    words = [f"mot{i}" for i in range(300)]
    for chunk in split_text(" ".join(words), chunk_size=120, overlap=30):
        assert all(w in words for w in chunk.split())


def test_overlap_trop_grand():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_fiche_courte_un_seul_passage():
    med = Medicament(cis="123", nom="TESTOL 500 mg, comprimé")
    chunks = chunk_medicament(med, chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].source == "TESTOL 500 mg, comprimé"
    assert chunks[0].url.endswith("/medicament/123/extrait")


def test_nom_repete_dans_chaque_passage():
    med = Medicament(cis="123", nom="TESTOL 500 mg, comprimé",
                     presentations=[f"boîte de {i} comprimés, prix {i},00 €" for i in range(60)])
    chunks = chunk_medicament(med, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.text.startswith("Médicament : TESTOL 500 mg, comprimé\n") for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
