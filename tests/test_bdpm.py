"""Lecture de fichiers BDPM miniatures, écrits dans un dossier temporaire."""

from src.bdpm import FILES, load_medicaments
from src.keywords import KeywordIndex, medicament_keywords, normalize

ROWS = {
    "specialites": [
        "111\tDOLITEST 1000 mg, comprimé\tcomprimé\torale\tAutorisation active\t"
        "Procédure nationale\tCommercialisée\t09/07/2002\t\t\t LABO TEST\tNon",
        "222\tANCIEN 5 mg, gélule\tgélule\torale\tAutorisation abrogée\t"
        "Procédure nationale\tNon commercialisée\t01/01/1990\t\t\t LABO TEST\tNon",
    ],
    "compositions": ["111\tcomprimé\t0001\tPARACÉTAMOL\t1000 mg\tun comprimé\tSA\t1"],
    "presentations": ["111\t123\tplaquette de 8 comprimé(s)\tPrésentation active\t"
                      "Déclaration de commercialisation\t16/03/2011\t340\toui\t65%\t2,18\t3,20\t1,02\t"],
    "smr": [
        "111\tCT-1\tInscription\t20150101\tModéré\tAncien avis",
        "111\tCT-2\tRéévaluation\t20200101\tImportant\tLe service médical rendu est<br>important.",
    ],
    "generiques": ["7\tPARACETAMOL 1000 mg - DOLITEST\t111\t0\t1"],
    "conditions": ["111\tliste II"],
}


def write_fake_bdpm(folder, encoding="cp1252"):
    for key, rows in ROWS.items():
        (folder / FILES[key]).write_text("\r\n".join(rows), encoding=encoding)


def test_assemblage_des_fichiers(tmp_path):
    write_fake_bdpm(tmp_path)
    meds = load_medicaments(tmp_path)
    assert [m.cis for m in meds] == ["111"]  # le médicament non commercialisé est exclu
    med = meds[0]
    assert med.substances == ["PARACÉTAMOL 1000 mg (pour un comprimé)"]
    assert med.conditions == ["liste II"]
    assert "remboursé à 65%, prix 3,20 €" in med.presentations[0]
    assert "princeps" in med.generique
    # seul l'avis HAS le plus récent est gardé, sans balise HTML
    assert "Important : Le service médical rendu est important." in med.smr
    assert "Ancien avis" not in med.smr

    text = med.to_text()
    assert text.startswith("Médicament : DOLITEST 1000 mg, comprimé")
    assert "liste II" in text


def test_encodage_utf8_et_tous_les_medicaments(tmp_path):
    write_fake_bdpm(tmp_path, encoding="utf-8")
    meds = load_medicaments(tmp_path, only_commercialised=False)
    assert {m.cis for m in meds} == {"111", "222"}
    assert meds[0].substance_names == ["PARACÉTAMOL"]


def test_mots_cles(tmp_path):
    write_fake_bdpm(tmp_path)
    med = load_medicaments(tmp_path)[0]
    assert medicament_keywords(med) == {"DOLITEST", "PARACETAMOL"}

    index = KeywordIndex()
    index.add(medicament_keywords(med), [0, 1])
    assert index.match("Le dolitest est-il remboursé ?") == (["DOLITEST"], [0, 1])
    assert index.match("Que contient le paracétamol ?") == (["PARACETAMOL"], [0, 1])
    assert index.match("Quel est le prix ?") == ([], [])


def test_normalize():
    assert normalize("Paracétamol, 1000 mg ?") == "PARACETAMOL 1000 MG"
