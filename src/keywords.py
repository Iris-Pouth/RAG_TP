"""Index par mots-clés : noms de médicaments et substances actives.

Les embeddings comprennent bien le sens d'une question (« effets sur la grossesse »),
mais distinguent mal deux noms propres comme DOLIPRANE et DAFALGAN.
Si la question cite un nom de médicament ou une substance, on limite donc
la recherche vectorielle aux passages de ces médicaments : c'est une recherche hybride.
"""

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .bdpm import Medicament

# Premiers mots de noms de médicaments qui sont aussi des mots courants
STOPWORDS = {
    "ACIDE", "ALCOOL", "AVEC", "COLLYRE", "CREME", "DOSE", "EAU", "ESTOMAC", "GEL",
    "HUILE", "LAIT", "PEAU", "POMMADE", "POUR", "QUEL", "QUELLE", "QUELS", "REIN",
    "SANS", "SIROP", "SOLUTION", "SUPPOSITOIRE", "VACCIN", "VITAMINE", "ZINC",
}


def normalize(text: str) -> str:
    """Majuscules, sans accents, sans ponctuation : « Paracétamol ? » -> « PARACETAMOL »."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[A-Z0-9]+", text.upper()))


def medicament_keywords(med: Medicament) -> set[str]:
    """Mots-clés d'un médicament : le premier mot de son nom + ses substances actives."""
    keywords = set()
    first_word = normalize(med.nom).split(" ")[0]
    if len(first_word) >= 4 and first_word.isalpha() and first_word not in STOPWORDS:
        keywords.add(first_word)
    for substance in med.substance_names:
        name = normalize(substance)
        if len(name) >= 4 and name not in STOPWORDS:
            keywords.add(name)
    return keywords


class KeywordIndex:
    def __init__(self, index: dict[str, list[int]] | None = None):
        self.index = index or {}  # mot-clé -> numéros des passages

    def add(self, keywords: set[str], chunk_ids: list[int]) -> None:
        for kw in keywords:
            self.index.setdefault(kw, []).extend(chunk_ids)

    def match(self, question: str) -> tuple[list[str], list[int]]:
        """Renvoie les mots-clés trouvés dans la question et les passages correspondants."""
        q = f" {normalize(question)} "
        found = [kw for kw in self.index if f" {kw} " in q]
        # « ACIDE ACETYLSALICYLIQUE » trouvé : inutile de garder aussi un mot-clé plus court inclus dedans
        found = [kw for kw in found if not any(kw != other and kw in other for other in found)]
        ids = sorted({i for kw in found for i in self.index[kw]})
        return found, ids

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.index, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KeywordIndex":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def build_keyword_index(meds: list[Medicament], chunk_ids: list[list[int]]) -> KeywordIndex:
    """`chunk_ids[i]` contient les numéros des passages du médicament `meds[i]`."""
    kw_index = KeywordIndex(defaultdict(list))
    for med, ids in zip(meds, chunk_ids):
        kw_index.add(medicament_keywords(med), ids)
    return KeywordIndex(dict(kw_index.index))
