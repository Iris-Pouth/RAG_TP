"""Découpage des fiches médicaments en passages (chunks) qui se chevauchent."""

import re
from dataclasses import dataclass

from .bdpm import FICHE_URL, Medicament


@dataclass
class Chunk:
    text: str
    source: str  # nom du médicament
    cis: str = ""  # code CIS, pour retrouver la fiche officielle

    @property
    def url(self) -> str:
        return FICHE_URL.format(cis=self.cis) if self.cis else ""


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Découpe un texte en morceaux d'environ `chunk_size` caractères.

    - On coupe de préférence à la fin d'une phrase, sinon entre deux mots.
    - Deux morceaux consécutifs partagent environ `overlap` caractères, pour
      qu'une idée coupée en deux reste lisible dans au moins un des morceaux.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap doit être plus petit que chunk_size")

    text = re.sub(r"\s+", " ", text).strip()  # normalise les espaces et retours à la ligne
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Fin de phrase dans la seconde moitié du morceau ?
            sentence_end = text.rfind(". ", start + chunk_size // 2, end)
            if sentence_end != -1:
                end = sentence_end + 1
            else:
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space

        chunks.append(text[start:end].strip())
        if end >= len(text):
            break

        # Le morceau suivant recommence `overlap` caractères plus tôt,
        # sans jamais reculer (sinon boucle infinie)
        start = max(end - overlap, start + 1)
        # On recale le début sur un début de mot
        space = text.find(" ", start, end)
        if space != -1:
            start = space + 1

    return [c for c in chunks if c]


def chunk_medicament(med: Medicament, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Découpe la fiche d'un médicament.

    Le nom du médicament est répété en tête de chaque morceau : sinon, un morceau
    du milieu de la fiche (ex. : la liste des prix) ne dirait pas de quel médicament il parle.
    """
    fiche = med.to_text()
    if len(fiche) <= chunk_size:
        return [Chunk(text=fiche, source=med.nom, cis=med.cis)]
    header = f"Médicament : {med.nom}"
    body = fiche.removeprefix(header)
    return [
        Chunk(text=f"{header}\n{piece}", source=med.nom, cis=med.cis)
        # place restante après l'en-tête (au moins 2 x overlap, si le nom est très long)
        for piece in split_text(body, max(chunk_size - len(header) - 1, 2 * overlap), overlap)
    ]


def chunk_medicaments(meds: list[Medicament], chunk_size: int = 1000,
                      overlap: int = 150) -> list[Chunk]:
    return [chunk for med in meds for chunk in chunk_medicament(med, chunk_size, overlap)]
