"""Téléchargement et lecture de la Base de Données Publique des Médicaments (BDPM).

Source : https://base-donnees-publique.medicaments.gouv.fr/telechargement
Chaque fichier est un tableau texte séparé par des tabulations, sans ligne d'en-tête.
Tous les fichiers sont reliés par le code CIS (identifiant unique d'un médicament).
"""

import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = "https://base-donnees-publique.medicaments.gouv.fr/download/file/"
FICHE_URL = "https://base-donnees-publique.medicaments.gouv.fr/medicament/{cis}/extrait"

FILES = {
    "specialites": "CIS_bdpm.txt",           # nom, forme, voie, titulaire, statut…
    "compositions": "CIS_COMPO_bdpm.txt",    # substances actives et dosages
    "presentations": "CIS_CIP_bdpm.txt",     # boîtes, prix, taux de remboursement
    "smr": "CIS_HAS_SMR_bdpm.txt",           # avis de la HAS (service médical rendu)
    "generiques": "CIS_GENER_bdpm.txt",      # groupes génériques
    "conditions": "CIS_CPD_bdpm.txt",        # conditions de prescription
}


@dataclass
class Medicament:
    cis: str
    nom: str
    forme: str = ""
    voies: str = ""
    statut_amm: str = ""
    commercialisation: str = ""
    date_amm: str = ""
    titulaire: str = ""
    surveillance_renforcee: bool = False
    substances: list[str] = field(default_factory=list)  # « PARACÉTAMOL 1000 mg (pour un comprimé) »
    substance_names: list[str] = field(default_factory=list)  # « PARACÉTAMOL »
    presentations: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    generique: str = ""
    smr: str = ""

    @property
    def url(self) -> str:
        return FICHE_URL.format(cis=self.cis)

    def to_text(self) -> str:
        """Fiche texte du médicament : c'est ce texte qui sera découpé et vectorisé."""
        lignes = [
            f"Médicament : {self.nom}",
            f"Forme : {self.forme}. Voie(s) d'administration : {self.voies}.",
        ]
        if self.substances:
            lignes.append("Substance(s) active(s) : " + " ; ".join(self.substances) + ".")
        lignes.append(f"Titulaire de l'autorisation : {self.titulaire}.")
        lignes.append(f"Statut : {self.statut_amm}, {self.commercialisation.lower()}, "
                      f"autorisé depuis le {self.date_amm}.")
        if self.surveillance_renforcee:
            lignes.append("Ce médicament fait l'objet d'une surveillance renforcée.")
        if self.conditions:
            lignes.append("Conditions de prescription : " + " ; ".join(self.conditions) + ".")
        if self.generique:
            lignes.append(self.generique)
        if self.presentations:
            lignes.append("Présentations : " + " ; ".join(self.presentations) + ".")
        if self.smr:
            lignes.append(self.smr)
        return "\n".join(lignes)


# ---------- Téléchargement ----------

def download(data_dir: Path, force: bool = False) -> None:
    """Télécharge les fichiers BDPM dans `data_dir` (sauf s'ils sont déjà présents)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        path = data_dir / name
        if path.exists() and not force:
            continue
        print(f"Téléchargement de {name}…")
        urllib.request.urlretrieve(BASE_URL + name, path)


def files_present(data_dir: Path) -> bool:
    return all((data_dir / name).exists() for name in FILES.values())


# ---------- Lecture ----------

def read_rows(path: Path) -> list[list[str]]:
    """Lit un fichier BDPM et renvoie ses lignes découpées en colonnes.

    Les fichiers n'ont pas tous le même encodage : certains sont en UTF-8,
    d'autres en Windows-1252 (accents « à l'ancienne »). On essaie UTF-8 d'abord.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [line.split("\t") for line in text.splitlines() if line.strip()]


def clean(text: str) -> str:
    """Retire les balises HTML (<br>…) et les espaces en trop."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def col(row: list[str], i: int) -> str:
    """Colonne i d'une ligne, ou chaîne vide si la ligne est trop courte."""
    return clean(row[i]) if i < len(row) else ""


def load_medicaments(data_dir: Path, only_commercialised: bool = True) -> list[Medicament]:
    """Assemble les 6 fichiers en une liste de Medicament."""
    meds: dict[str, Medicament] = {}

    # 1. Spécialités : une ligne = un médicament
    for r in read_rows(data_dir / FILES["specialites"]):
        med = Medicament(
            cis=col(r, 0), nom=col(r, 1), forme=col(r, 2), voies=col(r, 3),
            statut_amm=col(r, 4), commercialisation=col(r, 6), date_amm=col(r, 7),
            titulaire=col(r, 10), surveillance_renforcee=col(r, 11).lower() == "oui",
        )
        if only_commercialised and med.commercialisation != "Commercialisée":
            continue
        meds[med.cis] = med

    # 2. Compositions : on garde les substances actives (SA)
    for r in read_rows(data_dir / FILES["compositions"]):
        med = meds.get(col(r, 0))
        if med and col(r, 6) == "SA":
            substance = f"{col(r, 3)} {col(r, 4)}".strip()
            if col(r, 5):
                substance += f" (pour {col(r, 5)})"
            med.substances.append(substance)
            med.substance_names.append(col(r, 3))

    # 3. Présentations : boîte, taux de remboursement, prix
    for r in read_rows(data_dir / FILES["presentations"]):
        med = meds.get(col(r, 0))
        if not med:
            continue
        pres = col(r, 2)
        if col(r, 8):
            pres += f", remboursé à {col(r, 8)}"
        else:
            pres += ", non remboursé"
        if col(r, 10):
            pres += f", prix {col(r, 10)} €"
        med.presentations.append(pres)

    # 4. Conditions de prescription (liste I, liste II, réservé à l'hôpital…)
    for r in read_rows(data_dir / FILES["conditions"]):
        med = meds.get(col(r, 0))
        if med:
            med.conditions.append(col(r, 1))

    # 5. Groupes génériques (type 0 = princeps, c'est-à-dire le médicament d'origine)
    for r in read_rows(data_dir / FILES["generiques"]):
        med = meds.get(col(r, 2))
        if med:
            role = "le médicament de référence (princeps)" if col(r, 3) == "0" else "un générique"
            med.generique = f"Groupe générique : {col(r, 1)} ; ce médicament en est {role}."

    # 6. Avis SMR de la HAS : on garde les avis les plus récents
    avis = defaultdict(list)
    for r in read_rows(data_dir / FILES["smr"]):
        if col(r, 0) in meds:
            avis[col(r, 0)].append((col(r, 3), col(r, 4), col(r, 5)))  # date, niveau, texte
    for cis, liste in avis.items():
        derniere_date = max(date for date, _, _ in liste)
        textes = [f"{niveau} : {texte}" for date, niveau, texte in liste if date == derniere_date]
        meds[cis].smr = "Avis de la HAS (service médical rendu) — " + " ".join(textes)

    return list(meds.values())
