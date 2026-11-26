"""Télécharge la BDPM et construit l'index FAISS.

Exemples :
    python indexer.py                 # médicaments commercialisés (~13 600)
    python indexer.py --all           # tous les médicaments, y compris arrêtés
    python indexer.py --download      # force le re-téléchargement des fichiers
"""

import argparse
import sys
import time

from src import bdpm, config
from src.pipeline import RAGPipeline

sys.stdout.reconfigure(encoding="utf-8")  # accents corrects dans la console Windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexation de la BDPM")
    parser.add_argument("--all", action="store_true",
                        help="inclure les médicaments non commercialisés")
    parser.add_argument("--download", action="store_true",
                        help="re-télécharger les fichiers même s'ils existent")
    args = parser.parse_args()

    start = time.time()

    print("[1/3] Données BDPM")
    bdpm.download(config.DATA_DIR, force=args.download)

    print("[2/3] Chargement du modèle d'embeddings")
    pipeline = RAGPipeline()  # pas besoin de clé API pour indexer

    print("[3/3] Découpage, embeddings et index FAISS (quelques minutes)")
    n = pipeline.build_index(only_commercialised=not args.all)

    print(f"\n{n} passages indexés dans {config.INDEX_DIR} en {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
