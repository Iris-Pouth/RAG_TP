"""Assistant médicaments en ligne de commande.

Exemples :
    python rag.py                                         # conversation avec historique
    python rag.py --ask "Le Doliprane 1000 est-il remboursé ?"
    python rag.py --no-reformulation                      # sans reformulation des questions
"""

import argparse
import sys

from src.pipeline import RAGPipeline, Retrieval

sys.stdout.reconfigure(encoding="utf-8")  # accents corrects dans la console Windows


def print_sources(retrieval: Retrieval) -> None:
    if retrieval.search_query != retrieval.question:
        print(f"\n↪ Question reformulée : {retrieval.search_query}")
    if retrieval.keywords:
        print(f"Recherche ciblée sur : {', '.join(retrieval.keywords)}")
    print(f"Confiance : {retrieval.confidence:.0%} ({retrieval.label})")
    for i, (chunk, score) in enumerate(retrieval.results, start=1):
        print(f"  [{i}] {chunk.source}  (similarité {score:.2f})  {chunk.url}")


def answer(pipeline: RAGPipeline, question: str) -> None:
    retrieval = pipeline.retrieve(question)
    for text in pipeline.stream_answer(retrieval):
        print(text, end="", flush=True)
    print()
    print_sources(retrieval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant médicaments (BDPM + Claude)")
    parser.add_argument("--ask", metavar="QUESTION", help="poser une seule question")
    parser.add_argument("--no-reformulation", action="store_true",
                        help="désactiver la reformulation des questions de suivi")
    args = parser.parse_args()

    pipeline = RAGPipeline(reformulation=not args.no_reformulation)
    if not pipeline.load_index():
        sys.exit("Aucun index trouvé. Lance d'abord : python indexer.py")

    if args.ask:
        answer(pipeline, args.ask)
        return

    print("Assistant médicaments — information uniquement, ne remplace pas un avis médical.")
    print("Commandes : « reset » efface l'historique, « quit » pour sortir.")
    while True:
        try:
            question = input("\nVous > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "reset":
            pipeline.reset_history()
            print("Historique effacé.")
        elif question:
            print("Assistant > ", end="")
            answer(pipeline, question)


if __name__ == "__main__":
    main()
