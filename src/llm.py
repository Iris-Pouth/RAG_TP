"""Appels à Claude (API Anthropic) : reformulation de la question et génération de la réponse."""

from collections.abc import Iterator

import anthropic

from .chunker import Chunk

SYSTEM_PROMPT = """Tu es un assistant d'information sur les médicaments commercialisés en France.
Tu réponds à partir d'extraits de la Base de Données Publique des Médicaments (BDPM).

Règles :
- Appuie-toi uniquement sur les extraits entre balises <documents>.
- Cite tes sources avec leur numéro entre crochets, par exemple [1] ou [2][3].
- Si les extraits ne permettent pas de répondre, dis-le clairement au lieu d'inventer.
  La BDPM contient la composition, la forme, le statut, les prix, le remboursement,
  les conditions de prescription et les avis de la HAS, mais pas les notices
  (posologie, effets indésirables, contre-indications) : pour ces questions,
  oriente vers la notice, un pharmacien ou un médecin.
- Tu donnes une information, pas un avis médical : ne recommande jamais de commencer,
  arrêter ou modifier un traitement.
- Réponds en français, de façon claire et concise."""

REWRITE_PROMPT = """Réécris la dernière question de l'utilisateur pour qu'elle soit compréhensible
sans l'historique : remplace les pronoms (« il », « son prix », « ce médicament »…)
par le nom du médicament ou de la substance dont il est question.
Si la question est déjà autonome, recopie-la telle quelle.
Réponds uniquement avec la question réécrite, sans guillemets ni explication."""


def build_context(chunks: list[Chunk]) -> str:
    """Met les passages retrouvés en forme, numérotés pour que Claude puisse les citer."""
    parts = [
        f'<document index="{i}" source="{chunk.source}">\n{chunk.text}\n</document>'
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "<documents>\n" + "\n".join(parts) + "\n</documents>"


class ClaudeLLM:
    def __init__(self, model: str, effort: str = "low", max_tokens: int = 4000,
                 use_fallback: bool = True):
        # La clé est lue automatiquement dans la variable d'environnement ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        # Si le modèle refuse une requête, l'API la relance sur un modèle de secours
        self.options = (
            {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
            if use_fallback else {}
        )

    def rewrite_question(self, question: str, history: list[dict]) -> str:
        """Transforme une question de suivi en question autonome, pour la recherche."""
        conversation = "\n".join(f"{m['role']} : {m['content']}" for m in history)
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=1000,
            system=REWRITE_PROMPT,
            messages=[{"role": "user", "content":
                       f"Historique :\n{conversation}\n\nDernière question : {question}"}],
            output_config={"effort": "low"},
            **self.options,
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return text or question

    def stream_answer(
        self, question: str, chunks: list[Chunk], history: list[dict]
    ) -> Iterator[str]:
        """Envoie la question + les passages + l'historique à Claude, et renvoie
        la réponse morceau par morceau (streaming), pour l'afficher au fur et à mesure."""
        user_message = f"{build_context(chunks)}\n\nQuestion : {question}"
        messages = history + [{"role": "user", "content": user_message}]

        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": self.effort},  # profondeur de réflexion du modèle
            **self.options,
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()

        if final.stop_reason == "refusal":
            yield "\n\n_(Le modèle a refusé de répondre à cette question.)_"
        elif final.stop_reason == "max_tokens":
            yield "\n\n_(Réponse tronquée : limite de longueur atteinte.)_"
