# MediAssist — assistant médicaments RAG (BDPM + Claude)

Assistant qui répond à des questions sur les **médicaments commercialisés en France** à partir de la [Base de Données Publique des Médicaments](https://base-donnees-publique.medicaments.gouv.fr) (BDPM). Il cite ses sources, donne un **score de confiance** et refuse de répondre quand la base ne contient pas l'information.

> ⚠️ Projet pédagogique : les réponses sont informatives et ne remplacent pas l'avis d'un médecin ou d'un pharmacien.

<!-- Capture d'écran : ![Démo](docs/demo.png) -->

## Exemples de questions

- *Le Doliprane 1000 mg est-il remboursé ?* → oui, à 65 %, avec le prix de la boîte et un lien vers la fiche officielle
- *Quelle est la substance active du Spasfon ?*
- *Le tramadol est-il sur liste I ou liste II ?*
- puis une question de suivi : *Et son avis HAS ?* → la question est reformulée avec le nom du médicament avant la recherche

## Fonctionnement

```
                 ┌──────────────────── INDEXATION (indexer.py) ─────────────────────┐
 BDPM (6 fichiers) ─► fiche texte par ─► découpage en ─► embeddings ─► index FAISS
 téléchargés         médicament         passages        multilingues   + index mots-clés
                 └───────────────────────────────────────────────────────────────────┘

                 ┌──────────────────── QUESTION (rag.py / app.py) ──────────────────┐
 Question ─► reformulation ─► recherche hybride ─► score de ─► Claude ─► réponse
             (si suivi)       mots-clés + FAISS    confiance            + sources
                 └───────────────────────────────────────────────────────────────────┘
```

1. **Données** — `src/bdpm.py` télécharge 6 fichiers de la BDPM (spécialités, compositions, présentations et prix, conditions de prescription, groupes génériques, avis SMR de la HAS) et les assemble par code CIS en une fiche texte par médicament (~13 600 médicaments commercialisés).
2. **Découpage** — `src/chunker.py` coupe les fiches longues en passages de 1 000 caractères avec 150 caractères de chevauchement (la plupart des fiches tiennent en un seul passage) ; le nom du médicament est répété en tête de chaque passage.
3. **Embeddings** — `src/embedder.py` transforme chaque passage en vecteur normalisé de 384 dimensions avec `multilingual-e5-small`, un modèle entraîné pour la recherche question → document (préfixes `query:` et `passage:`).
4. **Index** — `src/vector_store.py` stocke les vecteurs dans FAISS (`IndexFlatIP` : produit scalaire = similarité cosinus) ; `src/keywords.py` construit un index des noms de médicaments et des substances actives.
5. **Recherche hybride** — si la question cite un médicament ou une substance (« Doliprane », « paracétamol »), la recherche vectorielle est limitée à ses passages ; sinon elle porte sur toute la base et seuls les passages très similaires (score ≥ 0,85) sont gardés.
6. **Score de confiance** — moyenne des similarités pondérée par le rang (1, 1/2, 1/3…), ramenée de l'intervalle [0,80 ; 0,90] à [0 ; 100 %] : élevée ≥ 70 %, moyenne ≥ 40 %, sinon faible. **Sans passage pertinent, Claude n'est pas appelé** et l'assistant répond qu'il n'a pas l'information.
7. **Génération** — `src/llm.py` envoie à Claude les passages numérotés, la question et l'historique ; la réponse s'affiche en streaming et cite ses sources `[1]`, `[2]`.
8. **Historique et reformulation** — les 3 derniers échanges sont conservés ; une question de suivi (« et son prix ? ») est d'abord réécrite en question autonome pour que la recherche trouve le bon médicament.

## Structure

| Fichier | Rôle |
|---|---|
| `indexer.py` | Télécharge la BDPM et construit l'index |
| `rag.py` | Assistant en ligne de commande |
| `app.py` | Interface Streamlit |
| `src/config.py` | Paramètres (lus depuis `.env`) |
| `src/bdpm.py` | Téléchargement et lecture des fichiers BDPM |
| `src/chunker.py` | Découpage des fiches en passages |
| `src/embedder.py` | Texte → vecteurs |
| `src/vector_store.py` | Index FAISS : ajout, recherche, sauvegarde |
| `src/keywords.py` | Index des noms de médicaments et substances (recherche hybride) |
| `src/llm.py` | Prompts et appels à Claude |
| `src/pipeline.py` | Assemble le tout : indexation, recherche, confiance, génération, historique |
| `tests/` | Tests unitaires (sans réseau ni clé API) |

## Installation

Python 3.10 ou plus récent.

```bash
git clone https://github.com/Iris-Pouth/RAG_TP.git
cd RAG_TP
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux : source .venv/bin/activate)
pip install -r requirements.txt
```

Copie `.env.example` en `.env` et renseigne ta clé API Anthropic :

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Utilisation

```bash
# 1. Télécharger la BDPM et construire l'index (une seule fois, quelques minutes)
python indexer.py

# 2a. Interface web
streamlit run app.py

# 2b. Ou en ligne de commande
python rag.py
python rag.py --ask "Le Doliprane 1000 mg est-il remboursé ?"
```

Options : `python indexer.py --all` indexe aussi les médicaments non commercialisés ; `python indexer.py --download` force le re-téléchargement ; `python rag.py --no-reformulation` désactive la reformulation.

## Tests

```bash
pytest
```

Les tests utilisent des fichiers BDPM miniatures et remplacent le modèle d'embeddings et Claude par de faux objets : ils tournent sans Internet ni clé API.

## Choix techniques

- **Pourquoi une recherche hybride ?** Les embeddings captent bien le sens (« est-il remboursé ? ») mais confondent les noms propres proches (DOLIPRANE / DAFALGAN). Le filtre par mots-clés garantit qu'on cherche dans les bons médicaments.
- **Pourquoi répéter le nom dans chaque passage ?** Un passage du milieu d'une fiche (la liste des prix, par exemple) ne dirait sinon pas de quel médicament il parle.
- **Pourquoi e5 plutôt que MiniLM ?** La première version utilisait `paraphrase-multilingual-MiniLM-L12-v2`, entraîné à comparer deux phrases entre elles. Testé sur la vraie base, il classait mal les fiches (« avis de la HAS sur le Doliprane » ne trouvait aucun passage Doliprane) et ne lisait que 128 tokens. `multilingual-e5-small`, de même taille mais entraîné pour la recherche, retrouve les bonnes fiches et lit 512 tokens.
- **Pourquoi remettre les scores à l'échelle ?** Les similarités d'e5 sont resserrées (~0,75 pour un texte sans rapport, ~0,90 pour un texte très proche) : un score brut de 0,83 ne veut pas dire « 83 % de confiance ».
- **Pourquoi `IndexFlatIP` ?** Recherche exacte, rapide pour quelques dizaines de milliers de vecteurs ; avec des vecteurs normalisés, le produit scalaire est la similarité cosinus.
- **Limites de la BDPM** — elle ne contient pas les notices (posologie, effets indésirables, contre-indications) : l'assistant le dit et oriente vers un professionnel de santé.
- **Questions par symptôme** — « un médicament contre la toux pour enfant » donne de mauvais résultats : la BDPM ne décrit pas les indications, la recherche ne peut donc pas relier un symptôme à un médicament. L'assistant marche bien quand la question cite un médicament ou une substance.
- **Claude** — modèle et niveau de réflexion (`CLAUDE_EFFORT`) réglables dans `.env`. En cas de refus du modèle, l'API relance la requête sur un modèle de secours (`USE_FALLBACK`).

## Pistes d'amélioration

- Ajouter les notices (RCP) pour couvrir posologie et effets indésirables
- Re-ranking des passages avec un cross-encoder
- Jeu d'évaluation (questions / réponses attendues) pour mesurer la qualité des réponses
- Mise à jour automatique de l'index quand la BDPM change

## Données

Données issues de la Base de Données Publique des Médicaments (ANSM, HAS, Assurance Maladie), téléchargées par `indexer.py` et non incluses dans ce dépôt.
