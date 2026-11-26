"""Interface Streamlit.  Lancement : streamlit run app.py"""

import streamlit as st

from src import config
from src.pipeline import RAGPipeline

st.set_page_config(page_title="MediAssist", page_icon="⚕️", layout="wide")


@st.cache_resource(show_spinner="Chargement du modèle et de l'index…")
def get_pipeline() -> RAGPipeline | None:
    # Créé une seule fois, puis réutilisé entre les interactions (le modèle est long à charger)
    pipeline = RAGPipeline()
    return pipeline if pipeline.load_index() else None


pipeline = get_pipeline()
if pipeline is None:
    st.error("Aucun index trouvé. Lance d'abord : `python indexer.py`")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []  # ce qui est affiché à l'écran

# ---------- Barre latérale ----------
with st.sidebar:
    st.title("⚕️ MediAssist")
    st.caption("Base de Données Publique des Médicaments + Claude")
    st.metric("Passages indexés", f"{len(pipeline.store):,}".replace(",", " "))

    st.header("Réglages")
    top_k = st.slider("Passages retrouvés (top k)", 1, 10, config.TOP_K)
    min_score = st.slider("Similarité minimale", 0.0, 1.0, config.MIN_SCORE, 0.05)
    pipeline.reformulation = st.toggle("Reformuler les questions de suivi",
                                       value=config.REFORMULATION)

    if st.button("Nouvelle conversation", use_container_width=True):
        pipeline.reset_history()
        st.session_state.messages = []
        st.rerun()

    st.warning("Information uniquement : ne remplace pas l'avis d'un médecin "
               "ou d'un pharmacien.", icon="⚠️")

# ---------- Conversation ----------
st.title("Assistant médicaments")
st.caption("Composition, prix, remboursement, conditions de prescription, avis de la HAS…")

COLORS = {"élevée": "green", "moyenne": "orange", "faible": "red"}


def show_details(details: dict) -> None:
    if details["search_query"] != details["question"]:
        st.caption(f"↪ Question reformulée : {details['search_query']}")
    if details["keywords"]:
        st.caption(f"Recherche ciblée sur : {', '.join(details['keywords'])}")
    label = details["label"]
    st.markdown(f"Confiance : :{COLORS[label]}[**{details['confidence']:.0%} ({label})**]")
    if details["sources"]:
        with st.expander("Sources"):
            for i, (name, score, url, text) in enumerate(details["sources"], start=1):
                st.markdown(f"**[{i}] [{name}]({url})** · similarité {score:.2f}")
                st.caption(text)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
            show_details(msg["details"])

if question := st.chat_input("Ex : le Doliprane 1000 mg est-il remboursé ?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la base…"):
            retrieval = pipeline.retrieve(question, top_k=top_k, min_score=min_score)
        answer = st.write_stream(pipeline.stream_answer(retrieval))
        details = {
            "question": retrieval.question,
            "search_query": retrieval.search_query,
            "keywords": retrieval.keywords,
            "confidence": retrieval.confidence,
            "label": retrieval.label,
            "sources": [(c.source, s, c.url, c.text) for c, s in retrieval.results],
        }
        show_details(details)

    st.session_state.messages.append({"role": "assistant", "content": answer,
                                      "details": details})
