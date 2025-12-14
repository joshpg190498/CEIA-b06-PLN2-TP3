# services/streamlit/main.py
from __future__ import annotations

import streamlit as st

from config import get_settings
from services.rag.vector_store import VectorStore, VectorStoreConfig
from services.agents.multi_agent import AgentRouter, RetrievedChunk


@st.cache_resource
def get_router() -> AgentRouter:
    settings = get_settings()
    vs_config = VectorStoreConfig(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        dimension=384,  # mismo que embeddings
    )
    store = VectorStore(vs_config)
    return AgentRouter(settings, store)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> None:
    settings = get_settings()
    st.sidebar.title("ℹ️ Info TP3")
    st.sidebar.markdown(
        """
**TP3 – Chatbot multi-agente sobre CVs**

- 1 agente por persona (cada CV del equipo).
- RAG sobre Pinecone con filtro por `person_id`.
- Si no se menciona a nadie, se usa el CV del alumno por defecto.
- Si se mencionan varios nombres, se combinan contextos y se responde por persona.
"""
    )
    st.sidebar.markdown("### Personas disponibles:")
    for p in settings.persons:
        default_mark = " *(por defecto)*" if p.is_default else ""
        st.sidebar.markdown(f"- {p.name}{default_mark}")


def main() -> None:
    st.set_page_config(
        page_title="Chatbot RAG multi-agente - CVs",
        page_icon="🤖",
    )

    router = get_router()
    init_session_state()
    render_sidebar()

    st.title("🤖 Chatbot RAG multi-agente")
    st.write(
        "Este asistente responde preguntas sobre los CVs de los integrantes del equipo. "
        "Si la pregunta no menciona ningún nombre, se usa tu CV por defecto."
    )

    # historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("chunks"):
                with st.expander("Ver fragmentos de CV usados como contexto"):
                    # agrupamos por persona para mostrar mejor
                    by_person = {}
                    for c in msg["chunks"]:
                        by_person.setdefault(c.person_name, []).append(c)
                    for person_name, plist in by_person.items():
                        st.markdown(f"### {person_name}")
                        for i, c in enumerate(plist, start=1):
                            st.markdown(f"**Fragmento {i}** (score: {c.score:.3f})")
                            st.write(c.text)

    user_input = st.chat_input("Escribe tu pregunta sobre uno o varios CVs...")
    if user_input:
        # usuario
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # agente(s)
        with st.chat_message("assistant"):
            with st.spinner("Consultando a los agentes..."):
                answer, chunks = router.answer(user_input)
                st.markdown(answer)

                with st.expander("Ver fragmentos de CV usados como contexto"):
                    by_person = {}
                    for c in chunks:
                        by_person.setdefault(c.person_name, []).append(c)
                    for person_name, plist in by_person.items():
                        st.markdown(f"### {person_name}")
                        for i, c in enumerate(plist, start=1):
                            st.markdown(f"**Fragmento {i}** (score: {c.score:.3f})")
                            st.write(c.text)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "chunks": chunks,
            }
        )


if __name__ == "__main__":
    main()
