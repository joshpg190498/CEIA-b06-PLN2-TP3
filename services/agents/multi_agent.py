# services/agents/multi_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from groq import Groq

from config import Settings, get_settings, PersonConfig
from services.rag.embeddings import embed_text
from services.rag.vector_store import VectorStore, VectorStoreConfig


@dataclass
class RetrievedChunk:
    person_id: str
    person_name: str
    id: str
    score: float
    text: str


@dataclass
class RAGAgent:
    person: PersonConfig
    vector_store: VectorStore
    groq_api_key: str
    embedding_model_name: str
    top_k: int
    llm_model_name: str

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        query_vec = embed_text(question, model_name=self.embedding_model_name)
        matches = self.vector_store.query(
            query_vec,
            top_k=self.top_k,
            metadata_filter={"person_id": self.person.id},
        )
        chunks: List[RetrievedChunk] = []
        for _id, score, meta in matches:
            chunks.append(
                RetrievedChunk(
                    person_id=self.person.id,
                    person_name=self.person.name,
                    id=_id,
                    score=score,
                    text=meta.get("text", ""),
                )
            )
        return chunks


class AgentRouter:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.settings = settings
        self.store = store
        self.agents: Dict[str, RAGAgent] = {}

        for person in settings.persons:
            self.agents[person.id] = RAGAgent(
                person=person,
                vector_store=store,
                groq_api_key=settings.groq_api_key,
                embedding_model_name=settings.embedding_model_name,
                top_k=settings.top_k,
                llm_model_name="llama-3.1-8b-instant",
            )

    @property
    def default_agent(self) -> RAGAgent:
        for agent in self.agents.values():
            if agent.person.is_default:
                return agent
        # fallback: el primero
        return list(self.agents.values())[0]

    def detect_agents(self, question: str) -> List[RAGAgent]:
        q = question.lower()
        selected: List[RAGAgent] = []

        for person in self.settings.persons:
            for alias in person.aliases:
                if alias.lower() in q:
                    selected.append(self.agents[person.id])
                    break  # no repetir por alias

        if not selected:
            selected = [self.default_agent]

        return selected

    def answer(self, question: str) -> Tuple[str, List[RetrievedChunk]]:
        selected_agents = self.detect_agents(question)

        # single persona → respondemos como antes
        if len(selected_agents) == 1:
            agent = selected_agents[0]
            chunks = agent.retrieve(question)
            answer = self._generate_single_answer(question, agent, chunks)
            return answer, chunks

        # multi-persona → juntamos contexto de todos
        all_chunks: List[RetrievedChunk] = []
        for agent in selected_agents:
            all_chunks.extend(agent.retrieve(question))

        answer = self._generate_multi_answer(question, all_chunks)
        return answer, all_chunks

    def _generate_single_answer(
        self,
        question: str,
        agent: RAGAgent,
        chunks: List[RetrievedChunk],
    ) -> str:
        context_lines = [f"- {c.text}" for c in chunks if c.text]
        context_block = "\n".join(context_lines) or "(sin contexto recuperado)"

        system_prompt = (
            "Eres un asistente que responde preguntas sobre el CV de una persona. "
            "Responde solo con la información presente en el contexto. "
            "Si no hay datos suficientes, dilo explícitamente. Responde en español."
        )

        user_prompt = f"""Persona: {agent.person.name}

Contexto (fragmentos del CV):
{context_block}

Pregunta del usuario:
{question}

Instrucciones:
- Responde solo sobre {agent.person.name}.
- No inventes información fuera del contexto.
- Sé claro y conciso.
"""

        client = Groq(api_key=agent.groq_api_key)

        completion = client.chat.completions.create(
            model=agent.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content

    def _generate_multi_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
    ) -> str:
        by_person: Dict[str, List[RetrievedChunk]] = {}
        for c in chunks:
            by_person.setdefault(c.person_name, []).append(c)

        sections = []
        for person_name, plist in by_person.items():
            lines = [f"- {p.text}" for p in plist if p.text]
            section = f"[CV de {person_name}]\n" + "\n".join(lines)
            sections.append(section)

        context_block = "\n\n".join(sections) or "(sin contexto recuperado)"

        system_prompt = (
            "Eres un asistente que compara o responde sobre varias personas a la vez, "
            "usando únicamente los fragmentos de CV provistos. "
            "Responde en español y deja claro qué información corresponde a cada persona."
        )

        user_prompt = f"""Contexto (fragmentos por persona):
{context_block}

Pregunta del usuario:
{question}

Instrucciones:
- Estructura la respuesta en secciones, una por persona.
- En cada sección, aclara el nombre de la persona.
- No inventes datos que no aparezcan en el contexto.
"""

        settings = self.settings
        client = Groq(api_key=settings.groq_api_key)

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
