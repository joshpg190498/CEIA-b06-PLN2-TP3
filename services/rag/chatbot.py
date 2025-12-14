# services/rag/chatbot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from groq import Groq

from services.rag.embeddings import embed_text
from services.rag.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    id: str
    score: float
    text: str


@dataclass
class RAGChatbot:
    vector_store: VectorStore
    groq_api_key: str
    embedding_model_name: str
    top_k: int = 4
    llm_model_name: str = "llama-3.1-8b-instant"

    def _retrieve(self, question: str) -> List[RetrievedChunk]:
        query_vec = embed_text(question, model_name=self.embedding_model_name)
        matches = self.vector_store.query(query_vec, top_k=self.top_k)

        chunks: List[RetrievedChunk] = []
        for _id, score, meta in matches:
            text = meta.get("text", "")
            chunks.append(RetrievedChunk(id=_id, score=score, text=text))
        return chunks

    def _build_prompt(
        self,
        question: str,
        chunks: List[RetrievedChunk],
    ) -> Tuple[str, str]:
        context_lines = [f"- {c.text}" for c in chunks if c.text]
        context_block = "\n".join(context_lines) if context_lines else "(sin contexto recuperado)"

        system_prompt = (
            "Eres un asistente que responde preguntas sobre el CV de un candidato. "
            "Responde solo con la información disponible en el contexto. "
            "Si la respuesta no está en el contexto, dilo claramente. "
            "Utiliza un tono profesional y conciso en español."
        )

        user_prompt = f"""Contexto (fragmentos del CV):

{context_block}

Pregunta del usuario:
{question}

Instrucciones:
- Responde en español.
- Si faltan datos, dilo explícitamente.
- Si corresponde, menciona cargos, tecnologías y años relevantes.
"""

        return system_prompt, user_prompt

    def answer(
        self,
        question: str,
    ) -> Tuple[str, List[RetrievedChunk]]:
        chunks = self._retrieve(question)
        system_prompt, user_prompt = self._build_prompt(question, chunks)

        client = Groq(api_key=self.groq_api_key)

        completion = client.chat.completions.create(
            model=self.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = completion.choices[0].message.content
        return answer, chunks
