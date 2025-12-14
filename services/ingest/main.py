# services/ingest/main.py
from __future__ import annotations

import pathlib
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

from config import get_settings
from services.rag.embeddings import embed_text
from services.rag.vector_store import VectorStore, VectorStoreConfig


nltk.download("punkt", quiet=True)


def split_paragraphs(text: str) -> List[str]:
    raw_paragraphs = text.split("\n\n")
    return [p.strip() for p in raw_paragraphs if p.strip()]


def chunk_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = sent_tokenize(paragraph, language="spanish")
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        candidate = (current + " " + sent).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str, max_chars: int = 400) -> List[str]:
    paragraphs = split_paragraphs(text)
    all_chunks: List[str] = []
    for p in paragraphs:
        all_chunks.extend(chunk_long_paragraph(p, max_chars))
    return [c.strip() for c in all_chunks if c.strip()]


def load_text(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {p.resolve()}")
    return p.read_text(encoding="utf-8")


def main() -> None:
    settings = get_settings()

    all_ids: List[str] = []
    all_vectors: List[List[float]] = []
    all_metadatas: List[dict] = []

    for person in settings.persons:
        print(f"Ingestando CV de {person.name} desde {person.cv_path}...")
        raw_text = load_text(person.cv_path)
        chunks = chunk_text(raw_text, max_chars=2000)
        print(f"  Chunks para {person.id}: {len(chunks)}")

        embeddings = embed_text(chunks, model_name=settings.embedding_model_name)

        ids = [f"{person.id}-chunk-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "person_id": person.id,
                "person_name": person.name,
                "text": chunk,
            }
            for chunk in chunks
        ]

        all_ids.extend(ids)
        all_vectors.extend(embeddings)
        all_metadatas.extend(metadatas)

    vs_config = VectorStoreConfig(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        dimension=len(all_vectors[0]),
    )
    store = VectorStore(vs_config)

    print("Subiendo todos los vectores al índice...")
    store.upsert(all_ids, all_vectors, all_metadatas)
    print("Ingesta multi-persona completa.")


if __name__ == "__main__":
    main()
