# services/rag/embeddings.py
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Union

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache
def _get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_text(
    text: Union[str, Iterable[str]],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Union[List[float], List[List[float]]]:
    """
    Devuelve embeddings como listas de floats (para fácil serialización).
    - Si text es str, devuelve List[float]
    - Si es iterable de str, devuelve List[List[float]]
    """
    model = _get_model(model_name)
    embeddings = model.encode(text)

    # SentenceTransformers puede devolver np.array o lista; lo normalizamos
    if isinstance(text, str):
        return embeddings.tolist()
    return embeddings.tolist()
