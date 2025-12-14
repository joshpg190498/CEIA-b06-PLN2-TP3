# services/rag/vector_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pinecone import Pinecone, ServerlessSpec


@dataclass
class VectorStoreConfig:
    api_key: str
    index_name: str
    cloud: str = "aws"
    region: str = "us-east-1"
    dimension: int = 384
    metric: str = "cosine"


class VectorStore:
    def __init__(self, config: VectorStoreConfig) -> None:
        self._config = config
        self._client = Pinecone(api_key=config.api_key)
        self._ensure_index()
        self._index = self._client.Index(config.index_name)

    def _ensure_index(self) -> None:
        existing = [idx["name"] for idx in self._client.list_indexes()]
        if self._config.index_name not in existing:
            self._client.create_index(
                name=self._config.index_name,
                dimension=self._config.dimension,
                metric=self._config.metric,
                spec=ServerlessSpec(
                    cloud=self._config.cloud,
                    region=self._config.region,
                ),
            )

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict],
    ) -> None:
        payload = []
        for _id, vec, meta in zip(ids, vectors, metadatas):
            payload.append(
                {"id": _id, "values": vec, "metadata": meta}
            )
        self._index.upsert(vectors=payload)

    def query(
        self,
        vector: List[float],
        top_k: int = 4,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Tuple[str, float, Dict]]:
        kwargs = dict(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        if metadata_filter:
            kwargs["filter"] = metadata_filter

        res = self._index.query(**kwargs)

        matches: List[Tuple[str, float, Dict]] = []
        for m in res["matches"]:
            matches.append(
                (m["id"], m["score"], m.get("metadata", {})),
            )
        return matches
