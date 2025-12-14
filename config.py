# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

@dataclass(frozen=True)
class PersonConfig:
    id: str             
    name: str           
    cv_path: str        
    aliases: List[str]  
    is_default: bool = False  


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    groq_api_key: str = ""
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    persons: List[PersonConfig] = None


@lru_cache
def get_settings() -> Settings:
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY no está definido en .env")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY no está definido en .env")
    
    persons = [
        PersonConfig(
            id="jose",
            name="José Pérez",
            cv_path="data/cv_1.txt",
            aliases=["jose"],
            is_default=True,
        ),
        PersonConfig(
            id="maria",
            name="Maria Rojas",
            cv_path="data/cv_2.txt",
            aliases=["maria"],
        ),
        PersonConfig(
            id="luis",
            name="Luis Hidalgo",
            cv_path="data/cv_3.txt",
            aliases=["luis"],
        ),
        PersonConfig(
            id="ana",
            name="Ana Morales",
            cv_path="data/cv_4.txt",
            aliases=["ana"],
        ),
    ]

    return Settings(
        pinecone_api_key=pinecone_api_key,
        groq_api_key=groq_api_key,
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "cv-multi-index"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        top_k=int(os.getenv("TOP_K", "4")),
        persons=persons
    )
