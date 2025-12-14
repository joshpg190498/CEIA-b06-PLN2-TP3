# RAG CV Chatbot – TP3 (Multi-Agente)
**Chatbot con Retrieval-Augmented Generation (RAG) multi-agente usando Streamlit, Pinecone, Groq, NLTK y Embeddings**

Este proyecto extiende el TP2 incorporando un **sistema multi-agente**, donde cada agente está especializado en responder preguntas sobre el **CV de una persona específica**.  
Cada integrante del equipo cuenta con su propio agente RAG, y el sistema decide automáticamente cuál o cuáles utilizar según la consulta del usuario.

El chatbot permite:
- Consultar un CV específico.
- Consultar múltiples CVs en una sola pregunta.
- Utilizar un **agente por defecto** cuando no se menciona ninguna persona.

La interfaz está construida con **Streamlit**, mientras que la recuperación semántica se realiza con **Pinecone** y la generación de respuestas con **Groq (LLaMA 3.1)**.


### Funcionalidades principales
- Sistema **multi-agente (1 agente por persona)**.
- Indexación de múltiples CVs en un único índice vectorial con metadata por persona.
- Chunking del texto con soporte para español (NLTK).
- Recuperación semántica filtrada por persona (`person_id`).
- Respuestas generadas exclusivamente a partir del contexto recuperado.
- Soporte para:
  - Preguntas sobre una persona.
  - Preguntas sobre varias personas.
  - Preguntas sin nombre → agente por defecto.
- Interfaz Streamlit con visualización del contexto utilizado.



### Arquitectura del Proyecto

```
.
├── app.py
├── config.py
├── data/
│   ├── cv_jose_perez.txt
│   ├── cv_ana.txt
│   └── cv_pedro.txt
├── services/
│   ├── rag/
│   ├── agents/
│   │   └── multi_agent.py
│   ├── ingest/
│   │   └── main.py
│   └── streamlit/
│       └── main.py
└── .env
```

### Tecnologías utilizadas

| Componente | Tecnología |
|-----------|------------|
| **Vector DB** | Pinecone |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **LLM** | Groq – LLaMA 3.1 |
| **NLP** | NLTK (tokenización en español) |
| **Frontend** | Streamlit |
| **Entorno** | uv |



### Instalación y configuración

### 1. Instalar dependencias
```bash
uv sync
```

### 2. Crear archivo `.env`
```env
PINECONE_API_KEY=tu_api_key
PINECONE_INDEX_NAME=cv-multi-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
GROQ_API_KEY=tu_api_key
TOP_K=8
```

### Ingestar el CV (construir el índice)

```bash
uv run python -m services.ingest.main
```

Este proceso:
- Segmenta los CVs en chunks,
- Genera embeddings,
- Los envía a Pinecone.

Repetir si alguno de los CV cambia.

### Ejecutar el chatbot

```bash
uv run streamlit run app.py
```

Disponible en:

```
http://localhost:8501
```

### Cómo funciona el RAG en este proyecto

1. Detección de personas en la consulta.
2. Selección automática del agente correspondiente.
3. Recuperación semántica filtrada por CV.
4. Generación de respuestas estructuradas por persona.

### Posibles mejoras
- Ajuste dinámico de top_k según la consulta.
- Re-ranking de resultados con cross-encoders.
- Agregar un chunk de CV completo como fallback.
- Uso de embeddings para detectar la persona mencionada (en lugar de matching por texto).
- Memoria conversacional entre preguntas.

### Autor
**José Luis Perez Galindo**  