# PDF Chatbot

FastAPI RAG chatbot for PDF question answering. It extracts PDF text, chunks it, embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`, stores vectors in Qdrant, retrieves relevant chunks for a question, and answers through Ollama by default.

## Architecture

```text
PDF -> Text Extraction -> Chunking -> Embeddings -> Qdrant
Question -> Embedding -> Similarity Search -> Top Chunks -> LLM -> Answer
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

By default, the app uses Qdrant local embedded storage in `qdrant_storage/`, so no Docker daemon is required for local development.

To use Qdrant as a server instead, start it:

```bash
docker compose up -d qdrant
```

Then set:

```text
QDRANT_URL=http://localhost:6333
```

Start Ollama and pull a model:

```bash
ollama pull llama3.1
ollama serve
```

Run the API:

```bash
uvicorn app.main:app --reload
```

## API

Upload and index a PDF:

```bash
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@Operating Systems Notes.pdf"
```

Ask a question:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is deadlock?"}'
```

## Configuration

Environment variables:

```text
QDRANT_URL=
QDRANT_PATH=qdrant_storage
QDRANT_COLLECTION=pdf_chunks
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

To use OpenAI instead:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
```
