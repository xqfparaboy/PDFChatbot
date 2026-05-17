from pathlib import Path

from app.config import get_settings
from app.db.qdrant import search_chunks, upsert_chunks
from app.models import ChatResponse
from app.rag.chunking import chunk_page_text
from app.services.embedding_service import embed_text, embed_texts
from app.services.llm_service import generate_answer
from app.services.pdf_service import extract_text_by_page


async def index_pdf(pdf_path: Path, source: str) -> int:
    settings = get_settings()
    page_texts = extract_text_by_page(pdf_path)
    chunks = chunk_page_text(
        page_texts,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    if not chunks:
        return 0

    vectors = embed_texts([chunk.text for chunk in chunks])
    upsert_chunks(chunks, vectors, source=source)
    return len(chunks)


async def answer_question(question: str) -> ChatResponse:
    settings = get_settings()
    query_vector = embed_text(question)
    chunks = search_chunks(query_vector, top_k=settings.search_top_k)
    if not chunks:
        return ChatResponse(
            answer="I do not know based on the PDF. No indexed chunks were found.",
            sources=[],
        )

    answer = await generate_answer(question, chunks)
    return ChatResponse(answer=answer, sources=chunks)
