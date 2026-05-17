import httpx
from openai import OpenAI

from app.config import get_settings
from app.models import SourceChunk


def build_prompt(question: str, chunks: list[SourceChunk]) -> str:
    context = "\n\n".join(
        f"[Source: {chunk.source}, page {chunk.page}]\n{chunk.text}"
        for chunk in chunks
    )

    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you do not know based on the PDF.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


async def generate_answer(question: str, chunks: list[SourceChunk]) -> str:
    settings = get_settings()
    prompt = build_prompt(question, chunks)

    if settings.llm_provider == "openai":
        return await _generate_with_openai(prompt)

    return await _generate_with_ollama(prompt)


async def _generate_with_ollama(prompt: str) -> str:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()


async def _generate_with_openai(prompt: str) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using the supplied PDF context.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""
