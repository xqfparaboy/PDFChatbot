from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int


def chunk_page_text(
    page_texts: list[tuple[int, str]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    step = chunk_size - overlap

    for page, text in page_texts:
        normalized = " ".join(text.split())
        if not normalized:
            continue

        for start in range(0, len(normalized), step):
            chunk = normalized[start : start + chunk_size].strip()
            if chunk:
                chunks.append(TextChunk(text=chunk, page=page))
            if start + chunk_size >= len(normalized):
                break

    return chunks
