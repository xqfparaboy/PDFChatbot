from functools import lru_cache
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import get_settings
from app.models import SourceChunk
from app.rag.chunking import TextChunk


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def ensure_collection(client: QdrantClient | None = None) -> None:
    settings = get_settings()
    client = client or get_qdrant_client()

    if client.collection_exists(settings.qdrant_collection):
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=settings.embedding_dimension,
            distance=Distance.COSINE,
        ),
    )


def upsert_chunks(
    chunks: list[TextChunk],
    vectors: list[list[float]],
    source: str,
    client: QdrantClient | None = None,
) -> None:
    settings = get_settings()
    client = client or get_qdrant_client()
    ensure_collection(client)
    delete_chunks_for_source(source, client)

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "text": chunk.text,
                "page": chunk.page,
                "source": source,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)


def delete_chunks_for_source(source: str, client: QdrantClient | None = None) -> None:
    settings = get_settings()
    client = client or get_qdrant_client()
    ensure_collection(client)

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            ]
        ),
    )


def search_chunks(
    query_vector: list[float],
    top_k: int,
    client: QdrantClient | None = None,
) -> list[SourceChunk]:
    settings = get_settings()
    client = client or get_qdrant_client()
    ensure_collection(client)

    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        SourceChunk(
            source=str(point.payload.get("source", "")),
            page=int(point.payload.get("page", 0)),
            score=float(point.score),
            text=str(point.payload.get("text", "")),
        )
        for point in results
    ]
