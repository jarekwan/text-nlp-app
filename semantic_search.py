import os
import uuid
from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models


COLLECTION_NAME = "documents"
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_LIMIT = 5


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Ładuje model embeddingów raz na instancję aplikacji."""
    return TextEmbedding(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Tworzy klienta Qdrant na podstawie zmiennych środowiskowych."""
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("Brak zmiennej środowiskowej QDRANT_URL.")
    if not api_key:
        raise RuntimeError("Brak zmiennej środowiskowej QDRANT_API_KEY.")
    return QdrantClient(url=url, api_key=api_key)


def split_into_chunks(content: str) -> list[str]:
    """Dzieli tekst na fragmenty z nakładaniem 150 znaków."""
    text = content.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            split_at = max(
                text.rfind(" ", start, end),
                text.rfind("\n", start, end),
            )
            if split_at > start + CHUNK_SIZE // 2:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def index_document(chunks: list[str], filename: str, gcs_path: str) -> None:
    """Tworzy embeddingi fragmentów i zapisuje je w kolekcji Qdrant."""
    vectors = list(get_embedding_model().embed(chunks))
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "text": chunk,
                "filename": filename,
                "gcs_path": gcs_path,
                "chunk_index": index,
            },
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    get_qdrant_client().upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def find_similar_chunks(query: str) -> list[dict]:
    """Wyszukuje maksymalnie pięć fragmentów podobnych do zapytania."""
    query_vector = next(
        get_embedding_model().query_embed(query)
    ).tolist()

    response = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=SEARCH_LIMIT,
        with_payload=True,
    )

    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            {
                "snippet": str(payload.get("text", "")),
                "filename": str(payload.get("filename", "Nieznany plik")),
                "gcs_path": str(payload.get("gcs_path", "")),
                "score": (
                    float(point.score)
                    if point.score is not None
                    else 0.0
                ),
            }
        )

    return results


def list_indexed_documents() -> list[dict]:
    """Zwraca unikalne dokumenty zapisane w Qdrant."""
    client = get_qdrant_client()
    documents = {}
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            gcs_path = str(payload.get("gcs_path", ""))
            if not gcs_path:
                continue

            documents[gcs_path] = {
                "filename": str(
                    payload.get("filename", "Nieznany plik")
                ),
                "gcs_path": gcs_path,
            }

        if next_offset is None:
            break
        offset = next_offset

    return sorted(
        documents.values(),
        key=lambda item: (
            item["filename"].casefold(),
            item["gcs_path"],
        ),
    )


def get_document_chunks(gcs_path: str) -> list[str]:
    """Pobiera fragmenty wybranego dokumentu w ich kolejności."""
    client = get_qdrant_client()
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="gcs_path",
                match=models.MatchValue(value=gcs_path),
            )
        ]
    )

    points = []
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)

        if next_offset is None:
            break
        offset = next_offset

    points.sort(
        key=lambda point: int(
            (point.payload or {}).get("chunk_index", 0)
        )
    )

    return [
        str((point.payload or {}).get("text", ""))
        for point in points
        if (point.payload or {}).get("text")
    ]
