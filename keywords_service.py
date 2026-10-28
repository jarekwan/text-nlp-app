from functools import lru_cache

import numpy as np
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sklearn.feature_extraction.text import CountVectorizer

from semantic_search import (
    get_document_chunks,
    get_embedding_model,
    list_indexed_documents,
)


POLISH_STOP_WORDS = [
    "aby", "albo", "bardzo", "bez", "bo", "by", "być", "było",
    "był", "była", "byli", "były", "co", "czy", "dla", "do",
    "gdy", "gdzie", "ich", "ile", "im", "jest", "jego", "jej",
    "już", "jak", "jaka", "jaki", "jakie", "jako", "jeśli",
    "kiedy", "kto", "która", "które", "który", "lub", "ma", "mam",
    "mi", "mnie", "może", "można", "na", "nad", "nam", "nas",
    "nasz", "nie", "nich", "nim", "niż", "od", "oraz", "po", "pod",
    "przez", "przy", "sam", "się", "są", "tak", "także", "te",
    "tego", "tej", "ten", "to", "tu", "tym", "we", "więc",
    "wszystko", "za", "że",
]


class FastEmbedBackend(BaseEmbedder):
    """Udostępnia model FastEmbed jako backend embeddingów dla KeyBERT."""

    def __init__(self):
        super().__init__()

    def embed(self, documents, verbose=False):
        texts = list(documents)
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        vectors = list(get_embedding_model().embed(texts))
        return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=1)
def get_keyword_model() -> KeyBERT:
    """Tworzy model KeyBERT raz na instancję aplikacji."""
    return KeyBERT(model=FastEmbedBackend())


def get_document_keywords(
    gcs_path: str,
) -> tuple[str, list[tuple[str, float]]]:
    """Zwraca nazwę dokumentu i ranking słów kluczowych."""
    documents = list_indexed_documents()
    document = next(
        (item for item in documents if item["gcs_path"] == gcs_path),
        None,
    )
    if document is None:
        raise ValueError("Nie znaleziono wybranego dokumentu w Qdrant.")

    chunks = get_document_chunks(gcs_path)
    if not chunks:
        raise ValueError("Wybrany dokument nie ma tekstowych fragmentów.")

    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        stop_words=POLISH_STOP_WORDS,
        token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b",
    )

    try:
        results_per_chunk = get_keyword_model().extract_keywords(
            chunks,
            vectorizer=vectorizer,
            top_n=10,
            use_mmr=True,
            diversity=0.35,
        )
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            return document["filename"], []
        raise

    # Ta sama fraza może pojawić się w kilku fragmentach.
    # Zachowujemy jej najwyższy wynik.
    best_scores: dict[str, float] = {}

    for chunk_results in results_per_chunk:
        for phrase, score in chunk_results:
            phrase = str(phrase).strip()
            best_scores[phrase] = max(
                best_scores.get(phrase, 0.0),
                float(score),
            )

    ranked_keywords = sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:15]

    return document["filename"], ranked_keywords
