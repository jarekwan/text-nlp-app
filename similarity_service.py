import numpy as np

from semantic_search import (
    get_document_chunks,
    get_embedding_model,
    list_indexed_documents,
)


def compare_documents(gcs_paths: list[str]) -> dict:
    """Oblicza podobieństwo znaczeniowe wybranych dokumentów."""
    selected_paths = list(dict.fromkeys(path.strip() for path in gcs_paths))

    if len(selected_paths) < 2:
        raise ValueError("Wybierz co najmniej dwa dokumenty.")

    documents_by_path = {
        document["gcs_path"]: document
        for document in list_indexed_documents()
    }

    document_vectors = []
    document_names = []

    for gcs_path in selected_paths:
        document = documents_by_path.get(gcs_path)

        if document is None:
            raise ValueError(
                "Wybrany dokument nie znajduje się w indeksie."
            )

        chunks = get_document_chunks(gcs_path)

        if not chunks:
            raise ValueError(
                f"Dokument „{document['filename']}” nie ma "
                "fragmentów w indeksie."
            )

        chunk_vectors = np.asarray(
            list(get_embedding_model().embed(chunks)),
            dtype=np.float32,
        )

        document_vector = chunk_vectors.mean(axis=0)
        vector_length = np.linalg.norm(document_vector)

        if vector_length == 0:
            raise ValueError(
                f"Nie można obliczyć wektora dokumentu "
                f"„{document['filename']}”."
            )

        normalized_vector = document_vector / vector_length
        document_vectors.append(normalized_vector)
        document_names.append(document["filename"])

    vectors_matrix = np.asarray(document_vectors)
    similarity_matrix = np.clip(
        vectors_matrix @ vectors_matrix.T,
        -1.0,
        1.0,
    )

    return {
        "document_names": document_names,
        "similarity_matrix": similarity_matrix.tolist(),
    }
