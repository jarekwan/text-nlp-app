import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from semantic_search import (
    get_document_chunks,
    get_embedding_model,
    list_indexed_documents,
)


def cluster_documents(n_clusters: int) -> dict:
    """Grupuje zaindeksowane dokumenty i wylicza ich położenie na wykresie PCA."""
    documents = list_indexed_documents()

    if len(documents) < 3:
        raise ValueError(
            "Do grupowania i wykresu PCA potrzebne są co najmniej "
            "trzy zaindeksowane dokumenty."
        )

    if n_clusters < 2 or n_clusters >= len(documents):
        raise ValueError(
            f"Liczba grup musi wynosić od 2 do {len(documents) - 1}."
        )

    document_vectors = []
    document_labels = []

    for document in documents:
        gcs_path = document["gcs_path"]
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

        document_vectors.append(document_vector / vector_length)

        file_id = gcs_path.rsplit("/", 1)[-1][-12:]
        document_labels.append(
            f"{document['filename']} — {file_id}"
        )

    vectors_matrix = np.asarray(document_vectors)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    cluster_numbers = kmeans.fit_predict(vectors_matrix)

    coordinates = PCA(n_components=2).fit_transform(vectors_matrix)

    points = []
    for label, cluster_number, coordinate in zip(
        document_labels,
        cluster_numbers,
        coordinates,
    ):
        points.append(
            {
                "label": label,
                "cluster": int(cluster_number) + 1,
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
            }
        )

    return {
        "points": points,
        "cluster_count": n_clusters,
    }
