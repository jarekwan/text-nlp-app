import os
import uuid

from google.cloud import storage


def save_document(content: str) -> str:
    """Zapisuje oryginalny tekst w Google Cloud Storage."""
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError("Brak zmiennej środowiskowej GCS_BUCKET_NAME.")

    bucket = storage.Client().bucket(bucket_name)
    object_name = f"text-nlp-app/documents/{uuid.uuid4().hex}.txt"
    bucket.blob(object_name).upload_from_string(
        content,
        content_type="text/plain; charset=utf-8",
    )
    return f"gs://{bucket_name}/{object_name}"