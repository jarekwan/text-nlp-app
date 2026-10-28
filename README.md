# Text NLP App

A FastAPI application for exploring user-provided text documents with semantic search, retrieval-augmented generation (RAG), BERT fill-mask, and keyword extraction.

## Features

- **Document input:** paste text into the app or upload a UTF-8 `.txt` file.
- **Document storage and preparation:** the original text is saved in Google Cloud Storage. The app normalizes and splits text into smaller passages.
- **Semantic search:** FastEmbed creates vector embeddings for passages. Qdrant stores the vectors and source metadata, then returns relevant passages with their source document and similarity score.
- **RAG:** the app retrieves relevant passages from Qdrant and uses Gemini through Vertex AI to produce an answer grounded in those sources. Results include supporting passages; the app reports when the documents do not provide enough information.
- **BERT fill-mask:** users enter a sentence containing `[MASK]`. The model suggests possible completions based on the surrounding context and displays model scores.
- **KeyBERT:** users select an indexed document and receive ranked keywords and phrases. Scores indicate similarity, not probability.

## Technologies

- Python 3.13
- FastAPI and Uvicorn
- FastEmbed and KeyBERT
- Qdrant Cloud
- Google Cloud Storage
- Gemini through Vertex AI
- Transformers

## Cloud architecture

The FastAPI application runs on Google Cloud Run. Original documents are stored in Google Cloud Storage. Text embeddings and passage metadata are stored in a Qdrant Cloud collection. Gemini is accessed through Vertex AI for RAG answers.

## Configuration

The application reads these environment variable names:

- `GCS_BUCKET_NAME` — Google Cloud Storage bucket for original documents.
- `GOOGLE_CLOUD_PROJECT` — Google Cloud project used by Vertex AI.
- `QDRANT_URL` — Qdrant cluster URL.
- `QDRANT_API_KEY` — Qdrant cluster API key.

Do not commit real credentials or secret values to this repository. Google Cloud authentication must also be configured for access to Cloud Storage and Vertex AI.

## Run locally

Install dependencies with `python -m pip install -r requirements.txt`. Configure the required environment variables and Google Cloud authentication, then start the app with `uvicorn app:app --host 0.0.0.0 --port 8080`.

## Demo

[Open the deployed application](https://text-nlp-app-983807437277.europe-west4.run.app)

The demo may be temporarily unavailable during updates.
