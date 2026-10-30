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

## Document similarity

Users can select two or more indexed documents and compare their semantic similarity. The app retrieves each document's passages from Qdrant, creates an embedding for each passage with the same FastEmbed model, and averages the passage embeddings into one normalized document vector. It then calculates cosine similarity for every pair and displays the results as a matrix.

The matrix is symmetric. A value of `1.0` on the diagonal represents a document compared with itself. Other values are similarity scores, not probabilities.

## Document clustering and visualization

Users can choose how many groups to create from the indexed documents. The app builds one normalized vector per document by averaging its passage embeddings, then applies scikit-learn KMeans to assign documents to groups. PCA reduces the document vectors to two dimensions for a 2D plot. The results page shows the plot and each document's assigned group.

The PCA plot is an exploratory visualization. It helps inspect how documents relate to one another; it does not prove that the clusters are objectively correct. At least three indexed documents are required.

## Relevant modules

- `app.py` — FastAPI routes for document upload, search, RAG, keywords, BERT, similarity, and clustering.
- `home_ui.py` — main page and links to the app's features.
- `semantic_search.py` — text chunking, FastEmbed embeddings, Qdrant indexing and search.
- `similarity_service.py` and `similarity_ui.py` — document similarity calculation and result pages.
- `clustering_service.py` and `clustering_ui.py` — KMeans clustering, PCA coordinates, and result pages.
- `keywords_service.py` — KeyBERT keyword extraction.
- `rag_service.py` — answer generation using retrieved passages.
- `bert_service.py` and `fill_mask_ui.py` — BERT fill-mask demonstration.

## Using the analysis features

1. Add and index at least two `.txt` documents on the main page.
2. Open **Compare documents** to view their pairwise similarity matrix.
3. Add at least three indexed documents, then open **Cluster documents** to choose the number of groups and view the KMeans/PCA results.
4. Use semantic search, RAG, KeyBERT, and BERT fill-mask from their sections in the application.
