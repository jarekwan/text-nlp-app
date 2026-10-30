from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

import fill_mask_ui
import home_ui
import similarity_ui
import clustering_ui
from bert_service import MASK_TOKEN, get_fill_mask_pipeline
from gcs_repository import save_document
from keywords_service import get_document_keywords
from rag_service import generate_rag_answer
from semantic_search import (
    COLLECTION_NAME,
    find_similar_chunks,
    index_document,
    list_indexed_documents,
    split_into_chunks,
)

from similarity_service import compare_documents
from clustering_service import cluster_documents


app = FastAPI(title="Aplikacja do analizy i wyszukiwania tekstów")


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        documents = list_indexed_documents()
    except Exception:
        documents = []

    return home_ui.home_page(documents)


@app.post("/documents", response_class=HTMLResponse)
async def add_document(text: str = Form(""), file: UploadFile | None = File(None)):
    content = text.strip()
    filename = "Tekst wklejony przez użytkownika"

    if file is not None and file.filename:
        if not file.filename.lower().endswith(".txt"):
            return home_ui.error_page(
                "Nieprawidłowy plik",
                "Wybierz plik z rozszerzeniem .txt.",
                400,
            )
        try:
            content = (await file.read()).decode("utf-8-sig").strip()
            filename = file.filename
        except UnicodeDecodeError:
            return home_ui.error_page(
                "Nie udało się odczytać pliku",
                "Plik TXT musi być zapisany w kodowaniu UTF-8.",
                400,
            )

    if not content:
        return home_ui.error_page(
            "Brak tekstu",
            "Wklej tekst albo wybierz plik TXT.",
            400,
        )

    try:
        gcs_path = save_document(content)
    except Exception as error:
        return home_ui.error_page(
            "Nie udało się zapisać tekstu",
            str(error),
            500,
        )

    chunks = split_into_chunks(content)
    try:
        index_document(chunks, filename, gcs_path)
    except Exception as error:
        return home_ui.error_page(
            "Tekst zapisany, indeksowanie nie powiodło się",
            f"Oryginał zapisano w GCS: {gcs_path}. "
            f"Błąd indeksowania: {error}",
            500,
        )

    return home_ui.document_saved_page(
        filename,
        len(content),
        len(chunks),
        gcs_path,
        COLLECTION_NAME,
        content[:5000],
    )


@app.post("/search", response_class=HTMLResponse)
def search(query: str = Form(...)):
    query = query.strip()
    if not query:
        return home_ui.error_page(
            "Brak zapytania",
            "Wpisz pytanie lub temat.",
            400,
        )

    try:
        results = find_similar_chunks(query)
    except Exception as error:
        return home_ui.error_page(
            "Wyszukiwanie nie powiodło się",
            str(error),
            500,
        )

    return home_ui.search_results_page(query, results)


@app.post("/rag", response_class=HTMLResponse)
def rag(query: str = Form(...)):
    """Wyszukuje fragmenty, generuje odpowiedź i pokazuje źródła."""
    query = query.strip()
    if not query:
        return home_ui.error_page(
            "Brak pytania",
            "Wpisz pytanie do RAG.",
            400,
        )

    try:
        sources = find_similar_chunks(query)
        answer = generate_rag_answer(query, sources)
    except Exception as error:
        return home_ui.error_page(
            "Nie udało się wygenerować odpowiedzi RAG",
            str(error),
            500,
        )

    return home_ui.rag_results_page(query, answer, sources)


@app.post("/keywords", response_class=HTMLResponse)
def keywords(gcs_path: str = Form(...)):
    """Wyodrębnia słowa kluczowe z wybranego dokumentu."""
    gcs_path = gcs_path.strip()
    if not gcs_path:
        return home_ui.error_page(
            "Nie wybrano dokumentu",
            "Wróć na stronę główną i wybierz dokument.",
            400,
        )

    try:
        filename, ranked_keywords = get_document_keywords(gcs_path)
    except Exception as error:
        return home_ui.error_page(
            "Nie udało się wyodrębnić słów kluczowych",
            str(error),
            500,
        )

    return home_ui.keyword_results_page(
        filename,
        gcs_path,
        ranked_keywords,
    )


@app.get("/similarity", response_class=HTMLResponse)
def show_similarity_form():
    """Pokazuje formularz wyboru dokumentów do porównania."""
    try:
        documents = list_indexed_documents()
    except Exception as error:
        return home_ui.error_page(
            "Nie udało się pobrać listy dokumentów",
            str(error),
            500,
        )

    return similarity_ui.similarity_form_page(documents)


@app.post("/similarity", response_class=HTMLResponse)
def compare_selected_documents(gcs_paths: list[str] = Form(...)):
    """Porównuje wybrane dokumenty i pokazuje macierz wyników."""
    try:
        result = compare_documents(gcs_paths)
    except Exception as error:
        try:
            documents = list_indexed_documents()
        except Exception:
            documents = []

        return similarity_ui.similarity_form_page(
            documents,
            selected_paths=gcs_paths,
            error=str(error),
        )

    return similarity_ui.similarity_results_page(
        result["document_names"],
        result["similarity_matrix"],
    )


@app.get("/clustering", response_class=HTMLResponse)
def show_clustering_form():
    """Pokazuje formularz wyboru liczby grup."""
    try:
        documents = list_indexed_documents()
    except Exception as error:
        return home_ui.error_page(
            "Nie udało się pobrać listy dokumentów",
            str(error),
            500,
        )

    return clustering_ui.clustering_form_page(len(documents))


@app.post("/clustering", response_class=HTMLResponse)
def run_clustering(n_clusters: int = Form(...)):
    """Uruchamia KMeans i PCA dla zaindeksowanych dokumentów."""
    try:
        result = cluster_documents(n_clusters)
    except Exception as error:
        try:
            documents = list_indexed_documents()
        except Exception:
            documents = []

        return clustering_ui.clustering_form_page(
            len(documents),
            error=str(error),
        )

    return clustering_ui.clustering_results_page(
        result["points"],
        result["cluster_count"],
    )


@app.get("/fill-mask", response_class=HTMLResponse)
def show_fill_mask_form():
    return fill_mask_ui.fill_mask_form_page(MASK_TOKEN)


@app.post("/fill-mask", response_class=HTMLResponse)
def fill_mask(sentence: str = Form(...)):
    sentence = sentence.strip()

    if sentence.count(MASK_TOKEN) != 1:
        return fill_mask_ui.fill_mask_form_page(
            MASK_TOKEN,
            sentence=sentence,
            error=f"Zdanie musi zawierać dokładnie jeden token {MASK_TOKEN}.",
            status_code=400,
        )

    try:
        predictions = get_fill_mask_pipeline()(sentence, top_k=5)
    except Exception as error:
        return fill_mask_ui.error_page(
            "BERT nie zwrócił wyniku",
            str(error),
            500,
        )

    return fill_mask_ui.fill_mask_results_page(sentence, predictions)
