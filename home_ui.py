import html

from fastapi.responses import HTMLResponse


MODEL_CARD_URL = "https://huggingface.co/dkleczek/bert-base-polish-cased-v1"


def _page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    safe_title = html.escape(title)
    return HTMLResponse(
        f"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
body {{font-family:Arial,sans-serif;max-width:900px;margin:36px auto;padding:0 20px;color:#222}}
textarea {{box-sizing:border-box;width:100%;min-height:150px}}
input,textarea,button,select {{padding:9px;margin:8px 0}} button {{cursor:pointer}}
pre,.result {{white-space:pre-wrap;overflow-wrap:anywhere;padding:14px;background:#f3f3f3}}
.section {{margin:28px 0}} .error {{color:#a00}} .score {{color:#555}}
table {{border-collapse:collapse;width:100%;margin-top:18px}}
th,td {{border:1px solid #ccc;padding:9px;text-align:left;vertical-align:top}}
th {{background:#f3f3f3}}
</style>
</head>
<body>
<h1>{safe_title}</h1>
{body}
</body>
</html>""",
        status_code=status_code,
    )


def home_page(documents: list[dict] | None = None) -> HTMLResponse:
    documents = documents or []

    if documents:
        options = []
        for document in documents:
            filename = html.escape(document["filename"])
            gcs_path = document["gcs_path"]
            safe_path = html.escape(gcs_path, quote=True)
            identifier = html.escape(gcs_path.rsplit("/", 1)[-1][-12:])
            options.append(
                f'<option value="{safe_path}">'
                f"{filename} — {identifier}</option>"
            )

        keywords_form = f"""
<form action="/keywords" method="post">
<label for="keyword-document">Dokument:</label><br>
<select id="keyword-document" name="gcs_path" required>
{''.join(options)}
</select>
<p><button type="submit">Wyodrębnij słowa kluczowe</button></p>
</form>
"""
    else:
        keywords_form = (
            "<p>Najpierw dodaj i zaindeksuj tekst, "
            "aby wybrać dokument do analizy.</p>"
        )

    body = f"""
<div class="section">
<h2>Dodaj tekst</h2>
<p>Oryginał zostanie zapisany w Google Cloud Storage, a fragmenty trafią do wyszukiwarki Qdrant.</p>
<form action="/documents" method="post" enctype="multipart/form-data">
<label for="text">Wklej tekst:</label>
<textarea id="text" name="text" placeholder="Wpisz lub wklej tekst"></textarea>
<p>Albo wybierz plik TXT:</p>
<input id="file" name="file" type="file" accept=".txt,text/plain">
<p><button type="submit">Zapisz i zaindeksuj tekst</button></p>
</form>
</div>

<div class="section">
<h2>Wyszukaj fragmenty</h2>
<p>Ta opcja pokazuje znalezione fragmenty bez generowania odpowiedzi przez LLM.</p>
<form action="/search" method="post">
<label for="search-query">Pytanie lub szukany temat:</label><br>
<input id="search-query" name="query" type="text"
style="width:100%;box-sizing:border-box"
placeholder="Wpisz pytanie własnymi słowami" required>
<p><button type="submit">Szukaj fragmentów</button></p>
</form>
</div>

<div class="section">
<h2>Zapytaj na podstawie dokumentów (RAG)</h2>
<p>Aplikacja wyszuka pasujące fragmenty w Qdrant i poprosi Gemini o odpowiedź opartą na tych źródłach. Wynik pokaże cytowane fragmenty.</p>
<form action="/rag" method="post">
<label for="rag-query">Pytanie:</label><br>
<input id="rag-query" name="query" type="text"
style="width:100%;box-sizing:border-box"
placeholder="Zadaj pytanie dotyczące dodanych dokumentów" required>
<p><button type="submit">Wygeneruj odpowiedź RAG</button></p>
</form>
</div>

<div class="section">
<h2>Słowa kluczowe przez KeyBERT</h2>
<p>Wybierz dodany dokument. Aplikacja oceni, które słowa i frazy najlepiej opisują jego fragmenty.</p>
{keywords_form}
</div>

<div class="section">
<h2>Demonstracja BERT-a</h2>
<p>Ta funkcja uzupełnia zamaskowany token na podstawie wzorców poznanych podczas wcześniejszego treningu. Nie korzysta z dodanych dokumentów, Qdrant ani internetu.</p>
<p>Model: <code>dkleczek/bert-base-polish-cased-v1</code>. Według karty był trenowany na polskich tekstach z OpenSubtitles, ParaCrawl, Polskiego Korpusu Parlamentarnego i Wikipedii.</p>
<p><a href="{MODEL_CARD_URL}" target="_blank" rel="noopener noreferrer">Karta modelu</a></p>
<p><a href="/fill-mask">Otwórz BERT fill-mask</a></p>
</div>
"""
    return _page("Wyszukiwarka semantyczna tekstów", body)


def error_page(
    title: str,
    message: str,
    status_code: int = 400,
) -> HTMLResponse:
    body = (
        f'<p class="error">{html.escape(message)}</p>'
        '<p><a href="/">Wróć do aplikacji</a></p>'
    )
    return _page(title, body, status_code)


def document_saved_page(
    filename: str,
    character_count: int,
    chunk_count: int,
    gcs_path: str,
    collection_name: str,
    preview: str,
) -> HTMLResponse:
    body = f"""
<p><strong>Źródło:</strong> {html.escape(filename)}</p>
<p><strong>Liczba znaków:</strong> {character_count}</p>
<p><strong>Liczba fragmentów:</strong> {chunk_count}</p>
<p><strong>Oryginał w GCS:</strong> {html.escape(gcs_path)}</p>
<p>Embeddingi zapisano w kolekcji <strong>{html.escape(collection_name)}</strong>.</p>
<h2>Podgląd tekstu</h2>
<pre>{html.escape(preview)}</pre>
<p><a href="/">Wróć do aplikacji</a></p>
"""
    return _page("Tekst zapisany i zaindeksowany", body)


def search_results_page(query: str, results: list[dict]) -> HTMLResponse:
    if not results:
        results_html = (
            "<p>Brak zaindeksowanych tekstów. "
            "Najpierw dodaj tekst na stronie głównej.</p>"
        )
    else:
        cards = []
        for rank, result in enumerate(results, start=1):
            cards.append(
                f"""
<div class="result">
<h2>Wynik {rank}</h2>
<p><strong>Źródło:</strong> {html.escape(result["filename"])}</p>
<p class="score">Podobieństwo: {result["score"]:.4f}</p>
<pre>{html.escape(result["snippet"])}</pre>
<p><strong>Oryginał w GCS:</strong> {html.escape(result["gcs_path"])}</p>
</div>
"""
            )
        results_html = "".join(cards)

    body = (
        f"<p><strong>Zapytanie:</strong> {html.escape(query)}</p>"
        f"{results_html}<p><a href='/'>Wróć do aplikacji</a></p>"
    )
    return _page("Wyniki wyszukiwania", body)


def rag_results_page(
    query: str,
    answer: str,
    sources: list[dict],
) -> HTMLResponse:
    if not sources:
        sources_html = "<p>Nie znaleziono zaindeksowanych fragmentów.</p>"
    else:
        cards = []
        for rank, source in enumerate(sources, start=1):
            cards.append(
                f"""
<div class="result">
<h3>Źródło [{rank}]: {html.escape(source["filename"])}</h3>
<p class="score">Podobieństwo: {source["score"]:.4f}</p>
<pre>{html.escape(source["snippet"])}</pre>
</div>
"""
            )
        sources_html = "".join(cards)

    body = f"""
<p><strong>Pytanie:</strong> {html.escape(query)}</p>
<h2>Odpowiedź</h2>
<pre>{html.escape(answer)}</pre>
<h2>Wykorzystane fragmenty</h2>
{sources_html}
<p>Odpowiedź została wygenerowana przez Gemini na podstawie pokazanych fragmentów. Sprawdź cytowania przy twierdzeniach.</p>
<p><a href="/">Wróć do aplikacji</a></p>
"""
    return _page("Odpowiedź RAG", body)


def keyword_results_page(
    filename: str,
    gcs_path: str,
    keywords: list[tuple[str, float]],
) -> HTMLResponse:
    if not keywords:
        results_html = (
            "<p>Nie znaleziono słów kluczowych. Dokument może być "
            "zbyt krótki albo zawierać głównie często występujące słowa.</p>"
        )
    else:
        rows = []
        for rank, (phrase, score) in enumerate(keywords, start=1):
            rows.append(
                "<tr>"
                f"<td>{rank}</td>"
                f"<td>{html.escape(phrase)}</td>"
                f"<td>{score:.4f}</td>"
                "</tr>"
            )
        results_html = f"""
<table>
<thead>
<tr>
<th>Miejsce</th>
<th>Słowo lub fraza</th>
<th>Wynik podobieństwa</th>
</tr>
</thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""

    body = f"""
<p><strong>Dokument:</strong> {html.escape(filename)}</p>
<p><strong>Oryginał w GCS:</strong> {html.escape(gcs_path)}</p>
<p>KeyBERT porównuje embedding dokumentu i kandydatów na frazy.
Wynik jest rankingiem podobieństwa, a nie prawdopodobieństwem.</p>
{results_html}
<p><a href="/">Wróć do aplikacji</a></p>
"""
    return _page("Słowa kluczowe dokumentu", body)
