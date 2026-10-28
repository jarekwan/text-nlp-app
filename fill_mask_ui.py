import html

from fastapi.responses import HTMLResponse

from home_ui import _page, MODEL_CARD_URL


def error_page(
    title: str,
    message: str,
    status_code: int = 400,
) -> HTMLResponse:
    body = (
        f'<p class="error">{html.escape(message)}</p>'
        '<p><a href="/fill-mask">Wróć do formularza BERT-a</a></p>'
    )
    return _page(title, body, status_code)


def fill_mask_form_page(
    mask_token: str,
    sentence: str = "",
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    token = html.escape(mask_token)
    safe_sentence = html.escape(sentence)
    error_html = (
        f'<p class="error">{html.escape(error)}</p>'
        if error
        else ""
    )

    body = f"""
<p>Wstaw dokładnie jeden <code>{token}</code> w miejscu brakującego tokenu. Możesz umieścić go na początku, w środku lub na końcu zdania. Gdy maska jest w środku, BERT wykorzystuje dostępny tekst przed nią i po niej.</p>
<p>Przykłady:</p>
<ul>
<li><code>Warszawa jest stolicą {token}.</code></li>
<li><code>Warszawa jest {token} Polski.</code></li>
<li><code>{token} jest stolicą Polski.</code></li>
</ul>
<p>Maska oznacza jedną pozycję tokenizera BERT-a. Zwykle odpowiada słowu, ale propozycją może być też fragment słowa lub znak interpunkcyjny.</p>
{error_html}
<form action="/fill-mask" method="post">
<label for="sentence">Zdanie z jednym {token}:</label>
<textarea id="sentence" name="sentence"
placeholder="Warszawa jest {token} Polski." required>{safe_sentence}</textarea>
<p><button type="submit">Uzupełnij maskę</button></p>
</form>
<h2>Skąd model bierze propozycje?</h2>
<p>Model <code>dkleczek/bert-base-polish-cased-v1</code> był trenowany na polskich tekstach z OpenSubtitles (wariant cased z usuniętymi duplikatami), ParaCrawl, Polskiego Korpusu Parlamentarnego i Wikipedii (zrzut z lutego 2020).</p>
<p>Nie przeszukuje Twoich dokumentów, Qdrant ani internetu. Wyniki są przewidywaniami na podstawie wzorców poznanych podczas treningu, a nie weryfikacją faktów.</p>
<p><a href="{MODEL_CARD_URL}" target="_blank" rel="noopener noreferrer">Karta modelu i źródła danych treningowych</a></p>
<p><a href="/">Wróć do aplikacji</a></p>
"""
    return _page("Demonstracja BERT fill-mask", body, status_code)


def fill_mask_results_page(
    sentence: str,
    predictions: list[dict],
) -> HTMLResponse:
    rows = []
    for rank, item in enumerate(predictions, start=1):
        rows.append(
            f"<tr><td>{rank}</td>"
            f"<td>{html.escape(str(item['token_str']).strip())}</td>"
            f"<td>{float(item['score']) * 100:.2f}%</td>"
            f"<td>{html.escape(str(item['sequence']))}</td></tr>"
        )

    body = f"""
<p><strong>Zdanie:</strong> {html.escape(sentence)}</p>
<table>
<thead>
<tr>
<th>Miejsce</th>
<th>Proponowany token</th>
<th>Wynik modelu</th>
<th>Zdanie po uzupełnieniu</th>
</tr>
</thead>
<tbody>{''.join(rows)}</tbody>
</table>
<p>Procent to wynik modelu dla propozycji, nie miara prawdziwości faktu ani gwarancja poprawności. BERT uzupełnia tekst, ale nie weryfikuje faktów.</p>
<p><a href="/fill-mask">Sprawdź inne zdanie</a></p>
<p><a href="/">Wróć do aplikacji</a></p>
"""
    return _page("Propozycje BERT fill-mask", body)
