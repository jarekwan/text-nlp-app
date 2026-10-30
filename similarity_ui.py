from html import escape


def similarity_form_page(
    documents: list[dict],
    selected_paths: list[str] | None = None,
    error: str = "",
) -> str:
    """Wyświetla formularz wyboru dokumentów do porównania."""
    selected_paths = set(selected_paths or [])

    document_options = []
    for document in documents:
        filename = escape(str(document["filename"]))
        gcs_path = str(document["gcs_path"])
        checked = "checked" if gcs_path in selected_paths else ""

        document_options.append(
            f"""
            <label class="document-option">
                <input
                    type="checkbox"
                    name="gcs_paths"
                    value="{escape(gcs_path, quote=True)}"
                    {checked}
                >
                <span>{filename}</span>
            </label>
            """
        )

    if document_options:
        document_list = "\n".join(document_options)
        form = f"""
        <form action="/similarity" method="post">
            <div class="document-list">
                {document_list}
            </div>
            <button type="submit">Porównaj dokumenty</button>
        </form>
        """
    else:
        form = """
        <p>Najpierw dodaj i zaindeksuj dokumenty na stronie głównej.</p>
        """

    error_message = (
        f'<p class="error">{escape(error)}</p>'
        if error
        else ""
    )

    return f"""<!doctype html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Podobieństwo dokumentów</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 36px auto;
            padding: 0 20px;
            color: #222;
        }}
        .document-list {{
            display: grid;
            gap: 10px;
            margin: 20px 0;
        }}
        .document-option {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            overflow-wrap: anywhere;
        }}
        .document-option input {{
            margin-top: 3px;
        }}
        button {{
            padding: 10px 14px;
            cursor: pointer;
        }}
        .error {{
            color: #a00000;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Podobieństwo tekstów</h1>
    <p>
        Wybierz co najmniej dwa zaindeksowane dokumenty.
        Aplikacja porówna ich znaczenie i pokaże macierz podobieństwa
        cosinusowego.
    </p>
    {error_message}
    {form}
    <p><a href="/">Wróć do aplikacji</a></p>
</body>
</html>
"""


def similarity_results_page(
    document_names: list[str],
    similarity_matrix: list[list[float]],
) -> str:
    """Wyświetla macierz podobieństwa wybranych dokumentów."""
    headers = "".join(
        f"<th>{escape(name)}</th>" for name in document_names
    )

    rows = []
    for name, scores in zip(document_names, similarity_matrix):
        cells = "".join(
            f"<td>{float(score):.4f}</td>" for score in scores
        )
        rows.append(
            f"<tr><th>{escape(name)}</th>{cells}</tr>"
        )

    table_rows = "\n".join(rows)

    return f"""<!doctype html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Wyniki podobieństwa dokumentów</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1100px;
            margin: 36px auto;
            padding: 0 20px;
            color: #222;
        }}
        .table-wrap {{
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            min-width: 600px;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 10px;
            text-align: left;
        }}
        thead, tr th {{
            background: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>Macierz podobieństwa dokumentów</h1>
    <p>
        Wyższy wynik oznacza większe podobieństwo znaczeniowe.
        Wynik nie jest prawdopodobieństwem.
    </p>
    <div class="table-wrap">
        <table>
            <thead>
                <tr><th>Dokument</th>{headers}</tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    <p><a href="/similarity">Porównaj inne dokumenty</a></p>
    <p><a href="/">Wróć do aplikacji</a></p>
</body>
</html>
"""
