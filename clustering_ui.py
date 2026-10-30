from html import escape


def clustering_form_page(document_count: int, error: str = "") -> str:
    """Pokazuje formularz wyboru liczby grup KMeans."""
    error_html = (
        f'<p class="error">{escape(error)}</p>'
        if error
        else ""
    )

    if document_count < 3:
        form_html = """
        <p>
            Do grupowania i wykresu PCA potrzebne są co najmniej
            trzy zaindeksowane dokumenty. Dodaj kolejne dokumenty
            na stronie głównej.
        </p>
        """
    else:
        form_html = f"""
        <form action="/clustering" method="post">
            <label for="n-clusters">Liczba grup KMeans:</label>
            <input
                id="n-clusters"
                name="n_clusters"
                type="number"
                min="2"
                max="{document_count - 1}"
                value="2"
                required
            >
            <button type="submit">Grupuj dokumenty</button>
        </form>
        <p>Liczba zaindeksowanych dokumentów: {document_count}</p>
        """

    return f"""<!doctype html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Grupowanie tekstów</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 36px auto;
            padding: 0 20px;
            color: #222;
        }}
        input, button {{
            padding: 9px;
            margin: 8px 0;
        }}
        button {{
            cursor: pointer;
        }}
        .error {{
            color: #a00000;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Grupowanie tekstów</h1>
    <p>
        KMeans przypisze dokumenty do grup na podstawie ich embeddingów.
        PCA pokaże dokumenty na wykresie 2D.
    </p>
    {error_html}
    {form_html}
    <p><a href="/">Wróć do aplikacji</a></p>
</body>
</html>
"""


def clustering_results_page(
    points: list[dict],
    cluster_count: int,
) -> str:
    """Pokazuje wykres PCA i tabelę grup przypisanych przez KMeans."""
    width = 800
    height = 500
    margin = 60

    x_values = [point["x"] for point in points]
    y_values = [point["y"] for point in points]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#c026d3",
        "#4d7c0f",
    ]

    circles = []
    table_rows = []

    for index, point in enumerate(points, start=1):
        x = margin + (point["x"] - x_min) / x_range * (
            width - 2 * margin
        )
        y = height - margin - (point["y"] - y_min) / y_range * (
            height - 2 * margin
        )
        color = palette[(point["cluster"] - 1) % len(palette)]
        label = escape(str(point["label"]))
        point_name = f"D{index}"

        circles.append(
            f"""
            <circle cx="{x:.2f}" cy="{y:.2f}" r="9" fill="{color}">
                <title>{point_name}: {label} — grupa {point['cluster']}</title>
            </circle>
            <text x="{x + 12:.2f}" y="{y + 5:.2f}">{point_name}</text>
            """
        )

        table_rows.append(
            f"<tr><td>{point_name}</td><td>{label}</td>"
            f"<td>Grupa {point['cluster']}</td></tr>"
        )

    legend_items = []
    for cluster_number in range(1, cluster_count + 1):
        color = palette[(cluster_number - 1) % len(palette)]
        legend_items.append(
            f'<span class="legend-item">'
            f'<i style="background:{color}"></i>'
            f'Grupa {cluster_number}</span>'
        )

    return f"""<!doctype html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Wyniki grupowania tekstów</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 36px auto;
            padding: 0 20px;
            color: #222;
        }}
        .chart-wrap {{
            width: 100%;
            overflow-x: auto;
        }}
        svg {{
            min-width: 650px;
            border: 1px solid #ccc;
            background: #fff;
        }}
        .axis {{
            stroke: #555;
            stroke-width: 1;
        }}
        .axis-label, text {{
            font-family: Arial, sans-serif;
            font-size: 13px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            margin: 14px 0 24px;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-item i {{
            width: 13px;
            height: 13px;
            border-radius: 50%;
            display: inline-block;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 9px;
            text-align: left;
        }}
        th {{
            background: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>Grupy dokumentów</h1>
    <p>
        KMeans utworzył {cluster_count} grupy.
        PCA przedstawiło embeddingi na wykresie 2D.
        Wykres pomaga zobaczyć układ dokumentów; sam w sobie
        nie potwierdza poprawności grup.
    </p>

    <div class="chart-wrap">
        <svg viewBox="0 0 {width} {height}" role="img"
             aria-label="Wykres PCA dokumentów pogrupowanych przez KMeans">
            <line class="axis"
                  x1="{margin}" y1="{height - margin}"
                  x2="{width - margin}" y2="{height - margin}"/>
            <line class="axis"
                  x1="{margin}" y1="{margin}"
                  x2="{margin}" y2="{height - margin}"/>
            <text class="axis-label"
                  x="{width / 2}" y="{height - 15}">PCA 1</text>
            <text class="axis-label"
                  x="15" y="{height / 2}"
                  transform="rotate(-90 15 {height / 2})">PCA 2</text>
            {''.join(circles)}
        </svg>
    </div>

    <div class="legend">
        {''.join(legend_items)}
    </div>

    <h2>Przypisanie dokumentów do grup</h2>
    <table>
        <thead>
            <tr><th>Oznaczenie</th><th>Dokument</th><th>Grupa</th></tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>

    <p><a href="/clustering">Zmień liczbę grup</a></p>
    <p><a href="/">Wróć do aplikacji</a></p>
</body>
</html>
"""
