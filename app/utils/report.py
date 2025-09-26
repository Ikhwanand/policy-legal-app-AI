from __future__ import annotations

import datetime as dt
from typing import Dict, List

from io import BytesIO
from markdown import markdown
from xhtml2pdf import pisa

def _format_location(hit: Dict) -> str:
    parts = []
    if hit.get("page"):
        parts.append(f"hal. {hit['page']}")
    if hit.get("section"):
        parts.append(f"paragraf {hit['section']}")
    if hit.get("section_chunk"):
        parts.append(f"bagian {hit['section_chunk']}")
    return ", ".join(parts)


def make_markdown_report(query: str, answer: str, hits: List[Dict], rec: str = "") -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["# Rekomendasi Kebijakan - Laporan", f"_Generated: {ts}_", ""]
    lines += ["## Pertanyaan", query, ""]
    lines += ["## Jawaban/Analisis", answer, ""]
    lines += ["## Referensi Konteks"]
    for i, hit in enumerate(hits, 1):
        source = hit.get("source") or hit.get("doc_id") or "-"
        score = hit.get("score", 0.0)
        location = _format_location(hit)
        location_note = f" ({location})" if location else ""
        lines.append(f"{i}. **{source}**{location_note} - skor: {score:.3f}")
        snippet = hit.get("text", "")[:800]
        lines.append(f"   > {snippet}")
    if rec:
        lines += ["", "## Rekomendasi Sistem", rec]
    return "\n".join(lines)


def make_pdf_report(query: str, answer: str, hits: List[Dict], rec: str = "") -> bytes:
    markdown_body = make_markdown_report(query, answer, hits, rec)
    html = markdown(
        markdown_body,
        extensions=['extra', 'tables', 'sane_lists', 'toc'],
        output_format='html5'
    )
    template = (
        "<html><head><meta charset='utf-8' />"
        "<style>"
        "body { font-family: 'Helvetica', sans-serif; line-height: 1.5; font-size: 12pt; }"
        "h1, h2, h3 { color: #1F4E79; margin-top: 1.2em; }"
        "p { margin: 0.4em 0; }"
        "blockquote { margin: 0.6em 0; padding-left: 0.6em; border-left: 3px solid #ccc; font-style: italic; }"
        "table { width: 100%; border-collapse: collapse; margin: 0.8em 0; }"
        "th, td { border: 1px solid #ccc; padding: 6px; text-align: left; }"
        "code { font-family: 'Courier New', monospace; }"
        "</style>"
        "</head><body>"
        f"{html}"
        "</body></html>"
    )
    
    buffer = BytesIO()
    result = pisa.CreatePDF(src=template, dest=buffer, encoding='utf-8')
    if result.err:
        raise RuntimeError("Gagal membuat PDF dari laporan markdown.")
    buffer.seek(0)
    return buffer.read()
