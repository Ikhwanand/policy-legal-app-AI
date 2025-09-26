from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class QAResult:
    answer: str
    context: List[Dict]
    mode: str  # "extractive", "llm", atau "llm_fallback"


ANSWER_CACHE: OrderedDict[Tuple[bool, str], QAResult] = OrderedDict()
CACHE_LIMIT = 20


def simple_extractive_answer(query: str, contexts: List[Dict], top_n: int = 3) -> str:
    lines = [f"**Pertanyaan:** {query}", "", "**Ringkasan Konteks Terkait:**"]
    for i, context in enumerate(contexts[:top_n], 1):
        snippet = context.get("text", "")[:600]
        lines.append(f"{i}. {snippet}")
    lines.append("")
    lines.append("_Catatan: ini jawaban berbasis konteks (tanpa LLM). Aktifkan LLM untuk jawaban generatif._")
    return "\n".join(lines)


# def llm_answer_with_openai(query: str, contexts: List[Dict]) -> str:
#     ...


# def llm_answer_with_gemini(query: str, contexts: List[Dict]) -> str:
#     ...


def llm_answer_with_agno(query: str, contexts: List[Dict]) -> str:
    """Jawaban generatif menggunakan Agno Agent + Google Gemini."""
    try:
        from agno.agent import Agent
        from agno.models.google import Gemini
        from agno.tools.duckduckgo import DuckDuckGoTools
        from agno.tools.googlesearch import GoogleSearchTools
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Gagal memuat dependensi Agno: {exc}") from exc

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY tidak ditemukan")

    context_text = "\n\n".join([c.get("text", "") for c in contexts[:6]])
    system_instructions = (
        "Anda adalah AI Legal/Policy Agent. Jawab ringkas dan berbasis bukti dari konteks. "
        "Sertakan rujukan pasal/ayat jika tersirat. Jika informasi tidak cukup, katakan dengan jujur."
        "Jika anda tidak memiliki informasi yang cukup, anda bisa gunakan tools seperti google search dan duckduckgo search."
        "Ketika anda melakukan pencarian, pastikan regulasi yang dirujuk merupakan versi terbaru."
    )

    agent = Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        instructions=system_instructions,
        markdown=True,
        tools=[
            GoogleSearchTools(),
            DuckDuckGoTools(),
        ],
    )

    prompt = (
        f"Pertanyaan: {query}\n\n"
        f"KONTEKS (kutipan dari dokumen hukum):\n{context_text}\n\n"
        "Jawab berdasarkan konteks di atas."
        "Apabila anda tidak memiliki informasi yang cukup, pertimbangkan penggunaan pencarian eksternal secara hemat. Seperti google search dan duckduckgo search."
    )

    try:
        run = agent.run(prompt)
    except Exception as exc:
        raise RuntimeError(f"Gagal memanggil Gemini via Agno: {exc}") from exc
    return getattr(run, "content", str(run)).strip()


def _make_cache_key(query: str, contexts: List[Dict]) -> str:
    hasher = hashlib.blake2s(digest_size=16)
    hasher.update(query.encode("utf-8"))
    for context in contexts:
        hasher.update(str(context.get("doc_id", "")).encode("utf-8"))
        hasher.update(str(context.get("page", "")).encode("utf-8"))
        hasher.update(str(context.get("section", "")).encode("utf-8"))
        hasher.update(str(context.get("vector_id", "")).encode("utf-8"))
        snippet = context.get("text", "")[:200]
        hasher.update(snippet.encode("utf-8"))
    return hasher.hexdigest()


def _remember(key: Tuple[bool, str], value: QAResult) -> QAResult:
    ANSWER_CACHE[key] = value
    ANSWER_CACHE.move_to_end(key)
    while len(ANSWER_CACHE) > CACHE_LIMIT:
        ANSWER_CACHE.popitem(last=False)
    return value


def answer_query(query: str, hits: List[Dict], use_llm: bool) -> QAResult:
    hits = hits or []
    cache_key = (use_llm, _make_cache_key(query, hits))
    if cache_key in ANSWER_CACHE:
        return ANSWER_CACHE[cache_key]

    if use_llm:
        try:
            answer_text = llm_answer_with_agno(query, hits)
            result = QAResult(answer=answer_text, context=hits, mode="llm")
        except Exception as exc:
            fallback = simple_extractive_answer(query, hits)
            warning = f"_Catatan: Mode LLM gagal dengan pesan: {exc}_"
            combined = f"{fallback}\n\n{warning}"
            result = QAResult(answer=combined, context=hits, mode="llm_fallback")
    else:
        answer_text = simple_extractive_answer(query, hits)
        result = QAResult(answer=answer_text, context=hits, mode="extractive")

    return _remember(cache_key, result)