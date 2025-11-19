from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://peraturan.bpk.go.id"
SEARCH_PATH = "/Search"
DEFAULT_HEADERS = {
    "User-Agent": "SIKAP-Agent/1.0 (+https://github.com/)",
}


@dataclass
class DownloadEntry:
    label: str
    url: str
    filename: str


@dataclass
class PeraturanResult:
    title: str
    description: str
    subjects: List[str]
    detail_url: str | None
    downloads: List[DownloadEntry]


def _absolute_url(href: str | None) -> str | None:
    if not href:
        return None
    return urljoin(BASE_URL, href)


def _parse_card(card) -> PeraturanResult | None:
    title_el = card.select_one("div.col-lg-10.fs-2.fw-bold.pe-4 a")
    if not title_el:
        return None
    title = title_el.get_text(strip=True)
    detail_href = title_el.get("href")
    description_el = card.select_one("div.col-lg-8.fw-semibold.fs-5.text-gray-600")
    description = description_el.get_text(strip=True) if description_el else ""
    subjects = [span.get_text(strip=True) for span in card.select("span.badge")]

    downloads: List[DownloadEntry] = []
    for link in card.select("a.download-file"):
        href = link.get("href")
        if not href:
            continue
        filename = link.get("title") or link.get_text(strip=True)
        label = link.get_text(strip=True)
        downloads.append(
            DownloadEntry(
                label=label,
                url=_absolute_url(href) or "",
                filename=filename or label or "document.pdf",
            )
        )
    if not downloads:
        return None

    return PeraturanResult(
        title=title,
        description=description,
        subjects=subjects,
        detail_url=_absolute_url(detail_href),
        downloads=downloads,
    )


def search_peraturan(
    keyword: str,
    limit: int = 5,
    tentang: str | None = None,
    nomor: str | None = None,
) -> List[PeraturanResult]:
    params = {
        "keywords": keyword or "",
        "tentang": tentang or "",
        "nomor": nomor or "",
    }
    response = requests.get(
        f"{BASE_URL}{SEARCH_PATH}",
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=60,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    cards = soup.select("div.card-body.p-xl-10.p-8.d-flex")
    results: List[PeraturanResult] = []
    for card in cards:
        parsed = _parse_card(card)
        if parsed:
            results.append(parsed)
        if len(results) >= limit:
            break
    return results


def download_file(url: str, destination: Path, timeout: int = 120) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    logger.info("Downloaded %s -> %s", url, destination)
    return destination
