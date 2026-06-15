"""arXiv publisher adapter — search via arxiv package, full text via ar5iv HTML."""

import time
from html.parser import HTMLParser

import arxiv
import requests

from config import (
    AR5IV_URL,
    ARXIV_CATEGORIES,
    ARXIV_MAX_RESULTS,
    ARXIV_QUERY,
    BOILERPLATE_SECTIONS,
    RATE_LIMIT_PAUSE,
)
from adapters.base import PublisherAdapter


# ── ar5iv HTML parser ─────────────────────────────────────────────────────

class _Ar5ivParser(HTMLParser):
    """
    Minimal parser for ar5iv HTML.
    Extracts sections from <section> or <div class="ltx_section"> elements.
    """

    def __init__(self):
        super().__init__()
        self._sections: dict[str, str] = {}
        self._current_title: str | None = None
        self._current_text: list[str] = []
        self._in_section: int = 0          # nesting depth of section elements
        self._in_title: bool = False
        self._title_tag: str | None = None

    def _is_section_tag(self, tag: str, attrs: dict) -> bool:
        if tag == "section":
            return True
        if tag == "div" and "ltx_section" in attrs.get("class", ""):
            return True
        return False

    def _is_title_tag(self, tag: str, attrs: dict) -> bool:
        if tag in ("h1", "h2", "h3", "h4"):
            return True
        cls = attrs.get("class", "")
        if "ltx_title" in cls:
            return True
        return False

    def handle_starttag(self, tag: str, attrs):
        attrs_dict = dict(attrs)
        if self._is_section_tag(tag, attrs_dict):
            if self._in_section > 0 and self._current_title:
                # Save the current section before nesting deeper
                self._flush_section()
            self._in_section += 1
            self._current_title = None
            self._current_text = []
        elif self._in_section and self._is_title_tag(tag, attrs_dict):
            self._in_title = True
            self._title_tag = tag

    def handle_endtag(self, tag: str):
        if self._in_title and tag == self._title_tag:
            self._in_title = False
            self._title_tag = None
        elif self._in_section and tag in ("section", "div"):
            self._in_section -= 1
            if self._in_section == 0:
                self._flush_section()

    def handle_data(self, data: str):
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self._current_title = (self._current_title or "") + text
        elif self._in_section:
            self._current_text.append(text)

    def _flush_section(self):
        title = (self._current_title or "unknown").strip().lower()
        text = " ".join(self._current_text).strip()
        if text and title not in BOILERPLATE_SECTIONS:
            # Merge with existing text if title already seen
            if title in self._sections:
                self._sections[title] += " " + text
            else:
                self._sections[title] = text
        self._current_title = None
        self._current_text = []

    @property
    def sections(self) -> dict[str, str]:
        return self._sections


def _parse_ar5iv_html(html: str) -> dict[str, str]:
    parser = _Ar5ivParser()
    parser.feed(html)
    return parser.sections


# ── arXiv adapter ─────────────────────────────────────────────────────────

class ArxivAdapter(PublisherAdapter):
    """Adapter for arXiv search + ar5iv HTML full-text retrieval."""

    def search(self, max_results: int = ARXIV_MAX_RESULTS) -> list[dict]:
        """Return papers from arXiv as {title, paper_id, abstract, source} dicts."""
        # Build category filter
        cat_filter = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
        full_query = f"({ARXIV_QUERY}) AND ({cat_filter})"

        client = arxiv.Client()
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results: list[dict] = []
        try:
            for result in client.results(search):
                arxiv_id = result.get_short_id()  # e.g. "2301.12345v1" → strip version
                arxiv_id = arxiv_id.split("v")[0]
                results.append({
                    "title": result.title.strip(),
                    "paper_id": arxiv_id,
                    "abstract": result.summary.strip(),
                    "journal": "arXiv",
                    "year": str(result.published.year) if result.published else "",
                    "source": "arxiv",
                })
                time.sleep(RATE_LIMIT_PAUSE * 0.5)
        except Exception as exc:
            print(f"  [arxiv] Search error: {exc}")

        return results

    def fetch_full_text(self, paper: dict) -> dict | None:
        """
        Fetch HTML from ar5iv and parse into sections.
        Falls back to abstract-only if ar5iv fetch fails.
        """
        arxiv_id = paper.get("paper_id", "")
        if not arxiv_id:
            return None

        url = AR5IV_URL.format(arxiv_id=arxiv_id)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            sections = _parse_ar5iv_html(resp.text)
        except requests.RequestException as exc:
            print(f"  [arxiv] ar5iv fetch failed for {arxiv_id}: {exc}. Falling back to abstract.")
            sections = {}

        # If parsing yielded nothing, fall back to abstract
        if not sections:
            abstract = paper.get("abstract", "")
            if abstract:
                sections = {"abstract_fallback": abstract}

        return {"sections": sections, "tables": []}
