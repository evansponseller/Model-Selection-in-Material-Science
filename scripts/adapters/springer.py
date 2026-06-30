"""Springer Nature publisher adapter (npj Computational Materials, open access).

Uses the Springer Nature OpenAccess API:
  - metadata search → JSON  (https://api.springernature.com/openaccess/json)
  - full text       → JATS  (https://api.springernature.com/openaccess/jats)

Requires SPRINGER_API_KEY (free tier from https://dev.springernature.com).
Only open-access content is retrievable this way — npj Computational Materials
is fully OA, so all of its articles are fetchable.
"""

import time
import xml.etree.ElementTree as ET

import requests

from config import (
    HTTP_REFERER,
    RATE_LIMIT_PAUSE,
    SPRINGER_API_KEY,
    SPRINGER_JATS_URL,
    SPRINGER_MAX_RESULTS,
    SPRINGER_META_URL,
    SPRINGER_QUERY,
)
from adapters.base import PublisherAdapter

# JATS uses no/default namespace for the tags we need (sec, title, p, abstract).


def _text_of(elem: ET.Element) -> str:
    """All descendant text of an element, whitespace-normalised."""
    return " ".join(t.strip() for t in elem.itertext() if t.strip())


def _localname(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


class SpringerAdapter(PublisherAdapter):
    """Adapter for Springer Nature OpenAccess search + JATS full-text retrieval."""

    def search(self, max_results: int = SPRINGER_MAX_RESULTS) -> list[dict]:
        """Return papers as {title, paper_id, doi, abstract, journal, year, source} dicts."""
        if not SPRINGER_API_KEY:
            print("  [springer] SPRINGER_API_KEY not set — skipping Springer search.")
            return []

        results: list[dict] = []
        start = 1            # Springer is 1-indexed
        page_size = 25       # Open Access free tier caps page size at 25 (p>25 → 403)

        while len(results) < max_results:
            params = {
                "q": SPRINGER_QUERY,
                "p": page_size,
                "s": start,
                "api_key": SPRINGER_API_KEY,
            }
            try:
                resp = requests.get(SPRINGER_META_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                print(f"  [springer] Search error: {exc}")
                break

            records = data.get("records", [])
            if start == 1:
                total = data.get("result", [{}])[0].get("total", "?")
                print(f"  Springer total results: {total}")
            if not records:
                break

            for rec in records:
                doi = (rec.get("doi") or "").strip()
                if not doi:
                    continue
                results.append({
                    "title": (rec.get("title") or "").strip(),
                    "paper_id": doi,
                    "doi": doi,
                    "abstract": _record_abstract(rec),
                    "journal": (rec.get("publicationName") or "").strip(),
                    "year": (rec.get("publicationDate") or "")[:4],
                    "source": "springer",
                })

            print(f"  Fetched {len(results)} Springer candidates so far …", end="\r")
            if len(records) < page_size:
                break
            start += page_size
            time.sleep(RATE_LIMIT_PAUSE)

        print(f"  Fetched {len(results)} Springer candidates total.      ")
        return results[:max_results]

    def fetch_full_text(self, paper: dict) -> dict | None:
        """Fetch JATS full text by DOI and parse into {sections, tables, intro_text}."""
        if not SPRINGER_API_KEY:
            raise EnvironmentError("SPRINGER_API_KEY is not set.")

        doi = paper.get("doi") or paper.get("paper_id", "")
        if not doi:
            return None

        params = {"q": f"doi:{doi}", "api_key": SPRINGER_API_KEY}
        headers = {"HTTP-Referer": HTTP_REFERER}
        try:
            resp = requests.get(SPRINGER_JATS_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [springer] Error fetching JATS for {doi}: {exc}")
            return None

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            print(f"  [springer] JATS parse error for {doi}: {exc}")
            return None

        result = self._parse_jats(root)
        # Recent online-first articles sometimes expose only the abstract in JATS.
        # Without a body there's nothing to extract — skip (returning None means
        # it isn't persisted, so it's retried on a later run once full text exists).
        if not any(k != "abstract" for k in result["sections"]):
            print(f"  [springer] {doi}: full-text body not available yet (abstract only).")
            return None
        return result

    def _parse_jats(self, root: ET.Element) -> dict:
        """Pull section titles + text and tables from a JATS <article>."""
        sections: dict[str, str] = {}
        tables: list[str] = []

        # Abstract (in <front>/<article-meta>/<abstract>)
        for abs in root.iter():
            if _localname(abs) == "abstract":
                txt = _text_of(abs)
                if txt:
                    sections["abstract"] = txt
                break

        # Body sections
        body = next((e for e in root.iter() if _localname(e) == "body"), None)
        if body is not None:
            for sec in body.iter():
                if _localname(sec) != "sec":
                    continue
                title_el = next(
                    (c for c in sec if _localname(c) == "title"), None
                )
                title = (
                    title_el.text.strip().lower()
                    if title_el is not None and title_el.text
                    else "unknown"
                )
                text = _text_of(sec)
                if text:
                    # Keep the first occurrence of a title (top-level section)
                    sections.setdefault(title, text)

        # Tables
        for tw in root.iter():
            if _localname(tw) == "table-wrap":
                table_text = _text_of(tw)
                if table_text:
                    tables.append(table_text)

        intro_text = sections.get("introduction", "")
        return {"sections": sections, "tables": tables, "intro_text": intro_text}


def _record_abstract(rec: dict) -> str:
    """Springer records carry the abstract as either a string or {p: ...}."""
    abs = rec.get("abstract", "")
    if isinstance(abs, dict):
        # occasionally structured; flatten any text values
        return " ".join(str(v) for v in abs.values() if isinstance(v, str)).strip()
    return str(abs).strip()
