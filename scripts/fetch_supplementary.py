"""
fetch_supplementary.py — Backfill supplementary-material text into the corpus.

For each Elsevier paper in data/paper_extractions.json:
  - downloads its supplementary files (mmcN objects: PDF / DOCX)
  - extracts the text and stores it under sections["supplementary"]

This makes feature lists, dataset details, and other info that authors put in
supplementary notes (reviewer-flagged) available to the extraction RAG step,
since EXTRACT_SKIP_SECTIONS keeps the "supplementary" section.

Crash-safe: writes after every paper and skips papers that already have a
non-empty sections["supplementary"]. Re-run freely.

Run:  python3 scripts/fetch_supplementary.py
"""

import json
import os
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import INPUT_JSON, RATE_LIMIT_PAUSE
from adapters.elsevier import ElsevierAdapter


def _atomic_dump(papers: list, path: Path):
    """Write to a temp file then rename, so a crash mid-write can't corrupt the corpus."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run():
    with open(INPUT_JSON) as f:
        papers = json.load(f)

    adapter = ElsevierAdapter()
    filled = 0

    for i, paper in enumerate(papers, 1):
        doi = paper.get("doi", "")
        title = paper.get("title", "")[:60]
        print(f"[{i}/{len(papers)}] {title}")

        sections = paper.setdefault("sections", {})
        if sections.get("supplementary", "").strip():
            print("  → already has supplementary, skipping.\n")
            continue
        # arXiv papers won't resolve against the Elsevier API
        if not doi or paper.get("source") == "arxiv":
            print("  → no Elsevier DOI, skipping.\n")
            continue

        text = adapter.fetch_supplementary_text(doi)
        if text.strip():
            sections["supplementary"] = text
            filled += 1
            print(f"  → added {len(text)} chars of supplementary text.\n")
        else:
            print("  → no extractable supplementary materials.\n")

        # Crash-safe: persist atomically after every paper
        _atomic_dump(papers, INPUT_JSON)
        time.sleep(RATE_LIMIT_PAUSE)

    print(f"Done. Supplementary text added to {filled} papers.")


if __name__ == "__main__":
    run()
