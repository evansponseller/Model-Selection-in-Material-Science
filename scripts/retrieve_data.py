"""
retrieve_data.py — Steps 1 & 2 of the Metalearning_AlloyDesign pipeline.

Step 1: Query Scopus (Elsevier) and arXiv for alloy + ML papers.
Step 2: Fetch full text via publisher adapters, then classify each paper as
        'trained', 'finetuned', or 'unclear' with the Claude API.
        Papers classified as 'unclear' are discarded.

Output: data/paper_extractions.json
"""

import json
import re
import sys
import time
from pathlib import Path

# Ensure scripts/ is on sys.path so adapters package resolves correctly
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import requests

from adapters import ArxivAdapter, ElsevierAdapter, SpringerAdapter
from config import (
    AI_GATEWAY_API_KEY,
    AI_GATEWAY_URL,
    ARXIV_MAX_RESULTS,
    CLASSIFY_INTRO_SENTENCES,
    CLASSIFY_MODEL,
    ELSEVIER_API_KEY,
    HTTP_REFERER,
    MAX_RETRIES,
    MAX_SCOPUS_RESULTS,
    OUTPUT_JSON,
    RATE_LIMIT_PAUSE,
    RATE_LIMIT_RETRY_PAUSE,
    SPRINGER_API_KEY,
    SPRINGER_MAX_RESULTS,
)

# ── Sentence utilities ────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def get_last_n_sentences(text: str, n: int = CLASSIFY_INTRO_SENTENCES) -> list[str]:
    """Return the last n sentences of text (chronological order)."""
    sentences = _split_sentences(text)
    return sentences[-n:]


# ── LLM classification (compat route — known working) ────────────────────

CLASSIFY_PROMPT = """\
You are classifying a materials science paper.

Respond with exactly one word:
- "trained"   — the authors trained or developed their OWN ML model, interatomic potential,
                neural network potential, force field, or surrogate model FROM SCRATCH on data
                (includes: fitting/training MLIPs, developing machine-learned potentials,
                training random forests, neural networks, Gaussian processes, etc.)
- "finetuned" — the authors started FROM a pre-trained model, foundation model, or universal
                potential (e.g. a published MLIP, ORB, MACE-MP, a universal force field) and
                fine-tuned, adapted, or retrained it on their own data. Language like
                "fine-tune", "fine-tuning", "adapted the pre-trained", "retrained on" → finetuned.
- "unclear"   — the paper does NOT train or fine-tune any model of its own. This INCLUDES
                papers that merely APPLY/USE an existing pre-trained model or off-the-shelf
                potential without further training, as well as pure reviews and purely
                experimental work that only cites ML.

Key distinction: building or fitting a model on data = "trained"/"finetuned".
Only running someone else's already-trained model off the shelf = "unclear".
When genuinely ambiguous between "trained" and "unclear", prefer "trained" only if the
text explicitly mentions fitting, training, developing, or constructing a model — not
merely using or applying one.

Abstract:
{abstract}

Last sentences of introduction:
{intro_sentences}

Classification:"""


def classify_paper(abstract: str, intro_text: str) -> str:
    """Return 'trained', 'finetuned', or 'unclear' via the compat route."""
    if not AI_GATEWAY_API_KEY:
        raise EnvironmentError("JHU_AI_GATEWAY_API_KEY is not set.")

    intro_sentences = get_last_n_sentences(intro_text)
    prompt = CLASSIFY_PROMPT.format(
        abstract=abstract[:1500],
        intro_sentences="\n".join(intro_sentences),
    )
    payload = {
        "model": CLASSIFY_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
    }
    headers = {
        "Authorization": f"Bearer {AI_GATEWAY_API_KEY}",
        "HTTP-Referer": HTTP_REFERER,
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(AI_GATEWAY_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429:
                wait = RATE_LIMIT_RETRY_PAUSE * attempt
                print(f"  [classify] Rate limited. Waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            label = resp.json()["choices"][0]["message"]["content"].strip().lower()
            if label in {"trained", "finetuned", "unclear"}:
                return label
            for valid in ("trained", "finetuned", "unclear"):
                if valid in label:
                    return valid
            return "unclear"
        except requests.RequestException as exc:
            wait = RATE_LIMIT_RETRY_PAUSE * attempt
            print(f"  [classify] Network error (attempt {attempt}): {exc}. Waiting {wait}s …")
            time.sleep(wait)

    print("  [classify] All retries exhausted. Defaulting to 'unclear'.")
    return "unclear"


# ── Deduplication ─────────────────────────────────────────────────────────

def load_existing_ids(path: Path) -> set[str]:
    """Return the set of paper_ids (DOIs + arXiv IDs) already in the output JSON."""
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    ids: set[str] = set()
    for p in data:
        if "doi" in p:
            ids.add(p["doi"])
        if "paper_id" in p:
            ids.add(p["paper_id"])
    return ids


# ── Main pipeline ─────────────────────────────────────────────────────────

def run():
    existing_ids = load_existing_ids(OUTPUT_JSON)
    if existing_ids:
        print(f"Skipping {len(existing_ids)} paper IDs already in {OUTPUT_JSON.name}.\n")

    kept: list[dict] = []
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON) as f:
            kept = json.load(f)

    # Collect candidates from all adapters
    adapters = []
    if ELSEVIER_API_KEY:
        adapters.append(("Elsevier/Scopus", ElsevierAdapter(), MAX_SCOPUS_RESULTS))
    else:
        print("ELSEVIER_API_KEY not set — skipping Elsevier adapter.\n")

    if SPRINGER_API_KEY:
        adapters.append(("Springer Nature", SpringerAdapter(), SPRINGER_MAX_RESULTS))
    else:
        print("SPRINGER_API_KEY not set — skipping Springer adapter.\n")

    # arXiv is always attempted (no key required)
    adapters.append(("arXiv", ArxivAdapter(), ARXIV_MAX_RESULTS))

    all_candidates: list[dict] = []
    for source_name, adapter, max_results in adapters:
        print(f"Searching {source_name} …")
        candidates = adapter.search(max_results=max_results)
        print(f"  Found {len(candidates)} candidates from {source_name}.\n")
        all_candidates.extend(candidates)

    # Deduplicate across sources before processing
    seen_in_run: set[str] = set()
    unique_candidates: list[dict] = []
    for paper in all_candidates:
        pid = paper.get("paper_id", "")
        doi = paper.get("doi", "")
        key = doi or pid
        if key and key not in seen_in_run:
            seen_in_run.add(key)
            unique_candidates.append(paper)

    print(f"Total unique candidates: {len(unique_candidates)}\n")

    # Instantiate adapters by source for full-text fetching
    adapter_map: dict[str, object] = {}
    if ELSEVIER_API_KEY:
        adapter_map["elsevier"] = ElsevierAdapter()
    if SPRINGER_API_KEY:
        adapter_map["springer"] = SpringerAdapter()
    adapter_map["arxiv"] = ArxivAdapter()

    for i, paper in enumerate(unique_candidates, 1):
        pid = paper.get("paper_id", "")
        doi = paper.get("doi", pid)
        source = paper.get("source", "unknown")
        print(f"[{i}/{len(unique_candidates)}] [{source}] {paper['title'][:80]}")

        # Skip already-processed papers
        if pid in existing_ids or (doi and doi in existing_ids):
            print("  → already processed, skipping.\n")
            continue

        # Fetch full text via the appropriate adapter
        adapter = adapter_map.get(source)
        if adapter is None:
            print(f"  → no adapter for source '{source}', skipping.\n")
            continue

        print(f"  Fetching full text …")
        full = adapter.fetch_full_text(paper)
        if full is None:
            print("  → could not fetch full text, skipping.\n")
            existing_ids.add(pid)
            continue

        # Introduction is excluded from extraction sections but needed for classification
        intro_text = full["sections"].get("introduction", "") or full.get("intro_text", "")
        label = classify_paper(paper["abstract"], intro_text)
        print(f"  Classification: {label}")

        if label == "unclear":
            print("  → discarded.\n")
            existing_ids.add(pid)
            if doi:
                existing_ids.add(doi)
            continue

        record = {
            "title": paper["title"],
            "paper_id": pid,
            "doi": doi,
            "source": source,
            "journal": paper.get("journal", ""),
            "year": paper.get("year", ""),
            "abstract": paper["abstract"],
            "ml_category": label,
            "sections": full["sections"],
            "tables": full["tables"],
        }
        kept.append(record)
        existing_ids.add(pid)
        if doi:
            existing_ids.add(doi)

        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(kept, f, indent=2)
        print(f"  → saved. Total kept: {len(kept)}\n")

        time.sleep(RATE_LIMIT_PAUSE)

    print(f"\nDone. {len(kept)} papers saved to {OUTPUT_JSON}.")


if __name__ == "__main__":
    run()
