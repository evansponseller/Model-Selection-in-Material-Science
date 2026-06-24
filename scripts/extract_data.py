"""
extract_data.py — Step 3 of the Metalearning_AlloyDesign pipeline.

For each paper in data/paper_extractions.json:
  1. Score every sentence in non-boilerplate sections against field keywords
     (RAG-style retrieval).
  2. Combine the top-scoring sentences into one context block.
  3. Make a single Claude API call with tool use to extract all fields at once.
  4. Write results row-by-row to results/extracted_results.csv (crash-safe).

Fields extracted per paper:
  title, authors, year, ml_category, ml_models, target_properties,
  dataset_size, data_type, features, num_features, confidence
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

# Ensure scripts/ is on sys.path so adapters/config resolve correctly
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import json as _json

import requests

from config import (
    AI_GATEWAY_API_KEY,
    AI_GATEWAY_URL,
    BOILERPLATE_SECTIONS,
    EXTRACT_MODEL,
    EXTRACTION_FIELDS,
    FIELD_KEYWORDS,
    HTTP_REFERER,
    INPUT_JSON,
    MAX_CONTEXT_CHARS,
    MAX_RETRIES,
    OUTPUT_CSV,
    RATE_LIMIT_PAUSE,
    RATE_LIMIT_RETRY_PAUSE,
)

CSV_COLUMNS = [
    "title", "authors", "year", "doi", "ml_category",
    "ml_models", "ml_models_quote",
    "target_properties", "target_properties_quote",
    "dataset_size", "dataset_size_quote",
    "data_type", "data_type_quote",
    "features", "features_quote",
    "num_features",
    "performance_metric", "performance_metric_quote",
    "metric_value", "metric_type",
    "confidence",
]


# Phrases the LLM uses when it cannot find the answer (kept for is_missing checks)
NOT_FOUND_PHRASES = {
    "nr", "not reported", "not mentioned", "not found", "not stated", "n/a",
    "not explicitly listed", "not explicitly stated", "not listed", "not specified",
    "not provided", "not given", "not available", "not applicable",
}


# ── Sentence scoring (RAG retrieval) ─────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def retrieve_context(sections: dict[str, str], field: str) -> str:
    """
    Score every sentence in non-boilerplate sections against field keywords,
    return the top-scoring sentences up to MAX_CONTEXT_CHARS.
    """
    keywords = [kw.lower() for kw in FIELD_KEYWORDS.get(field, [])]
    scored: list[tuple[float, str]] = []

    for section_title, text in sections.items():
        if section_title.lower() in BOILERPLATE_SECTIONS:
            continue
        for sentence in _split_sentences(text):
            lower = sentence.lower()
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    context_parts: list[str] = []
    total = 0
    for _, sentence in scored:
        if total + len(sentence) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(sentence)
        total += len(sentence) + 1

    return " ".join(context_parts)


# ── Claude extraction via structured JSON prompt ──────────────────────────

_EXTRACTION_PROMPT = """\
You are extracting structured metadata from a materials science paper about machine learning.

Below are the most relevant sentences from the paper, grouped by the field they address.
Extract all fields using ONLY information found in these sentences.

{context_block}

Reply with ONLY a valid JSON object — no markdown, no preamble, no trailing text.
Use exactly these keys:

{{
  "ml_models": "<comma-separated model names, or NR>",
  "ml_models_quote": "<exact verbatim sentence from the context that best supports ml_models, or NR>",
  "target_properties": "<comma-separated property names, or NR. For MLIP papers: physical behavior studied, NOT energies/forces>",
  "target_properties_quote": "<exact verbatim sentence from the context that best supports target_properties, or NR>",
  "dataset_size": <integer or null>,
  "dataset_size_quote": "<exact verbatim sentence from the context that best supports dataset_size, or NR>",
  "data_type": "<experimental | computational | both | NR>",
  "data_type_quote": "<exact verbatim sentence from the context that best supports data_type, or NR>",
  "features": "<comma-separated feature names, or NR>",
  "features_quote": "<exact verbatim sentence from the context that best supports features, or NR>",
  "num_features": <integer or null>,
  "performance_metric": "<best reported metric string, e.g. 'R²=0.95' or 'RMSE=0.12 eV/Å', or NR>",
  "performance_metric_quote": "<exact verbatim sentence from the context that best supports performance_metric, or NR>",
  "metric_value": <numeric value as float, e.g. 0.95 for R²=0.95 or 0.12 for RMSE=0.12, or null>,
  "metric_type": "<R2 | RMSE | MAE | MAPE | accuracy | other | NR>",
  "confidence": "<high | medium | low>"
}}

Rules:
- For every _quote field: copy the sentence VERBATIM from the context above — do not paraphrase or summarize. If no sentence supports that field, use NR.
- data_type: 'computational' = DFT/MD/CALPHAD only. 'both' = training set explicitly mixes lab + simulation data. DFT-trained MLIP run in MD is still 'computational'.
- performance_metric: report the BEST result the authors highlight for their primary model. Prefer test/validation metrics over training metrics. For MLIPs, RMSE on energy/force is fine.
- metric_value: extract just the number (e.g. 0.95 from R²=0.95, 12.3 from RMSE=12.3 MPa).
- metric_type: use R2 for R²/coefficient of determination, RMSE for root-mean-square error, MAE for mean absolute error.
- confidence: high = all fields found, medium = 1-2 null/NR fields, low = 3+ null/NR fields.
"""


def extract_with_claude(sections: dict[str, str]) -> dict:
    """
    Run RAG retrieval for all fields, then make ONE API call with function/tool use
    via the compat route to extract all fields simultaneously.
    """
    if not AI_GATEWAY_API_KEY:
        raise EnvironmentError("JHU_AI_GATEWAY_API_KEY is not set.")

    _NR_RESULT = {
        "ml_models": "NR", "ml_models_quote": "NR",
        "target_properties": "NR", "target_properties_quote": "NR",
        "dataset_size": "NR", "dataset_size_quote": "NR",
        "data_type": "NR", "data_type_quote": "NR",
        "features": "NR", "features_quote": "NR",
        "num_features": "NR",
        "performance_metric": "NR", "performance_metric_quote": "NR",
        "metric_value": "NR", "metric_type": "NR",
        "confidence": "low",
    }

    context_parts: list[str] = []
    for field in EXTRACTION_FIELDS:
        ctx = retrieve_context(sections, field)
        if ctx:
            context_parts.append(f"[{field}]\n{ctx}")

    if not context_parts:
        return _NR_RESULT

    context_block = "\n\n".join(context_parts)
    prompt = _EXTRACTION_PROMPT.format(context_block=context_block)

    headers = {
        "Authorization": f"Bearer {AI_GATEWAY_API_KEY}",
        "HTTP-Referer": HTTP_REFERER,
        "Content-Type": "application/json",
    }
    payload = {
        "model": EXTRACT_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(AI_GATEWAY_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code == 429:
                wait = RATE_LIMIT_RETRY_PAUSE * attempt
                print(f"    [extract] Rate limited. Waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences if present
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            args = _json.loads(text)
            def _str(v) -> str:
                return str(v) if v is not None and str(v).strip() else "NR"

            return {
                "ml_models":                  _str(args.get("ml_models")),
                "ml_models_quote":            _str(args.get("ml_models_quote")),
                "target_properties":          _str(args.get("target_properties")),
                "target_properties_quote":    _str(args.get("target_properties_quote")),
                "dataset_size":               str(args["dataset_size"]) if args.get("dataset_size") is not None else "NR",
                "dataset_size_quote":         _str(args.get("dataset_size_quote")),
                "data_type":                  _str(args.get("data_type")),
                "data_type_quote":            _str(args.get("data_type_quote")),
                "features":                   _str(args.get("features")),
                "features_quote":             _str(args.get("features_quote")),
                "num_features":               str(args["num_features"]) if args.get("num_features") is not None else "NR",
                "performance_metric":         _str(args.get("performance_metric")),
                "performance_metric_quote":   _str(args.get("performance_metric_quote")),
                "metric_value":               str(args["metric_value"]) if args.get("metric_value") is not None else "NR",
                "metric_type":                _str(args.get("metric_type")),
                "confidence":                 _str(args.get("confidence")) if args.get("confidence") else "low",
            }
        except _json.JSONDecodeError as exc:
            print(f"    [extract] JSON parse error (attempt {attempt}): {exc}.")
        except requests.RequestException as exc:
            wait = RATE_LIMIT_RETRY_PAUSE * attempt
            print(f"    [extract] Network error (attempt {attempt}): {exc}. Waiting {wait}s …")
            time.sleep(wait)

    print("    [extract] All retries exhausted. Returning NR for all fields.")
    return _NR_RESULT


# ── Per-paper extraction ──────────────────────────────────────────────────

def extract_paper(paper: dict) -> dict:
    """Run RAG + Claude tool-use extraction for all fields; return a result dict."""
    sections = paper.get("sections", {})
    extractions = extract_with_claude(sections)

    for field, val in extractions.items():
        print(f"    {field}: {str(val)[:80]}")

    return extractions


# ── CSV helpers ───────────────────────────────────────────────────────────

def load_processed_dois(output_path: Path) -> set[str]:
    """Return DOIs already written to the CSV (for crash-safe resumption)."""
    if not output_path.exists():
        return set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["doi"] for row in reader if "doi" in row}


def append_row(output_path: Path, row: dict):
    """Append a single result row to the CSV (creates file with header if needed)."""
    write_header = not output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Main pipeline ─────────────────────────────────────────────────────────

def run():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"{INPUT_JSON} not found. Run retrieve_data.py first."
        )

    with open(INPUT_JSON) as f:
        papers = json.load(f)

    processed = load_processed_dois(OUTPUT_CSV)
    if processed:
        print(f"Resuming: {len(processed)} papers already in {OUTPUT_CSV.name}.\n")

    for i, paper in enumerate(papers, 1):
        doi = paper.get("doi", f"unknown_{i}")
        title = paper.get("title", "")[:80]
        print(f"[{i}/{len(papers)}] {title}")

        if doi in processed:
            print("  → already extracted, skipping.\n")
            continue

        extractions = extract_paper(paper)

        row = {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year", ""),
            "doi": doi,
            "ml_category": paper.get("ml_category", ""),
            **extractions,
        }
        append_row(OUTPUT_CSV, row)
        processed.add(doi)
        print(f"  → written. Confidence: {extractions['confidence']}\n")

        time.sleep(RATE_LIMIT_PAUSE)

    print(f"\nDone. Results saved to {OUTPUT_CSV}.")


if __name__ == "__main__":
    run()
