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
    EXTRACT_SKIP_SECTIONS,
    EXTRACT_MODEL,
    EXTRACT_FALLBACK_MODEL,
    EXTRACTION_FIELDS,
    EXCLUDED_CSV,
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
    "num_features", "num_features_explanation",
    "performance_metric", "performance_metric_quote",
    "metric_value", "metric_type",
    "confidence", "confidence_explanation",
    "recommend_exclude", "exclude_reason",
    "extraction_model",
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


def intro_overview(sections: dict[str, str], n: int = 6) -> str:
    """
    Return the closing sentences of the introduction — where papers typically
    state their contributions, core ML approach, and the paper's structure.
    Used as a multi-stage prime: the LLM reads this first to orient itself.
    """
    intro = sections.get("introduction", "")
    sentences = _split_sentences(intro)
    return " ".join(sentences[-n:]) if sentences else ""


def retrieve_context(sections: dict[str, str], field: str) -> str:
    """
    Score every sentence in non-boilerplate sections against field keywords,
    return the top-scoring sentences up to MAX_CONTEXT_CHARS.
    """
    keywords = [kw.lower() for kw in FIELD_KEYWORDS.get(field, [])]
    scored: list[tuple[float, str]] = []

    for section_title, text in sections.items():
        if section_title.lower() in EXTRACT_SKIP_SECTIONS:
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


# ── Per-item value/quote formatting ───────────────────────────────────────

def _format_items(arr) -> tuple[str, str]:
    """
    Turn a list of {"value", "quote"} objects into (values, quotes) strings for the CSV.
    - values:  comma-separated item names           e.g. "Random Forest, XGBoost"
    - quotes:  one line per item, each with its own verbatim quote, e.g.
                 Random Forest — "We trained a random forest …"
                 XGBoost — "An XGBoost model was also fit …"
    Returns ("NR", "NR") if there are no supportable items.
    """
    if not isinstance(arr, list):
        # Back-compat: a bare string still works, just without per-item quotes
        s = str(arr).strip() if arr is not None else ""
        return (s, "NR") if s and s.upper() != "NR" else ("NR", "NR")

    values, lines = [], []
    for item in arr:
        if isinstance(item, dict):
            value = str(item.get("value") or "").strip()
            quote = str(item.get("quote") or "").strip()
        else:
            value, quote = str(item).strip(), ""
        if not value:
            continue
        values.append(value)
        lines.append(f'{value} — "{quote}"' if quote and quote.upper() != "NR" else f"{value} — NR")

    if not values:
        return "NR", "NR"
    return ", ".join(values), "\n".join(lines)


# ── Claude extraction via structured JSON prompt ──────────────────────────

_EXTRACTION_PROMPT = """\
You are extracting structured metadata from a materials science paper about machine learning.

STEP 1 — Read [paper_overview] first (if present). These are the closing sentences of the
introduction, where authors state their contributions, the core ML method, and the paper's
structure. Use it to understand what the paper actually does before extracting anything.

STEP 2 — Then extract each field using ONLY information found in the sentences below, which
are grouped by the field they address.

{context_block}

Reply with ONLY a valid JSON object — no markdown, no preamble, no trailing text.
Use exactly these keys:

{{
  "ml_models": [{{"value": "<one model name>", "quote": "<exact verbatim sentence that directly supports THIS model>"}}, ...],
  "target_properties": [{{"value": "<one property>", "quote": "<exact verbatim sentence that directly supports THIS property>"}}, ...],
  "dataset_size": <integer or null>,
  "dataset_size_quote": "<exact verbatim sentence from the context that best supports dataset_size, or NR>",
  "data_type": "<experimental | computational | both | NR>",
  "data_type_quote": "<exact verbatim sentence from the context that best supports data_type, or NR>",
  "features": [{{"value": "<one feature/descriptor>", "quote": "<exact verbatim sentence that directly supports THIS feature>"}}, ...],
  "num_features": <integer or null>,
  "num_features_explanation": "<explain how you arrived at the num_features count — cite the relevant evidence from the context, e.g. which sentence named the features or stated the count explicitly, or how you inferred it from the features list>",
  "performance_metric": "<best reported metric string, e.g. 'R²=0.95' or 'RMSE=0.12 eV/Å', or NR>",
  "performance_metric_quote": "<exact verbatim sentence from the context that best supports performance_metric, or NR>",
  "metric_value": <numeric value as float, e.g. 0.95 for R²=0.95 or 0.12 for RMSE=0.12, or null>,
  "metric_type": "<R2 | RMSE | MAE | MAPE | accuracy | other | NR>",
  "confidence": "<high | medium | low>",
  "confidence_explanation": "<explain why you assigned this confidence level — list which fields are null/NR and why, or confirm all fields were found>",
  "recommend_exclude": "<yes | no>",
  "exclude_reason": "<if yes, a brief reason; else NR>"
}}

Rules:
- ml_models, target_properties, features are LISTS — one object per distinct item, each with its OWN verbatim quote.
  * Provide a separate quote for EACH item. A quote may only be reused if the same sentence genuinely names multiple of those items.
  * PRECISION over recall: include an item ONLY if you can supply a verbatim sentence from the context that DIRECTLY supports it. If a candidate item has no clear supporting sentence, OMIT it entirely (do not invent or guess). If a list would have no supportable items, return an empty list [].
  * Quotes must be copied VERBATIM from the context — never paraphrase or summarize.
- For every other _quote field (dataset_size_quote, data_type_quote, performance_metric_quote): copy the sentence VERBATIM; the quote must DIRECTLY state the value, else NR.
- features: report the INPUT variables/descriptors fed INTO the model. Do NOT report the optimization objectives, design goals, or target properties here (those belong in target_properties). If the paper only names design targets and never lists model inputs, return [].
- recommend_exclude (we prioritize ACCURACY over a complete dataset — when in doubt, exclude):
  * Set "yes" if the paper's only ML component is a machine-learned interatomic potential (MLIP) or
    force field used to drive molecular dynamics / molecular statics simulations — i.e. the model
    predicts energies/forces to run MD, rather than predicting an alloy property from a feature
    dataset. Note this in exclude_reason (e.g. "MLIP/force field for MD, not property prediction").
    Still fill in the other fields as best you can.
  * Set "yes" if the core fields are NOT clearly stated: you must have ml_models AND target_properties
    AND at least one of (dataset_size, features) clearly supported by the text. If not, exclude and
    say what is missing in exclude_reason.
  * Otherwise set "no" and exclude_reason = NR.
- data_type: classify by the source of the data the model was TRAINED ON — nothing else. 'experimental' = lab-measured/fabricated data. 'computational' = DFT/MD/CALPHAD-generated training data. 'both' = the training set explicitly mixes lab + simulation data. A passing mention of DFT/MD used only for validation, comparison, or cited from prior work does NOT make the data_type computational. If the training data is experimental and simulations are merely referenced, the answer is 'experimental'. A DFT-trained MLIP later run in MD is still 'computational'.
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
        "num_features": "NR", "num_features_explanation": "NR",
        "performance_metric": "NR", "performance_metric_quote": "NR",
        "metric_value": "NR", "metric_type": "NR",
        "confidence": "low", "confidence_explanation": "NR",
        "recommend_exclude": "NR", "exclude_reason": "NR",
        "extraction_model": "none",
    }

    context_parts: list[str] = []
    overview = intro_overview(sections)
    if overview:
        context_parts.append(f"[paper_overview]\n{overview}")
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

    def _str(v) -> str:
        return str(v) if v is not None and str(v).strip() else "NR"

    def _parse(text: str, model: str) -> dict:
        # Strip markdown code fences if present, then parse JSON
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        args = _json.loads(text)
        ml_models, ml_models_quote = _format_items(args.get("ml_models"))
        target_props, target_props_quote = _format_items(args.get("target_properties"))
        features, features_quote = _format_items(args.get("features"))
        return {
            "ml_models":                  ml_models,
            "ml_models_quote":            ml_models_quote,
            "target_properties":          target_props,
            "target_properties_quote":    target_props_quote,
            "dataset_size":               str(args["dataset_size"]) if args.get("dataset_size") is not None else "NR",
            "dataset_size_quote":         _str(args.get("dataset_size_quote")),
            "data_type":                  _str(args.get("data_type")),
            "data_type_quote":            _str(args.get("data_type_quote")),
            "features":                   features,
            "features_quote":             features_quote,
            "num_features":               str(args["num_features"]) if args.get("num_features") is not None else "NR",
            "num_features_explanation":   _str(args.get("num_features_explanation")),
            "performance_metric":         _str(args.get("performance_metric")),
            "performance_metric_quote":   _str(args.get("performance_metric_quote")),
            "metric_value":               str(args["metric_value"]) if args.get("metric_value") is not None else "NR",
            "metric_type":                _str(args.get("metric_type")),
            "confidence":                 _str(args.get("confidence")) if args.get("confidence") else "low",
            "confidence_explanation":     _str(args.get("confidence_explanation")),
            "recommend_exclude":          _str(args.get("recommend_exclude")),
            "exclude_reason":             _str(args.get("exclude_reason")),
            "extraction_model":           model.split("/")[-1],
        }

    # Try the primary model (Opus); if it's rate-limited right now, fall back to
    # the secondary model (Sonnet) for THIS paper only — the next paper starts
    # over on Opus. (max_429 = how many 429s to tolerate before falling back.)
    models = [
        (EXTRACT_MODEL,          2,           15),                       # Opus: brief, then fall back
        (EXTRACT_FALLBACK_MODEL, MAX_RETRIES, RATE_LIMIT_RETRY_PAUSE),   # Sonnet: full patience
    ]
    for model, max_429, wait_429 in models:
        name = model.split("/")[-1]
        payload = {"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": prompt}]}
        rl_hits = 0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(AI_GATEWAY_URL, json=payload, headers=headers, timeout=120)
                if resp.status_code == 429:
                    rl_hits += 1
                    if rl_hits >= max_429:
                        print(f"    [extract] {name} rate-limited — falling back to next model.")
                        break
                    print(f"    [extract] {name} rate limited. Waiting {wait_429}s …")
                    time.sleep(wait_429)
                    continue
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                if choice.get("finish_reason") == "length":
                    print("    [extract] Response hit max_tokens (truncated) — raise max_tokens.")
                return _parse(choice["message"]["content"].strip(), model)
            except _json.JSONDecodeError as exc:
                print(f"    [extract] {name} JSON parse error (attempt {attempt}): {exc}.")
            except requests.RequestException as exc:
                wait = RATE_LIMIT_RETRY_PAUSE * attempt
                print(f"    [extract] {name} network error (attempt {attempt}): {exc}. Waiting {wait}s …")
                time.sleep(wait)
        # fall through to the next model

    # Both models exhausted. Return None so the caller SKIPS this paper without
    # writing it — a re-run will retry it. (Writing an all-NR row would mark the
    # paper "done" and poison the dataset.)
    print("    [extract] All models exhausted — skipping (will retry on re-run).")
    return None


# ── Per-paper extraction ──────────────────────────────────────────────────

def extract_paper(paper: dict) -> dict | None:
    """Run RAG + Claude tool-use extraction for all fields; return a result dict
    (or None if the API call could not be completed — caller should skip)."""
    sections = paper.get("sections", {})
    extractions = extract_with_claude(sections)
    if extractions is None:
        return None

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

    # Resume across BOTH files: a paper already in either is done
    processed = load_processed_dois(OUTPUT_CSV) | load_processed_dois(EXCLUDED_CSV)
    if processed:
        print(f"Resuming: {len(processed)} papers already processed.\n")

    kept_n = excluded_n = skipped_n = 0
    for i, paper in enumerate(papers, 1):
        doi = paper.get("doi", f"unknown_{i}")
        title = paper.get("title", "")[:80]
        print(f"[{i}/{len(papers)}] {title}")

        if doi in processed:
            print("  → already extracted, skipping.\n")
            continue

        extractions = extract_paper(paper)
        if extractions is None:
            # API failure — do NOT write or mark processed; a re-run will retry.
            skipped_n += 1
            print("  → API failure, left for a later re-run.\n")
            time.sleep(RATE_LIMIT_PAUSE)
            continue

        row = {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year", ""),
            "doi": doi,
            "ml_category": paper.get("ml_category", ""),
            **extractions,
        }
        # Route flagged papers to the excluded file so the main results stay clean
        if str(extractions.get("recommend_exclude", "")).strip().lower() == "yes":
            append_row(EXCLUDED_CSV, row)
            excluded_n += 1
            print(f"  → EXCLUDED ({extractions.get('exclude_reason','')[:60]}).\n")
        else:
            append_row(OUTPUT_CSV, row)
            kept_n += 1
            print(f"  → written. Confidence: {extractions['confidence']}\n")

        processed.add(doi)
        time.sleep(RATE_LIMIT_PAUSE)

    msg = (f"\nDone. {kept_n} to review in {OUTPUT_CSV.name}, "
           f"{excluded_n} set aside in {EXCLUDED_CSV.name}.")
    if skipped_n:
        msg += f"\n{skipped_n} skipped due to API failures — re-run to retry them."
    print(msg)


if __name__ == "__main__":
    run()
