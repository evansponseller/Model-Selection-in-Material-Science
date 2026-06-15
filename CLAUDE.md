# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CLAUDE.md — Metalearning_AlloyDesign
Project by Johns Hopkins University, 2026.
Automated pipeline to build a knowledge base of ML methodology from alloy
research papers, as the foundation for a meta-learning model recommender.

---

## What this project is trying to do

Materials scientists use many ML models (Random Forest, Gaussian Process,
Neural Networks, MLIPs, etc.) to predict alloy properties. Choosing the wrong
model wastes time and reduces accuracy. The long-term goal is a meta-classifier
that recommends the best model for a new task, trained on structured metadata
extracted from the literature.

This repo covers Phase 1: automated knowledge-base construction.

**Phase roadmap:**
1. Build structured knowledge base from literature (this repo, current work)
2. Train and compare base ML models on the extracted datasets
3. Train the meta-classifier on meta-features
4. Incorporate physical constraints and domain priors
5. Analysis and writing

---

## Repository structure

```
Metalearning_AlloyDesign/
├── scripts/
│   ├── config.py             # Centralized config (keys, paths, model names, limits)
│   ├── retrieve_data.py      # Step 1+2 — Scopus search + LLM classification → JSON
│   └── extract_data.py       # Step 3 — RAG + LLM field extraction → CSV
├── data/
│   └── paper_extractions.json  # 29 papers, full text + tables + ml_category
├── results/
│   ├── claude_extracted_results.csv   # Best results (agentic Claude Code run)
│   ├── codex_extracted_results.csv    # High accuracy (agentic Codex run)
│   ├── claude_notes.json              # Per-paper extraction notes from Claude
│   └── (extracted_results.csv)        # Automated Llama baseline if you run step 3
├── bib/                      # Reference PDFs (Brazdil metalearning book, surveys)
├── requirements.txt
├── readme.md
└── CLAUDE.md                 # This file
```

---

## How the pipeline works

### retrieve_data.py — Step 1 + 2
1. Queries Scopus API for alloy + ML papers in *Acta Materialia* after 2019
2. Fetches full XML text (sections + tables) via Elsevier Article Retrieval API
3. Classifies each paper as `trained`, `finetuned`, or `unclear` using Llama
   3.3-70B via OpenRouter — based on abstract + last 12 sentences of intro
4. Discards `unclear` papers (those that only cite ML, don't train it)
5. Saves kept papers to `data/paper_extractions.json`

**Key config constants** (in `config.py`):
- `SCOPUS_QUERY` — the search string; don't change without re-running
- `LLM_MODEL` — currently `meta-llama/llama-3.3-70b-instruct` via OpenRouter
- `ARTICLE_URL` — uses DOI: `https://api.elsevier.com/content/article/doi/{doi}`
- `ARTICLE_PII_URL` — uses PII: `https://api.elsevier.com/content/article/pii/{pii}`
- `OUTPUT_FILE` — `../data/paper_extractions.json`

**Environment variables required:**
```bash
export ELSEVIER_API_KEY=<key from dev.elsevier.com>
export JHU_AI_GATEWAY_API_KEY=<key from gateway.engineering.jhu.edu>
```

### extract_data.py — Step 3
For each paper in the JSON:
1. Scores every sentence in every non-boilerplate section against
   field-specific keywords (RAG retrieval)
2. Sends the top-scoring sentences (≤3000 chars) as context to the LLM
3. Asks one focused question per field
4. Writes results row-by-row to CSV (crash-safe)

**Fields extracted per paper:**
`title`, `authors`, `year`, `ml_category`, `ml_models`, `target_properties`,
`dataset_size`, `data_type`, `features`, `n_features_total`, `confidence`

**Sections always skipped** (they cite others' methods, not this paper's):
abstract, introduction, related work, acknowledgements, references, etc.

**Environment variables required:**
```bash
export JHU_AI_GATEWAY_API_KEY=<key from gateway.engineering.jhu.edu>
```

---

## Known bugs and issues

### High priority
1. **`n_features_total` is wrong** — currently counts commas in the features
   string. Breaks when the LLM returns a descriptive sentence instead of a
   clean list. Should parse the list properly, or ask the LLM for a count
   separately.
2. **`HTTP-Referer` header is a placeholder** — both scripts had:
   ```python
   "HTTP-Referer": "https://github.com/your-repo"
   ```
   Fixed to `https://github.com/Metalearning_AlloyDesign`.
3. **No DOI deduplication** — re-running the pipeline re-fetches and
   re-classifies papers already in the JSON. Should skip DOIs already present.

### Medium priority
4. **No retry logic beyond 429s** — `extract_data.py` has a basic 30s wait on
   rate limits but no exponential backoff. Network errors crash the run.
5. **No confidence scoring** — extractions have no flag for low-certainty
   results. Added a `confidence` column.
6. **Config is scattered** — query string, model name, rate limits, and file
   paths are hardcoded constants across both files. Moved to `config.py`.

### Low priority
7. **`get_last_n_sentences` reverses the sentences** before sending to the LLM
   (most recent first). This is unusual and may confuse some models — worth
   testing without reversal.

---

## Goals for improvement

### 1. Expand to other publishers
The pipeline currently only works with Elsevier (Scopus + ScienceDirect).
Target additions in priority order:
- **arXiv** (free, no key needed) — use the `arxiv` Python package;
  query `cond-mat` + `materials-science` categories
- **Springer/Nature** — Springer API; covers *npj Computational Materials*,
  *Nature Materials*
- **Wiley** — covers *Advanced Materials*, *Materials Science & Engineering*

**Architecture needed:** a publisher adapter layer — a shared interface that
`retrieve_data.py` calls, so adding a new source means writing one new adapter
class rather than duplicating the whole script.

**PII support** — Elsevier articles can also be fetched directly by PII
(visible in ScienceDirect URLs, e.g. `S0927025620300902`). The API supports:
```
https://api.elsevier.com/content/article/pii/{pii}
```

### 2. Replace OpenRouter with Claude API directly
The `anthropic` SDK (v0.100.0) is already in `requirements.txt`.

**Classification (retrieve_data.py):** swap `classify_paper_with_llm()` to use
`anthropic.Anthropic().messages.create()` with `claude-haiku-4-5-20251001`.

**Extraction (extract_data.py):** swap to `claude-sonnet-4-6` and use tool use
/ structured output to enforce the JSON schema. Benefits:
- All 6 fields extracted in a single API call per paper (not 5-6 separate calls)
- Schema enforcement means no more regex post-processing
- Add an `evidence` field (quote the source sentence)

**Example extraction schema to enforce:**
```json
{
  "ml_models": "string",
  "target_properties": "string",
  "dataset_size": "integer or null",
  "data_type": "experimental | synthetic | both | NR",
  "features": ["list", "of", "feature", "names"],
  "num_features": "integer or null",
  "evidence": {
    "ml_models": "quoted sentence from paper",
    "dataset_size": "quoted sentence from paper"
  }
}
```

### 3. Broader queries
Current query is limited to *Acta Materialia*. Widen to:
- Other Elsevier journals: *Acta Materialia*, *Scripta Materialia*,
  *Computational Materials Science*, *Journal of Nuclear Materials*,
  *Materials Today Communications*
- Keyword expansions: `"high-entropy alloys"`, `"refractory alloys"`,
  `"nuclear materials"`, `"interatomic potential"` alongside `"machine learning"`

---

## Results summary (current knowledge base)

- **29 papers** kept from *Acta Materialia* (2020–2026), all trained/finetuned
- **1 paper** discarded (unclear classification)
- **3 extraction approaches** compared:

| Approach | Output file | Notes |
|---|---|---|
| Agentic Claude Code | `claude_extracted_results.csv` | Highest accuracy |
| Agentic OpenAI Codex | `codex_extracted_results.csv` | High accuracy |
| Automated RAG + Llama 70B | run `extract_data.py` | Automated baseline |

**ML model families found in the corpus:**
MLIPs (MACE, MTP, NNP), CNN, Random Forest, XGBoost, Gradient Boosting,
Gaussian Process Regression, Deep Neural Networks / MLP, Linear Regression

**Dataset sizes:** ~8K to 85K data points

**Data types:** experimental, computational (DFT/MD/CALPHAD), mixed

**Target properties:** strength, liquidus temperature, bandgap, SRO, etc.

---

## Reference papers (bib/)

- `978-3-540-73263-1.pdf` — Brazdil et al. (2009), *Metalearning: Applications
  to Data Mining* — the core theoretical reference for the project
- `1703.04977v2.pdf` — survey on meta-learning
- `2003.04919v6.pdf` — survey on meta-learning
- `2005.00707v2.pdf` — related survey
- `dann24a.pdf` — related work
- `s41586-018-0337-2.pdf` — foundational materials ML paper
- `AI_OCR_Paper (1).pdf` — related work

**External review papers of interest (not in bib, fetch via Elsevier API):**
- `S0927025620300902` — ML for alloy design (broad review, *Comp. Mat. Sci.*)
- `S0022311519306671` — ML in nuclear materials (*J. Nuclear Materials*)
- `S2352492820328828` — ML for HEA yield strength (*Mat. Today Comm.*)

Note: these three are review papers — they will be classified as `unclear`
by the pipeline because they don't train their own models. Use them as
background reading / bib references, not as knowledge-base data sources.
