# Metalearning_AlloyDesign

**Phase 1** of a meta-learning pipeline for ML model recommendation in alloy design.
Automated extraction of ML methodology metadata from the materials science literature.


---

## Setup

```bash
pip install -r requirements.txt

export ELSEVIER_API_KEY=<your key from dev.elsevier.com>
export JHU_AI_GATEWAY_API_KEY=<your key from gateway.engineering.jhu.edu>
```

---

## Running the pipeline

### Step 1 & 2 — Retrieve and classify papers

```bash
cd scripts
python retrieve_data.py
```

Queries Scopus, fetches full XML from Elsevier, classifies each paper as
`trained`, `finetuned`, or `unclear`, and writes kept papers to
`data/paper_extractions.json`.

Re-running is safe — DOIs already in the JSON are skipped automatically.

### Step 3 — Extract structured metadata

```bash
cd scripts
python extract_data.py
```

Reads `data/paper_extractions.json`, runs RAG + LLM extraction for each paper,
and appends rows to `results/extracted_results.csv`.

Re-running is safe — DOIs already in the CSV are skipped automatically.

---

## Output fields

| Field | Description |
|---|---|
| `title` | Paper title |
| `authors` | Author list |
| `year` | Publication year |
| `doi` | Digital Object Identifier |
| `ml_category` | `trained` or `finetuned` |
| `ml_models` | ML algorithms used |
| `target_properties` | Material properties predicted |
| `dataset_size` | Number of data points |
| `data_type` | `experimental`, `computational`, or `both` |
| `features` | Input feature/descriptor names |
| `n_features_total` | Total number of input features |
| `confidence` | Extraction quality: `high`, `medium`, `low` |

---

## Configuration

All tunable constants (query string, model names, rate limits, file paths)
are in `scripts/config.py`.

---

## Results

Pre-computed extraction results are in `results/`:
- `claude_extracted_results.csv` — agentic Claude Code run (highest accuracy)
- `codex_extracted_results.csv` — agentic OpenAI Codex run
- `extracted_results.csv` — automated RAG + Llama 70B baseline (from running step 3)
