"""
reclassify.py — Re-run classification on the papers already in
data/paper_extractions.json using the current CLASSIFY_PROMPT / CLASSIFY_MODEL.

For each paper:
  - feeds its abstract + introduction to classify_paper()
  - prints old ml_category -> new label
Papers that now come back 'unclear' are MOVED to data/dropped_unclear.json
(the pipeline discards unclear papers). The original JSON is backed up to
data/paper_extractions.backup.json before anything is overwritten.

Run:  python3 scripts/reclassify.py
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import INPUT_JSON, DATA_DIR, RATE_LIMIT_PAUSE
from retrieve_data import classify_paper

BACKUP_JSON = DATA_DIR / "paper_extractions.backup.json"
DROPPED_JSON = DATA_DIR / "dropped_unclear.json"


def run():
    with open(INPUT_JSON) as f:
        papers = json.load(f)

    print(f"Reclassifying {len(papers)} papers with the updated prompt …\n")

    kept, dropped, changed = [], [], 0
    for i, p in enumerate(papers, 1):
        sections = p.get("sections", {})
        abstract = sections.get("abstract", "") or p.get("abstract", "")
        intro = sections.get("introduction", "")
        old = p.get("ml_category", "")

        new = classify_paper(abstract, intro)
        flag = "  <-- CHANGED" if new != old else ""
        print(f"[{i:2}/{len(papers)}] {p.get('title','')[:52]:<54} {old:>18} -> {new}{flag}")

        if new != old:
            changed += 1
        p["ml_category"] = new
        (dropped if new == "unclear" else kept).append(p)
        time.sleep(RATE_LIMIT_PAUSE)

    # Back up the original before overwriting
    shutil.copy(INPUT_JSON, BACKUP_JSON)

    def _atomic_dump(obj, path):
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    _atomic_dump(kept, INPUT_JSON)
    if dropped:
        _atomic_dump(dropped, DROPPED_JSON)

    print(f"\n{changed} papers changed label.")
    print(f"Kept: {len(kept)}  |  Dropped as unclear: {len(dropped)}")
    print(f"Backup written to {BACKUP_JSON.name}")
    if dropped:
        print(f"Dropped papers saved to {DROPPED_JSON.name}:")
        for p in dropped:
            print(f"  - {p.get('title','')[:70]}")


if __name__ == "__main__":
    run()
