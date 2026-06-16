#!/usr/bin/env python3
"""Compare freshly produced eval CSVs against the reported benchmark numbers.

Run after `run_eval_detect.sh` / `run_eval_segment.sh`. It scans the validation
results dir, matches CSVs by their `-n` note, and prints actual vs expected means
for CIDEr / F1 / accuracy / LLM-score.

    python tools/compare_eval.py --model detection
    python tools/compare_eval.py --model segmentation
"""
import argparse
import glob
import os
import sys

import pandas as pd

from radgrounder.paths import VALIDATION_RESULTS_DIR

# Reported means for the released checkpoints. SLAKE + combined VQA-RAD are the paper's
# bootstrap means (10k resamples); VQA-RAD open/closed are derived from the early eval
# CSVs (the combined run's per-sample scores grouped by q_type).
EXPECTED = {
    "detection": {
        "slake_vqa_vqa_open":   {"cider": 3.00, "f1": 0.87, "accuracy": 0.83, "llm_score": 4.52},
        "slake_vqa_vqa_closed": {"cider": 2.33, "f1": 0.91, "accuracy": 0.91, "llm_score": 4.65},
        "vqa_rad_vqa":          {"cider": 0.99, "f1": 0.50, "accuracy": 0.44, "llm_score": 3.22},
        "vqa_rad_vqa_open":     {"cider": 0.625, "f1": 0.281, "accuracy": 0.149, "llm_score": 2.63},
        "vqa_rad_vqa_closed":   {"cider": 1.250, "f1": 0.659, "accuracy": 0.648, "llm_score": 3.64},
    },
    "segmentation": {
        "slake_vqa_vqa_open":   {"cider": 3.02, "f1": 0.86, "accuracy": 0.82, "llm_score": 4.49},
        "slake_vqa_vqa_closed": {"cider": 2.29, "f1": 0.90, "accuracy": 0.90, "llm_score": 4.59},
        "vqa_rad_vqa":          {"cider": 1.00, "f1": 0.50, "accuracy": 0.45, "llm_score": 3.30},
        "vqa_rad_vqa_open":     {"cider": 0.552, "f1": 0.268, "accuracy": 0.139, "llm_score": 2.76},
        "vqa_rad_vqa_closed":   {"cider": 1.307, "f1": 0.670, "accuracy": 0.660, "llm_score": 3.68},
    },
}
METRICS = ["cider", "f1", "accuracy", "llm_score"]


def latest_csv(model: str, note: str):
    # Match on the note (anchored with "_20" — the timestamp — so "vqa_rad_vqa" doesn't
    # also match "vqa_rad_vqa_open"), then split detection vs segmentation by filename.
    # eval_groundedgemma names files "segmentation_eval_..." (run name prefix);
    # eval_detectgemma uses an md5 hash prefix. So segmentation files start with
    # "segmentation" and detection files do not — works whether or not the note is
    # additionally model-tagged.
    pattern = os.path.join(str(VALIDATION_RESULTS_DIR), f"*tested_on_{note}_20*.csv")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    is_seg = lambda p: os.path.basename(p).startswith("segmentation")
    if model == "segmentation":
        matches = [m for m in matches if is_seg(m)]
    else:
        matches = [m for m in matches if not is_seg(m)]
    return matches[-1] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(EXPECTED), required=True)
    args = ap.parse_args()

    print(f"Results dir: {VALIDATION_RESULTS_DIR}\n")
    header = f"{'split':22} {'metric':10} {'actual':>8} {'expected':>9} {'diff':>8}"
    print(header)
    print("-" * len(header))

    missing = False
    for note, exp in EXPECTED[args.model].items():
        csv = latest_csv(args.model, note)
        if not csv:
            print(f"{note:22} (no CSV found — run the eval first)")
            if exp:  # only the rows with published targets gate reproducibility
                missing = True
            continue
        df = pd.read_csv(csv)
        for m in METRICS:
            if m not in df.columns:
                print(f"{note:22} {m:10} {'n/a':>8} (column missing)")
                continue
            actual = pd.to_numeric(df[m], errors="coerce").mean()
            e = exp.get(m)
            if e is None:  # info-only row (no published target)
                print(f"{note:22} {m:10} {actual:8.3f} {'-':>9} {'-':>8}")
            else:
                print(f"{note:22} {m:10} {actual:8.3f} {e:9.2f} {actual - e:+8.3f}")
        print()

    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
