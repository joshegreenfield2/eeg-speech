#!/usr/bin/env python3
"""run_pipeline.py — end-to-end batch: preprocess + baseline classify all 14 subjects.

Usage:
    uv run python scripts/run_pipeline.py            # all subjects
    uv run python scripts/run_pipeline.py MM05 MM08  # specific subjects
    uv run python scripts/run_pipeline.py --skip-preprocess  # only re-run classifier

Outputs:
    data/processed/<subject>-clean-epo.fif    — preprocessed epochs
    data/processed/<subject>-log.json         — preprocessing log
    outputs/results/run_YYYYMMDD_HHMMSS/      — classification results per subject per task
    outputs/summary_results.csv               — aggregate of all runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Make `src.*` importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.classifier import cross_val_classify, relabel_for_task, PHONOLOGICAL_TASKS
from src.evaluate import save_run, summary_csv
from src.features import band_power_features, DEFAULT_BANDS
from src.loader import EVENT_ID, list_available_subjects, SUBJECTS
from src.preprocessor import preprocess_subject, load_clean_epochs


def labels_from_epochs(epochs):
    """Recover string labels from the integer event codes."""
    inv = {v: k for k, v in EVENT_ID.items()}
    return np.array([inv[e] for e in epochs.events[:, 2]])


def run_one_subject(
    subject: str,
    run_id: str,
    skip_preprocess: bool = False,
    permutation_test: bool = False,
):
    """Preprocess + classify one subject. Returns dict of {task: result_or_error}."""
    t0 = time.time()
    print(f"\n{'='*60}\n{subject}\n{'='*60}")
    out: dict[str, dict] = {}

    # ---- Phase 2: preprocess ----
    if not skip_preprocess:
        try:
            t = time.time()
            print(f"  [preprocess] starting...")
            _, log = preprocess_subject(subject, skip_if_exists=True, verbose=False)
            print(f"  [preprocess] done in {time.time()-t:.0f}s — "
                  f"{log['n_epochs_after_reject']}/{log['n_epochs_before_reject']} epochs, "
                  f"{log['n_bad_channels']} bad channels, ICA rejected {len(log['ica_components_rejected'])}")
            out["preprocess"] = {"ok": True, "log": log}
        except Exception as e:
            print(f"  [preprocess] FAILED: {e}")
            traceback.print_exc()
            out["preprocess"] = {"ok": False, "error": str(e)}
            return out

    # ---- Phase 3: classify ----
    try:
        epochs, log = load_clean_epochs(subject)
    except Exception as e:
        out["classify"] = {"ok": False, "error": f"could not load preprocessed: {e}"}
        return out

    labels = labels_from_epochs(epochs)
    n_classes_observed = len(set(labels))
    if n_classes_observed < 2:
        out["classify"] = {"ok": False, "error": f"only {n_classes_observed} classes after rejection"}
        return out

    feats = band_power_features(epochs, labels, bands=DEFAULT_BANDS)
    print(f"  [features] X={feats.X.shape}, channels={len(feats.channel_names)}, classes={n_classes_observed}")

    # 11-class (will be fewer if some classes were dropped during rejection)
    try:
        t = time.time()
        result = cross_val_classify(
            feats, subject=subject, task=f"{n_classes_observed}-class",
            cv_splits=5,
            n_permutations=1000 if permutation_test else 0,
        )
        save_run(result, run_id=run_id)
        print(f"  [{n_classes_observed}-class] acc={result.mean_accuracy:.3f} "
              f"bal={result.mean_balanced_accuracy:.3f} f1={result.f1_macro:.3f} "
              f"(chance={result.chance_level:.3f}, took {time.time()-t:.0f}s)")
        out["multiclass"] = {"ok": True, "mean_acc": result.mean_accuracy,
                             "mean_bal": result.mean_balanced_accuracy}
    except Exception as e:
        print(f"  [multiclass] FAILED: {e}")
        out["multiclass"] = {"ok": False, "error": str(e)}

    # Binary subproblems — only if all required classes are present
    for task_name in PHONOLOGICAL_TASKS:
        try:
            mapping = PHONOLOGICAL_TASKS[task_name]
            classes_in_data = set(labels)
            classes_needed = set(mapping.keys())
            if not classes_needed.issubset(classes_in_data):
                missing = classes_needed - classes_in_data
                print(f"  [{task_name}] SKIP — missing classes after rejection: {missing}")
                continue

            bin_labels = relabel_for_task(labels, task_name)
            bin_feats = band_power_features(epochs, bin_labels, bands=DEFAULT_BANDS)
            t = time.time()
            r = cross_val_classify(bin_feats, subject=subject, task=task_name, cv_splits=5)
            save_run(r, run_id=run_id)
            print(f"  [{task_name}] acc={r.mean_accuracy:.3f} bal={r.mean_balanced_accuracy:.3f} "
                  f"f1={r.f1_macro:.3f} ({time.time()-t:.0f}s)")
            out[task_name] = {"ok": True, "mean_acc": r.mean_accuracy,
                              "mean_bal": r.mean_balanced_accuracy}
        except Exception as e:
            print(f"  [{task_name}] FAILED: {e}")
            out[task_name] = {"ok": False, "error": str(e)}

    print(f"  [{subject}] total wall time: {time.time()-t0:.0f}s")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="*", help="subjects to run; default = all on disk")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="skip preprocessing (use existing .fif files)")
    parser.add_argument("--permutation-test", action="store_true",
                        help="run 1000-shuffle permutation test (slow)")
    parser.add_argument("--run-id", default=None,
                        help="custom run_id (default: phase3_batch_YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    available = list_available_subjects("data/raw")
    subjects = args.subjects if args.subjects else available
    if not subjects:
        print(f"No subjects available in data/raw/. Run scripts/download_karaone.py first.")
        sys.exit(1)
    missing = [s for s in subjects if s not in available]
    if missing:
        print(f"Warning: requested subjects not on disk: {missing}")
        subjects = [s for s in subjects if s in available]

    run_id = args.run_id or f"phase3_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nBatch pipeline — {len(subjects)} subjects → run_id={run_id}")
    print(f"  Subjects: {subjects}")
    print(f"  Skip preprocess: {args.skip_preprocess}")
    print(f"  Permutation test: {args.permutation_test}")

    overall_t0 = time.time()
    results = {}
    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}]", end=" ")
        try:
            results[subject] = run_one_subject(
                subject, run_id=run_id,
                skip_preprocess=args.skip_preprocess,
                permutation_test=args.permutation_test,
            )
        except Exception as e:
            print(f"  TOP-LEVEL ERROR for {subject}: {e}")
            traceback.print_exc()
            results[subject] = {"top_error": str(e)}

    # Save aggregate summary
    df = summary_csv()
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE — total wall time: {(time.time()-overall_t0)/60:.1f} min")
    print(f"{'='*60}")
    if df is not None and not df.empty:
        print(f"\n{len(df)} runs in summary CSV")
        # Print quick summary by task
        for task, group in df.groupby("task"):
            mean_acc = group["mean_accuracy"].mean()
            print(f"  {task}: mean acc across subjects = {mean_acc:.3f} (n={len(group)})")

    # Write final batch summary
    summary_path = Path("outputs/results") / run_id / "_batch_summary.json"
    summary_path.write_text(json.dumps({
        "run_id": run_id,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "total_minutes": (time.time() - overall_t0) / 60,
        "subjects": subjects,
        "results": {k: {ok_key: v[ok_key] for ok_key in ("ok", "error", "mean_acc", "mean_bal")
                        if ok_key in v}
                    for sub_d in results.values() for k, v in sub_d.items()
                    if isinstance(v, dict)},
    }, indent=2, default=str))
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
