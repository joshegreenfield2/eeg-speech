#!/usr/bin/env python3
"""run_dl.py — train EEGNet + CNN-BiLSTM on every preprocessed subject.

Use AFTER preprocessing has finished (data/processed/<subj>-clean-epo.fif exist).
Saves results to outputs/results/<run_id>/ in the same format as the SVM baseline,
so the summary CSV ranks SVM vs EEGNet vs CNN-BiLSTM side by side.

Usage:
    uv run python scripts/run_dl.py                    # both models, all subjects
    uv run python scripts/run_dl.py --model eegnet     # one model only
    uv run python scripts/run_dl.py MM05 MM08          # specific subjects
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.evaluate import save_run, summary_csv
from src.loader import EVENT_ID
from src.preprocessor import load_clean_epochs
from src.train import cross_val_train


def labels_from_epochs(epochs):
    inv = {v: k for k, v in EVENT_ID.items()}
    return np.array([inv[e] for e in epochs.events[:, 2]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="*", help="default = all preprocessed subjects")
    parser.add_argument("--model", choices=["eegnet", "cnn_bilstm", "both"], default="both")
    parser.add_argument("--epochs-max", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    processed_dir = Path("data/processed")
    available = sorted(f.stem.replace("-clean-epo", "")
                       for f in processed_dir.glob("*-clean-epo.fif"))
    subjects = args.subjects if args.subjects else available
    if not subjects:
        print("No preprocessed subjects in data/processed/. Run scripts/run_pipeline.py first.")
        sys.exit(1)
    missing = [s for s in subjects if s not in available]
    if missing:
        print(f"Warning: not preprocessed yet: {missing}")
        subjects = [s for s in subjects if s in available]

    if args.model == "both":
        models = ["eegnet", "cnn_bilstm"]
    else:
        models = [args.model]

    run_id = args.run_id or f"phase4_dl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nDL pipeline — {len(subjects)} subjects × {len(models)} models → {run_id}")
    print(f"  Subjects: {subjects}")
    print(f"  Models: {models}")
    print(f"  epochs_max={args.epochs_max} batch_size={args.batch_size} lr={args.lr}")

    overall_t0 = time.time()
    for i, subject in enumerate(subjects, 1):
        try:
            epochs, log = load_clean_epochs(subject)
            labels = labels_from_epochs(epochs)
            n_classes_observed = len(set(labels))
            if n_classes_observed < 2:
                print(f"\n[{i}/{len(subjects)}] {subject}: SKIP — only {n_classes_observed} classes")
                continue
            print(f"\n[{i}/{len(subjects)}] {subject} — {epochs.get_data().shape}, "
                  f"{n_classes_observed} classes")

            for model_name in models:
                t = time.time()
                try:
                    result = cross_val_train(
                        epochs, labels, subject=subject, model_name=model_name,
                        epochs_max=args.epochs_max, batch_size=args.batch_size, lr=args.lr,
                        verbose=False,
                    )
                    save_run(result, run_id=run_id)
                    print(f"  [{model_name}] acc={result.mean_accuracy:.3f} "
                          f"bal={result.mean_balanced_accuracy:.3f} f1={result.f1_macro:.3f} "
                          f"({time.time()-t:.0f}s)")
                except Exception as e:
                    print(f"  [{model_name}] FAILED: {e}")
                    traceback.print_exc()
        except Exception as e:
            print(f"\n[{i}/{len(subjects)}] {subject}: TOP-LEVEL ERROR — {e}")
            traceback.print_exc()

    df = summary_csv()
    print(f"\nDL batch complete in {(time.time()-overall_t0)/60:.1f} min")
    if df is not None and not df.empty:
        for task, grp in df.groupby("task"):
            print(f"  {task}: mean_acc across subjects = {grp['mean_accuracy'].mean():.3f} (n={len(grp)})")


if __name__ == "__main__":
    main()
