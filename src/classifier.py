"""classifier.py — baseline SVM with proper cross-validation.

For WITHIN-SUBJECT classification:
  - StratifiedKFold(5) preserves class balance per fold
  - Each trial is one independent sample (no segment-shuffling leakage)
  - StandardScaler fit on train fold only (NEVER fit_transform on whole dataset)

Per-class chance level for 11-class KaraOne = 1/11 = 9.1%.
SOTA on KaraOne 11-class: ~57% (wavelet+DNN).
Sanity-check binary subproblems should hit 65-70% (vowel/consonant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.features import FeatureSet


@dataclass
class CVResult:
    """Result of one cross-validated classification run."""
    subject: str
    task: str                              # "11-class" | "vowel_v_consonant" | "nasal" | etc.
    n_classes: int
    classes: list[str]                     # sorted unique class labels
    fold_accuracies: list[float]           # one per fold (raw accuracy)
    mean_accuracy: float
    std_accuracy: float
    fold_balanced_accuracies: list[float]  # one per fold (balanced — guards against majority-class wins)
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    chance_level: float
    f1_macro: float
    confusion_matrix: list[list[int]]      # JSON-serializable
    n_train_total: int
    n_test_total: int
    permutation_pvalue: float | None       # None if not run
    n_permutations: int                    # 0 if not run
    config: dict                           # the config dict passed in


def build_pipeline(
    n_features_in: int,
    k_best: int | Literal["all"] = "all",
    C: float = 1.0,
    gamma: str | float = "scale",
    kernel: str = "rbf",
    class_weight: str | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    """Standard EEG ML pipeline: scale → (optional feature select) → SVM.

    `class_weight='balanced'` is the default — imagined-speech subproblems are
    often imbalanced (e.g. 30 vowels vs 135 consonants in KaraOne), and an
    unweighted SVM will just predict the majority class and pretend to win.
    """
    steps: list[tuple] = [("scale", StandardScaler())]
    if k_best != "all" and isinstance(k_best, int) and k_best < n_features_in:
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_best)))
    steps.append(("svm", SVC(
        C=C, kernel=kernel, gamma=gamma,
        class_weight=class_weight,
        random_state=random_state,
    )))
    return Pipeline(steps)


def cross_val_classify(
    feats: FeatureSet,
    subject: str,
    task: str = "11-class",
    cv_splits: int = 5,
    k_best: int | Literal["all"] = "all",
    C: float = 1.0,
    gamma: str | float = "scale",
    n_permutations: int = 0,
    random_state: int = 42,
    verbose: bool = False,
) -> CVResult:
    """Run StratifiedKFold cross-validation for within-subject classification.

    No data leakage: scaler/selector fit on train fold only via Pipeline.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(feats.y)
    classes = list(le.classes_)
    n_classes = len(classes)

    pipe = build_pipeline(
        n_features_in=feats.X.shape[1],
        k_best=k_best, C=C, gamma=gamma,
        random_state=random_state,
    )

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    fold_accs: list[float] = []
    fold_bal_accs: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    n_train_total = 0
    n_test_total = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(feats.X, y_enc)):
        X_train, X_test = feats.X[train_idx], feats.X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        fold_accs.append(float(acc))
        fold_bal_accs.append(float(bal_acc))
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        n_train_total += len(y_train)
        n_test_total += len(y_test)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{cv_splits}: acc={acc:.3f}, bal_acc={bal_acc:.3f}, n_test={len(y_test)}")

    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(n_classes))).tolist()
    f1m = f1_score(all_y_true, all_y_pred, average="macro")

    # Permutation test — confirms accuracy is significantly above chance
    perm_p = None
    if n_permutations > 0:
        if verbose:
            print(f"  Permutation test ({n_permutations} shuffles)...")
        _, _, perm_p = permutation_test_score(
            pipe, feats.X, y_enc,
            scoring="accuracy",
            cv=skf,
            n_permutations=n_permutations,
            n_jobs=1,
            random_state=random_state,
        )

    return CVResult(
        subject=subject,
        task=task,
        n_classes=n_classes,
        classes=classes,
        fold_accuracies=fold_accs,
        mean_accuracy=float(np.mean(fold_accs)),
        std_accuracy=float(np.std(fold_accs)),
        fold_balanced_accuracies=fold_bal_accs,
        mean_balanced_accuracy=float(np.mean(fold_bal_accs)),
        std_balanced_accuracy=float(np.std(fold_bal_accs)),
        chance_level=1.0 / n_classes,
        f1_macro=float(f1m),
        confusion_matrix=cm,
        n_train_total=n_train_total,
        n_test_total=n_test_total,
        permutation_pvalue=float(perm_p) if perm_p is not None else None,
        n_permutations=n_permutations,
        config={
            "cv_splits": cv_splits, "k_best": k_best, "C": C, "gamma": gamma,
            "class_weight": "balanced",
            "random_state": random_state,
        },
    )


# ---------- Binary subproblem helpers ----------
# From the AshrithSagar repo + KaraOne paper. Pipeline sanity check: these
# binary tasks should hit 65-70% if preprocessing is sound.
PHONOLOGICAL_TASKS = {
    "vowel_v_consonant": {
        # 0 = vowel only, 1 = consonant present
        "/diy/": 1, "/iy/": 0, "/m/": 1, "/n/": 1,
        "/piy/": 1, "/tiy/": 1, "/uw/": 0,
        "gnaw": 1, "knew": 1, "pat": 1, "pot": 1,
    },
    "nasal_v_nonnasal": {
        # 1 = nasal (m, n), 0 = non-nasal
        "/diy/": 0, "/iy/": 0, "/m/": 1, "/n/": 1,
        "/piy/": 0, "/tiy/": 0, "/uw/": 0,
        "gnaw": 1, "knew": 1, "pat": 0, "pot": 0,
    },
    "bilabial_v_nonbilabial": {
        # 1 = bilabial (m, p), 0 = non-bilabial
        "/diy/": 0, "/iy/": 0, "/m/": 1, "/n/": 0,
        "/piy/": 1, "/tiy/": 0, "/uw/": 0,
        "gnaw": 0, "knew": 0, "pat": 1, "pot": 1,
    },
}


def relabel_for_task(labels: np.ndarray, task: str) -> np.ndarray:
    """Map the 11-class labels into a binary task per PHONOLOGICAL_TASKS."""
    if task not in PHONOLOGICAL_TASKS:
        raise ValueError(f"Unknown task '{task}'. Available: {list(PHONOLOGICAL_TASKS.keys())}")
    mapping = PHONOLOGICAL_TASKS[task]
    return np.array([mapping[lbl] for lbl in labels])
