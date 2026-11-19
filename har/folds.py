"""Subject-wise folds, inner validation and leak-free normalisation statistics (plan Section 2.2).

Every fold: HARTH subjects split into train / validation (3 subjects held out from training) / test
("general"); all 18 HAR70+ subjects are test-only ("older") in every fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import CHANNELS, Windows, require_harth

N_FOLDS, N_VAL, SEED = 5, 3, 42
MIN_VAL_WINDOWS = 1000  # validation subjects need >= 1,000 non-overlapping windows


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train: tuple[str, ...]          # HARTH, used for training (and normalisation statistics)
    val: tuple[str, ...]            # HARTH, held out: early stopping, tuning, gate, tau
    test_general: tuple[str, ...]   # HARTH, held-out test subjects
    test_older: tuple[str, ...]     # HAR70+, all subjects, test-only

    def role(self, subject: str) -> str:
        for name in ("train", "val", "test_general", "test_older"):
            if subject in getattr(self, name):
                return name
        raise KeyError(subject)

    def check(self, harth_subjects: Sequence[str], older_subjects: Sequence[str]) -> None:
        """Raise unless the sets are disjoint, HARTH is fully covered and HAR70+ is test-only."""
        sets = {"train": set(self.train), "val": set(self.val), "test_general": set(self.test_general),
                "test_older": set(self.test_older)}
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if sets[a] & sets[b]:
                    raise ValueError(f"fold {self.fold}: subjects in both {a} and {b}: {sorted(sets[a] & sets[b])}")
        if sets["train"] | sets["val"] | sets["test_general"] != set(harth_subjects):
            raise ValueError(f"fold {self.fold}: HARTH subjects not exactly covered by train/val/test")
        if sets["test_older"] != set(older_subjects):
            raise ValueError(f"fold {self.fold}: test_older must be all HAR70+ subjects")
        if (sets["train"] | sets["val"]) & set(older_subjects):
            raise ValueError(f"fold {self.fold}: HAR70+ subject in train or validation")


def assign_folds(window_counts: dict[str, int], n_folds: int = N_FOLDS) -> dict[str, int]:
    """Deterministic greedy balancing of HARTH subjects over folds by test-window count.

    Subjects sorted by count (desc, ties by ID) go to the fold with the fewest windows so far,
    among folds that may still grow: fold sizes are floor(n/k) or floor(n/k)+1 and only n % k folds
    get the larger size (22 subjects -> 5, 5, 4, 4, 4). Ties -> lowest fold index."""
    n = len(window_counts)
    base, n_big = divmod(n, n_folds)
    sizes, totals = [0] * n_folds, [0] * n_folds
    out = {}
    for s in sorted(window_counts, key=lambda s: (-window_counts[s], s)):
        n_full = sum(sz == base + 1 for sz in sizes)
        allowed = [f for f in range(n_folds) if sizes[f] < base or (sizes[f] == base and n_full < n_big)]
        f = min(allowed, key=lambda f: (totals[f], f))
        out[s] = f
        sizes[f] += 1
        totals[f] += window_counts[s]
    return out


def inner_val_subjects(train_subjects: Sequence[str], window_counts: dict[str, int], n: int = N_VAL,
                       seed: int = SEED, fold: int = 0, min_windows: int = MIN_VAL_WINDOWS) -> list[str]:
    """n of the fold's non-test HARTH subjects with >= min_windows windows, drawn with seed + fold
    (one seed per fold, so the same few subjects are not the validation set of every fold)."""
    eligible = sorted(s for s in train_subjects if window_counts[s] >= min_windows)
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} subjects with >= {min_windows} windows")
    rng = np.random.default_rng(seed + fold)
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())


def make_folds(harth_test_counts: dict[str, int], older_subjects: Sequence[str], n_folds: int = N_FOLDS,
               n_val: int = N_VAL, seed: int = SEED) -> list[FoldSplit]:
    """harth_test_counts: non-overlapping (test-split) window count per HARTH subject."""
    test_fold = assign_folds(harth_test_counts, n_folds)
    folds = []
    for k in range(n_folds):
        test = sorted(s for s, f in test_fold.items() if f == k)
        rest = sorted(s for s in harth_test_counts if s not in test)
        val = inner_val_subjects(rest, harth_test_counts, n_val, seed, fold=k)
        train = sorted(s for s in rest if s not in val)
        fs = FoldSplit(k, tuple(train), tuple(val), tuple(test), tuple(sorted(older_subjects)))
        fs.check(list(harth_test_counts), older_subjects)
        folds.append(fs)
    return folds


def tuning_sets(fold: FoldSplit) -> dict[str, tuple[str, ...]]:
    """Model-selection tuning (plan 2.9, Q6): train on the fold's training subjects, score on its 3
    validation subjects only. Its test subjects are returned so callers can assert they are unused."""
    return {"tuning_train": fold.train, "tuning_validation": fold.val, "never_used": fold.test_general}


def channel_stats(w: Windows, fold: FoldSplit, subjects: Sequence[str] | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std (population) over every sample of the training windows.

    `subjects` defaults to fold.train. Raises if `w` is not HARTH, or if any requested subject is
    not one of the fold's training subjects (validation, test, HAR70+ or unknown IDs)."""
    require_harth(w)
    subjects = list(fold.train if subjects is None else subjects)
    bad = sorted(set(subjects) - set(fold.train))
    if bad:
        roles = {s: (fold.role(s) if s in fold.val + fold.test_general + fold.test_older else "unknown") for s in bad}
        raise ValueError(f"fold {fold.fold}: normalisation statistics may only use training subjects; got {roles}")
    if not subjects:
        raise ValueError("no training subjects given")
    total = np.zeros(len(CHANNELS)); total_sq = np.zeros(len(CHANNELS)); n = 0
    for s in subjects:
        Xs = w.X[w.subject == s]
        if len(Xs) == 0:
            raise ValueError(f"training subject {s} has no windows")
        total += Xs.sum(axis=(0, 2), dtype=np.float64)
        total_sq += np.square(Xs, dtype=np.float64).sum(axis=(0, 2))
        n += Xs.shape[0] * Xs.shape[2]
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))
    if (std <= 0).any():
        raise ValueError("a channel has zero variance")
    return mean.astype(np.float32), std.astype(np.float32)
