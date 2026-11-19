"""Subject-wise folds, inner validation and leak-free normalisation statistics (plan Section 2.2).

Every fold: HARTH subjects split into train / validation (3 subjects held out from training) / test
("general"); all 18 HAR70+ subjects are test-only ("older") in every fold.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import CHANNELS, CLASS_NAMES, Windows, require_harth

N_FOLDS, N_VAL, SEED = 5, 3, 42
MIN_VAL_WINDOWS = 1000  # validation subjects need >= 1,000 non-overlapping windows
N_EXPLAIN_PER_SUBJECT = 30
N_BACKGROUND = 100
TEST_SPLITS = ("harth_test", "har70plus_test")  # non-overlapping windows; the only explain sources


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


def stratified_pick(classes: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Class-stratified sample of positions: shuffle each class's positions with `rng` (classes in
    index order), then take one per class per round, round-robin over the classes present (in index
    order), until n are chosen or every class is used up. Exhausted classes simply drop out."""
    perms = [rng.permutation(np.flatnonzero(classes == c)) for c in range(len(CLASS_NAMES))]
    out, r = [], 0
    while len(out) < n and any(r < len(p) for p in perms):
        for p in perms:
            if r < len(p):
                out.append(int(p[r]))
                if len(out) == n:
                    break
        r += 1
    return np.array(out, dtype=np.int64)


def select_explain_windows(w: Windows, split: str, per_subject: int = N_EXPLAIN_PER_SUBJECT,
                           seed: int = SEED) -> np.ndarray:
    """Indices into a test split (`harth_test` or `har70plus_test`, non-overlapping windows): up to
    `per_subject` windows per subject, stratified by class with `stratified_pick`. Each subject has
    its own generator, seeded with (seed, crc32(subject)), so a subject's choice does not depend on
    which other subjects are present. Order: subjects sorted by ID, then pick order."""
    if split not in TEST_SPLITS:
        raise ValueError(f"explain windows come from test splits only {TEST_SPLITS}, got {split!r}")
    out = []
    for s in w.subjects:
        pos = np.flatnonzero(w.subject == s)
        rng = np.random.default_rng([seed, zlib.crc32(s.encode())])
        out.append(pos[stratified_pick(w.y[pos], per_subject, rng)])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def select_background(w: Windows, split: str, fold: FoldSplit, n: int = N_BACKGROUND,
                      seed: int = SEED) -> np.ndarray:
    """Indices into `harth_train` of the fold's SHAP background: n windows from the fold's TRAINING
    subjects only (validation, test and HAR70+ excluded), class-stratified with `stratified_pick`
    over the pooled training windows, generator seeded with (seed, fold)."""
    require_harth(w)
    if split != "harth_train":
        raise ValueError(f"backgrounds come from harth_train windows, got {split!r}")
    pos = np.flatnonzero(np.isin(w.subject, fold.train))
    bad = set(w.subject[pos].tolist()) - set(fold.train)
    assert not bad, bad
    rng = np.random.default_rng([seed, fold.fold])
    idx = pos[stratified_pick(w.y[pos], n, rng)]
    if len(idx) < n:
        raise ValueError(f"fold {fold.fold}: only {len(idx)} training windows for a background of {n}")
    return idx


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
