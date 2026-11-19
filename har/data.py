"""Load HARTH / HAR70+, resample to 50 Hz, split at gaps and cut labelled windows (plan Section 2.1).

Pipeline per recording:
    load_recording  -> drop extra index columns, parse timestamps, sanity asserts
    to_50hz         -> S006 is recorded at 100 Hz: FIR anti-alias + decimate by 2 per gap-free segment
    lowpass_20hz    -> HARTH's 4th-order 20 Hz Butterworth low-pass, per gap-free segment (both datasets)
    make_windows    -> 128-sample windows inside runs of gap-free, shared-label rows
                       (step 64 for training, 128 for testing); window label = majority label,
                       kept only if it covers > 50% of the window (>= 65 of 128 rows)
build_all() runs this for every subject and caches the result in data/processed/har/ (git-ignored).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, decimate, sosfiltfilt

from .paths import (MANIFEST, PROCESSED_DIR, RAW_DIR, RECORDING_STATS_CSV, WINDOW_CACHE_META,
                    WINDOW_CACHE_NPZ, WINDOW_STATS_CSV)

SHARED_LABELS = (1, 3, 4, 5, 6, 7, 8)
LABEL_TO_INDEX = {label: i for i, label in enumerate(SHARED_LABELS)}
CLASS_NAMES = ("walking", "shuffling", "stairs_up", "stairs_down", "standing", "sitting", "lying")
CHANNELS = ("back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z")   # tensor channel order
REQUIRED_COLUMNS = ("timestamp", *CHANNELS, "label")
FS, WIN, SEG = 50, 128, 16             # 2.56 s windows, 8 segments of 0.32 s
PERIOD_S = 1.0 / FS
MAX_GAP_S = 2 * PERIOD_S               # a step longer than 40 ms splits the recording
TRAIN_STEP, TEST_STEP = WIN // 2, WIN  # 50% overlap for training, none for testing
MIN_MAJORITY = WIN // 2 + 1            # 65 of 128 rows: strictly more than 50%
# HARTH's "20 Hz low-pass fourth-order Butterworth filter" [Logacjov et al. 2021, Sec 3.3, p10],
# applied to both datasets (decided 2026-11-19). Zero-phase filtering (sosfiltfilt) is our choice:
# the paper does not say causal or zero-phase; zero phase keeps signals aligned with the labels.
LOWPASS_HZ, LOWPASS_ORDER = 20.0, 4
_SOS_LOWPASS = butter(LOWPASS_ORDER, LOWPASS_HZ, btype="low", fs=FS, output="sos")
_LOWPASS_PADLEN = 3 * (2 * len(_SOS_LOWPASS) + 1
                       - min(int((_SOS_LOWPASS[:, 2] == 0).sum()), int((_SOS_LOWPASS[:, 5] == 0).sum())))
NON_STAIR = (0, 1, 4, 5, 6)
STAIRS = (2, 3)

DATASET_DIRS = {"harth": "harth", "har70plus": "har70plus"}
KNOWN_LABELS = {
    "harth": frozenset({1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140}),
    "har70plus": frozenset(SHARED_LABELS),
}
LABEL_NAMES = {1: "walking", 2: "running", 3: "shuffling", 4: "stairs_up", 5: "stairs_down",
               6: "standing", 7: "sitting", 8: "lying", 13: "cycling_sit", 14: "cycling_stand",
               130: "cycling_sit_inactive", 140: "cycling_stand_inactive"}
GROUP = {"harth": "general", "har70plus": "older"}

# Walking-aid users, INFERRED: the only HAR70+ subjects with no stairs rows at all.
WALKING_AID_INFERRED = ("503", "505", "513", "514", "515")
WA_YES, WA_NO, WA_NA = "inferred_walking_aid", "inferred_no_walking_aid", "not_applicable"

# Window splits built by build_all(): (dataset, step)
SPLITS = {"harth_train": ("harth", TRAIN_STEP), "harth_test": ("harth", TEST_STEP),
          "har70plus_test": ("har70plus", TEST_STEP)}

PIPELINE_VERSION = 1  # bump when windowing logic changes, so the cache is rebuilt


def walking_aid_status(dataset: str, subject: str) -> str:
    if dataset != "har70plus":
        return WA_NA
    return WA_YES if subject in WALKING_AID_INFERRED else WA_NO


@dataclass
class Windows:
    X: np.ndarray               # (n, 6, 128) float32, raw acceleration in g (not normalized)
    y: np.ndarray               # (n,) int64 class index 0..6
    subject: np.ndarray         # (n,) str, e.g. "S006" or "501"
    dataset: str                # "harth" or "har70plus"
    group: np.ndarray           # (n,) str, "general" (HARTH) or "older" (HAR70+)
    walking_aid: np.ndarray     # (n,) str, inferred_walking_aid / inferred_no_walking_aid / not_applicable
    majority_share: np.ndarray  # (n,) float32, share of the 128 rows carrying the window label
    start_time: np.ndarray      # (n,) datetime64[ms], timestamp of the first row
    start_row: np.ndarray       # (n,) int64, first row in the subject's 50 Hz recording

    ARRAYS = ("X", "y", "subject", "group", "walking_aid", "majority_share", "start_time", "start_row")

    def __len__(self) -> int:
        return len(self.y)

    def select(self, idx) -> "Windows":
        return Windows(dataset=self.dataset, **{a: getattr(self, a)[idx] for a in self.ARRAYS})

    def for_subjects(self, subjects: Sequence[str]) -> "Windows":
        return self.select(np.isin(self.subject, list(subjects)))

    @property
    def subjects(self) -> list[str]:
        return sorted(set(self.subject.tolist()))

    @staticmethod
    def concat(parts: Sequence["Windows"], dataset: str) -> "Windows":
        assert all(p.dataset == dataset for p in parts)
        return Windows(dataset=dataset, **{a: np.concatenate([getattr(p, a) for p in parts])
                                           for a in Windows.ARRAYS})

    @staticmethod
    def empty(dataset: str) -> "Windows":
        return Windows(X=np.zeros((0, len(CHANNELS), WIN), np.float32), y=np.zeros(0, np.int64),
                       subject=np.zeros(0, "<U4"), dataset=dataset, group=np.zeros(0, "<U7"),
                       walking_aid=np.zeros(0, "<U23"), majority_share=np.zeros(0, np.float32),
                       start_time=np.zeros(0, "datetime64[ms]"), start_row=np.zeros(0, np.int64))


def window_ids(w: Windows, split: str) -> np.ndarray:
    """Stable window IDs "<split>:<subject>:<start_row>", e.g. "harth_test:S006:1408". They stay valid
    as long as PIPELINE_VERSION (the windowing rules) is unchanged."""
    if split not in SPLITS or SPLITS[split][0] != w.dataset:
        raise ValueError(f"split {split!r} does not match dataset {w.dataset!r}")
    return np.char.add(np.char.add(f"{split}:", np.char.add(w.subject, ":")), w.start_row.astype(str))


def find_windows(w: Windows, split: str, ids: Sequence[str]) -> np.ndarray:
    """Indices into `w` of the given window IDs, in the given order; raises if any is missing."""
    pos = {k: i for i, k in enumerate(window_ids(w, split).tolist())}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise KeyError(f"{len(missing)} window IDs not found, e.g. {missing[:3]}")
    return np.array([pos[i] for i in ids], dtype=np.int64)


def require_harth(*ws: Windows) -> None:
    """Leakage guard: training-side code must only ever see HARTH windows."""
    for w in ws:
        if w.dataset != "harth":
            raise ValueError(f"expected HARTH windows, got dataset={w.dataset!r} (HAR70+ is test-only)")


# --------------------------------------------------------------------------------------- loading

def recording_path(dataset: str, subject: str, raw_dir: Path = RAW_DIR) -> Path:
    return Path(raw_dir) / DATASET_DIRS[dataset] / f"{subject}.csv"


def list_subjects(dataset: str, raw_dir: Path = RAW_DIR) -> list[str]:
    d = Path(raw_dir) / DATASET_DIRS[dataset]
    subjects = sorted(p.stem for p in d.glob("*.csv"))
    if not subjects:
        raise FileNotFoundError(f"no CSVs in {d}; run `python data/download.py`")
    return subjects


def _ns(df: pd.DataFrame) -> np.ndarray:
    return df["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)


def load_recording(dataset: str, subject: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One subject's CSV with exactly REQUIRED_COLUMNS. Extra index columns (`index` in S015/S021,
    an unnamed one in S023) are dropped and listed in df.attrs["dropped_columns"]."""
    path = recording_path(dataset, subject, raw_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python data/download.py`")  # no synthetic fallback
    df = pd.read_csv(path)
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    unexpected = [c for c in extra if c != "index" and not str(c).startswith("Unnamed")]
    if unexpected:
        raise ValueError(f"{path.name}: unexpected columns {unexpected}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df = df[list(REQUIRED_COLUMNS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
    if df.isna().any().any():
        raise ValueError(f"{path.name}: contains NaN")
    if (np.diff(_ns(df)) <= 0).any():
        raise ValueError(f"{path.name}: timestamps not strictly increasing")
    bad = set(np.unique(df["label"]).tolist()) - KNOWN_LABELS[dataset]
    if bad:
        raise ValueError(f"{path.name}: unknown labels {sorted(bad)}")
    df.attrs["dropped_columns"] = extra
    return df


def sampling_period(df: pd.DataFrame) -> float:
    """Median step between rows, in seconds."""
    return float(np.median(np.diff(_ns(df)))) / 1e9


def contiguous_segments(df: pd.DataFrame, max_gap_s: float = MAX_GAP_S) -> list[slice]:
    """Row ranges without a step longer than max_gap_s."""
    breaks = np.flatnonzero(np.diff(_ns(df)) > max_gap_s * 1e9) + 1
    edges = np.r_[0, breaks, len(df)]
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def to_50hz(df: pd.DataFrame) -> pd.DataFrame:
    """50 Hz -> unchanged. 100 Hz (S006) -> per gap-free segment, FIR anti-alias filter and keep every
    2nd sample (scipy.signal.decimate, zero phase); labels and timestamps take every 2nd row."""
    p = sampling_period(df)
    if abs(p - PERIOD_S) < 1e-3:
        return df.reset_index(drop=True)
    if abs(p - PERIOD_S / 2) < 5e-4:
        parts = []
        for s in contiguous_segments(df, max_gap_s=2 * p):
            seg = df.iloc[s]
            out = seg.iloc[::2].copy()
            if len(seg) >= 2:
                x = decimate(seg[list(CHANNELS)].to_numpy(np.float64), 2, ftype="fir", axis=0, zero_phase=True)
                assert len(x) == len(out)
                out[list(CHANNELS)] = x
            parts.append(out)
        res = pd.concat(parts, ignore_index=True)
        res.attrs = dict(df.attrs)
        return res
    raise ValueError(f"unsupported sampling period {p * 1000:.2f} ms (expected 20 ms or 10 ms)")


def lowpass_20hz(df: pd.DataFrame) -> pd.DataFrame:
    """4th-order 20 Hz Butterworth low-pass on the 6 channels of a 50 Hz recording, applied to each
    gap-free segment separately (like the resampling), so no gap is filtered across. Segments of
    <= 15 rows (too short for the filter's edge padding) are left unfiltered; they are far shorter
    than a window, so no window contains them. Labels and timestamps are unchanged."""
    if abs(sampling_period(df) - PERIOD_S) > 1e-3:
        raise ValueError("lowpass_20hz expects a 50 Hz recording (call to_50hz first)")
    vals = df[list(CHANNELS)].to_numpy(np.float64).copy()
    for s in contiguous_segments(df):
        if s.stop - s.start > _LOWPASS_PADLEN:
            vals[s] = sosfiltfilt(_SOS_LOWPASS, vals[s], axis=0)
    out = df.copy()
    out[list(CHANNELS)] = vals
    out.attrs = dict(df.attrs)
    return out


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """The full signal pipeline before windowing: to_50hz, then lowpass_20hz."""
    return lowpass_20hz(to_50hz(raw))


# ------------------------------------------------------------------------------------- windowing

def _class_index(labels: np.ndarray) -> np.ndarray:
    cls = np.full(len(labels), -1, np.int64)
    for label, i in LABEL_TO_INDEX.items():
        cls[labels == label] = i
    return cls


def shared_runs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Maximal [start, end) row runs that are gap-free and carry only shared labels. Windows are cut
    inside these runs, so no window crosses a gap or contains a non-shared row."""
    shared = _class_index(df["label"].to_numpy()) >= 0
    runs = []
    for s in contiguous_segments(df):
        m = np.r_[False, shared[s], False].astype(np.int8)
        d = np.flatnonzero(np.diff(m))
        runs += [(s.start + int(a), s.start + int(b)) for a, b in zip(d[::2], d[1::2])]
    return runs


def make_windows(df: pd.DataFrame, dataset: str, subject: str, step: int) -> tuple[Windows, dict]:
    """Windows of one 50 Hz recording plus counts of what was dropped and why."""
    n = len(df)
    cls = _class_index(df["label"].to_numpy())
    runs = shared_runs(df)
    starts = [np.arange(a, b - WIN + 1, step) for a, b in runs if b - a >= WIN]
    starts = np.concatenate(starts) if starts else np.zeros(0, np.int64)

    onehot = np.zeros((n, len(SHARED_LABELS)), np.int32)
    onehot[cls >= 0, cls[cls >= 0]] = 1
    cs = np.zeros((n + 1, len(SHARED_LABELS)), np.int32)
    np.cumsum(onehot, axis=0, out=cs[1:])
    counts = cs[starts + WIN] - cs[starts] if len(starts) else np.zeros((0, len(SHARED_LABELS)), np.int32)
    top = counts.max(axis=1) if len(starts) else np.zeros(0, np.int32)
    keep = top >= MIN_MAJORITY
    idx = starts[keep]

    covered = np.zeros(n + 1, np.int32)               # rows inside at least one candidate window
    np.add.at(covered, starts, 1)
    np.add.at(covered, starts + WIN, -1)
    covered = np.cumsum(covered[:-1]) > 0
    short = np.zeros(n, bool)
    for a, b in runs:
        if b - a < WIN:
            short[a:b] = True
    shared = cls >= 0

    if len(idx):
        X = sliding_window_view(df[list(CHANNELS)].to_numpy(np.float32), WIN, axis=0)[idx].copy()
        ts = df["timestamp"].to_numpy().astype("datetime64[ms]")[idx]
        k = len(idx)
        w = Windows(X=X, y=counts[keep].argmax(axis=1).astype(np.int64),
                    subject=np.full(k, subject, "<U4"), dataset=dataset,
                    group=np.full(k, GROUP[dataset], "<U7"),
                    walking_aid=np.full(k, walking_aid_status(dataset, subject), "<U23"),
                    majority_share=(top[keep] / WIN).astype(np.float32),
                    start_time=ts, start_row=idx.astype(np.int64))
    else:
        w = Windows.empty(dataset)
    stats = {
        "dataset": dataset, "subject": subject, "step": step,
        "n_rows_50hz": n, "n_gaps": len(contiguous_segments(df)) - 1,
        "n_shared_runs": len(runs), "n_candidate_windows": len(starts),
        "n_windows": int(keep.sum()), "n_single_label": int((top == WIN).sum()),
        "n_mixed_kept": int((keep & (top < WIN)).sum()), "n_dropped_no_majority": int((~keep).sum()),
        "rows_short_runs": int((short & shared).sum()),
        "rows_not_covered_run_tails": int((shared & ~short & ~covered).sum()),
    }
    return w, stats


# ----------------------------------------------------------------------------------- build + cache

def _cache_key() -> str:
    lines = [l for l in MANIFEST.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#") and l.split()[-1].split("/")[0] in DATASET_DIRS.values()]
    cfg = {"version": PIPELINE_VERSION, "win": WIN, "steps": SPLITS, "min_majority": MIN_MAJORITY,
           "max_gap_s": MAX_GAP_S, "shared": SHARED_LABELS, "walking_aid": WALKING_AID_INFERRED,
           "lowpass": [LOWPASS_HZ, LOWPASS_ORDER, "butterworth", "sosfiltfilt per gap-free segment"]}
    return hashlib.sha256((json.dumps(cfg, sort_keys=True) + "\n".join(sorted(lines))).encode()).hexdigest()


def _build(raw_dir: Path) -> tuple[dict[str, Windows], pd.DataFrame, pd.DataFrame]:
    parts = {k: [] for k in SPLITS}
    wstats, rstats = [], []
    for dataset in DATASET_DIRS:
        for subject in list_subjects(dataset, raw_dir):
            raw = load_recording(dataset, subject, raw_dir)
            df = preprocess(raw)
            labels = raw["label"].to_numpy()
            r = {"dataset": dataset, "subject": subject, "group": GROUP[dataset],
                 "walking_aid": walking_aid_status(dataset, subject),
                 "n_rows_raw": len(raw), "raw_period_ms": round(sampling_period(raw) * 1000, 3),
                 "n_rows_50hz": len(df), "dropped_columns": ";".join(map(str, raw.attrs["dropped_columns"])),
                 "hours_50hz": len(df) * PERIOD_S / 3600}
            lab50 = df["label"].to_numpy()
            for code in sorted(KNOWN_LABELS["harth"]):
                r[f"rows_label_{code}"] = int((lab50 == code).sum())
            r["has_stairs_rows"] = bool(np.isin(labels, [4, 5]).any())
            rstats.append(r)
            for key, (ds, step) in SPLITS.items():
                if ds != dataset:
                    continue
                w, s = make_windows(df, dataset, subject, step)
                parts[key].append(w)
                wstats.append({"split": key, **s})
    windows = {k: Windows.concat(v, SPLITS[k][0]) for k, v in parts.items()}
    return windows, pd.DataFrame(wstats), pd.DataFrame(rstats)


def build_all(force: bool = False, raw_dir: Path = RAW_DIR) -> dict[str, Windows]:
    """{"harth_train" (step 64), "harth_test" (step 128), "har70plus_test" (step 128)}, cached in
    data/processed/har/ and keyed by the pinned raw-file checksums and the windowing config."""
    use_cache = Path(raw_dir).resolve() == RAW_DIR.resolve()
    key = _cache_key() if use_cache else None
    if use_cache and not force and WINDOW_CACHE_NPZ.exists() and WINDOW_CACHE_META.exists():
        meta = json.loads(WINDOW_CACHE_META.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            return _load_cache()
    windows, wstats, rstats = _build(raw_dir)
    if use_cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(WINDOW_CACHE_NPZ, **{f"{k}__{a}": getattr(w, a) for k, w in windows.items() for a in Windows.ARRAYS})
        wstats.to_csv(WINDOW_STATS_CSV, index=False)
        rstats.to_csv(RECORDING_STATS_CSV, index=False)
        WINDOW_CACHE_META.write_text(json.dumps({"key": key, "version": PIPELINE_VERSION}), encoding="utf-8")
    return windows


def _load_cache(splits: Sequence[str] = tuple(SPLITS)) -> dict[str, Windows]:
    with np.load(WINDOW_CACHE_NPZ) as z:   # NpzFile reads only the arrays that are accessed
        return {k: Windows(dataset=SPLITS[k][0], **{a: z[f"{k}__{a}"] for a in Windows.ARRAYS}) for k in splits}


HARTH_SPLITS = ("harth_train", "harth_test")


def load_harth_windows() -> dict[str, Windows]:
    """Only the HARTH splits of the window cache, for training-side steps (model choice). HAR70+
    arrays are never read and no raw CSV is opened. Raises if the cache is missing or out of date:
    build it first with `python -m har.build_windows`."""
    if not (WINDOW_CACHE_NPZ.exists() and WINDOW_CACHE_META.exists()):
        raise FileNotFoundError(f"window cache {WINDOW_CACHE_NPZ} missing; run `python -m har.build_windows`")
    meta = json.loads(WINDOW_CACHE_META.read_text(encoding="utf-8"))
    if meta.get("key") != _cache_key():
        raise RuntimeError("window cache is out of date (raw checksums or windowing config changed); "
                           "run `python -m har.build_windows --force`")
    return _load_cache(HARTH_SPLITS)


def cache_key() -> str:
    return _cache_key()


def build_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(window stats per subject and split, recording stats per subject) of the cached build."""
    build_all()
    return pd.read_csv(WINDOW_STATS_CSV, dtype={"subject": str}), pd.read_csv(RECORDING_STATS_CSV, dtype={"subject": str})
