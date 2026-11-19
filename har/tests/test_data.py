"""Tests for har/data.py (plan Section 2.10, test_data.py). Synthetic tests always run; real-data
tests need data/raw/ (python data/download.py) and use the window cache."""

import numpy as np
import pandas as pd
import pytest

from har import data as D
from har.paths import RAW_DIR

HAS_RAW = (RAW_DIR / "harth").is_dir() and (RAW_DIR / "har70plus").is_dir()
real = pytest.mark.skipif(not HAS_RAW, reason="raw data not downloaded")


def make_df(labels, period_s=0.02, gaps=(), start="2020-01-01 00:00:00", values=None) -> pd.DataFrame:
    """Synthetic recording. gaps: row indices before which a 1 s gap is inserted."""
    n = len(labels)
    steps = np.full(n, period_s)
    steps[0] = 0.0
    for g in gaps:
        steps[g] = 1.0
    t = pd.Timestamp(start) + pd.to_timedelta(np.round(np.cumsum(steps), 6), unit="s")
    vals = np.arange(n, dtype=float) if values is None else values
    df = pd.DataFrame({"timestamp": t})
    for i, c in enumerate(D.CHANNELS):
        df[c] = vals + i if np.ndim(vals) == 1 else vals[:, i]
    df["label"] = np.asarray(labels)
    return df


def write_csv(df: pd.DataFrame, path, extra=None):
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    if extra == "index":
        out.insert(1, "index", np.arange(len(out)))
        out.to_csv(path, index=False)
    elif extra == "unnamed":
        out.to_csv(path, index=True)  # leading unnamed index column, as in S023
    else:
        out.to_csv(path, index=False)


# ------------------------------------------------------------------------------- loading quirks

@pytest.mark.parametrize("extra,dropped", [("index", ["index"]), ("unnamed", ["Unnamed: 0"]), (None, [])])
def test_load_drops_extra_index_columns(tmp_path, extra, dropped):
    (tmp_path / "harth").mkdir()
    write_csv(make_df([6] * 300), tmp_path / "harth" / "S999.csv", extra)
    df = D.load_recording("harth", "S999", raw_dir=tmp_path)
    assert list(df.columns) == list(D.REQUIRED_COLUMNS)
    assert df.attrs["dropped_columns"] == dropped
    assert len(df) == 300 and df["timestamp"].dtype.kind == "M"


def test_load_rejects_bad_files(tmp_path):
    (tmp_path / "harth").mkdir()
    (tmp_path / "har70plus").mkdir()
    bad_col = make_df([6] * 10)
    bad_col["mystery"] = 1.0
    write_csv(bad_col, tmp_path / "harth" / "S900.csv")
    with pytest.raises(ValueError, match="unexpected columns"):
        D.load_recording("harth", "S900", raw_dir=tmp_path)
    write_csv(make_df([6] * 9 + [2]), tmp_path / "har70plus" / "599.csv")  # running is not a HAR70+ label
    with pytest.raises(ValueError, match="unknown labels"):
        D.load_recording("har70plus", "599", raw_dir=tmp_path)
    back = make_df([6] * 10)
    back.loc[5, "timestamp"] = back.loc[2, "timestamp"]
    write_csv(back, tmp_path / "harth" / "S901.csv")
    with pytest.raises(ValueError, match="strictly increasing"):
        D.load_recording("harth", "S901", raw_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        D.load_recording("harth", "S000", raw_dir=tmp_path)


# ---------------------------------------------------------------------------------- resampling

def test_to_50hz_halves_100hz_and_keeps_2hz_sine():
    n = 4000
    t = np.arange(n) * 0.01
    sine = np.sin(2 * np.pi * 2.0 * t)
    df = make_df([7] * n, period_s=0.01, values=np.tile(sine[:, None], (1, 6)))
    out = D.to_50hz(df)
    assert len(out) == n // 2
    assert abs(D.sampling_period(out) - 0.02) < 1e-9
    t_out = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    expected = np.sin(2 * np.pi * 2.0 * t_out)
    inner = slice(50, -50)  # ignore filter edge effects
    assert np.max(np.abs(out["back_x"].to_numpy()[inner] - expected[inner])) < 1e-2
    assert (out["label"] == 7).all()


def test_to_50hz_resamples_each_segment_separately():
    df = make_df([7] * 1000 + [6] * 1000, period_s=0.01, gaps=[1000])
    out = D.to_50hz(df)
    assert len(out) == 1000
    assert len(D.contiguous_segments(out)) == 2      # the gap survives resampling
    assert (out["label"].to_numpy()[:500] == 7).all() and (out["label"].to_numpy()[500:] == 6).all()


def test_to_50hz_keeps_50hz_and_rejects_other_rates():
    df = make_df([6] * 200)
    out = D.to_50hz(df)
    assert len(out) == 200 and np.array_equal(out["back_x"], df["back_x"])
    with pytest.raises(ValueError, match="unsupported sampling period"):
        D.to_50hz(make_df([6] * 200, period_s=0.04))


# ----------------------------------------------------------------------------- 20 Hz low-pass

def test_lowpass_keeps_slow_signals_and_removes_fast_ones():
    t = np.arange(2000) * 0.02
    slow, fast = np.sin(2 * np.pi * 2.0 * t), np.sin(2 * np.pi * 24.0 * t)    # 2 Hz and 24 Hz (Nyquist 25)
    out = D.lowpass_20hz(make_df([7] * 2000, values=np.tile((slow + fast)[:, None], (1, 6))))
    inner = slice(100, -100)
    assert np.max(np.abs(out["thigh_z"].to_numpy()[inner] - slow[inner])) < 0.05   # 2 Hz passes, 24 Hz gone
    assert len(out) == 2000 and (out["label"] == 7).all()
    assert out["timestamp"].equals(make_df([7] * 2000)["timestamp"])


def test_lowpass_is_applied_per_gap_free_segment():
    a, b = np.zeros(300), np.full(300, 5.0)
    df = make_df([6] * 600, gaps=[300], values=np.tile(np.r_[a, b][:, None], (1, 6)))
    out = D.lowpass_20hz(df)
    assert np.allclose(out["back_x"].to_numpy()[:300], 0.0, atol=1e-9)    # nothing leaks across the gap
    assert np.allclose(out["back_x"].to_numpy()[300:], 5.0, atol=1e-9)
    filtered_once = D.lowpass_20hz(make_df([6] * 300, values=np.tile(np.sin(np.arange(300.0))[:, None], (1, 6))))
    together = D.lowpass_20hz(make_df([6] * 600, gaps=[300],
                                      values=np.tile(np.r_[np.sin(np.arange(300.0)), b][:, None], (1, 6))))
    assert np.allclose(together["back_x"].to_numpy()[:300], filtered_once["back_x"].to_numpy())


def test_lowpass_leaves_tiny_segments_and_rejects_100hz():
    df = make_df([6] * 10 + [6] * 300, gaps=[10], values=np.tile(np.arange(310.0)[:, None] % 2, (1, 6)))
    out = D.lowpass_20hz(df)
    assert np.array_equal(out["back_x"].to_numpy()[:10], df["back_x"].to_numpy()[:10])   # 10 rows: unfiltered
    with pytest.raises(ValueError, match="50 Hz"):
        D.lowpass_20hz(make_df([6] * 400, period_s=0.01))


# ----------------------------------------------------------------------------------- windowing

def rows_of(w, i):
    return np.arange(w.start_row[i], w.start_row[i] + D.WIN)


def test_no_window_crosses_a_gap():
    df = make_df([1] * 300, gaps=[200])   # segments: rows 0-199 and 200-299
    w, st = D.make_windows(df, "harth", "S999", D.TRAIN_STEP)
    assert st["n_gaps"] == 1
    assert w.start_row.tolist() == [0, 64]  # 200-row segment fits 2 windows at step 64; 100-row one none
    for i in range(len(w)):
        r = rows_of(w, i)
        assert (r < 200).all() or (r >= 200).all()
    assert st["rows_short_runs"] == 100


def test_windows_never_contain_non_shared_labels():
    labels = [1] * 400
    labels[100:110] = [2] * 10            # running (HARTH-only) in the middle
    df = make_df(labels)
    w, st = D.make_windows(df, "harth", "S999", D.TEST_STEP)
    for i in range(len(w)):
        assert not np.isin(rows_of(w, i), np.arange(100, 110)).any()
    assert w.start_row.tolist() == [110, 238]   # windows restart after the non-shared rows
    assert st["rows_short_runs"] == 100          # rows 0-99 form a run shorter than 128


def test_majority_rule_65_kept_64_dropped():
    kept = make_df([1] * 65 + [6] * 63)
    w, st = D.make_windows(kept, "harth", "S999", D.TEST_STEP)
    assert len(w) == 1 and w.y[0] == D.LABEL_TO_INDEX[1]
    assert w.majority_share[0] == pytest.approx(65 / 128)
    assert st["n_mixed_kept"] == 1 and st["n_single_label"] == 0

    tie = make_df([1] * 64 + [6] * 64)
    w, st = D.make_windows(tie, "harth", "S999", D.TEST_STEP)
    assert len(w) == 0 and st["n_dropped_no_majority"] == 1

    three = make_df([1] * 50 + [6] * 40 + [7] * 38)   # plurality but no majority
    w, st = D.make_windows(three, "harth", "S999", D.TEST_STEP)
    assert len(w) == 0 and st["n_dropped_no_majority"] == 1


def test_train_step_64_and_test_step_128():
    df = make_df([7] * 512)
    w_tr, _ = D.make_windows(df, "harth", "S999", D.TRAIN_STEP)
    w_te, _ = D.make_windows(df, "harth", "S999", D.TEST_STEP)
    assert w_tr.start_row.tolist() == [0, 64, 128, 192, 256, 320, 384]
    assert w_te.start_row.tolist() == [0, 128, 256, 384]
    assert w_tr.X.shape == (7, 6, 128) and w_tr.X.dtype == np.float32


def test_window_values_are_the_raw_rows():
    df = make_df([4] * 300)
    w, _ = D.make_windows(df, "harth", "S999", D.TRAIN_STEP)
    expected = df[list(D.CHANNELS)].to_numpy(np.float32)[64:192].T
    assert np.array_equal(w.X[1], expected)      # (channel, time) layout
    assert w.start_time[1] == df["timestamp"].to_numpy().astype("datetime64[ms]")[64]


def test_group_and_walking_aid_tags():
    df = make_df([1] * 256)
    for dataset, subject, group, wa in [("harth", "S006", "general", D.WA_NA),
                                        ("har70plus", "503", "older", D.WA_YES),
                                        ("har70plus", "501", "older", D.WA_NO)]:
        w, _ = D.make_windows(df, dataset, subject, D.TEST_STEP)
        assert w.dataset == dataset
        assert set(w.subject) == {subject} and set(w.group) == {group} and set(w.walking_aid) == {wa}
    assert D.WALKING_AID_INFERRED == ("503", "505", "513", "514", "515")


def test_require_harth():
    w, _ = D.make_windows(make_df([1] * 256), "har70plus", "501", D.TEST_STEP)
    with pytest.raises(ValueError, match="HAR70\\+ is test-only"):
        D.require_harth(w)


# ----------------------------------------------------------------------------------- real data

@pytest.fixture(scope="module")
def windows():
    return D.build_all()


@real
def test_real_subjects_classes_and_tags(windows):
    tr, te, old = windows["harth_train"], windows["harth_test"], windows["har70plus_test"]
    assert len(tr.subjects) == 22 and tr.subjects == te.subjects
    assert len(old.subjects) == 18 and not set(old.subjects) & set(tr.subjects)
    for w in (tr, te, old):
        assert set(np.unique(w.y)) == set(range(7))
        assert w.X.shape[1:] == (6, 128) and w.X.dtype == np.float32
        assert (w.majority_share > 0.5).all()
    assert set(tr.group) == set(te.group) == {"general"} and set(tr.walking_aid) == {D.WA_NA}
    assert set(old.group) == {"older"}
    wa = sorted(set(old.subject[old.walking_aid == D.WA_YES]))
    assert wa == list(D.WALKING_AID_INFERRED)


@real
def test_real_walking_aid_subjects_are_exactly_those_without_stairs(windows):
    no_stairs_rows = [s for s in D.list_subjects("har70plus")
                      if not D.load_recording("har70plus", s)["label"].isin([4, 5]).any()]
    assert no_stairs_rows == list(D.WALKING_AID_INFERRED)
    old = windows["har70plus_test"]
    for s in D.WALKING_AID_INFERRED:
        assert not np.isin(old.y[old.subject == s], D.STAIRS).any()


@real
@pytest.mark.parametrize("dataset,subject", [("harth", "S006"), ("harth", "S010"), ("harth", "S015"),
                                             ("harth", "S023"), ("harth", "S026"), ("har70plus", "507")])
def test_real_windows_match_raw_rows_and_never_cross_gaps(windows, dataset, subject):
    """Re-derive every window of a subject from its raw file: same values, no step > 40 ms inside,
    only shared labels, and the stored label is the > 50% majority."""
    df = D.preprocess(D.load_recording(dataset, subject))     # 50 Hz + 20 Hz low-pass, as in build_all
    ns = df["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    steps = np.diff(ns) / 1e9
    vals = df[list(D.CHANNELS)].to_numpy(np.float32)
    labels = df["label"].to_numpy()
    keys = ["harth_train", "harth_test"] if dataset == "harth" else ["har70plus_test"]
    for key in keys:
        w = windows[key].for_subjects([subject])
        assert len(w) > 0
        for i in range(len(w)):
            a = int(w.start_row[i])
            assert steps[a:a + D.WIN - 1].max() <= D.MAX_GAP_S + 1e-9
            lab = labels[a:a + D.WIN]
            assert np.isin(lab, D.SHARED_LABELS).all()
            maj = D.SHARED_LABELS[w.y[i]]
            assert (lab == maj).sum() >= D.MIN_MAJORITY
            assert np.array_equal(w.X[i], vals[a:a + D.WIN].T)


@real
def test_real_s006_resampled_to_50hz(windows):
    raw = D.load_recording("harth", "S006")
    assert abs(D.sampling_period(raw) - 0.01) < 1e-4
    df = D.to_50hz(raw)
    assert abs(D.sampling_period(df) - 0.02) < 1e-6
    assert abs(len(df) - len(raw) / 2) <= len(D.contiguous_segments(raw, max_gap_s=0.02))
    w = windows["harth_train"].for_subjects(["S006"])
    gaps = np.diff(w.start_time.astype(np.int64))  # ms between consecutive window starts
    assert (gaps >= 64 * 20).all()                  # 64 samples at 20 ms, never 64 at 10 ms
    assert np.median(gaps) == 64 * 20
