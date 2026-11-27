"""Gap study v4: ONE held-out test confirmation of the gap selected on validation (run once, after selection).

    $env:PYTHONPATH = "bit-reduction-test"
    python -m gap_study_v4.test_confirm --folds 0 --models B=<pattern> X=<pattern> ... --threads 3
    python -m gap_study_v4.test_confirm --analyze --models B X ...

Test populations (never used for any choice): each fold's HARTH test subjects ("general", 22 subjects over the 5
folds) and all 18 HAR70+ subjects ("older", in every fold, averaged over folds as in 9A). Explain windows, background,
targets and the original's SHAP (seeds 0 and 1) are 9A's / the noise floor's (har.run_noise_floor.explain_sets; their
SHAP cache is reused). Every model's checkpoint is fold-specific and was trained / built on the fold's training
subjects only.

Per model and subject: LCI with independent SHAP noise (model SHAP seed 1 vs original seed 0, the v3/v4 primary),
same-seed LCI (model seed 0), occlusion LCI, agreement with the original's prediction, accuracy on all of the
subject's test windows; the uncompressed ceiling = original seed 1 vs seed 0 on the same windows.

--models label=path_pattern; {k} in the pattern is replaced by the fold.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "bit-reduction-test"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gap_study_v3.confirm as C3  # noqa: E402
from har.attributions import shap_cached  # noqa: E402
from har.data import load_harth_windows, load_test_windows  # noqa: E402
from har.folds import load_folds  # noqa: E402
from har.paths import MODEL_SELECTION_DIR, SHAP_CACHE_DIR  # noqa: E402
from har.run_noise_floor import SMOKE as NF_SMOKE, explain_sets, fold_background  # noqa: E402
from har.run_sanity_walking_aid import load_fold_model  # noqa: E402
from har.train import predict_proba  # noqa: E402
from lci.metric import aggregate_time_segments  # noqa: E402

OUT = ROOT / "results" / "gap-study-v4" / "test"
P = "gap-study-v4-test-tinyhar"
GROUPS = ("general", "older")
NSAMPLES = 200


def load_net(net0, path: Path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    net = copy.deepcopy(net0).cpu()
    net.load_state_dict(c["state_dict"])
    net.prune_masks = {n: m.bool() for n, m in c.get("prune_masks", {}).items()}
    return net.eval()


def run_fold(k: int, a, harth, test, folds) -> None:
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"{P}-fold{k}-per-subject.csv"
    done = set(pd.read_csv(csv).label) if (csv.exists() and a.resume) else set()
    if csv.exists() and not a.resume:
        raise SystemExit(f"{csv} exists; pass --resume")
    frozen = json.loads((MODEL_SELECTION_DIR / "har-frozen-model-config.json").read_text(encoding="utf-8"))
    fold = folds[k]
    net0, _ = load_fold_model(k, frozen["architecture"])
    net0 = net0.eval()
    bg = fold_background(harth, k)
    bg_mean = bg.mean(0)
    sets = explain_sets(fold, harth, test, a.smoke)
    nsamples = NF_SMOKE["nsamples"] if a.smoke else NSAMPLES
    cache = SHAP_CACHE_DIR / ("smoke" if a.smoke else "")
    evalw = {"general": test["harth_test"].for_subjects(fold.test_general),
             "older": test["har70plus_test"].for_subjects(fold.test_older)}
    ref = {}
    for g in GROUPS:
        X, ids, subj = sets[g]
        Xt = torch.from_numpy(X)
        tgt = predict_proba(net0, X, "cpu").argmax(1)
        A0 = shap_cached(net0, bg, Xt, tgt, ids, 0, nsamples, cache, f"har-noise-floor-fold{k}-{g}", log=lambda x: None)
        A1 = shap_cached(net0, bg, Xt, tgt, ids, 1, nsamples, cache, f"har-noise-floor-fold{k}-{g}", log=lambda x: None)
        ref[g] = {"X": Xt, "ids": ids, "subj": np.asarray(subj), "tgt": tgt, "S0": aggregate_time_segments(A0),
                  "S1": aggregate_time_segments(A1), "O0": C3.occlusion(net0, Xt, tgt, bg_mean)}
    specs = [("original", None)] + [tuple(m.split("=", 1)) for m in a.models]
    for label, pat in specs:
        if label in done:
            continue
        t0 = time.perf_counter()
        if pat is None:
            net = net0
        else:
            path = Path(pat.format(k=k))
            if not path.exists():
                raise FileNotFoundError(path)
            net = load_net(net0, path)
        rows = []
        for g in GROUPS:
            r = ref[g]
            if pat is None:
                S0, S1, O = r["S0"], r["S1"], r["O0"]
            else:
                tag = f"gs4-test-fold{k}-{g}-{label}"
                S0 = aggregate_time_segments(shap_cached(net, bg, r["X"], r["tgt"], r["ids"], 0, nsamples, cache, tag,
                                                         log=lambda x: None))
                S1 = aggregate_time_segments(shap_cached(net, bg, r["X"], r["tgt"], r["ids"], 1, nsamples, cache, tag,
                                                         log=lambda x: None))
                O = C3.occlusion(net, r["X"], r["tgt"], bg_mean)
            pred_ex = predict_proba(net, r["X"].numpy(), "cpu").argmax(1)
            w = evalw[g]
            pred = predict_proba(net, w.X, "cpu").argmax(1)
            for s in sorted(set(r["subj"].tolist())):
                m = r["subj"] == s
                ms = w.subject == s
                rows.append({"fold": k, "label": label, "checkpoint": "" if pat is None else Path(pat.format(k=k)).name,
                             "group": g, "subject": s, "n_explain": int(m.sum()),
                             "lci_same": C3.lci_mean(r["S0"][m], S0[m], r["subj"][m]),
                             "lci_indep": C3.lci_mean(r["S0"][m], S1[m], r["subj"][m]),
                             "ceiling": C3.lci_mean(r["S0"][m], r["S1"][m], r["subj"][m]),
                             "occ_lci": C3.lci_mean(r["O0"][m], O[m], r["subj"][m]),
                             "agree": float((pred_ex[m] == r["tgt"][m]).mean()),
                             "accuracy": float((pred[ms] == w.y[ms]).mean()), "n_eval": int(ms.sum()),
                             "y": "", "pred": ""})
            # pooled F1 per group needs the windows: store compact label / prediction strings per subject
            for row in rows:
                if row["group"] == g and row["y"] == "":
                    ms = w.subject == row["subject"]
                    row["y"] = "".join(map(str, w.y[ms].tolist()))
                    row["pred"] = "".join(map(str, pred[ms].tolist()))
        C3._append(csv, rows)
        d = pd.DataFrame(rows)
        print(f"fold {k} {label}: " + "; ".join(
            f"{g} LCI indep {d[d.group == g].lci_indep.mean():.3f} ceiling {d[d.group == g].ceiling.mean():.3f} "
            f"acc {d[d.group == g].accuracy.mean():.3f}" for g in GROUPS) + f" [{time.perf_counter() - t0:.0f}s]",
              flush=True)


def analyze(a) -> None:
    from scipy import stats
    from sklearn.metrics import f1_score
    out = Path(a.out) if a.out else OUT
    d = pd.concat([pd.read_csv(f, dtype={"subject": str, "y": str, "pred": str})
                   for f in sorted(out.glob(f"{P}-fold*-per-subject.csv"))])
    # subject level: HARTH subjects appear in one fold, HAR70+ subjects in all 5 (mean over folds)
    num = ["lci_same", "lci_indep", "ceiling", "occ_lci", "agree", "accuracy"]
    subj = d.groupby(["label", "group", "subject"])[num].mean().reset_index()
    labels = [l.split("=")[0] for l in a.models]
    rows = []
    for g in GROUPS + ("all",):
        sg = subj if g == "all" else subj[subj.group == g]
        piv = {l: sg[sg.label == l].set_index(["group", "subject"]) for l in ["original"] + labels}
        for l in ["original"] + labels:
            x = piv[l]
            dd = d if g == "all" else d[d.group == g]
            dl = dd[dd.label == l]
            y = np.array([int(c) for s in dl.y for c in s])
            pr = np.array([int(c) for s in dl.pred for c in s])
            rec = {"group": g, "label": l, "n_subjects": len(x), "lci_indep": x.lci_indep.mean(),
                   "lci_same": x.lci_same.mean(), "ceiling": x.ceiling.mean(), "occ_lci": x.occ_lci.mean(),
                   "agree": x.agree.mean(), "accuracy_subject_mean": x.accuracy.mean(),
                   "accuracy_pooled_fold_instances": float((y == pr).mean()),
                   "macro_f1_pooled_fold_instances": float(f1_score(y, pr, average="macro")),
                   "gap_to_ceiling_indep": (x.ceiling - x.lci_indep).mean(),
                   "subjects_gap_to_ceiling_ge_0.10": int(((x.ceiling - x.lci_indep) >= 0.10).sum())}
            for ref_l in labels:
                if ref_l == l:
                    continue
                diff = (piv[ref_l].lci_indep - x.lci_indep).dropna()
                rec[f"{ref_l}_minus_this_lci_indep"] = diff.mean()
                rec[f"{ref_l}_minus_this_subjects_positive"] = int((diff > 0).sum())
                rec[f"{ref_l}_minus_this_wilcoxon_p_greater"] = (float(stats.wilcoxon(diff, alternative="greater").pvalue)
                                                                  if len(diff) > 5 else np.nan)
                da = (piv[ref_l].accuracy - x.accuracy).dropna()
                rec[f"{ref_l}_minus_this_accuracy_pt"] = 100 * da.mean()
            rows.append(rec)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(out / f"{P}-summary-per-group-and-model.csv", index=False, lineterminator="\n")
    subj.round(4).to_csv(out / f"{P}-per-subject-pooled-over-folds.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.analyze:
        analyze(a)
        return
    harth, test = load_harth_windows(), load_test_windows()
    counts = pd.Series(harth["harth_test"].subject).value_counts().to_dict()
    folds = load_folds(harth_test_counts={s: int(c) for s, c in counts.items()})
    for k in a.folds:
        run_fold(k, a, harth, test, folds)


if __name__ == "__main__":
    main()
