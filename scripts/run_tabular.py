"""Tabular experiments: {LR, RF, XGB} x feature sets x seeds.

Protocol: fit on steps 1-29, early stopping / threshold on 30-34, test on 35-49.
Writes results/tabular_runs.csv and results/tabular_per_step.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from kgaml import kg as K
from kgaml.config import RESULTS_DIR, SEEDS, ensure_dirs
from kgaml.data import load
from kgaml.experiment import eval_masks, score_run, summarise
from kgaml.features import matrix, unseen_entity_mask
from kgaml.tabular import fit_predict

FEATURE_SETS = ["ALL", "CLEAN", "CLEAN+KG", "CLEAN+KG+NBR", "ALL+KG+NBR"]
MODELS = ["LR", "RF", "XGB"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--fsets", nargs="*", default=FEATURE_SETS)
    args = ap.parse_args()
    ensure_dirs()

    d = load()
    g = K.build(d)
    masks = eval_masks(d, unseen_entity_mask(d, g))
    y = d.y.astype(int)
    rows, steps = [], []
    for fset in args.fsets:
        x = matrix(d, g, fset).to_numpy(np.float32)
        for model in args.models:
            seeds = SEEDS[: 1 if model == "LR" else args.seeds]
            for seed in seeds:
                _, p, fit_s = fit_predict(model, seed, x, y, masks["train"], masks["val"])
                meta = {"family": "tabular", "model": model, "features": fset, "seed": seed,
                        "n_features": x.shape[1], "fit_seconds": fit_s}
                r, s = score_run(p, y, d.step, masks, meta)
                rows += r
                steps += s
                t = next(q for q in r if q["view"] == "test")
                print(f"{model:4s} {fset:14s} seed={seed} test F1={t['f1']:.3f} "
                      f"PR-AUC={t['pr_auc']:.3f} ({fit_s:.1f}s)", flush=True)

    runs, per_step = pd.DataFrame(rows), pd.DataFrame(steps)
    if args.fsets != FEATURE_SETS or args.models != MODELS:
        # partial re-run: replace only the (model, feature set) cells that were recomputed
        for name, new in (("tabular_runs.csv", runs), ("tabular_per_step.csv", per_step)):
            path = RESULTS_DIR / name
            if path.exists():
                old = pd.read_csv(path)
                keep = ~(old.features.isin(args.fsets) & old.model.isin(args.models))
                new = pd.concat([old[keep], new], ignore_index=True)
            new.to_csv(path, index=False)
        runs = pd.read_csv(RESULTS_DIR / "tabular_runs.csv")
    else:
        runs.to_csv(RESULTS_DIR / "tabular_runs.csv", index=False)
        per_step.to_csv(RESULTS_DIR / "tabular_per_step.csv", index=False)
    summ = summarise(runs, ["model", "features", "view"])
    summ.to_csv(RESULTS_DIR / "tabular_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ[summ.view.isin(["test", "test_unseen_entity"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
