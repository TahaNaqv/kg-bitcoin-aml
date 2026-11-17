"""RQ2: class-imbalance strategies for the tree ensembles on CLEAN+KG features.

Strategies: none, class weighting, SMOTE, SMOTE-ENN (resampling applied to the
training window only), each evaluated at the default 0.5 threshold and at the
validation-tuned threshold. Writes results/imbalance_runs.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from kgaml import kg as K
from kgaml.config import RESULTS_DIR, SEEDS, ensure_dirs
from kgaml.data import load
from kgaml.experiment import eval_masks, summarise
from kgaml.features import matrix, unseen_entity_mask
from kgaml.metrics import best_f1_threshold, evaluate
from kgaml.tabular import fit_predict

STRATEGIES = ["none", "class_weight", "SMOTE", "SMOTE-ENN"]


def main() -> None:
    ensure_dirs()
    d = load()
    g = K.build(d)
    masks = eval_masks(d, unseen_entity_mask(d, g))
    y = d.y.astype(int)
    x = matrix(d, g, "CLEAN+KG").to_numpy(np.float32)
    tr, va, te = masks["train"], masks["val"], masks["test"]
    rows = []
    for model in ("XGB", "RF"):
        for strat in STRATEGIES:
            for seed in SEEDS[:3]:
                sample = None
                if strat == "SMOTE":
                    sample = SMOTE(random_state=seed).fit_resample(x[tr], y[tr])
                elif strat == "SMOTE-ENN":
                    sample = SMOTEENN(random_state=seed).fit_resample(x[tr], y[tr])
                _, p, fit_s = fit_predict(model, seed, x, y, tr, va, sample_xy=sample,
                                          weighted=(strat == "class_weight"))
                for thr_name, thr in (("0.5", 0.5), ("val-tuned", best_f1_threshold(y[va], p[va]))):
                    r = evaluate(y[te], p[te], thr)
                    rows.append({"model": model, "strategy": strat, "threshold_rule": thr_name,
                                 "seed": seed, "fit_seconds": fit_s, **r})
                last = rows[-1]
                print(f"{model} {strat:12s} seed={seed} F1(val-thr)={last['f1']:.3f} "
                      f"PR-AUC={last['pr_auc']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "imbalance_runs.csv", index=False)
    s = summarise(df, ["model", "strategy", "threshold_rule"])
    s.to_csv(RESULTS_DIR / "imbalance_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
