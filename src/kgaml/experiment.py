"""Shared evaluation bookkeeping for all model families."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TEST_STEPS
from .metrics import best_f1_threshold, evaluate


def score_run(p: np.ndarray, y: np.ndarray, step: np.ndarray, masks: dict[str, np.ndarray],
              meta: dict) -> tuple[list[dict], list[dict]]:
    """Choose the threshold on validation, then score every evaluation view.

    Returns (summary rows, per-step rows).
    """
    va = masks["val"]
    thr = best_f1_threshold(y[va], p[va])
    rows = []
    for view in ("val", "test", "test_unseen_entity", "test_pre_shutdown", "test_post_shutdown"):
        m = masks[view]
        rows.append({**meta, "view": view, "n": int(m.sum()), "n_illicit": int(y[m].sum()),
                     **evaluate(y[m], p[m], thr)})
    per_step = []
    for s in TEST_STEPS:
        m = masks["test"] & (step == s)
        if y[m].sum() == 0:
            continue
        r = evaluate(y[m], p[m], thr)
        per_step.append({**meta, "step": s, "f1": r["f1"], "pr_auc": r["pr_auc"],
                         "recall": r["recall"], "precision": r["precision"]})
    return rows, per_step


def eval_masks(d, unseen: np.ndarray) -> dict[str, np.ndarray]:
    from .config import DARK_MARKET_SHUTDOWN_STEP
    test = d.split_mask("test")
    return {
        "train": d.split_mask("train"),
        "val": d.split_mask("val"),
        "test": test,
        "test_unseen_entity": test & unseen,
        "test_pre_shutdown": test & (d.step < DARK_MARKET_SHUTDOWN_STEP),
        "test_post_shutdown": test & (d.step >= DARK_MARKET_SHUTDOWN_STEP),
    }


def summarise(df: pd.DataFrame, by: list[str], metrics=("f1", "pr_auc", "roc_auc",
              "precision", "recall", "precision_at_1pct")) -> pd.DataFrame:
    agg = df.groupby(by)[list(metrics)].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()
