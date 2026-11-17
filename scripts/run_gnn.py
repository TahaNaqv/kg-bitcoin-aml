"""Graph experiments on the time-sliced knowledge graph.

Every variant receives the same transaction inputs as the CLEAN+KG tree models
(CLEAN features + tabular KG features); variants differ only in which relations
messages are passed over. Protocol identical to the tabular script: fit on steps
1-29, early stopping and threshold on 30-34, test on 35-49.

Writes results/gnn_runs.csv and results/gnn_per_step.csv (appending per run so
that partial results survive interruption).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from kgaml import kg as K
from kgaml.config import CACHE_DIR, RESULTS_DIR, SEEDS, TRAIN_STEPS, ensure_dirs
from kgaml.data import load
from kgaml.experiment import eval_masks, score_run
from kgaml.features import kg_tables, unseen_entity_mask
from kgaml.gnn import VARIANTS, build_heterodata, train_eval

MAIN = ["MLP", "SAGE-FLOW", "KG-SAGE"]
ABLATIONS = ["KG-SAGE -flow", "KG-SAGE -input", "KG-SAGE -output", "KG-SAGE shuffled"]
LEARNING_RATES = (1e-3, 5e-3)
TUNE_SEEDS = (100, 101)  # disjoint from evaluation seeds


def _append(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=MAIN + ABLATIONS)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--ablation-seeds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--fresh", action="store_true", help="delete previous gnn results")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ensure_dirs()
    runs_p, steps_p = RESULTS_DIR / "gnn_runs.csv", RESULTS_DIR / "gnn_per_step.csv"
    if args.fresh:
        runs_p.unlink(missing_ok=True)
        steps_p.unlink(missing_ok=True)
    done = set()
    if runs_p.exists():
        prev = pd.read_csv(runs_p)
        done = set(zip(prev["model"], prev["seed"]))

    d = load()
    g = K.build(d)
    masks = eval_masks(d, unseen_entity_mask(d, g))
    y = d.y.astype(int)
    struct, _ = kg_tables(d, g)
    graphs = {"real": build_heterodata(d, g, TRAIN_STEPS, tx_extra=struct)}

    # ---- learning-rate selection on the validation window only ----
    # For each base model, pick the learning rate with the best mean validation
    # PR-AUC over two tuning seeds; ablations inherit the KG-SAGE choice.
    tune_p = RESULTS_DIR / "gnn_tuning.csv"
    tuned = set()
    if tune_p.exists():
        prev = pd.read_csv(tune_p)
        tuned = set(zip(prev["model"], prev["lr"].round(6), prev["seed"]))
    for variant in MAIN:
        for lr in LEARNING_RATES:
            for seed in TUNE_SEEDS:
                if (variant, round(lr, 6), seed) in tuned:
                    continue
                _, info = train_eval(graphs["real"], y, masks, VARIANTS[variant]["relations"],
                                     seed=seed, lr=lr, epochs=args.epochs, patience=args.patience)
                _append(tune_p, [{"model": variant, "lr": lr, "seed": seed,
                                  "val_pr_auc": info["val_pr_auc_select"],
                                  "best_epoch": info["best_epoch"],
                                  "fit_seconds": info["fit_seconds"]}])
                print(f"tune {variant:10s} lr={lr:g} seed={seed} "
                      f"val PR-AUC={info['val_pr_auc_select']:.3f}", flush=True)
    tune = pd.read_csv(tune_p).groupby(["model", "lr"])["val_pr_auc"].mean().reset_index()
    best_lr = {m: float(t.sort_values("val_pr_auc").iloc[-1]["lr"]) for m, t in tune.groupby("model")}
    print("selected learning rates:", best_lr, flush=True)

    for variant in args.variants:
        cfg = VARIANTS[variant]
        lr = best_lr.get(variant, best_lr["KG-SAGE"])
        n_seeds = args.seeds if variant in MAIN else args.ablation_seeds
        for seed in SEEDS[:n_seeds]:
            if (variant, seed) in done:
                continue
            key = ("shuffled", seed) if cfg["shuffle"] else "real"
            if key not in graphs:
                graphs[key] = build_heterodata(
                    d, g, TRAIN_STEPS, tx_extra=struct,
                    shuffle_seed=seed if cfg["shuffle"] else None)
            p, info = train_eval(graphs[key], y, masks, cfg["relations"], seed=seed, lr=lr,
                                 epochs=args.epochs, patience=args.patience)
            meta = {"family": "graph", "model": variant, "features": "CLEAN+KG", "seed": seed,
                    "lr": lr,
                    "n_features": int(graphs[key]["tx"].x.shape[1]),
                    "fit_seconds": info["fit_seconds"], "best_epoch": info["best_epoch"],
                    "n_params": info["n_params"]}
            rows, steps = score_run(p, y, d.step, masks, meta)
            _append(runs_p, rows)
            _append(steps_p, steps)
            if seed == 0:
                np.save(CACHE_DIR / f"proba_{variant.replace(' ', '_')}.npy", p)
                if variant == "KG-SAGE":
                    torch.save(info["model"].state_dict(), CACHE_DIR / "kg_sage_seed0.pt")
            t = next(r for r in rows if r["view"] == "test")
            print(f"{variant:18s} seed={seed} test F1={t['f1']:.3f} PR-AUC={t['pr_auc']:.3f} "
                  f"epochs={info['best_epoch']} ({info['fit_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
