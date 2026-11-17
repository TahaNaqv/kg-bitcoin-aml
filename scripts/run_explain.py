"""RQ3: which features drive predictions, and do model families agree?

* TreeSHAP for XGBoost on ALL features (shows reliance on leaky address features)
  and on CLEAN+KG features.
* Gradient x input attributions for KG-SAGE (seed 0) on the same CLEAN+KG inputs.
* Agreement: Spearman rank correlation and top-10 overlap between XGBoost |SHAP|
  and KG-SAGE |grad x input| feature rankings.

Writes results/shap_groups.csv, results/feature_importance.csv,
results/explain_agreement.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr

from kgaml import kg as K
from kgaml.config import CACHE_DIR, FEATURE_GROUP_OF, RESULTS_DIR, TRAIN_STEPS, ensure_dirs
from kgaml.data import load
from kgaml.experiment import eval_masks
from kgaml.features import kg_tables, matrix, unseen_entity_mask
from kgaml.gnn import VARIANTS, KGSage, build_heterodata, mean_adjacency
from kgaml.tabular import fit_predict


def group_of(col: str) -> str:
    if col in FEATURE_GROUP_OF:
        return FEATURE_GROUP_OF[col]
    if col == "aug_missing":
        return "tx-intrinsic"
    if col.startswith("kg_deg_"):
        return "KG relation degree"
    if col.startswith("kg_entity"):
        return "KG entity component"
    if col.startswith("kg_"):
        return "KG address history"
    if col.startswith("nbr_"):
        return "KG neighbour mean"
    return "other"


def shap_importance(fset, d, g, masks, y):
    xdf = matrix(d, g, fset)
    model, _, _ = fit_predict("XGB", 0, xdf.to_numpy(np.float32), y, masks["train"], masks["val"])
    te = masks["test"]
    sv = shap.TreeExplainer(model).shap_values(xdf.to_numpy(np.float32)[te])
    imp = pd.Series(np.abs(sv).mean(0), index=xdf.columns)
    return imp


def main() -> None:
    ensure_dirs()
    torch.set_num_threads(4)
    d = load()
    g = K.build(d)
    masks = eval_masks(d, unseen_entity_mask(d, g))
    y = d.y.astype(int)

    imps, groups = {}, []
    for fset in ("ALL", "CLEAN+KG"):
        imp = shap_importance(fset, d, g, masks, y)
        imps[fset] = imp
        gr = imp.groupby(imp.index.map(group_of)).sum()
        for k, v in (gr / gr.sum()).items():
            groups.append({"model": f"XGB {fset}", "group": k, "share": v})
        print(fset, (gr / gr.sum()).round(3).to_dict(), flush=True)

    # ---- KG-SAGE gradient x input on the same CLEAN+KG inputs ----
    struct, _ = kg_tables(d, g)
    h = build_heterodata(d, g, TRAIN_STEPS, tx_extra=struct)
    rel = VARIANTS["KG-SAGE"]["relations"]
    model = KGSage({"tx": h["tx"].x.shape[1], "addr": h["addr"].x.shape[1]}, rel)
    model.load_state_dict(torch.load(CACHE_DIR / "kg_sage_seed0.pt"))
    model.eval()
    n = {"tx": h["tx"].num_nodes, "addr": h["addr"].num_nodes}
    adj = {r: mean_adjacency(h[r].edge_index, n[r[0]], n[r[2]]) for r in model.relations}
    x_tx = h["tx"].x.clone().requires_grad_(True)
    rows = torch.from_numpy(np.flatnonzero(masks["test"]))
    out = model({"tx": x_tx, "addr": h["addr"].x}, adj, rows=rows)
    out.sum().backward()
    gxi = (x_tx.grad * x_tx).detach().abs()[rows].mean(0).numpy()
    cols = list(matrix(d, g, "CLEAN").columns) + list(struct.columns)
    gnn_imp = pd.Series(gxi, index=cols)
    gr = gnn_imp.groupby(gnn_imp.index.map(group_of)).sum()
    for k, v in (gr / gr.sum()).items():
        groups.append({"model": "KG-SAGE (grad x input)", "group": k, "share": v})
    print("KG-SAGE", (gr / gr.sum()).round(3).to_dict())

    xgb = imps["CLEAN+KG"].reindex(cols)
    rho = spearmanr(xgb.to_numpy(), gnn_imp.to_numpy()).statistic
    top_x, top_g = set(xgb.nlargest(10).index), set(gnn_imp.nlargest(10).index)
    agreement = {"spearman_rho_all_features": float(rho),
                 "top10_overlap": len(top_x & top_g),
                 "top10_xgb": list(xgb.nlargest(10).index),
                 "top10_kg_sage": list(gnn_imp.nlargest(10).index)}
    (RESULTS_DIR / "explain_agreement.json").write_text(json.dumps(agreement, indent=2))
    pd.DataFrame(groups).to_csv(RESULTS_DIR / "shap_groups.csv", index=False)
    fi = pd.DataFrame({"xgb_all_shap": imps["ALL"], }).join(
        pd.DataFrame({"xgb_cleankg_shap": xgb, "kg_sage_gxi": gnn_imp}), how="outer")
    fi["group"] = fi.index.map(group_of)
    fi.to_csv(RESULTS_DIR / "feature_importance.csv")
    print(json.dumps(agreement, indent=2))


if __name__ == "__main__":
    main()
