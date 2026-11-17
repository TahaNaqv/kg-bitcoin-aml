"""Where do KG-SAGE's false alarms come from? (seed-0 predictions)

Splits false positives by entity view (recurring vs unseen entities) and tests
whether they are explained by out-of-range inputs, either the transaction-level
KG features or the address-node attributes. Writes results/gnn_diagnosis.json.
"""

from __future__ import annotations

import json

import numpy as np

from kgaml import kg as K
from kgaml.config import CACHE_DIR, RESULTS_DIR, TRAIN_STEPS
from kgaml.data import load
from kgaml.experiment import eval_masks
from kgaml.features import kg_tables, unseen_entity_mask
from kgaml.metrics import best_f1_threshold


def main() -> None:
    d = load()
    g = K.build(d)
    un = unseen_entity_mask(d, g)
    m = eval_masks(d, un)
    y = d.y.astype(int)
    p = np.load(CACHE_DIR / "proba_KG-SAGE.npy")
    thr = best_f1_threshold(y[m["val"]], p[m["val"]])
    te = m["test"]
    fp, tn = te & (y == 0) & (p >= thr), te & (y == 0) & (p < thr)
    seen = ~un

    # out-of-range inputs relative to the training period
    struct, _ = kg_tables(d, g)
    tr = np.isin(d.step, list(TRAIN_STEPS))
    tx_oor = (struct.to_numpy() > struct[tr].max().to_numpy()).any(1)
    x = g.inst_x.to_numpy()
    inst_oor = (x > x[np.isin(g.inst_step, list(TRAIN_STEPS))].max(0)).any(1)
    addr_oor = np.zeros(len(y), bool)
    addr_oor[g.spends[inst_oor[g.spends[:, 0]], 1]] = True
    addr_oor[g.pays[inst_oor[g.pays[:, 1]], 0]] = True

    out = {
        "false_positives": int(fp.sum()),
        "fp_recurring_entity_share": float((fp & seen).sum() / fp.sum()),
        "fp_rate_recurring_licit": float((fp & seen).sum() / (te & (y == 0) & seen).sum()),
        "fp_rate_unseen_licit": float((fp & un).sum() / (te & (y == 0) & un).sum()),
        "recurring_fp_with_oor_address_input": float(addr_oor[fp & seen].mean()),
        "recurring_tn_with_oor_address_input": float(addr_oor[tn & seen].mean()),
        "fp_with_oor_tx_kg_feature": float(tx_oor[fp].mean()),
        "tn_with_oor_tx_kg_feature": float(tx_oor[tn].mean()),
    }
    (RESULTS_DIR / "gnn_diagnosis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
