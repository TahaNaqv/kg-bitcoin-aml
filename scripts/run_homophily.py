"""Knowledge-graph statistics and per-relation label homophily (RO1, motivation).

Writes results/homophily.csv and results/kg_stats.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kgaml import kg as K
from kgaml.config import LAST_TRAINING_STEP, RESULTS_DIR, ensure_dirs
from kgaml.data import load
from kgaml.features import unseen_entity_mask


def main() -> None:
    ensure_dirs()
    d = load()
    g = K.build(d)
    y, step = d.y, d.step

    tables = []
    for name, mask in (("train (1-34)", step <= LAST_TRAINING_STEP),
                       ("test (35-49)", step > LAST_TRAINING_STEP)):
        t = K.homophily_table(g, y, mask)
        t.insert(0, "period", name)
        tables.append(t)
    hom = pd.concat(tables)
    hom.to_csv(RESULTS_DIR / "homophily.csv", index=False)

    # address reuse across time
    first = pd.Series(g.inst_step).groupby(g.inst_addr).min()
    last = pd.Series(g.inst_step).groupby(g.inst_addr).max()
    unseen = unseen_entity_mask(d, g)
    test_lab = d.split_mask("test")
    ill_test = test_lab & (y == 1)
    stats = {
        "transactions": int(g.n_tx),
        "addresses": int(d.n_addresses),
        "address_instances": int(g.n_inst),
        "hub_instances_gt_200": int(g.hub.sum()),
        "edges_flow": int(len(g.flows)),
        "edges_spends_in": int(len(g.spends)),
        "edges_pays": int(len(g.pays)),
        "cross_step_flow_edges": int((step[g.flows[:, 0]] != step[g.flows[:, 1]]).sum()),
        "addresses_active_in_multiple_steps": int((g.inst_x.groupby(g.inst_addr).size() > 1).sum()),
        "addresses_spanning_train_and_test": int(((first <= LAST_TRAINING_STEP)
                                                  & (last > LAST_TRAINING_STEP)).sum()),
        "test_labelled": int(test_lab.sum()),
        "test_illicit": int(ill_test.sum()),
        "test_labelled_touching_train_address": int((test_lab & ~unseen).sum()),
        "test_illicit_touching_train_address": int((ill_test & ~unseen).sum()),
        "mean_flow_degree": float(2 * len(g.flows) / g.n_tx),
    }
    (RESULTS_DIR / "kg_stats.json").write_text(json.dumps(stats, indent=2))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(hom.round(3).to_string(index=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
