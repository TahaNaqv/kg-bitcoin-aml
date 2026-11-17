# kg-bitcoin-aml

**Knowledge Graph Enrichment for Leakage-Free Illicit Bitcoin Transaction Detection: When Does Graph Learning Help?**

Code for the DS-5002 (Data Science Tools and Techniques, FAST-NUCES Islamabad, Fall 2026) course project paper by S Muhammad Taha Abbas Naqvi.

Graph neural networks have repeatedly failed to beat tree ensembles on the Elliptic Bitcoin benchmark once evaluation is done strictly. This project asks whether that is because graph structure carries no signal or because models were given the wrong graph. It:

1. **Audits feature leakage.** Following the reverse-engineered feature semantics of Šafář et al. (DFRWS 2026), 72 of the 93 "local" Elliptic features are address statistics computed after the test period, and 36 aggregates look at future (successor) transactions. We compare the usual `ALL` feature set with a `CLEAN` set that only uses information observable at transaction time.
2. **Builds a time-sliced transaction–address knowledge graph (TS-KG)** from Elliptic++. Addresses are instantiated per time step (`Address@t`), their history enters only through causal attributes, and every edge stays inside one time step. Training, validation and test periods are therefore disconnected sub-graphs, so evaluation is strictly inductive by construction.
3. **Measures why graph learning should help.** The knowledge graph's relations are far more homophilous than the transaction graph used by prior GNN work (class-adjusted homophily 0.92 vs 0.38 on the test period). The analysis also isolates the part of that signal that is an artefact of how labels were assigned.
4. **Compares models under one leakage-controlled protocol.** It separates three effects: feature leakage, KG knowledge supplied as tabular features, and message passing over the KG (relational GraphSAGE). The comparison includes relation ablations, edge rewiring, an unseen-entity test view, imbalance strategies, SHAP and gradient attributions, and compute cost, over multiple seeds.

The paper source is in [`paper/`](paper/). Results (CSV) are in [`results/`](results/) and figures in [`figures/`](figures/).

## Key results

All numbers are illicit-class metrics on test steps 35–49: mean over 5 seeds, threshold tuned on steps 30–34. Full tables are in `results/` and the paper.

| Model | Features | F1 | PR-AUC |
|---|---|---|---|
| XGBoost | All (leaky, as in prior work) | 0.803 | 0.796 |
| XGBoost | Clean (leakage removed) | 0.561 | 0.608 |
| **XGBoost** | **Clean + time-sliced KG features** | **0.651** | **0.683** |
| MLP | Clean + KG | 0.500 | 0.526 |
| GraphSAGE | transaction graph only | 0.461 | 0.359 |
| KG-SAGE | Clean + KG, message passing over the KG | 0.376 | 0.227 |

What the table shows:

- **Leakage.** Features computed after the test period account for about 30% of the F1 reported for standard baselines.
- **Homophily.** The knowledge graph's entity relations are far more homophilous than the transaction graph: class-adjusted homophily is 0.92 vs 0.38 on the test period.
- **KG features help trees.** Causal KG features improve a leakage-controlled XGBoost by +0.09 F1.
- **KG message passing is fragile under drift.** KG-SAGE has the best *validation* score. On transactions of previously unseen entities it beats the other neural models (PR-AUC 0.646 vs 0.567 for the MLP). It still raises many false alarms on recurring licit entities in later periods.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The project pins Python 3.12 and CPU-only PyTorch.

```bash
uv sync
```

## Data

The Elliptic / Elliptic++ data are **not** included. The Elliptic dataset is licensed CC BY-NC-ND 4.0, so neither raw nor derived data may be redistributed.

1. Download the Elliptic++ CSVs from the authors' repository: <https://github.com/git-disl/EllipticPlusPlus> (Google Drive link in their README). You need these files:
   - `txs_features.csv`
   - `txs_classes.csv`
   - `txs_edgelist.csv`
   - `AddrTx_edgelist.csv`
   - `TxAddr_edgelist.csv`
2. Put them in a folder and point `KGAML_DATA_DIR` at it. The default is `../Elliptic++ Dataset` relative to this repository.

```bash
export KGAML_DATA_DIR="/path/to/Elliptic++ Dataset"
```

The first run caches parsed data as parquet under `data/cache/`, which is git-ignored.

## Reproduce

```bash
./scripts/run_all.sh
```

or step by step:

| Script | Output | Paper |
|---|---|---|
| `scripts/run_homophily.py` | `results/homophily.csv`, `results/kg_stats.json` | Table II |
| `scripts/run_tabular.py` | `results/tabular_*.csv` | Tables III, V; Fig. 2 |
| `scripts/run_gnn.py` | `results/gnn_*.csv`, `results/gnn_tuning.csv` | Tables III–V; Fig. 2 |
| `scripts/run_imbalance.py` | `results/imbalance_*.csv` | Table VI (RQ2) |
| `scripts/run_explain.py` | `results/shap_groups.csv`, `results/explain_agreement.json` | Fig. 3 (RQ3) |
| `scripts/diagnose_gnn.py` | `results/gnn_diagnosis.json` | Sec. VI-C |
| `scripts/make_tables.py` | `paper/tables/*.tex` | all result tables |
| `scripts/make_figures.py` | `figures/*.pdf`, `paper/figures/*.pdf` | all plots |
| `scripts/make_paper_zip.sh` | `kg-bitcoin-aml-paper.zip` | submission archive |

The full pipeline takes about 6–8 hours on a laptop CPU (8 threads, 14 GB RAM), almost all of it graph-model training. Tree-ensemble experiments take under an hour.

## Protocol and leakage controls

- **Chronological split.** Fit on steps 1–29, early-stop and select thresholds on 30–34, test on 35–49. This is identical for every model.
- **`CLEAN` features only.** These are tx-intrinsic features (Local 1–21), predecessor aggregates (Aggregate 1–36) and the Elliptic++ tx-intrinsic additions. Excluded are the address statistics computed at block 575,059 (Local 22–93), the successor aggregates (Aggregate 37–72) and `out_txs_degree`.
- **No released Elliptic++ address features.** Those are lifetime summaries; address knowledge comes only from the time-sliced KG.
- **Training-period statistics only.** Standardisation and imputation use training-period statistics.
- **No cross-period message passing.** KG edges never cross time steps, which `kgaml.kg.build` asserts.
- **No test data in tuning.** Neural learning rates are selected on validation PR-AUC with tuning seeds disjoint from the evaluation seeds.
- **Unseen-entity view.** Test transactions whose addresses never appeared in training (Šafář et al.'s address-disjointness requirement) are scored separately.

## Repository layout

```
src/kgaml/
  config.py      paths, splits, leakage-aware feature groups
  data.py        Elliptic++ loading and caching
  kg.py          time-sliced knowledge graph, KG features, homophily
  features.py    feature matrices, unseen-entity mask
  tabular.py     LR / RF / XGBoost
  gnn.py         relational GraphSAGE (sparse CPU implementation)
  metrics.py     imbalance-aware metrics
  experiment.py  shared evaluation views and per-step scoring
scripts/         experiment, table and figure entry points
results/         CSV outputs used in the paper
figures/         generated figures
paper/           IEEE LaTeX source
```

## License

Code: MIT (see `LICENSE`). Data: see the Elliptic and Elliptic++ licences.
