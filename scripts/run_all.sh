#!/usr/bin/env bash
# Reproduce every result and figure in the paper.
# Requires the Elliptic++ CSVs in $KGAML_DATA_DIR (default: ../Elliptic++ Dataset).
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
uv run python scripts/run_homophily.py      # KG statistics + Table II
uv run python scripts/run_tabular.py        # LR / RF / XGB x feature sets x 5 seeds
uv run python scripts/run_gnn.py --fresh    # MLP / SAGE-Flow / KG-SAGE + ablations
uv run python scripts/run_imbalance.py      # RQ2
uv run python scripts/run_explain.py        # RQ3 (needs the KG-SAGE seed-0 checkpoint)
uv run python scripts/diagnose_gnn.py       # where KG-SAGE false alarms come from
uv run python scripts/make_tables.py        # LaTeX tables from results/*.csv
uv run python scripts/make_figures.py       # paper figures
./scripts/make_paper_zip.sh                 # submission archive
