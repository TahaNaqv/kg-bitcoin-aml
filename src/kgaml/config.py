"""Paths, temporal splits and leakage-aware feature groups.

Feature semantics follow the reverse-engineering of the Elliptic dataset by
Safar et al. (DFRWS USA 2026; mapping at
github.com/nesfit/DeanonymizedEllipticBitcoinDataset). Elliptic++ renames the
original columns: original feature 1 is ``Time step``, original features 2..94
are ``Local_feature_1..93`` and original features 95..166 are
``Aggregate_feature_1..72``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Raw Elliptic++ CSVs. The dataset is licensed CC BY-NC-ND and must never be
# committed; point KGAML_DATA_DIR at your local copy.
DATA_DIR = Path(os.environ.get("KGAML_DATA_DIR", REPO_ROOT.parent / "Elliptic++ Dataset"))
CACHE_DIR = REPO_ROOT / "data" / "cache"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# ---------------------------------------------------------------------------
# Temporal protocol (time steps are ~2 weeks each; edges never cross steps).
# Weber et al. train on 1-34 and test on 35-49; we carve the last five training
# steps out as a validation window used only for early stopping and threshold
# selection, identically for every model family.
# ---------------------------------------------------------------------------
TRAIN_STEPS = range(1, 30)  # 1..29
VAL_STEPS = range(30, 35)  # 30..34
TEST_STEPS = range(35, 50)  # 35..49
LAST_TRAINING_STEP = 34
DARK_MARKET_SHUTDOWN_STEP = 43  # Weber et al.: detection collapses after this step

SEEDS = (0, 1, 2, 3, 4)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Original features 2..22: scoped to the transaction itself (value, fee,
# #inputs/#outputs, unique addresses, input/output value statistics).
LOCAL_TX = [f"Local_feature_{i}" for i in range(1, 22)]
# Original features 23..94: statistics of the participating addresses computed
# once at block 575,059, i.e. after the end of the test period -> look-ahead
# (temporal) leakage and entity-level non-independence (Safar et al., Sec. 4).
LOCAL_ADDR_LEAKY = [f"Local_feature_{i}" for i in range(22, 94)]
# Original features 95..130: aggregates over *input* (predecessor)
# transactions, which by definition precede the transaction.
AGG_PRED = [f"Aggregate_feature_{i}" for i in range(1, 37)]
# Original features 131..166: aggregates over *output* (successor)
# transactions, i.e. transactions that spend this one's outputs later.
AGG_SUCC_LEAKY = [f"Aggregate_feature_{i}" for i in range(37, 73)]
# Elliptic++ additions computed from the transaction itself.
AUG_TX = [
    "in_txs_degree",
    "total_BTC",
    "fees",
    "size",
    "num_input_addresses",
    "num_output_addresses",
    "in_BTC_min",
    "in_BTC_max",
    "in_BTC_mean",
    "in_BTC_median",
    "in_BTC_total",
    "out_BTC_min",
    "out_BTC_max",
    "out_BTC_mean",
    "out_BTC_median",
    "out_BTC_total",
]
# Number of successor transactions: depends on future spending.
AUG_SUCC_LEAKY = ["out_txs_degree"]

FEATURE_SETS: dict[str, list[str]] = {
    # Everything, as used throughout the literature.
    "ALL": LOCAL_TX + LOCAL_ADDR_LEAKY + AGG_PRED + AGG_SUCC_LEAKY + AUG_TX + AUG_SUCC_LEAKY,
    # Only features observable at (or, for predecessors, before) the transaction time.
    "CLEAN": LOCAL_TX + AGG_PRED + AUG_TX,
}

FEATURE_GROUP_OF: dict[str, str] = {}
for _cols, _name in [
    (LOCAL_TX, "tx-intrinsic"),
    (LOCAL_ADDR_LEAKY, "address (leaky)"),
    (AGG_PRED, "predecessor agg."),
    (AGG_SUCC_LEAKY, "successor agg. (leaky)"),
    (AUG_TX, "tx-intrinsic"),
    (AUG_SUCC_LEAKY, "successor agg. (leaky)"),
]:
    for _c in _cols:
        FEATURE_GROUP_OF[_c] = _name


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
