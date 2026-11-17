"""Assemble feature matrices for the tabular experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import kg as K
from .config import CACHE_DIR, FEATURE_SETS, LAST_TRAINING_STEP
from .data import Elliptic


def kg_tables(d: Elliptic, g: K.TSKG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Structural and neighbour-aggregate KG features (cached)."""
    p_s, p_n = CACHE_DIR / "kg_struct.parquet", CACHE_DIR / "kg_nbr.parquet"
    if p_s.exists() and p_n.exists():
        return pd.read_parquet(p_s), pd.read_parquet(p_n)
    struct = K.structural_features(g)
    cols = FEATURE_SETS["CLEAN"]
    x = log_scale(d.tx[cols], d.step).to_numpy(np.float32)
    cols = cols + ["aug_missing"]
    nbr = K.neighbour_mean_features(g, x, cols)
    struct.to_parquet(p_s)
    nbr.to_parquet(p_n)
    return struct, nbr


def log_scale(df: pd.DataFrame, step: np.ndarray | None = None) -> pd.DataFrame:
    """Signed log1p for the heavy-tailed raw Elliptic++ BTC/size columns, plus
    imputation of the 965 transactions Elliptic++ could not de-anonymise.

    The original Elliptic columns are already z-scored (affine), so the transform
    only changes the un-normalised Elliptic++ additions. Missing values are
    filled with the median of training-period transactions and flagged.
    """
    out = df.copy()
    raw = [c for c in out.columns if not c.startswith(("Local_", "Aggregate_"))]
    out[raw] = np.sign(out[raw]) * np.log1p(np.abs(out[raw]))
    if raw:
        missing = out[raw].isna().any(axis=1)
        fit = np.ones(len(out), bool) if step is None else step <= LAST_TRAINING_STEP - 5
        out[raw] = out[raw].fillna(out.loc[fit, raw].median())
        out["aug_missing"] = missing.astype(np.float32)
    return out


def matrix(d: Elliptic, g: K.TSKG, name: str) -> pd.DataFrame:
    """Feature set by name: ALL, CLEAN, CLEAN+KG, CLEAN+KG+NBR, ALL+KG+NBR."""
    base, *extra = name.split("+")
    df = log_scale(d.tx[FEATURE_SETS[base]], d.step)
    if extra:
        struct, nbr = kg_tables(d, g)
        parts = [df, struct]
        if "NBR" in extra:
            parts.append(nbr)
        df = pd.concat(parts, axis=1)
    return df


def unseen_entity_mask(d: Elliptic, g: K.TSKG) -> np.ndarray:
    """Transactions none of whose addresses were active in the training period.

    Implements the address-disjointness requirement of Safar et al. (2026):
    evaluating on these transactions measures generalisation to previously
    unseen entities rather than re-identification of known ones.
    """
    first_step = pd.Series(g.inst_step).groupby(g.inst_addr).min()
    known_addr = first_step[first_step <= LAST_TRAINING_STEP].index.to_numpy()
    known_inst = np.isin(g.inst_addr, known_addr)
    touches = np.zeros(g.n_tx, dtype=bool)
    touches[g.spends[known_inst[g.spends[:, 0]], 1]] = True
    touches[g.pays[known_inst[g.pays[:, 1]], 0]] = True
    return ~touches
