"""Loading and caching the Elliptic++ CSVs.

Raw CSVs are parsed once and cached as parquet under ``data/cache`` (git-ignored).
Transactions are re-indexed to contiguous integers ``0..N-1`` ordered as in
``txs_features.csv``; addresses are mapped to integers ``0..A-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CACHE_DIR, DATA_DIR, TEST_STEPS, TRAIN_STEPS, VAL_STEPS, ensure_dirs

# Elliptic class codes: 1 = illicit, 2 = licit, 3 = unknown.
LABEL_MAP = {1: 1, 2: 0, 3: -1}


@dataclass
class Elliptic:
    tx: pd.DataFrame  # one row per transaction: txId, step, y (1/0/-1) + features
    tx_edges: np.ndarray  # (E, 2) int64 tx index -> tx index (fund flow)
    spends: np.ndarray  # (E, 2) int64 [address index, tx index]: address is an input of tx
    pays: np.ndarray  # (E, 2) int64 [tx index, address index]: address is an output of tx
    n_addresses: int

    @property
    def step(self) -> np.ndarray:
        return self.tx["step"].to_numpy()

    @property
    def y(self) -> np.ndarray:
        return self.tx["y"].to_numpy()

    def split_mask(self, split: str, labelled: bool = True) -> np.ndarray:
        steps = {"train": TRAIN_STEPS, "val": VAL_STEPS, "test": TEST_STEPS}[split]
        m = np.isin(self.step, list(steps))
        if labelled:
            m &= self.y >= 0
        return m


def _read_raw() -> Elliptic:
    header = pd.read_csv(DATA_DIR / "txs_features.csv", nrows=0).columns
    # txId values exceed 2**24, so they must not pass through float32.
    dtypes = {c: np.float32 for c in header} | {"txId": np.int64}
    feats = pd.read_csv(DATA_DIR / "txs_features.csv", dtype=dtypes, engine="pyarrow")
    feats = feats.rename(columns={"Time step": "step"})
    feats["step"] = feats["step"].astype(np.int16)

    classes = pd.read_csv(DATA_DIR / "txs_classes.csv", engine="pyarrow")
    feats = feats.merge(classes, on="txId", how="left", validate="one_to_one")
    feats["y"] = feats.pop("class").map(LABEL_MAP).astype(np.int8)
    tx_index = pd.Series(np.arange(len(feats)), index=feats["txId"].to_numpy())

    e = pd.read_csv(DATA_DIR / "txs_edgelist.csv", engine="pyarrow")
    tx_edges = np.stack([tx_index[e["txId1"]].to_numpy(), tx_index[e["txId2"]].to_numpy()], 1)

    a_in = pd.read_csv(DATA_DIR / "AddrTx_edgelist.csv", engine="pyarrow")
    a_out = pd.read_csv(DATA_DIR / "TxAddr_edgelist.csv", engine="pyarrow")
    addrs = pd.Index(pd.unique(pd.concat([a_in["input_address"], a_out["output_address"]])))
    spends = np.stack(
        [addrs.get_indexer(a_in["input_address"]), tx_index[a_in["txId"]].to_numpy()], 1
    )
    pays = np.stack(
        [tx_index[a_out["txId"]].to_numpy(), addrs.get_indexer(a_out["output_address"])], 1
    )
    return Elliptic(feats, tx_edges.astype(np.int64), spends.astype(np.int64),
                    pays.astype(np.int64), len(addrs))


def load() -> Elliptic:
    """Load Elliptic++ (cached after the first call)."""
    ensure_dirs()
    paths = {k: CACHE_DIR / f"{k}.parquet" for k in ("tx", "tx_edges", "spends", "pays")}
    if not all(p.exists() for p in paths.values()):
        d = _read_raw()
        d.tx.to_parquet(paths["tx"])
        for k in ("tx_edges", "spends", "pays"):
            pd.DataFrame(getattr(d, k), columns=["src", "dst"]).to_parquet(paths[k])
        (CACHE_DIR / "n_addresses.txt").write_text(str(d.n_addresses))
        return d
    tx = pd.read_parquet(paths["tx"])
    arrs = {k: pd.read_parquet(paths[k]).to_numpy() for k in ("tx_edges", "spends", "pays")}
    n_addr = int((CACHE_DIR / "n_addresses.txt").read_text())
    return Elliptic(tx, arrs["tx_edges"], arrs["spends"], arrs["pays"], n_addr)
