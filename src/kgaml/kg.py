"""Time-sliced transaction-address knowledge graph (TS-KG).

Entities
    Transaction  -- one node per Elliptic transaction (lives in one time step).
    Address@t    -- one node per (address, time step) in which the address is
                    active. Splitting an address into per-step instances keeps
                    every edge inside a single time step, so the graph of the
                    test period is disconnected from the training graph
                    (strictly inductive by construction), and an address's
                    *history* enters only through causal node attributes.
Relations
    (tx)-[FLOWS_TO]->(tx)        fund flow between transactions (Elliptic edges)
    (addr@t)-[SPENDS_IN]->(tx)   address is an input of tx    (AddrTx)
    (tx)-[PAYS]->(addr@t)        address is an output of tx   (TxAddr)
Derived meta-relations between transactions (2-hop paths through Address@t)
    CO_SPEND   tx-addr-tx via two SPENDS_IN  (shared input address;
               transaction labels are inherited from the input entity, so this
               relation is homophilous *by construction*)
    CO_RECEIVE tx-addr-tx via two PAYS       (shared output address)
    CHAIN      tx-PAYS->addr-SPENDS_IN->tx   (payment chain through an address)

Address attributes are computed only from steps strictly before t (history)
plus counts inside step t, so no feature is observed after the end of the
step in which the transaction occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .data import Elliptic

# Address instances touching more than this many transactions in a single step
# (exchanges, mixers, pools) are excluded from the *meta-path* relations and
# component statistics, which would otherwise become dense cliques. They remain
# in the knowledge graph used by the GNN.
HUB_DEGREE = 200

ADDR_FEATURES = [
    "n_spend_t",
    "n_recv_t",
    "seen_before",
    "n_prior_steps",
    "steps_since_first",
    "steps_since_last",
    "prior_n_spend",
    "prior_n_recv",
]


@dataclass
class TSKG:
    n_tx: int
    n_inst: int
    inst_addr: np.ndarray  # address id of each instance
    inst_step: np.ndarray  # step of each instance
    inst_x: pd.DataFrame  # causal instance attributes (ADDR_FEATURES)
    spends: np.ndarray  # (E, 2) [inst, tx]
    pays: np.ndarray  # (E, 2) [tx, inst]
    flows: np.ndarray  # (E, 2) [tx, tx]
    hub: np.ndarray  # bool per instance

    # ----- incidence matrices (tx x inst), hubs removed -----
    def incidence(self, which: str, drop_hubs: bool = True) -> sp.csr_matrix:
        e = self.spends[:, ::-1] if which == "in" else self.pays
        if drop_hubs:
            e = e[~self.hub[e[:, 1]]]
        data = np.ones(len(e), dtype=np.float32)
        m = sp.csr_matrix((data, (e[:, 0], e[:, 1])), shape=(self.n_tx, self.n_inst))
        m.data[:] = 1.0
        return m

    def meta_relation(self, name: str) -> sp.csr_matrix:
        """Binary symmetric tx x tx adjacency of a relation (no self loops)."""
        b_in, b_out = self.incidence("in"), self.incidence("out")
        if name == "FLOW":
            f = self.flows
            m = sp.csr_matrix((np.ones(len(f), np.float32), (f[:, 0], f[:, 1])),
                              shape=(self.n_tx, self.n_tx))
        elif name == "CO_SPEND":
            m = b_in @ b_in.T
        elif name == "CO_RECEIVE":
            m = b_out @ b_out.T
        elif name == "CHAIN":
            m = b_out @ b_in.T
        else:
            raise ValueError(name)
        m = ((m + m.T) > 0).astype(np.float32).tocsr()
        m.setdiag(0)
        m.eliminate_zeros()
        return m


def build(d: Elliptic) -> TSKG:
    step = d.step
    # ---- address instances (address, step) ----
    a_sp, t_sp = d.spends[:, 0], d.spends[:, 1]
    t_pa, a_pa = d.pays[:, 0], d.pays[:, 1]
    key_sp = a_sp * 64 + step[t_sp]
    key_pa = a_pa * 64 + step[t_pa]
    keys, inv = np.unique(np.concatenate([key_sp, key_pa]), return_inverse=True)
    inst_sp, inst_pa = inv[: len(key_sp)], inv[len(key_sp):]
    inst_addr, inst_step = keys // 64, (keys % 64).astype(np.int16)
    n_inst = len(keys)

    # ---- causal attributes ----
    n_spend_t = np.bincount(inst_sp, minlength=n_inst)
    n_recv_t = np.bincount(inst_pa, minlength=n_inst)
    df = pd.DataFrame({"addr": inst_addr, "step": inst_step,
                       "n_spend_t": n_spend_t, "n_recv_t": n_recv_t})
    # keys are sorted by (addr, step) already, so groupby preserves temporal order
    g = df.groupby("addr", sort=False)
    df["n_prior_steps"] = g.cumcount()
    df["seen_before"] = (df["n_prior_steps"] > 0).astype(np.int8)
    first = g["step"].transform("min")
    prev = g["step"].shift(1)
    df["steps_since_first"] = (df["step"] - first).astype(np.int16)
    df["steps_since_last"] = (df["step"] - prev).fillna(0).astype(np.int16)
    df["prior_n_spend"] = g["n_spend_t"].cumsum() - df["n_spend_t"]
    df["prior_n_recv"] = g["n_recv_t"].cumsum() - df["n_recv_t"]
    inst_x = df[ADDR_FEATURES].astype(np.float32)

    hub = (n_spend_t + n_recv_t) > HUB_DEGREE
    return TSKG(
        n_tx=len(step), n_inst=n_inst, inst_addr=inst_addr, inst_step=inst_step,
        inst_x=inst_x,
        spends=np.stack([inst_sp, t_sp], 1), pays=np.stack([t_pa, inst_pa], 1),
        flows=d.tx_edges, hub=hub,
    )


# ---------------------------------------------------------------------------
# Tabular "knowledge" features derived from the TS-KG (all causal)
# ---------------------------------------------------------------------------
def structural_features(kg: TSKG) -> pd.DataFrame:
    """Per-transaction features describing its position in the TS-KG."""
    out = {}
    b_in_all, b_out_all = kg.incidence("in", False), kg.incidence("out", False)
    seen = kg.inst_x["seen_before"].to_numpy(np.float32)
    nprior = np.log1p(kg.inst_x["n_prior_steps"].to_numpy(np.float32))
    since = kg.inst_x["steps_since_first"].to_numpy(np.float32)
    for side, b in (("in", b_in_all), ("out", b_out_all)):
        cnt = np.asarray(b.sum(1)).ravel()
        out[f"kg_{side}_addr_n"] = cnt
        out[f"kg_{side}_addr_seen_frac"] = np.asarray(b @ seen).ravel() / np.maximum(cnt, 1)
        out[f"kg_{side}_addr_prior_mean"] = np.asarray(b @ nprior).ravel() / np.maximum(cnt, 1)
        out[f"kg_{side}_addr_age_max"] = _rowmax(b, since)
        out[f"kg_{side}_hub_n"] = np.asarray(b @ kg.hub.astype(np.float32)).ravel()
    for rel in ("FLOW", "CO_SPEND", "CO_RECEIVE", "CHAIN"):
        out[f"kg_deg_{rel.lower()}"] = np.asarray(kg.meta_relation(rel).sum(1)).ravel()
    # size of the transaction's entity component (tx-address bipartite, hubs removed)
    comp = entity_components(kg)
    sizes = np.bincount(comp)
    out["kg_entity_component_size"] = sizes[comp]
    return pd.DataFrame(out).astype(np.float32)


def entity_components(kg: TSKG) -> np.ndarray:
    """Connected components of the step-local tx-address graph (hubs removed)."""
    b = (kg.incidence("in") + kg.incidence("out")).tocoo()
    n = kg.n_tx + kg.n_inst
    adj = sp.coo_matrix((np.ones(b.nnz), (b.row, b.col + kg.n_tx)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    labels = labels[: kg.n_tx]
    # relabel to tx-only contiguous ids
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def neighbour_mean_features(kg: TSKG, x: np.ndarray, cols: list[str],
                            relations=("CO_SPEND", "CO_RECEIVE", "CHAIN")) -> pd.DataFrame:
    """Hand-crafted one-hop aggregation over KG meta-relations (no labels used)."""
    m = sum(kg.meta_relation(r) for r in relations)
    m = (m > 0).astype(np.float32)
    deg = np.asarray(m.sum(1)).ravel()
    agg = (m @ x) / np.maximum(deg, 1)[:, None]
    # No neighbours -> missing, not 0 (0 is the dataset mean of z-scored features).
    agg[deg == 0] = np.nan
    df = pd.DataFrame(agg, columns=[f"nbr_{c}" for c in cols]).astype(np.float32)
    df["nbr_count"] = deg.astype(np.float32)
    return df


def _rowmax(b: sp.csr_matrix, v: np.ndarray) -> np.ndarray:
    m = b.multiply(v[None, :] + 1).tocsr()  # +1 so that zero ages survive
    r = np.asarray(m.max(1).todense()).ravel() - 1
    r[np.asarray(b.sum(1)).ravel() == 0] = 0
    return r.astype(np.float32)


# ---------------------------------------------------------------------------
# Homophily diagnostics
# ---------------------------------------------------------------------------
def homophily_table(kg: TSKG, y: np.ndarray, node_mask: np.ndarray) -> pd.DataFrame:
    """Label homophily of each relation, restricted to labelled nodes in node_mask."""
    rows = []
    lab = (y >= 0) & node_mask
    for rel in ("FLOW", "CO_SPEND", "CO_RECEIVE", "CHAIN", "KG_UNION"):
        if rel == "KG_UNION":
            m = sum(kg.meta_relation(r) for r in ("CO_SPEND", "CO_RECEIVE", "CHAIN"))
            m = (m > 0).astype(np.float32).tocsr()
        else:
            m = kg.meta_relation(rel)
        coo = sp.triu(m, 1).tocoo()
        keep = lab[coo.row] & lab[coo.col]
        r, c = coo.row[keep], coo.col[keep]
        same = y[r] == y[c]
        h_edge = same.mean() if len(r) else np.nan
        # class-adjusted homophily (Platonov et al., 2023)
        degs = np.bincount(np.concatenate([r, c]), minlength=len(y))
        p = np.array([degs[(y == k) & lab].sum() for k in (0, 1)], dtype=float)
        p = p / p.sum() if p.sum() else p
        h_adj = (h_edge - (p ** 2).sum()) / (1 - (p ** 2).sum()) if len(r) else np.nan
        sub = m[lab][:, lab]
        yl = y[lab]
        deg_l = np.asarray(sub.sum(1)).ravel()
        ill_nb = np.asarray(sub @ (yl == 1).astype(np.float32)).ravel()
        row = {"relation": rel, "labelled_edges": int(len(r)), "edge_homophily": h_edge,
               "adjusted_homophily": h_adj}
        for k, name in ((1, "illicit"), (0, "licit")):
            sel = yl == k
            covered = deg_l[sel] > 0
            row[f"{name}_coverage"] = covered.mean()
            row[f"{name}_illicit_nbr_share"] = (ill_nb[sel][covered].sum()
                                                / max(deg_l[sel][covered].sum(), 1))
        rows.append(row)
    return pd.DataFrame(rows)
