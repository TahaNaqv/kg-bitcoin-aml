"""Graph models on the time-sliced knowledge graph.

Because every edge of the TS-KG stays inside one time step, the training,
validation and test periods form disconnected sub-graphs. A full-batch forward
pass therefore never propagates information across periods: the evaluation is
strictly inductive (test-period nodes and edges are never seen while
training), even though all periods are stored in one tensor for convenience.
No normalisation layer that pools statistics across nodes is used.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.data import HeteroData

from . import kg as K
from .config import FEATURE_SETS
from .data import Elliptic
from .features import log_scale

TX_RELATIONS = {
    "flows": ("tx", "flows", "tx"),
    "rev_flows": ("tx", "rev_flows", "tx"),
    "spends": ("addr", "spends", "tx"),
    "rev_spends": ("tx", "rev_spends", "addr"),
    "pays": ("tx", "pays", "addr"),
    "rev_pays": ("addr", "rev_pays", "tx"),
}

# Model variants: which relations are kept, and whether address-tx edges are
# randomly rewired within each time step (destroys entity structure while
# preserving every transaction's number of addresses).
VARIANTS = {
    "MLP": {"relations": [], "shuffle": False},
    "SAGE-FLOW": {"relations": ["flows", "rev_flows"], "shuffle": False},
    "KG-SAGE": {"relations": list(TX_RELATIONS), "shuffle": False},
    "KG-SAGE -flow": {"relations": ["spends", "rev_spends", "pays", "rev_pays"], "shuffle": False},
    "KG-SAGE -input": {"relations": ["flows", "rev_flows", "pays", "rev_pays"], "shuffle": False},
    "KG-SAGE -output": {"relations": ["flows", "rev_flows", "spends", "rev_spends"],
                        "shuffle": False},
    "KG-SAGE shuffled": {"relations": list(TX_RELATIONS), "shuffle": True},
}


def _standardise(x: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    mu = x[fit_rows].mean(0)
    sd = x[fit_rows].std(0)
    sd[sd < 1e-6] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def build_heterodata(d: Elliptic, g: K.TSKG, train_steps, tx_extra=None,
                     shuffle_seed: int | None = None) -> HeteroData:
    """Build the PyG graph used by every graph model.

    * Transaction inputs are the CLEAN features plus (optionally) the tabular KG
      features ``tx_extra``, i.e. exactly the inputs of the CLEAN+KG tree models,
      so that graph models and trees differ only in message passing.
    * Address@t instances linked to a single transaction cannot connect two
      transactions; their attributes are already summarised in ``tx_extra``, so
      they are pruned (920,691 -> 246,545 instances) to make CPU training feasible.
    * Features are standardised with statistics of training-period nodes only.
    """
    tx_df = log_scale(d.tx[FEATURE_SETS["CLEAN"]], d.step)
    if tx_extra is not None:
        tx_df = pd.concat([tx_df, np.log1p(tx_extra.clip(lower=0))], axis=1)
    tx_fit = np.isin(d.step, list(train_steps))
    tx_x = _standardise(tx_df.to_numpy(np.float32), tx_fit)

    # prune instances that touch fewer than two distinct transactions
    pairs = np.unique(np.concatenate([g.spends, g.pays[:, ::-1]]), axis=0)
    n_tx_per_inst = np.bincount(pairs[:, 0], minlength=g.n_inst)
    keep = n_tx_per_inst >= 2
    new_id = np.full(g.n_inst, -1, np.int64)
    new_id[keep] = np.arange(keep.sum())
    spends = g.spends[keep[g.spends[:, 0]]].copy()
    spends[:, 0] = new_id[spends[:, 0]]
    pays = g.pays[keep[g.pays[:, 1]]].copy()
    pays[:, 1] = new_id[pays[:, 1]]
    inst_step = g.inst_step[keep]
    a_x = np.log1p(g.inst_x.to_numpy(np.float32)[keep])
    a_x = _standardise(a_x, np.isin(inst_step, list(train_steps)))

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        for e, col in ((spends, 0), (pays, 1)):
            st = inst_step[e[:, col]]
            for s in np.unique(st):
                idx = np.flatnonzero(st == s)
                e[idx, col] = e[rng.permutation(idx), col]

    h = HeteroData()
    h["tx"].x = torch.from_numpy(tx_x)
    h["addr"].x = torch.from_numpy(a_x)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T)).long()  # noqa: E731
    h["tx", "flows", "tx"].edge_index = t(g.flows)
    h["tx", "rev_flows", "tx"].edge_index = t(g.flows[:, ::-1])
    h["addr", "spends", "tx"].edge_index = t(spends)
    h["tx", "rev_spends", "addr"].edge_index = t(spends[:, ::-1])
    h["tx", "pays", "addr"].edge_index = t(pays)
    h["addr", "rev_pays", "tx"].edge_index = t(pays[:, ::-1])
    return h


def mean_adjacency(edge_index: torch.Tensor, n_src: int, n_dst: int) -> torch.Tensor:
    """Row-normalised sparse matrix A (n_dst x n_src) so that A @ H is the mean of
    in-neighbour features, i.e. GraphSAGE mean aggregation."""
    src, dst = edge_index
    deg = torch.bincount(dst, minlength=n_dst).clamp(min=1).float()
    a = torch.sparse_coo_tensor(torch.stack([dst, src]), 1.0 / deg[dst], (n_dst, n_src))
    a = a.coalesce()
    return a.to_sparse_csr(), a.t().coalesce().to_sparse_csr()


class _SpMM(torch.autograd.Function):
    """A @ H with a cached transpose for the backward pass (A is a constant)."""

    @staticmethod
    def forward(ctx, a, a_t, h):
        ctx.a_t = a_t
        return torch.sparse.mm(a, h)

    @staticmethod
    def backward(ctx, grad):
        return None, None, torch.sparse.mm(ctx.a_t, grad)


class RelSAGELayer(nn.Module):
    """Relational GraphSAGE layer (mean aggregator):
    h'_v = W_root^{type(v)} h_v + sum_r W_r mean_{u in N_r(v)} h_u.
    Equivalent to PyG HeteroConv({r: SAGEConv}, aggr='sum') with shared root weights,
    implemented with sparse matrix products for CPU speed."""

    def __init__(self, relations: list[tuple], node_types: list[str], hidden: int):
        super().__init__()
        self.relations = relations
        self.root = nn.ModuleDict({k: nn.Linear(hidden, hidden) for k in node_types})
        self.rel = nn.ModuleDict({"__".join(r): nn.Linear(hidden, hidden, bias=False)
                                  for r in relations})

    def forward(self, h: dict, adj: dict) -> dict:
        out = {k: self.root[k](v) for k, v in h.items()}
        for r in self.relations:
            src, _, dst = r
            out[dst] = out[dst] + self.rel["__".join(r)](_SpMM.apply(*adj[r], h[src]))
        return out


class KGSage(nn.Module):
    """Relation-specific GraphSAGE over the TS-KG with jumping-knowledge
    concatenation of every transaction layer for the classifier. With no
    relations it reduces to an MLP of matched capacity."""

    def __init__(self, in_dims: dict[str, int], relations: list[str], hidden: int = 64,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.relations = [TX_RELATIONS[r] for r in relations]
        types = sorted({t for r in self.relations for t in (r[0], r[2])} | {"tx"})
        self.types = types
        self.enc = nn.ModuleDict({k: nn.Linear(in_dims[k], hidden) for k in types})
        n_layers = layers if self.relations else 0
        self.convs = nn.ModuleList(RelSAGELayer(self.relations, types, hidden)
                                   for _ in range(n_layers))
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden * (1 + n_layers), hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_dict, adj, rows: torch.Tensor | None = None,
                return_embedding: bool = False):
        """``rows`` restricts the classifier head to the nodes that are scored.

        Dropout is applied to the transaction input encoding and inside the head;
        it is omitted on the 247k address nodes because CPU random-number
        generation dominated the runtime.
        """
        h = {k: F.relu(self.enc[k](x_dict[k])) for k in self.types}
        h["tx"] = F.dropout(h["tx"], self.dropout, self.training)
        outs = [h["tx"]]
        for conv in self.convs:
            h = {k: F.relu(v) for k, v in conv(h, adj).items()}
            outs.append(h["tx"])
        z = torch.cat(outs, 1)
        if rows is not None:
            z = z[rows]
        return z if return_embedding else self.head(z).squeeze(-1)


def train_eval(h: HeteroData, y: np.ndarray, masks: dict[str, np.ndarray], relations: list[str],
               seed: int, epochs: int = 300, patience: int = 40, lr: float = 5e-3,
               weight_decay: float = 5e-4, hidden: int = 64, layers: int = 2,
               verbose: bool = False) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = KGSage({"tx": h["tx"].x.shape[1], "addr": h["addr"].x.shape[1]}, relations,
                   hidden=hidden, layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    yt = torch.from_numpy(y.astype(np.float32))
    tr = torch.from_numpy(np.flatnonzero(masks["train"]))
    va = masks["val"]
    pos_weight = torch.tensor((y[masks["train"]] == 0).sum() / (y[masks["train"]] == 1).sum())
    x_dict = {"tx": h["tx"].x, "addr": h["addr"].x}
    n = {"tx": h["tx"].num_nodes, "addr": h["addr"].num_nodes}
    ei = {r: mean_adjacency(h[r].edge_index, n[r[0]], n[r[2]]) for r in model.relations}

    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.perf_counter()
    va_rows = torch.from_numpy(np.flatnonzero(va))
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_dict, ei, rows=tr)
        loss = F.binary_cross_entropy_with_logits(logits, yt[tr], pos_weight=pos_weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(x_dict, ei, rows=va_rows)).numpy()
        ap = average_precision_score(y[va], p_va)
        if ap > best:
            best, best_state, best_epoch, bad = ap, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d} loss {loss.item():.4f} val AP {ap:.4f}", flush=True)
        if bad >= patience:
            break
    fit_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x_dict, ei)).numpy()
    return p, {"fit_seconds": fit_s, "best_epoch": best_epoch, "val_pr_auc_select": best,
               "n_params": sum(q.numel() for q in model.parameters()), "model": model}
