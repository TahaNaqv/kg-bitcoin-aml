"""Paper figures (PDF, IEEE single-column width) from results/*.csv."""

from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from kgaml.config import DARK_MARKET_SHUTDOWN_STEP, FIGURES_DIR, REPO_ROOT, RESULTS_DIR  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet);
# low-contrast slots are always paired with markers/labels and tables in the paper.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "pdf.fonttype": 42,
})
W = 3.5  # IEEE column width (inches)
PAPER_FIG = REPO_ROOT / "paper" / "figures"


def save(fig, name, in_paper=True):
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    PAPER_FIG.mkdir(exist_ok=True, parents=True)
    p = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    if in_paper:
        shutil.copy(p, PAPER_FIG / p.name)
    plt.close(fig)


def homophily():
    h = pd.read_csv(RESULTS_DIR / "homophily.csv")
    rels = ["FLOW", "CO_SPEND", "CO_RECEIVE", "CHAIN", "KG_UNION"]
    labels = ["Flow", "Co-spend", "Co-recv.", "Chain", "KG union"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.55), sharey=True)
    for ax, metric, title in ((axes[0], "adjusted_homophily", "Adjusted homophily"),
                              (axes[1], "illicit_coverage", "Illicit coverage")):
        for i, (period, color) in enumerate(zip(h.period.unique(), C)):
            sub = h[h.period == period].set_index("relation").loc[rels]
            xs = [j + (i - 0.5) * 0.38 for j in range(len(rels))]
            ax.bar(xs, sub[metric], width=0.36, color=color, label=period.split(" ")[0],
                   edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(rels)), labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["train (steps 1-34)", "test (steps 35-49)"], frameon=False, ncols=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_homophily", in_paper=False)  # homophily is Table II in the paper


def per_step():
    tab = pd.read_csv(RESULTS_DIR / "tabular_per_step.csv")
    gnn = pd.read_csv(RESULTS_DIR / "gnn_per_step.csv")
    tab["label"] = tab["model"] + " " + tab["features"]
    gnn["label"] = gnn["model"]
    df = pd.concat([tab, gnn])
    series = [("XGB ALL", "XGB, All (leaky)"), ("XGB CLEAN", "XGB, Clean"),
              ("XGB CLEAN+KG", "XGB, Clean+KG"), ("SAGE-FLOW", "SAGE-Flow"),
              ("KG-SAGE", "KG-SAGE")]
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for i, (key, name) in enumerate(series):
        s = df[df.label == key].groupby("step")["f1"].mean()
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=C[i], marker=MARKERS[i], markersize=3.2,
                linestyle="--" if "leaky" in name else "-", label=name)
    ax.axvline(DARK_MARKET_SHUTDOWN_STEP - 0.5, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(DARK_MARKET_SHUTDOWN_STEP - 0.3, 0.97, "dark-market\nshutdown", fontsize=6,
            color=INK2, va="top")
    ax.set_xlabel("Test time step")
    ax.set_ylabel("Illicit F1")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(35, 50, 2))
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6)
    fig.tight_layout()
    save(fig, "fig_per_step")


def shap_groups():
    g = pd.read_csv(RESULTS_DIR / "shap_groups.csv")
    order = ["tx-intrinsic", "predecessor agg.", "address (leaky)", "successor agg. (leaky)",
             "KG address history", "KG relation degree", "KG entity component"]
    models = ["XGB ALL", "XGB CLEAN+KG", "KG-SAGE (grad x input)"]
    names = ["XGB, All", "XGB, Clean+KG", "KG-SAGE"]
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for j, m in enumerate(models):
        sub = g[g.model == m].set_index("group")["share"].reindex(order).fillna(0)
        left = 0.0
        for i, grp in enumerate(order):
            v = sub[grp]
            if v <= 0:
                continue
            ax.barh(j, v, left=left, color=C[i], edgecolor="white", linewidth=0.8, height=0.62,
                    label=grp if j == 0 or grp.startswith("KG") else None)
            if v >= 0.08:
                ax.text(left + v / 2, j, f"{v:.0%}", ha="center", va="center", fontsize=5.8,
                        color="white" if i in (0, 1, 5, 6) else INK)
            left += v
    ax.set_yticks(range(len(models)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of total attribution")
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, ncols=3, fontsize=5.6,
              loc="upper center", bbox_to_anchor=(0.42, -0.32))
    fig.tight_layout()
    save(fig, "fig_attribution")


if __name__ == "__main__":
    for fn in (homophily, per_step, shap_groups):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)
