"""Generate the LaTeX result tables from results/*.csv (no hand-typed numbers)."""

from __future__ import annotations

import pandas as pd

from kgaml.config import REPO_ROOT, RESULTS_DIR

OUT = REPO_ROOT / "paper" / "tables"


def ms(df: pd.DataFrame, col: str, bold: bool = False) -> str:
    m, s = df[col].mean(), df[col].std()
    txt = f"{m:.3f}" if pd.isna(s) or len(df) < 2 else f"{m:.3f}\\,$\\pm$\\,{s:.3f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def runs() -> pd.DataFrame:
    tab = pd.read_csv(RESULTS_DIR / "tabular_runs.csv")
    gnn = pd.read_csv(RESULTS_DIR / "gnn_runs.csv") if (RESULTS_DIR / "gnn_runs.csv").exists() \
        else pd.DataFrame(columns=tab.columns)
    return pd.concat([tab, gnn], ignore_index=True)


MAIN_ROWS = [
    ("Leaky features (as in prior work)", None),
    ("RF", "ALL"), ("XGB", "ALL"),
    ("Leakage-controlled features", None),
    ("LR", "CLEAN"), ("RF", "CLEAN"), ("XGB", "CLEAN"),
    ("+ knowledge-graph features", None),
    ("RF", "CLEAN+KG"), ("XGB", "CLEAN+KG"),
    ("RF", "CLEAN+KG+NBR"), ("XGB", "CLEAN+KG+NBR"),
    ("Neural models (inputs = Clean+KG)", None),
    ("MLP", "CLEAN+KG"), ("SAGE-FLOW", "CLEAN+KG"), ("KG-SAGE", "CLEAN+KG"),
]
NAME = {"ALL": "All", "CLEAN": "Clean", "CLEAN+KG": "Clean+KG", "CLEAN+KG+NBR": "Clean+KG+Nbr",
        "SAGE-FLOW": "SAGE-Flow", "KG-SAGE": "KG-SAGE", "MLP": "MLP"}


def main_table(r: pd.DataFrame) -> str:
    test, unseen = r[r.view == "test"], r[r.view == "test_unseen_entity"]
    clean = test[~test.features.str.startswith("ALL")]
    best = {c: clean.groupby(["model", "features"])[c].mean().max()
            for c in ("f1", "pr_auc", "roc_auc", "precision_at_1pct")}
    lines = [r"\begin{table*}[t]",
             r"\caption{Illicit-class results on the test period (steps 35--49), mean $\pm$ std over seeds. "
             r"Thresholds are tuned on the validation window. Best leakage-controlled value in bold.}",
             r"\label{tab:main}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{@{}llccccccr@{}}", r"\toprule",
             r"Model & Features & F1 & Precision & Recall & PR-AUC & ROC-AUC & P@1\% & Fit (s) \\",
             r"\midrule"]
    for model, fset in MAIN_ROWS:
        if fset is None:
            lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{model}}}}} \\\\")
            continue
        t = test[(test.model == model) & (test.features == fset)]
        if t.empty:
            continue
        leaky = fset.startswith("ALL")
        cells = []
        for c in ("f1", "precision", "recall", "pr_auc", "roc_auc", "precision_at_1pct"):
            b = (not leaky) and c in best and abs(t[c].mean() - best[c]) < 1e-9
            cells.append(ms(t, c, b))
        name = NAME.get(model, model)
        lines.append(f"{name} & {NAME[fset]} & " + " & ".join(cells)
                     + f" & {t.fit_seconds.mean():.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def unseen_table(r: pd.DataFrame) -> str:
    rows = [("XGB", "ALL"), ("XGB", "CLEAN"), ("XGB", "CLEAN+KG"), ("SAGE-FLOW", "CLEAN+KG"),
            ("KG-SAGE", "CLEAN+KG")]
    views = [("test", "All test"), ("test_unseen_entity", "Unseen entities"),
             ("test_pre_shutdown", "Steps 35--42"), ("test_post_shutdown", "Steps 43--49")]
    lines = [r"\begin{table}[t]", r"\caption{Illicit F1 by evaluation view (mean over seeds).}",
             r"\label{tab:views}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}l" + "c" * len(views) + "@{}}", r"\toprule",
             "Model & " + " & ".join(v[1] for v in views) + r" \\", r"\midrule"]
    for model, fset in rows:
        sub = r[(r.model == model) & (r.features == fset)]
        if sub.empty:
            continue
        cells = [f"{sub[sub.view == v].f1.mean():.3f}" for v, _ in views]
        label = f"{NAME.get(model, model)}" + (f", {NAME[fset]}" if model == "XGB" else "")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    n = r[(r.model == "XGB") & (r.features == "ALL")].groupby("view")["n_illicit"].first()
    lines.append(r"\midrule")
    lines.append("\\# illicit & " + " & ".join(f"{int(n.get(v, 0))}" for v, _ in views) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def ablation_table(r: pd.DataFrame) -> str:
    variants = [("KG-SAGE", "Full TS-KG"), ("KG-SAGE -flow", "-- FLOWS\\_TO"),
                ("KG-SAGE -input", "-- SPENDS\\_IN (inputs)"), ("KG-SAGE -output", "-- PAYS (outputs)"),
                ("KG-SAGE shuffled", "Address edges rewired"), ("SAGE-FLOW", "FLOWS\\_TO only"),
                ("MLP", "No edges (MLP)")]
    test = r[r.view == "test"]
    # compare every variant on the seeds that all ablation variants share
    abl = test[test.model.isin([v for v, _ in variants[1:5]])]
    base_seeds = sorted(set(abl.seed.unique())) if not abl.empty else [0]
    full = test[(test.model == "KG-SAGE") & test.seed.isin(base_seeds)]
    lines = [r"\begin{table}[t]",
             r"\caption{KG-SAGE relation ablations (test period; mean $\pm$ std over the two ablation seeds).}",
             r"\label{tab:ablation}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"& \multicolumn{3}{c}{All test} & Unseen \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-5}",
             r"Graph & F1 & PR-AUC & $\Delta$PR-AUC & PR-AUC \\", r"\midrule"]
    for v, name in variants:
        t = test[(test.model == v) & test.seed.isin(base_seeds)]
        if t.empty:
            continue
        delta = t.pr_auc.mean() - full.pr_auc.mean()
        u = r[(r.view == "test_unseen_entity") & (r.model == v) & r.seed.isin(base_seeds)]
        lines.append(f"{name} & {ms(t, 'f1')} & {ms(t, 'pr_auc')} & "
                     f"{'--' if v == 'KG-SAGE' else f'{delta:+.3f}'} & {u.pr_auc.mean():.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def imbalance_table() -> str:
    p = RESULTS_DIR / "imbalance_runs.csv"
    if not p.exists():
        return "% imbalance results pending"
    df = pd.read_csv(p)
    lines = [r"\begin{table}[t]",
             r"\caption{Imbalance strategies on Clean+KG features (test F1; three seeds). "
             r"PR-AUC does not depend on the threshold.}",
             r"\label{tab:imbalance}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Model & Strategy & F1 @0.5 & F1 @val & PR-AUC \\", r"\midrule"]
    for model in ("XGB", "RF"):
        for strat in ("none", "class_weight", "SMOTE", "SMOTE-ENN"):
            s = df[(df.model == model) & (df.strategy == strat)]
            a, b = s[s.threshold_rule == "0.5"], s[s.threshold_rule == "val-tuned"]
            name = {"none": "None", "class_weight": "Class weights"}.get(strat, strat)
            lines.append(f"{model} & {name} & {ms(a, 'f1')} & {ms(b, 'f1')} & {ms(b, 'pr_auc')} \\\\")
        if model == "XGB":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def numbers(r: pd.DataFrame) -> str:
    """LaTeX macros for every number quoted in the prose."""
    test = r[r.view == "test"]
    out = []

    def mac(name, model, fset, col, view="test"):
        sub = r[(r.view == view) & (r.model == model) & (r.features == fset)]
        if not sub.empty:
            fmt = ".0f" if col == "fit_seconds" else ".3f"
            out.append(f"\\newcommand{{\\{name}}}{{{sub[col].mean():{fmt}}}}")

    for prefix, model in (("Xgb", "XGB"), ("Rf", "RF")):
        for tag, fset in (("All", "ALL"), ("Clean", "CLEAN"), ("CleanKg", "CLEAN+KG"),
                          ("CleanKgNbr", "CLEAN+KG+NBR")):
            mac(f"{prefix}{tag}Fone", model, fset, "f1")
            mac(f"{prefix}{tag}Pr", model, fset, "pr_auc")
            mac(f"{prefix}{tag}UnseenFone", model, fset, "f1", "test_unseen_entity")
            mac(f"{prefix}{tag}UnseenPr", model, fset, "pr_auc", "test_unseen_entity")
            mac(f"{prefix}{tag}Fit", model, fset, "fit_seconds")
            mac(f"{prefix}{tag}ValPr", model, fset, "pr_auc", "val")
    for tag, model in (("Mlp", "MLP"), ("SageFlow", "SAGE-FLOW"), ("KgSage", "KG-SAGE")):
        mac(f"{tag}Fone", model, "CLEAN+KG", "f1")
        mac(f"{tag}Pr", model, "CLEAN+KG", "pr_auc")
        mac(f"{tag}Roc", model, "CLEAN+KG", "roc_auc")
        mac(f"{tag}UnseenFone", model, "CLEAN+KG", "f1", "test_unseen_entity")
        mac(f"{tag}UnseenPr", model, "CLEAN+KG", "pr_auc", "test_unseen_entity")
        mac(f"{tag}ValPr", model, "CLEAN+KG", "pr_auc", "val")
        sub = test[(test.model == model)]
        if not sub.empty:
            out.append(f"\\newcommand{{\\{tag}Fit}}{{{sub.fit_seconds.mean():.0f}}}")
    def m(model, fset, col="f1"):
        sub = test[(test.model == model) & (test.features == fset)]
        return sub[col].mean()
    if not test.empty:
        leak = m("XGB", "ALL") - m("XGB", "CLEAN")
        out += [f"\\newcommand{{\\XgbLeakDrop}}{{{leak:.2f}}}",
                f"\\newcommand{{\\XgbLeakShare}}{{{100 * leak / m('XGB', 'ALL'):.0f}\\%}}",
                f"\\newcommand{{\\XgbKgGain}}{{{m('XGB', 'CLEAN+KG') - m('XGB', 'CLEAN'):.2f}}}",
                f"\\newcommand{{\\XgbKgPrGain}}{{{m('XGB', 'CLEAN+KG', 'pr_auc') - m('XGB', 'CLEAN', 'pr_auc'):.2f}}}"]
    # explanation shares and agreement (RQ3)
    gp, ag = RESULTS_DIR / "shap_groups.csv", RESULTS_DIR / "explain_agreement.json"
    if gp.exists() and ag.exists():
        import json
        g = pd.read_csv(gp)

        def share(model, pred):
            s = g[(g.model == model) & g.group.map(pred)].share.sum()
            return f"{100 * s:.0f}\\%"

        out += [
            f"\\newcommand{{\\ShapAllLeaky}}{{{share('XGB ALL', lambda x: 'leaky' in x)}}}",
            f"\\newcommand{{\\ShapAllAddr}}{{{share('XGB ALL', lambda x: x == 'address (leaky)')}}}",
            f"\\newcommand{{\\ShapCleanKgKg}}{{{share('XGB CLEAN+KG', lambda x: x.startswith('KG'))}}}",
            f"\\newcommand{{\\GxiKg}}{{{share('KG-SAGE (grad x input)', lambda x: x.startswith('KG'))}}}",
            f"\\newcommand{{\\GxiAddrHist}}{{{share('KG-SAGE (grad x input)', lambda x: x == 'KG address history')}}}",
            f"\\newcommand{{\\ShapCleanKgAddrHist}}{{{share('XGB CLEAN+KG', lambda x: x == 'KG address history')}}}",
        ]
        a = json.loads(ag.read_text())
        out += [f"\\newcommand{{\\ExplainRho}}{{{a['spearman_rho_all_features']:.2f}}}",
                f"\\newcommand{{\\ExplainOverlap}}{{{a['top10_overlap']}}}"]
    # ablations, compared on the seeds shared by all ablation variants
    abl_models = ["KG-SAGE -flow", "KG-SAGE -input", "KG-SAGE -output", "KG-SAGE shuffled"]
    abl = r[r.model.isin(abl_models)]
    if not abl.empty:
        seeds = sorted(abl.seed.unique())
        for tag, model in (("Full", "KG-SAGE"), ("Shuf", "KG-SAGE shuffled"),
                           ("NoOut", "KG-SAGE -output"), ("NoIn", "KG-SAGE -input"),
                           ("NoFlow", "KG-SAGE -flow")):
            for vtag, view in (("", "test"), ("Unseen", "test_unseen_entity")):
                sub = r[(r.model == model) & (r.view == view) & r.seed.isin(seeds)]
                if not sub.empty:
                    out.append(f"\\newcommand{{\\Abl{tag}{vtag}Pr}}{{{sub.pr_auc.mean():.3f}}}")
    dg = RESULTS_DIR / "gnn_diagnosis.json"
    if dg.exists():
        import json
        z = json.loads(dg.read_text())
        pct = lambda v: f"{100 * v:.0f}\\%"  # noqa: E731
        out += [f"\\newcommand{{\\DiagFp}}{{{z['false_positives']:,}}}".replace(",", "{,}"),
                f"\\newcommand{{\\DiagFpRecurShare}}{{{pct(z['fp_recurring_entity_share'])}}}",
                f"\\newcommand{{\\DiagFpRateRecur}}{{{pct(z['fp_rate_recurring_licit'])}}}",
                f"\\newcommand{{\\DiagFpRateUnseen}}{{{pct(z['fp_rate_unseen_licit'])}}}",
                f"\\newcommand{{\\DiagOorFp}}{{{pct(z['recurring_fp_with_oor_address_input'])}}}",
                f"\\newcommand{{\\DiagOorTn}}{{{pct(z['recurring_tn_with_oor_address_input'])}}}"]
    return "\n".join(out) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = runs()
    (OUT / "numbers.tex").write_text(numbers(r))
    (OUT / "main.tex").write_text(main_table(r))
    (OUT / "views.tex").write_text(unseen_table(r))
    (OUT / "ablation.tex").write_text(ablation_table(r))
    (OUT / "imbalance.tex").write_text(imbalance_table())
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
