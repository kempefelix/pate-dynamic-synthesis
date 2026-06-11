#!/usr/bin/env python3
"""
make_results.py — Canonical evaluation pipeline for the thesis
"Improving the Privacy of Federated Learning via Optimized Synthetic Data
Generation" (Felix Kempe, TU Wien).

This single script reproduces the entire quantitative evaluation from the
(epsilon-recomputed, see recompute_epsilon.py) result JSONs in
results/main (MNIST) and results/cifar (CIFAR-10), and replaces the
defective evaluate.py as the canonical pipeline (findings K11.1-K11.3, G2):

  * loads every result JSON and aggregates PER (dataset, beta,
    CONFIGURATION) over seeds — NOT pooled across configurations
    (fixes K11.2: evaluate.py pooled the three configs of each variant
    as if they were seeds);
  * accuracy mean +/- population std (ddof=0), macro-F1, epsilon and
    query counts of the final round;
  * paired t-tests (scipy.stats.ttest_rel) over per-seed final values,
    each configuration vs. the Static baseline, paired on common seeds,
    for BOTH accuracy AND macro-F1 (fixes K11.3: evaluate.py pooled
    configurations into the t-test as well);
  * epsilon savings in % per configuration vs. Static;
  * regenerates all 8 thesis figures in PER-CONFIGURATION resolution
    under the original thesis file names (fixes G2; the CIFAR Pareto
    front now also includes B1 — fix M7);
  * prints the complete new tables 7.2 (tab:main_results) and
    7.5 (tab:main_results_cifar) as LaTeX rows, plus the t-test
    Delta/p values for tables 7.4 (tab:ttest_results) and
    7.6 (tab:ttest_results_cifar);
  * prints auxiliary analyses used in the text: query reductions of
    variant A vs. Static (K5), B-vs-A query comparison (K2/K3),
    realized samples per round (K7/M3), constant variant-B query
    distributions from round 3 (M8), per-class accuracies (G4),
    duplicate-run deviations (K8), and the grid-search summary under
    the corrected accounting (M9).

Usage:
    python3 make_results.py [--repo-root PATH] [--fig-dir PATH]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# ---------------------------------------------------------------------------
# Configuration matrix (filename tokens) and display order
# ---------------------------------------------------------------------------
CONFIGS = {
    "Static": ("static",    "static_s40.0"),
    "A1":     ("variant_a", "varA_s40.0_a1.0_m0.5"),
    "A2":     ("variant_a", "varA_s40.0_a2.0_m0.3"),
    "A3":     ("variant_a", "varA_s40.0_a4.0_m0.1"),
    "B1":     ("variant_b", "varB_s40.0_b1.0_l0.1"),
    "B2":     ("variant_b", "varB_s40.0_b1.0_l0.3"),
    "B3":     ("variant_b", "varB_s40.0_b2.0_l0.5"),
}
CONFIG_ORDER = ["Static", "A1", "A2", "A3", "B1", "B2", "B3"]
VARIANT_ORDER = ["A1", "A2", "A3", "B1", "B2", "B3"]
CONFIG_LABEL = {
    "Static": "Static Baseline",
    "A1": "Variant A1 (cons.)", "A2": "Variant A2 (best)", "A3": "Variant A3 (aggr.)",
    "B1": "Variant B1 (cons.)", "B2": "Variant B2 (best)", "B3": "Variant B3 (aggr.)",
}
COLORS = {
    "Static": "#555555",
    "A1": "#90CAF9", "A2": "#2196F3", "A3": "#0D47A1",
    "B1": "#F48FB1", "B2": "#E91E63", "B3": "#880E4F",
}
MARKERS = {"Static": "s", "A1": "^", "A2": "^", "A3": "^", "B1": "o", "B2": "o", "B3": "o"}

DATASETS = {"MNIST": "main", "CIFAR10": "cifar"}
BETAS = [0.1, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]
NUM_CLASSES = 10

# Documented exclusions (finding K9): one CLIENT received 0 samples ->
# DataLoader abort. MNIST beta=0.1 seed 1; CIFAR-10 beta=0.1 seed 4.
# The corresponding JSONs do not exist; aggregation simply uses the seeds
# that are present, and t-tests pair over the common seeds.

# RDP accounting (corrected, Delta_2 = sqrt(2)) — for the grid summary
ALPHA_ORDERS = [1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256]
DELTA = 1e-5


def eps_from_queries(q, sigma):
    L = math.log(1.0 / DELTA)
    return min(q * a * 2.0 / (2.0 * sigma ** 2) + L / (a - 1.0) for a in ALPHA_ORDERS)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_runs(repo_root: Path):
    """runs[dataset][beta][config][seed] = strategy block (dict of series)."""
    runs = {ds: {b: {c: {} for c in CONFIG_ORDER} for b in BETAS} for ds in DATASETS}
    baselines = {ds: {b: {} for b in BETAS} for ds in DATASETS}  # [seed] = file dict
    for ds, sub in DATASETS.items():
        d = repo_root / "results" / sub
        for beta in BETAS:
            for seed in SEEDS:
                # baseline file (upper_bound + classical_pate)
                bf = d / f"{ds}_beta{beta}_seed{seed}_s40.0_.json"
                if bf.exists():
                    with open(bf) as fh:
                        baselines[ds][beta][seed] = json.load(fh)
                for cfg, (skey, token) in CONFIGS.items():
                    f = d / f"{ds}_beta{beta}_seed{seed}_{token}.json"
                    if f.exists():
                        with open(f) as fh:
                            runs[ds][beta][cfg][seed] = json.load(fh)[skey]
    return runs, baselines


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def agg_final(runs, ds, beta, cfg):
    """Per-seed final values + mean/std aggregates for one cell."""
    cell = runs[ds][beta][cfg]
    seeds = sorted(cell)
    acc = np.array([cell[s]["accuracy"][-1] for s in seeds])
    f1 = np.array([cell[s]["macro_f1"][-1] for s in seeds])
    eps = np.array([cell[s]["epsilon"][-1] for s in seeds])
    q = np.array([cell[s]["teacher_queries"][-1] for s in seeds])
    return {
        "seeds": seeds,
        "acc": acc, "f1": f1, "eps": eps, "q": q,
        "acc_mean": acc.mean() * 100, "acc_std": acc.std(ddof=0) * 100,
        "f1_mean": f1.mean(),
        "eps_mean": eps.mean(), "q_mean": q.mean(),
    }


def paired_test(runs, ds, beta, cfg, metric):
    """Paired t-test cfg vs Static over common seeds for final `metric`."""
    a = runs[ds][beta][cfg]
    b = runs[ds][beta]["Static"]
    common = sorted(set(a) & set(b))
    x = np.array([a[s][metric][-1] for s in common])
    y = np.array([b[s][metric][-1] for s in common])
    t, p = ttest_rel(x, y)
    return (x - y).mean(), p, common


def de(x, nd=2):
    """German decimal format."""
    return f"{x:.{nd}f}".replace(".", "{,}")


def thousands(n):
    s = f"{int(round(n)):,}".replace(",", "\\,")
    return s


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def print_main_table(runs, baselines, ds, table_no, label):
    print(f"\n=== Table {table_no} ({label}) — {ds}: Main results "
          f"(new ε under Δ₂=√2) ===")
    print("LaTeX rows (columns: β & Configuration & Accuracy [%] & F1 & ε & Queries):\n")
    for beta in BETAS:
        # baselines
        ub_acc = np.array([baselines[ds][beta][s]["upper_bound"]["accuracy"]
                           for s in sorted(baselines[ds][beta])])
        cp_acc = np.array([baselines[ds][beta][s]["classical_pate"]["accuracy"]
                           for s in sorted(baselines[ds][beta])])
        cp_eps = np.array([baselines[ds][beta][s]["classical_pate"]["epsilon"]
                           for s in sorted(baselines[ds][beta])])
        beta_str = "$" + de(beta, 1) + "$"
        print(f"{beta_str} & Upper Bound            & "
              f"${de(ub_acc.mean()*100)} \\pm {de(ub_acc.std(ddof=0)*100)}$ & ---    & ---     & ---     \\\\")
        print(f"        & Classical PATE         & "
              f"${de(cp_acc.mean()*100)} \\pm {de(cp_acc.std(ddof=0)*100)}$ & ---    & ${de(cp_eps.mean())}$ & 5\\,000 \\\\")
        cells = {cfg: agg_final(runs, ds, beta, cfg) for cfg in CONFIG_ORDER}
        best_acc = max(cells.values(), key=lambda c: c["acc_mean"])["acc_mean"]
        best_f1 = max(cells.values(), key=lambda c: c["f1_mean"])["f1_mean"]
        best_eps = min(cells.values(), key=lambda c: c["eps_mean"])["eps_mean"]
        for cfg in CONFIG_ORDER:
            c = cells[cfg]
            acc_s = f"{de(c['acc_mean'])} \\pm {de(c['acc_std'])}"
            if abs(c["acc_mean"] - best_acc) < 1e-9:
                acc_s = f"\\mathbf{{{acc_s}}}"
            f1_s = de(c["f1_mean"], 3)
            if abs(c["f1_mean"] - best_f1) < 1e-9:
                f1_s = f"\\mathbf{{{f1_s}}}"
            eps_s = de(c["eps_mean"])
            if abs(c["eps_mean"] - best_eps) < 1e-9:
                eps_s = f"\\mathbf{{{eps_s}}}"
            name = CONFIG_LABEL[cfg] if cfg != "Static" else "Static Baseline"
            print(f"        & {name:<22} & ${acc_s}$ & ${f1_s}$ & ${eps_s}$ & {thousands(c['q_mean'])} \\\\")
        if beta != BETAS[-1]:
            print("\\midrule")
    print()


def print_ttest_table(runs, ds, table_no, label):
    print(f"\n=== Table {table_no} ({label}) — {ds}: paired t-tests vs. Static "
          f"(pairing over common seeds; Accuracy AND Macro-F1) ===\n")
    hdr = f"{'Comparison':<10}"
    for beta in BETAS:
        hdr += f" | b={beta}: dAcc(pp)   p_acc   dF1      p_F1   n"
    print(hdr)
    latex_rows = []
    for cfg in VARIANT_ORDER:
        row = f"{cfg:<10}"
        lat = f"{cfg[0]}{cfg[1]} ({'cons.' if cfg.endswith('1') else 'best' if cfg.endswith('2') else 'aggr.'}) vs.\\ Static"
        for beta in BETAS:
            da, pa, common = paired_test(runs, ds, beta, cfg, "accuracy")
            df, pf, _ = paired_test(runs, ds, beta, cfg, "macro_f1")
            row += (f" | {da*100:+7.2f}  {pa:7.3f}  {df:+7.4f} {pf:7.3f}  {len(common)}")
            sig = pa < 0.05
            cell = f"${'+' if da>=0 else ''}{de(da*100)}\\%$ (${'p=' + de(pa,3)}$)"
            if sig:
                cell = f"$\\mathbf{{{'+' if da>=0 else ''}{de(da*100)}\\%}}$ ($\\mathbf{{p={de(pa,3)}}}$)"
            lat += f"  & {cell}"
        print(row)
        latex_rows.append(lat + " \\\\")
    print("\nLaTeX rows (accuracy difference, as in the thesis table format):")
    for r in latex_rows:
        print(r)
    print()


def print_savings(runs, ds):
    print(f"\n=== ε savings vs. Static (new ε, Δ₂=√2) — {ds} ===")
    print(f"{'Config':<8}" + "".join(f"  β={b}: ε / sav%   " for b in BETAS))
    sav = {b: {} for b in BETAS}
    for cfg in CONFIG_ORDER:
        line = f"{cfg:<8}"
        for beta in BETAS:
            c = agg_final(runs, ds, beta, cfg)
            st = agg_final(runs, ds, beta, "Static")
            s = (st["eps_mean"] - c["eps_mean"]) / st["eps_mean"] * 100
            sav[beta][cfg] = s
            line += f"  {c['eps_mean']:7.2f} /{s:+7.2f}%"
        print(line)
    return sav


def print_query_analyses(runs):
    print("\n=== K5: Query reduction of the A configurations vs. Static "
          "(mean of the final cumulative queries) ===")
    for ds in DATASETS:
        for beta in BETAS:
            st = agg_final(runs, ds, beta, "Static")["q_mean"]
            reds = []
            for cfg in ["A1", "A2", "A3"]:
                q = agg_final(runs, ds, beta, cfg)["q_mean"]
                reds.append((cfg, (st - q) / st * 100))
            line = ", ".join(f"{c}: {r:+.1f}%" for c, r in reds)
            print(f"  {ds} β={beta}: Static Q={st:.0f} | Reduction: {line}")

    print("\n=== K2/K3: Queries B configurations vs. A configurations ===")
    for ds in DATASETS:
        for beta in BETAS:
            qa = {c: agg_final(runs, ds, beta, c)["q_mean"] for c in ["A1", "A2", "A3"]}
            qb = {c: agg_final(runs, ds, beta, c)["q_mean"] for c in ["B1", "B2", "B3"]}
            min_a = min(qa.values())
            flags = {c: ("<" if q < min_a else ">=") + " min(A)" for c, q in qb.items()}
            print(f"  {ds} β={beta}: A={ {k: round(v) for k, v in qa.items()} } "
                  f"B={ {k: round(v) for k, v in qb.items()} } -> "
                  + ", ".join(f"{c} {flags[c]}" for c in ["B1", "B2", "B3"]))


def print_realized_samples(runs):
    print("\n=== K7/M3: realized samples per round (from cumulative queries) ===")
    for ds in DATASETS:
        for beta in BETAS:
            cell = runs[ds][beta]["Static"]
            per_round = []
            for s, blk in cell.items():
                tq = blk["teacher_queries"]
                per_round += [tq[0]] + [tq[i] - tq[i - 1] for i in range(1, len(tq))]
            print(f"  {ds} β={beta} Static: mean/round = {np.mean(per_round):.0f} "
                  f"(min {np.min(per_round)}, max {np.max(per_round)})")
        # B3 maximum realized per-round volume relative to N_syn = 5000
        for beta in BETAS:
            cell = runs[ds][beta]["B3"]
            mx = 0
            for s, blk in cell.items():
                tq = blk["teacher_queries"]
                inc = [tq[0]] + [tq[i] - tq[i - 1] for i in range(1, len(tq))]
                mx = max(mx, max(inc))
            print(f"  {ds} β={beta} B3: max samples/round = {mx} "
                  f"({mx / 5000 * 100:.0f}% of N_syn=5000)")


def print_constant_b_queries(runs):
    print("\n=== M8: Variant B runs (MNIST) with constant synthesis distribution "
          "from round index 3 ===")
    const, total = 0, 0
    for beta in BETAS:
        for cfg in ["B1", "B2", "B3"]:
            for s, blk in runs["MNIST"][beta][cfg].items():
                sd = blk.get("synthesis_distribution")
                if not sd:
                    continue
                total += 1
                tail = [tuple(r) for r in sd[3:]]
                if len(set(tail)) == 1:
                    const += 1
    print(f"  {const}/{total} MNIST Variant B runs have a constant per-class "
          f"query distribution from round 3 onward.")


def print_per_class(runs, ds="MNIST", beta=0.5):
    print(f"\n=== G4: final per-class accuracy (mean over seeds), {ds} β={beta} ===")
    for cfg in CONFIG_ORDER:
        cell = runs[ds][beta][cfg]
        pc = np.array([blk["per_class_accuracy"][-1] for blk in cell.values()])
        m = pc.mean(axis=0)
        print(f"  {cfg:<7}: " + " ".join(f"{v:.3f}" for v in m)
              + f"   | weakest class: {int(np.argmin(m))} ({m.min():.3f})")
    cell = runs[ds][beta]["Static"]
    pc = np.array([blk["per_class_accuracy"][-1] for blk in cell.values()]).mean(axis=0)
    order = np.argsort(pc)
    print("  Static, classes sorted (weakest first): "
          + ", ".join(f"{int(k)}={pc[k]:.3f}" for k in order[:4]))


def print_duplicates(repo_root):
    print("\n=== K8: Duplicate runs (identical configuration+seed in grid_search "
          "(σ=40) and main) — accuracy deviation across separate runs ===")
    pairs = [
        ("static_s40.0", "static"),
        ("varA_s40.0_a1.0_m0.5", "variant_a"), ("varA_s40.0_a2.0_m0.3", "variant_a"),
        ("varA_s40.0_a4.0_m0.1", "variant_a"),
        ("varB_s40.0_b1.0_l0.1", "variant_b"), ("varB_s40.0_b1.0_l0.3", "variant_b"),
        ("varB_s40.0_b2.0_l0.5", "variant_b"),
    ]
    mx = 0.0
    for token, key in pairs:
        f1 = repo_root / "results" / "grid_search" / f"MNIST_beta0.1_seed0_{token}.json"
        f2 = repo_root / "results" / "main" / f"MNIST_beta0.1_seed0_{token}.json"
        if not (f1.exists() and f2.exists()):
            continue
        a = json.load(open(f1))[key]["accuracy"][-1]
        b = json.load(open(f2))[key]["accuracy"][-1]
        d = abs(a - b) * 100
        mx = max(mx, d)
        print(f"  {token:<24}: grid={a*100:6.2f}%  main={b*100:6.2f}%  |Δ|={d:5.2f} pp")
    print(f"  -> maximum deviation of identical configurations: {mx:.2f} pp")


def print_grid_summary(repo_root):
    print("\n=== M9: Grid search (sensitivity analysis), 1 seed, MNIST β=0.1, "
          "new ε under Δ₂=√2 ===")
    d = repo_root / "results" / "grid_search"
    rows = []
    for f in sorted(d.glob("*.json")):
        data = json.load(open(f))
        for key in ("static", "variant_a", "variant_b"):
            if key in data:
                blk = data[key]
                import re
                sig = float(re.search(r"_s(\d+(?:\.\d+)?)", f.name).group(1))
                rows.append({
                    "file": f.name, "sigma": sig, "key": key,
                    "rounds": len(blk["accuracy"]),
                    "acc": blk["accuracy"][-1],
                    "eps": blk["epsilon"][-1],
                    "q": blk["teacher_queries"][-1],
                })
    for sig in (10.0, 20.0, 40.0):
        rs = [r for r in rows if r["sigma"] == sig]
        rounds = sorted({r["rounds"] for r in rs})
        eps_rng = (min(r["eps"] for r in rs), max(r["eps"] for r in rs))
        print(f"  σ={sig:>4}: {len(rs)} runs, rounds completed: {rounds}, "
              f"final ε ∈ [{eps_rng[0]:.1f}, {eps_rng[1]:.1f}]")
    # sigma = 40 details for the text
    print("  σ=40 details (MNIST β=0.1, 1 seed):")
    for r in rows:
        if r["sigma"] == 40.0:
            print(f"    {r['file']:<46} acc={r['acc']*100:6.2f}%  "
                  f"ε={r['eps']:6.2f}  Q={r['q']}")
    st = next(r for r in rows if r["sigma"] == 40.0 and r["key"] == "static")
    b03 = next(r for r in rows if r["sigma"] == 40.0 and "b1.0_l0.3" in r["file"])
    b05 = next(r for r in rows if r["sigma"] == 40.0 and "b1.0_l0.5" in r["file"])
    print(f"    -> Static: acc={st['acc']*100:.2f}%, ε={st['eps']:.2f} | "
          f"B(λ=0.3): acc={b03['acc']*100:.2f}% ({(b03['acc']-st['acc'])*100:+.2f} pp), "
          f"ε={b03['eps']:.2f} | B(λ=0.5): acc={b05['acc']*100:.2f}% "
          f"({(b05['acc']-st['acc'])*100:+.2f} pp), ε={b05['eps']:.2f}")


def print_validation(baselines):
    print("\n=== Validation (Tab. 7.1): Upper Bound & Classical PATE, new ε ===")
    for ds in DATASETS:
        ub, cp, cpe = [], [], []
        for beta in BETAS:
            for s, d in baselines[ds][beta].items():
                ub.append(d["upper_bound"]["accuracy"])
                cp.append(d["classical_pate"]["accuracy"])
                cpe.append(d["classical_pate"]["epsilon"])
        print(f"  {ds}: UpperBound acc={np.mean(ub)*100:.2f}±{np.std(ub):.4f}  "
              f"ClassicalPATE acc={np.mean(cp)*100:.2f}%, ε={np.mean(cpe):.4f} "
              f"(Q=5000, identical for all runs)")


# ---------------------------------------------------------------------------
# Figures (original thesis file names, per-configuration resolution)
# ---------------------------------------------------------------------------
def fig_accuracy_over_rounds(runs, ds, fname, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, beta in zip(axes, BETAS):
        for cfg in CONFIG_ORDER:
            cell = runs[ds][beta][cfg]
            if not cell:
                continue
            curves = np.array([blk["accuracy"] for blk in cell.values()]) * 100
            mean = curves.mean(axis=0)
            ax.plot(range(len(mean)), mean, color=COLORS[cfg],
                    marker=MARKERS[cfg], markersize=3.5, label=cfg, lw=1.8)
        ax.set_title(f"$\\beta = {beta}$")
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test accuracy [%]")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{ds}: Accuracy over synthesis rounds (mean over seeds, "
                 f"per configuration)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


def fig_pareto(runs, ds, fname, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, beta in zip(axes, BETAS):
        for cfg in CONFIG_ORDER:
            c = agg_final(runs, ds, beta, cfg)
            ax.errorbar(c["eps_mean"], c["acc_mean"], yerr=c["acc_std"],
                        fmt=MARKERS[cfg], color=COLORS[cfg], capsize=3,
                        markersize=7, label=cfg)
            ax.annotate(cfg, (c["eps_mean"], c["acc_mean"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_title(f"$\\beta = {beta}$")
        ax.set_xlabel("final $\\epsilon$ ($\\Delta_2=\\sqrt{2}$)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("final test accuracy [%]")
    fig.suptitle(f"{ds}: Privacy-utility points per configuration "
                 f"(mean ± std over seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


def fig_savings(runs, ds, fname, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, beta in zip(axes, BETAS):
        st = agg_final(runs, ds, beta, "Static")["eps_mean"]
        vals = []
        for cfg in VARIANT_ORDER:
            e = agg_final(runs, ds, beta, cfg)["eps_mean"]
            vals.append((st - e) / st * 100)
        bars = ax.bar(VARIANT_ORDER, vals,
                      color=[COLORS[c] for c in VARIANT_ORDER])
        ax.axhline(0, color="black", lw=0.8)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.set_title(f"$\\beta = {beta}$")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("$\\epsilon$ savings vs. Static [%]")
    fig.suptitle(f"{ds}: Privacy budget savings per configuration "
                 f"(negative = higher consumption; $\\Delta_2=\\sqrt{{2}}$)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


def fig_per_class(runs, ds, fname, fig_dir, beta=0.5, cfgs=("Static", "A2", "B2")):
    panel_color = {"Static": "#404040", "A2": "#FF7F0E", "B2": "#9C27B0"}
    fig, axes = plt.subplots(1, len(cfgs), figsize=(15, 4.5), sharey=True)
    x = np.arange(NUM_CLASSES)
    for ax, cfg in zip(axes, cfgs):
        cell = runs[ds][beta][cfg]
        pc = np.array([blk["per_class_accuracy"][-1]
                       for blk in cell.values()]).mean(axis=0) * 100
        bars = ax.bar(x, pc, color=panel_color.get(cfg, "#404040"))
        for b, v in zip(bars, pc):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8)
        label = "Static Baseline" if cfg == "Static" else CONFIG_LABEL[cfg]
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xlabel("Class")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Per-class accuracy [%]")
    fig.suptitle(f"Per-class accuracy ($\\beta = {beta}$, $\\sigma = 40$)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


def fig_synthesis_distribution(runs, ds, fname, fig_dir, beta=0.5):
    fig, axes = plt.subplots(1, 7, figsize=(20, 3.4), sharey=True)
    for ax, cfg in zip(axes, CONFIG_ORDER):
        cell = runs[ds][beta][cfg]
        mats = []
        for blk in cell.values():
            sd = np.array(blk["synthesis_distribution"], dtype=float)  # (rounds, K)
            sd = sd / sd.sum(axis=1, keepdims=True)
            mats.append(sd)
        m = np.mean(mats, axis=0).T  # rows = classes, cols = rounds
        im = ax.imshow(m, aspect="auto", cmap="magma", vmin=0)
        ax.set_title(cfg, fontsize=10)
        ax.set_xlabel("Round")
    axes[0].set_ylabel("Class")
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.005,
                 label="share of round budget")
    fig.suptitle(f"{ds}: per-class query distribution over rounds "
                 f"(rows = classes, columns = rounds; $\\beta = {beta}$, "
                 f"mean over seeds)", y=1.06)
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()
    root = Path(args.repo_root)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs, baselines = load_runs(root)

    # sanity: report missing runs (documented exclusions K9)
    print("=== Data basis ===")
    for ds in DATASETS:
        n = sum(len(runs[ds][b][c]) for b in BETAS for c in CONFIG_ORDER)
        nb = sum(len(baselines[ds][b]) for b in BETAS)
        missing = [(b, c, s) for b in BETAS for c in CONFIG_ORDER for s in SEEDS
                   if s not in runs[ds][b][c]]
        print(f"  {ds}: {n} strategy runs + {nb} baseline files; "
              f"missing (documented exclusions): "
              f"{sorted(set((b, s) for b, c, s in missing))}")

    print_validation(baselines)
    print_main_table(runs, baselines, "MNIST", "7.2", "tab:main_results")
    print_main_table(runs, baselines, "CIFAR10", "7.5", "tab:main_results_cifar")
    print_ttest_table(runs, "MNIST", "7.4", "tab:ttest_results")
    print_ttest_table(runs, "CIFAR10", "7.6", "tab:ttest_results_cifar")
    print_savings(runs, "MNIST")
    print_savings(runs, "CIFAR10")
    print_query_analyses(runs)
    print_realized_samples(runs)
    print_constant_b_queries(runs)
    print_per_class(runs, "MNIST", 0.5)
    print_duplicates(root)
    print_grid_summary(root)

    # ---- figures ---------------------------------------------------------
    fig_accuracy_over_rounds(runs, "MNIST", "accuracy_over_rounds.pdf", fig_dir)
    fig_pareto(runs, "MNIST", "pareto_front.pdf", fig_dir)
    fig_savings(runs, "MNIST", "epsilon_savings.pdf", fig_dir)
    fig_per_class(runs, "MNIST", "per_class_accuracy.pdf", fig_dir)
    fig_synthesis_distribution(runs, "MNIST", "synthesis_distribution.pdf", fig_dir)
    fig_accuracy_over_rounds(runs, "CIFAR10", "cifar_accuracy_over_rounds.pdf", fig_dir)
    fig_pareto(runs, "CIFAR10", "cifar_pareto_front.pdf", fig_dir)  # includes B1 (M7)
    fig_savings(runs, "CIFAR10", "cifar_epsilon_savings.pdf", fig_dir)
    print(f"\n8 figures written with original file names to: {fig_dir}/")


if __name__ == "__main__":
    main()
