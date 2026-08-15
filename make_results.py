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
  * prints the complete new tables 7.4 (tab:main_results) and
    7.6 (tab:main_results_cifar) as LaTeX rows, plus the t-test
    Delta/p values for tables 7.5 (tab:ttest_results) and
    7.7 (tab:ttest_results_cifar);
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
import decimal
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figures are included in the thesis at width=\linewidth, and the text block is
# 418.4 pt = 5.81 in wide. A figure drawn at FIG_WIDTH inches is therefore scaled
# by 5.81/FIG_WIDTH on the page, and every label shrinks with it. At the former
# 15 in the factor was 0.39, so an 8 pt tick label was printed at 3.1 pt against
# a 10.95 pt body text. FIG_WIDTH is the width the figures are drawn at; keep it
# close to the text block so that the scaling stays near 1.
FIG_WIDTH = 7.5
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
})

from scipy.stats import t as student_t
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
    "A1": "Variant A1 (cons.)", "A2": "Variant A2 (bal.)", "A3": "Variant A3 (aggr.)",
    "B1": "Variant B1 (cons.)", "B2": "Variant B2 (bal.)", "B3": "Variant B3 (aggr.)",
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
        lat = f"{cfg[0]}{cfg[1]} ({'cons.' if cfg.endswith('1') else 'bal.' if cfg.endswith('2') else 'aggr.'}) vs.\\ Static"
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

    # Macro-F1 family (thesis Section 7.4.7). The eighteen macro-F1 tests per
    # dataset are printed cell by cell in the console table above but are not
    # tabulated in the thesis; this block recomputes them to summarise scope and
    # outcome, and is the producer for the sentence that reports them.
    f1 = []
    for beta in BETAS:
        for cfg in VARIANT_ORDER:
            df, pf, common = paired_test(runs, ds, beta, cfg, "macro_f1")
            f1.append((beta, cfg, df, pf, len(common)))
    f1_sig = [c for c in f1 if c[3] < ALPHA_STAT]
    f1_rest = [c for c in f1 if c[3] >= ALPHA_STAT]
    print(f"\nMacro-F1 family, {DS_LABEL[ds]}: {len(f1)} paired tests vs. Static, "
          f"{len(f1_sig)} significant at alpha = {ALPHA_STAT}"
          + (" ("
             + "; ".join(f"{c[1]} at beta={c[0]}: dF1={c[2]:+.4f}, p={c[3]:.4f}"
                         for c in f1_sig)
             + ")" if f1_sig else "")
          + ".")
    print(f"  of the significant ones, {sum(1 for c in f1_sig if c[2] > 0)} "
          f"favour the variant; smallest p among the remaining {len(f1_rest)}: "
          f"{min(c[3] for c in f1_rest):.4f}")
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

    print("\n=== K5b: Query change of the B configurations vs. Static "
          "(mean of the final cumulative queries; positive = more queries) ===")
    for ds in DATASETS:
        for beta in BETAS:
            st = agg_final(runs, ds, beta, "Static")["q_mean"]
            chg = []
            for cfg in ["B1", "B2", "B3"]:
                q = agg_final(runs, ds, beta, cfg)["q_mean"]
                chg.append((cfg, (q - st) / st * 100))
            line = ", ".join(f"{c}: {r:+.1f}%" for c, r in chg)
            print(f"  {ds} β={beta}: Static Q={st:.0f} | Change: {line}")

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
    eps_rows = []
    for token, key in pairs:
        f1 = repo_root / "results" / "grid_search" / f"MNIST_beta0.1_seed0_{token}.json"
        f2 = repo_root / "results" / "main" / f"MNIST_beta0.1_seed0_{token}.json"
        if not (f1.exists() and f2.exists()):
            continue
        j1 = json.load(open(f1))[key]
        j2 = json.load(open(f2))[key]
        a, b = j1["accuracy"][-1], j2["accuracy"][-1]
        d = abs(a - b) * 100
        mx = max(mx, d)
        print(f"  {token:<24}: grid={a*100:6.2f}%  main={b*100:6.2f}%  |Δ|={d:5.2f} pp")
        eps_rows.append((token, j1["epsilon"][-1], j2["epsilon"][-1],
                         j1["teacher_queries"][-1], j2["teacher_queries"][-1]))
    print(f"  -> maximum deviation of identical configurations: {mx:.2f} pp")
    print("  ε and query deviation of the same seven pairs:")
    mxe = mxq = 0.0
    for token, e1, e2, q1, q2 in eps_rows:
        de_, dq = abs(e1 - e2), abs(q1 - q2)
        mxe, mxq = max(mxe, de_), max(mxq, dq)
        print(f"    {token:<24}: ε grid={e1:6.2f} main={e2:6.2f} |Δ|={de_:5.2f}  |  "
              f"Q grid={q1:>6} main={q2:>6} |Δ|={dq:>6}")
    print(f"  -> maximum ε deviation: {mxe:.2f}  |  maximum query deviation: {int(mxq)}")


def print_epsilon_dispersion(runs):
    print("\n=== K8b: Per-seed ε dispersion — ε is exact given Q, but Q varies across runs ===")
    for ds in DATASETS:
        for beta in BETAS:
            for cfg in CONFIG_ORDER:
                if not runs[ds][beta][cfg]:
                    continue
                a = agg_final(runs, ds, beta, cfg)
                e, q = a["eps"], a["q"]
                print(f"  {ds:<7} β={beta:<4} {cfg:<8}: ε mean={e.mean():6.2f} "
                      f"sd={e.std(ddof=0):5.2f} range={e.max()-e.min():6.2f} "
                      f"(n={len(e)})  |  Q range={int(q.max()-q.min()):>6}")
    sds = [agg_final(runs, "MNIST", b, c)["eps"].std(ddof=0)
           for b in (0.5, 1.0) for c in ("B1", "B2", "B3")
           if runs["MNIST"][b][c]]
    print(f"  -> MNIST β∈{{0.5,1.0}}, Variant B (the cells carrying the 23–35 % claim): "
          f"ε sd ∈ [{min(sds):.2f}, {max(sds):.2f}]")


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
    print("\n=== Validation (Tab. 7.3): Upper Bound & Classical PATE, new ε ===")
    print("    (first line per dataset pools all beta levels as a sanity check and")
    print("     appears in no table; of the per-beta rows below, MNIST ClassicalPATE")
    print("     is Table 7.3, CIFAR-10 ClassicalPATE is Table 7.6, and the six")
    print("     UpperBound rows are Tables 7.4 and 7.6)")
    for ds in DATASETS:
        ub, cp, cpe = [], [], []
        for beta in BETAS:
            for s, d in baselines[ds][beta].items():
                ub.append(d["upper_bound"]["accuracy"])
                cp.append(d["classical_pate"]["accuracy"])
                cpe.append(d["classical_pate"]["epsilon"])
        # *100 on the deviation as well: without it this line printed 99.21+-0.0005,
        # and 0.0005 read as a percentage is the "+-0.05 %" that CON-007 had to
        # remove from Section 7.2.1
        print(f"  {ds}: UpperBound acc={np.mean(ub)*100:.2f}±{np.std(ub)*100:.4f}  "
              f"ClassicalPATE acc={np.mean(cp)*100:.2f}%, ε={np.mean(cpe):.4f} "
              f"(Q=5000, identical for all runs)")
        # the pooled figure above is a sanity check and appears in no table cell;
        # Table 7.3 is per beta, so the tabulated rows are printed here too
        for beta in BETAS:
            cells = baselines[ds][beta]
            if not cells:
                continue
            u = [d["upper_bound"]["accuracy"] for d in cells.values()]
            c = [d["classical_pate"]["accuracy"] for d in cells.values()]
            print(f"    beta={beta:<4} n={len(cells)} | UpperBound "
                  f"{np.mean(u)*100:6.2f}±{np.std(u)*100:5.2f}  "
                  f"ClassicalPATE {np.mean(c)*100:6.2f}±{np.std(c)*100:5.2f}")


# ---------------------------------------------------------------------------
# Figures (original thesis file names, per-configuration resolution)
# ---------------------------------------------------------------------------
def fig_accuracy_over_rounds(runs, ds, fname, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.9), sharey=True)
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
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.9), sharey=True)
    for ax, beta in zip(axes, BETAS):
        for cfg in CONFIG_ORDER:
            c = agg_final(runs, ds, beta, cfg)
            # the baseline is the reference every other point is read against,
            # so it is drawn on top: at beta = 0.1 on CIFAR-10 the three
            # Variant-A points sit within 0.7 of its epsilon and would bury it
            ax.errorbar(c["eps_mean"], c["acc_mean"], yerr=c["acc_std"],
                        fmt=MARKERS[cfg], color=COLORS[cfg], capsize=3,
                        markersize=7, label=cfg,
                        zorder=4 if cfg == "Static" else 3)
        ax.set_title(f"$\\beta = {beta}$")
        ax.set_xlabel("final $\\epsilon$ ($\\Delta_2=\\sqrt{2}$)")
        ax.grid(alpha=0.3)
        ax.margins(x=0.12)
    axes[0].set_ylabel("final test accuracy [%]")
    # One shared legend rather than a label next to every point: at the width of
    # the text block the neighbouring configurations overprint each other.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONFIG_ORDER),
               bbox_to_anchor=(0.5, -0.09), frameon=False)
    fig.suptitle(f"{ds}: Privacy-utility points per configuration "
                 f"(mean ± std over seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


def fig_savings(runs, ds, fname, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.9), sharey=True)
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
            # upright: six bars across the width of the text block leave about
            # 30 pt per bar, and a horizontal "+34.5%" needs more than that
            ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top",
                        rotation=90, fontsize=7)
        ax.set_title(f"$\\beta = {beta}$")
        ax.grid(alpha=0.3, axis="y")
        ax.margins(y=0.22)          # headroom for the upright bar labels
    axes[0].set_ylabel("$\\epsilon$ savings vs. Static [%]")
    fig.suptitle(f"{ds}: Privacy budget savings per configuration "
                 f"(negative = higher consumption; $\\Delta_2=\\sqrt{{2}}$)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


def fig_per_class(runs, ds, fname, fig_dir, beta=0.5, cfgs=("Static", "A2", "B2")):
    panel_color = {"Static": "#404040", "A2": "#FF7F0E", "B2": "#9C27B0"}
    fig, axes = plt.subplots(1, len(cfgs), figsize=(FIG_WIDTH, 3.1), sharey=True)
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
    # Seven panels as a 2x4 grid rather than a single 1x7 row: at the width of
    # the text block a single row would leave each panel about 1 in wide.
    axgrid = plt.subplots(2, 4, figsize=(FIG_WIDTH, 4.4), sharey=True,
                          layout="constrained")
    fig, axes = axgrid[0], axgrid[1].flatten()
    for ax, cfg in zip(axes, CONFIG_ORDER):
        cell = runs[ds][beta][cfg]
        mats = []
        for blk in cell.values():
            sd = np.array(blk["synthesis_distribution"], dtype=float)  # (rounds, K)
            sd = sd / sd.sum(axis=1, keepdims=True)
            mats.append(sd)
        m = np.mean(mats, axis=0).T  # rows = classes, cols = rounds
        im = ax.imshow(m, aspect="auto", cmap="magma", vmin=0)
        ax.set_title(cfg)
        ax.set_xlabel("Round")
        # rounds are integers, and an automatic integer locator settles on a
        # step of 5, which leaves two labels on a ten-round axis
        ax.set_xticks(range(0, m.shape[1], 2))
    axes[len(CONFIG_ORDER)].remove()          # the grid holds one cell more than
    used = list(axes[:len(CONFIG_ORDER)])     # there are configurations
    axes[0].set_ylabel("Class")
    axes[4].set_ylabel("Class")
    fig.colorbar(im, ax=used, fraction=0.015, pad=0.005,
                 label="share of round budget")
    fig.suptitle(f"{ds}: per-class query distribution over rounds\n"
                 f"(rows = classes, columns = rounds; $\\beta = {beta}$, "
                 f"mean over seeds)")
    fig.savefig(fig_dir / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Accuracy equivalence (TOST) and literal Pareto dominance — thesis Section 7.1.3
# ---------------------------------------------------------------------------
# Equivalence margins specified in advance in NFR1 (thesis Section 4.4.3): the smallest
# practically relevant accuracy difference, in percentage points.
TOST_MARGIN = {"MNIST": 1.0, "CIFAR10": 2.0}
ALPHA_STAT = 0.05


def tost(diff_pp, margin, alpha=ALPHA_STAT):
    """Two one-sided tests for equivalence of a paired difference to zero.

    diff_pp: per-seed paired differences (configuration minus Static), in
    percentage points. Returns (p, lo, hi) where p = max(p1, p2) and [lo, hi]
    is the (1 - 2*alpha) confidence interval — the interval that must lie
    inside (-margin, +margin) for equivalence at level alpha.
    """
    n = len(diff_pp)
    m = float(np.mean(diff_pp))
    se = float(np.std(diff_pp, ddof=1)) / math.sqrt(n)
    df = n - 1
    p1 = 1.0 - student_t.cdf((m + margin) / se, df)   # H0: mu <= -margin
    p2 = student_t.cdf((m - margin) / se, df)         # H0: mu >= +margin
    crit = student_t.ppf(1.0 - alpha, df)
    return max(p1, p2), m - crit * se, m + crit * se


def _paired_acc_diff(runs, ds, beta, cfg):
    a, b = runs[ds][beta][cfg], runs[ds][beta]["Static"]
    common = sorted(set(a) & set(b))
    return np.array([(a[s]["accuracy"][-1] - b[s]["accuracy"][-1]) * 100
                     for s in common]), common


def print_tost_table(runs):
    """Accuracy equivalence against the NFR1 margins (thesis Section 7.1.3)."""
    print("\n=== Accuracy equivalence (TOST) vs. Static, margins from NFR1 "
          "(MNIST 1 pp, CIFAR-10 2 pp), alpha = 0.05 ===\n")
    print(f"{'Dataset':<9} {'beta':<5} {'cfg':<4} {'n':>2} {'dAcc pp':>8} "
          f"{'90% CI':>20} {'p_TOST':>8}  verdict")
    carriers = {"pass": 0, "total": 0}
    for ds in DATASETS:
        margin = TOST_MARGIN[ds]
        for beta in BETAS:
            for cfg in VARIANT_ORDER:
                d, common = _paired_acc_diff(runs, ds, beta, cfg)
                if len(d) < 2:
                    continue
                p, lo, hi = tost(d, margin)
                ok = p < ALPHA_STAT
                if beta in (0.5, 1.0) and cfg.startswith("B"):
                    carriers["total"] += 1
                    carriers["pass"] += int(ok)
                print(f"{ds:<9} {beta:<5} {cfg:<4} {len(common):>2} "
                      f"{d.mean():+8.3f} [{lo:+8.3f},{hi:+8.3f}] {p:8.4f}  "
                      f"{'EQUIVALENT' if ok else 'not established'}")
    print(f"\n  Variant-B cells at beta in {{0.5, 1.0}}: "
          f"{carriers['pass']} of {carriers['total']} establish equivalence "
          f"(the count quoted in Sections 7.1.3, 7.4.7, 7.9 and Chapter 8).")


def print_tost_multiplicity(runs):
    """Multiplicity of the equivalence tests (thesis Section 7.4.7).

    The twelve Variant-B equivalence tests at beta in {0.5, 1.0} are reported
    uncorrected. This routine states what a Holm step-down would leave under
    each defensible family definition, so that the numbers quoted in the
    thesis have a producer.
    """
    print("\n=== Multiplicity of the equivalence tests (Holm, alpha = 0.05) ===\n")

    cells = []          # (ds, beta, cfg, p) for every A and B cell at beta in {0.5, 1.0}
    for ds in DATASETS:
        margin = TOST_MARGIN[ds]
        for beta in (0.5, 1.0):
            for cfg in VARIANT_ORDER:
                d, common = _paired_acc_diff(runs, ds, beta, cfg)
                if len(d) < 2:
                    continue
                p, _, _ = tost(d, margin)
                cells.append((ds, beta, cfg, p))

    b_cells = [c for c in cells if c[2].startswith("B")]
    uncorrected = sum(1 for c in b_cells if c[3] < ALPHA_STAT)

    def survivors(groups):
        """groups: list of lists of (ds, beta, cfg, p). Returns surviving B cells."""
        n = 0
        for g in groups:
            keep = holm([c[3] for c in g], ALPHA_STAT)
            n += sum(1 for i, c in enumerate(g) if keep[i] and c[2].startswith("B"))
        return n

    by_ds_beta = [[c for c in cells if (c[0], c[1]) == k]
                  for k in sorted({(c[0], c[1]) for c in cells})]
    by_ds_beta_b = [[c for c in b_cells if (c[0], c[1]) == k]
                    for k in sorted({(c[0], c[1]) for c in b_cells})]
    by_ds_b = [[c for c in b_cells if c[0] == ds] for ds in DATASETS]

    print(f"  uncorrected                                  : "
          f"{uncorrected} of {len(b_cells)}")
    print(f"  Holm per dataset x beta, A and B (6 tests)    : "
          f"{survivors(by_ds_beta)} of {len(b_cells)}")
    print(f"  Holm per dataset x beta, B only  (3 tests)    : "
          f"{survivors(by_ds_beta_b)} of {len(b_cells)}")
    print(f"  Holm per dataset, B only         (6 tests)    : "
          f"{survivors(by_ds_b)} of {len(b_cells)}")
    print(f"  Holm over all twelve as one family           : "
          f"{survivors([b_cells])} of {len(b_cells)}")

    print("\n  Cells surviving Holm per dataset x beta (A and B):")
    for g in by_ds_beta:
        keep = holm([c[3] for c in g], ALPHA_STAT)
        kept = [f"{c[2]} (p={c[3]:.4f})" for i, c in enumerate(g) if keep[i]]
        print(f"    {g[0][0]:<9} beta={g[0][1]}: {', '.join(kept) if kept else 'none'}")


def print_dominance_counts(runs):
    """Equation 3.7 applied literally to the seed means (thesis Section 7.1.3).

    Acc(X) >= Acc(Static) AND eps(X) <= eps(Static), at least one strict.
    Reported because the thesis evaluates a WEAKENED criterion and must state
    how many cells the original one actually admits.
    """
    print("\n=== Equation 3.7 applied literally to the seed means ===\n")
    for ds in DATASETS:
        hits = []
        total = 0
        for beta in BETAS:
            base = agg_final(runs, ds, beta, "Static")
            for cfg in VARIANT_ORDER:
                c = agg_final(runs, ds, beta, cfg)
                total += 1
                ge = c["acc_mean"] >= base["acc_mean"]
                le = c["eps_mean"] <= base["eps_mean"]
                strict = (c["acc_mean"] > base["acc_mean"]) or \
                         (c["eps_mean"] < base["eps_mean"])
                if ge and le and strict:
                    hits.append((beta, cfg, c["acc_mean"] - base["acc_mean"],
                                 c["eps_mean"] - base["eps_mean"]))
        print(f"  {DS_LABEL[ds]}: {len(hits)} of {total} cells")
        for beta, cfg, da, de_ in hits:
            print(f"      beta={beta:<4} {cfg}: dAcc={da:+.3f} pp, deps={de_:+.3f}")
        b1 = [h for h in hits if h[1] == "B1"]
        print(f"      B1 satisfies it in {len(b1)} of {len(BETAS)} levels")


# ---------------------------------------------------------------------------
# Strategy contrast: Variant B against Variant A (thesis Section 7.5.4)
# ---------------------------------------------------------------------------
# EXPLORATORY, and declared as such in the thesis. The test family discussed in
# Section 7.4.7 is the six accuracy tests against the Static baseline within one
# dataset and one beta level; it is NOT enlarged by what follows.
#
# These contrasts answer the ranking half of RQ1, which the
# variant-versus-Static design does not address: every Variant-B configuration
# is paired with the Variant-A configuration of the same aggressiveness over
# the common seeds, so teacher ensembles, cGAN initializations and Dirichlet
# partitionings cancel exactly as they do in the Static comparison. Nine
# contrasts per dataset (3 matched pairs x 3 beta levels), 18 in total, per
# criterion.
BVA_PAIRS = [("B1", "A1"), ("B2", "A2"), ("B3", "A3")]


def paired_contrast(runs, ds, beta, cfg_x, cfg_y, metric):
    """Paired t-test of cfg_x against cfg_y over common seeds for final `metric`."""
    a, b = runs[ds][beta][cfg_x], runs[ds][beta][cfg_y]
    common = sorted(set(a) & set(b))
    x = np.array([a[s][metric][-1] for s in common])
    y = np.array([b[s][metric][-1] for s in common])
    t, p = ttest_rel(x, y)
    return (x - y).mean(), p, common


def print_bva_contrasts(runs):
    """Variant B against Variant A at matched aggressiveness (Section 7.5.4)."""
    print("\n=== RQ1, ranking half: Variant B vs. Variant A at matched "
          "aggressiveness (EXPLORATORY) ===")
    print("    Nine contrasts per dataset (3 matched pairs x 3 beta levels), "
          "18 in total, per criterion.")
    print("    Not part of the accuracy test family of Section 7.4.7.\n")

    def label(c):
        return f"{c[1]}-{c[2]} at beta={c[0]}"

    for metric, unit, scale in (("accuracy", "pp", 100.0),
                                ("macro_f1", "F1", 1.0)):
        print(f"  --- {metric} ---")
        for ds in DATASETS:
            cells = []
            for beta in BETAS:
                for bx, ay in BVA_PAIRS:
                    d, p, common = paired_contrast(runs, ds, beta, bx, ay, metric)
                    cells.append((beta, bx, ay, d * scale, p, len(common)))
                    print(f"    {DS_LABEL[ds]:<9} beta={beta:<4} {bx}-{ay}  "
                          f"d={d * scale:+8.4f} {unit:<2} p={p:7.4f}  "
                          f"n={len(common)}{'   *' if p < ALPHA_STAT else ''}")
            sig = [c for c in cells if c[4] < ALPHA_STAT]
            rest = [c for c in cells if c[4] >= ALPHA_STAT]
            print(f"    {DS_LABEL[ds]}: {len(sig)} of {len(cells)} significant "
                  f"uncorrected"
                  + (" (" + ", ".join(label(c) for c in sig) + ")" if sig else "")
                  + f"; smallest p among the remaining {len(rest)}: "
                    f"{min(c[4] for c in rest):.4f}")
            print(f"      numerically in favour of B: "
                  f"{sum(1 for c in cells if c[3] > 0)} of {len(cells)}")
            keep = holm([c[4] for c in cells], ALPHA_STAT)
            surv = [c for i, c in enumerate(cells) if keep[i]]
            print(f"      Holm over all nine of the dataset: "
                  f"{', '.join(label(c) for c in surv) or 'none'}")
            surv_beta = []
            for beta in BETAS:
                g = [c for c in cells if c[0] == beta]
                k = holm([c[4] for c in g], ALPHA_STAT)
                surv_beta += [c for i, c in enumerate(g) if k[i]]
            print(f"      Holm within one beta level (3 tests): "
                  f"{', '.join(label(c) for c in surv_beta) or 'none'}")
        print()


# ---------------------------------------------------------------------------
# Post-hoc round analysis (thesis Section 7.7)
# ---------------------------------------------------------------------------
DS_LABEL = {"MNIST": "MNIST", "CIFAR10": "CIFAR-10"}


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. Returns {index: survives}."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = {i: False for i in range(m)}
    for rank, i in enumerate(order):
        # <=, not <: Holm-Bonferroni rejects AT the threshold (CTV-070). No cell
        # of this thesis sits on a boundary, so no printed verdict changes.
        if pvals[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break  # step-down: everything above the first failure also fails
    return out


def _round_seed_means(runs, ds, beta):
    """Per-seed round-0 and round-9 accuracy, averaged over the seven configs.

    THE UNIT OF ANALYSIS IS THE SEED, NOT THE RUN. Round 0 is procedurally
    identical for all seven configurations: the coordinator falls back to the
    uniform static query while no student feedback exists yet (see
    src/synthesis/coordinator.py). The seven runs of one seed are therefore
    procedural replicates, not independent observations. Pooling them as
    independent pairs would inflate the degrees of freedom roughly sevenfold.
    """
    seeds = sorted(set.intersection(*[set(runs[ds][beta][c]) for c in CONFIG_ORDER]))
    r0 = np.array([np.mean([runs[ds][beta][c][s]["accuracy"][0]
                            for c in CONFIG_ORDER]) for s in seeds]) * 100
    r9 = np.array([np.mean([runs[ds][beta][c][s]["accuracy"][-1]
                            for c in CONFIG_ORDER]) for s in seeds]) * 100
    q0 = np.mean([runs[ds][beta][c][s]["teacher_queries"][0]
                  for c in CONFIG_ORDER for s in seeds])
    q9 = np.mean([runs[ds][beta][c][s]["teacher_queries"][-1]
                  for c in CONFIG_ORDER for s in seeds])
    return seeds, r0, r9, q0, q9


def print_round_table(runs):
    """Thesis Table 7.8 (tab:eval_rounds): round 0 against round 9."""
    print("\n=== Table 7.8 (tab:eval_rounds) — round 0 vs. round 9 "
          "(unit of analysis: seed) ===\n")
    print(f"{'Dataset':<9} {'beta':<5} {'n':>2} | {'Acc r0 [%]':>15} "
          f"{'Acc r9 [%]':>15} | {'eps r0':>7} {'eps r9':>7} | "
          f"{'dAcc':>6} {'p':>9}")
    cells, pvals = [], []
    for ds in DATASETS:
        for beta in BETAS:
            seeds, r0, r9, q0, q9 = _round_seed_means(runs, ds, beta)
            _, p = ttest_rel(r9, r0)
            cells.append(dict(ds=ds, beta=beta, n=len(seeds),
                              r0=r0.mean(), r0sd=r0.std(ddof=0),
                              r9=r9.mean(), r9sd=r9.std(ddof=0),
                              e0=eps_from_queries(q0, 40.0),
                              e9=eps_from_queries(q9, 40.0),
                              d=r9.mean() - r0.mean(), p=p))
            pvals.append(p)
    surv = holm(pvals)
    for i, c in enumerate(cells):
        print(f"{c['ds']:<9} {c['beta']:<5} {c['n']:>2} | "
              f"{c['r0']:8.2f} +- {c['r0sd']:<4.2f} {c['r9']:8.2f} +- {c['r9sd']:<4.2f} | "
              f"{c['e0']:7.2f} {c['e9']:7.2f} | {c['d']:+6.2f} {c['p']:9.4g}"
              f"{'  (survives Holm over the 6 tests)' if surv[i] else ''}")
    print("\nLaTeX rows (as in the thesis table):")
    for i, c in enumerate(cells):
        d = f"{c['d']:+.2f}"
        dcell = f"$\\mathbf{{{d}}}$" if surv[i] else f"${d}$"
        print(f"{DS_LABEL[c['ds']]:<8} & ${c['beta']}$ & "
              f"${c['r0']:.2f} \\pm {c['r0sd']:.2f}$ & ${c['e0']:.2f}$ & "
              f"${c['r9']:.2f} \\pm {c['r9sd']:.2f}$ & ${c['e9']:.2f}$ & "
              f"{dcell} & ${c['p']:.2g}$ \\\\")
    print()


def _half_up(x, nd):
    """Round half away from zero, so that a value landing exactly on a .xx5
    boundary is decided by a stated rule and not by the IEEE-754 binary
    representation (CTV-020). Called from print_round_truncation (including its
    block (e)) and from print_round_profile; both are blocks added on or after
    2026-08-02, so no output block that predates the rule changes because of it.

    Accumulation noise is absorbed before the rule is applied. A mean such as
    (2.17 + 5.29 + 4.88 + 4.76)/4 is exactly 4.275, but floating-point summation
    returns 4.2749999999999995, which would round down and defeat the purpose.
    The inputs derive from accuracies stored to four decimals, so no true value
    carries information beyond ~1e-6; collapsing at 1e-9 is therefore safe and
    restores the exact tie before rounding it.
    """
    q = decimal.Decimal(10) ** -nd
    d = decimal.Decimal(repr(float(x))).quantize(
        decimal.Decimal(10) ** -9, rounding=decimal.ROUND_HALF_UP)
    r = d.quantize(q, rounding=decimal.ROUND_HALF_UP)
    return float(r) + 0.0  # normalise -0.0 to 0.0


def print_round_profile(runs):
    """Full per-round accuracy profile, averaged over the seven configurations.

    Produces the figures Section 7.7.2 quotes for the CIFAR-10 decline -- in
    particular the round at which the profile bottoms out, which Table 7.8 does
    not show because it tabulates round 0 and round 9 only (CON-015). Same unit
    of analysis as _round_seed_means: the seven runs of a seed are procedural
    replicates, so the mean is taken over configurations and seeds together.
    """
    print("\n=== Per-round accuracy profile (mean over the seven configurations) "
          "— the intermediate rounds Table 7.8 does not tabulate ===\n")
    hdr = "  {:<9} {:<5} {:>2} | ".format("Dataset", "beta", "n")
    print(hdr + " ".join("r%-5d" % r for r in range(10)) + " | min at")
    for ds in DATASETS:
        for beta in BETAS:
            cells = [runs[ds][beta][c] for c in CONFIG_ORDER]
            if not all(cells):
                continue
            seeds = sorted(set.intersection(*[set(c) for c in cells]))
            prof = []
            for r in range(10):
                vals = [runs[ds][beta][c][s]["accuracy"][r]
                        for c in CONFIG_ORDER for s in seeds
                        if len(runs[ds][beta][c][s]["accuracy"]) > r]
                prof.append(np.mean(vals) * 100 if vals else float("nan"))
            lo = min(range(10), key=lambda r: prof[r])
            print("  {:<9} {:<5} {:>2} | ".format(DS_LABEL[ds], beta, len(seeds))
                  + " ".join("%5.2f" % _half_up(v, 2) for v in prof)
                  + " | r%d (%.2f)" % (lo, _half_up(prof[lo], 2)))


def print_round_truncation(runs, trunc=(1, 3, 5, 10)):
    """Quantifies the round-count lever against the strategy lever (RED-103).

    Section 7.7.4 already concedes that "reducing the number of rounds addresses
    the same axis with a considerably larger lever" than the 23-35 % saving of
    Variant B. That concession is qualitative; this routine puts a number on it.

    THIS IS A READ-OFF, NOT A NEW EXPERIMENT. Round r of a committed run is
    identical to a run that had been configured for R = r. The reason is a
    property of the producing code, not of the data: in the code path that
    produced every committed result, `num_rounds` occurs exactly once and only
    as the bound of the synthesis loop (run_experiment.py:279), and the per-round
    volume is the fixed `samples_per_round` (run_experiment.py:297), never a
    total divided by R. No per-round quantity is therefore a function of the
    configured R, and the trajectory up to round r is the same whichever R the
    run was configured for. Every value below is read from the committed
    trajectories at index r-1; no experiment was re-executed.

    The routine additionally reports one corroborating data property for the
    Static baseline it truncates: its realized query count per round is exactly
    constant. For the adaptive variants it is not, and that is the documented
    adaptivity effect (the realized query volume is the driver identified in
    Section 7.7), not a counter-example to the read-off.
    """
    print("\n=== Round-count lever vs. strategy lever (read-off from the "
          "committed trajectories; RED-103) ===\n")

    # (a) corroborating property for the truncated baseline, checked not assumed
    bad_static, adaptive_varies = [], 0
    for ds in DATASETS:
        for beta in BETAS:
            for cfg in CONFIG_ORDER:
                for s, blk in runs[ds][beta][cfg].items():
                    q = blk["teacher_queries"]
                    inc = {q[i] - q[i - 1] for i in range(1, len(q))}
                    constant = (len(inc) == 1 and inc == {q[0]})
                    if cfg == "Static" and not constant:
                        bad_static.append((ds, beta, s, q[0], sorted(inc)))
                    elif cfg != "Static" and not constant:
                        adaptive_varies += 1
    print(f"  Static, queries per round constant and equal to round 0: "
          f"{'HOLDS in every Static run' if not bad_static else 'VIOLATED in %d runs' % len(bad_static)}")
    for b in bad_static[:5]:
        print(f"    violation: {b}")
    print(f"  Adaptive variants with a varying per-round query count: "
          f"{adaptive_varies} runs (expected — this is the adaptivity effect "
          f"of Section 7.7, not a counter-example)")

    # (b) Static truncated at R, against Static at R = 10.
    #     dAcc is reported WITH its seed-level dispersion and the sign count,
    #     because most of these deltas sit inside the seed noise and a bare mean
    #     would overstate them (CTV-022). eps carries no such caveat: it is a
    #     pure function of the cumulative query count and is exact.
    print(f"\n  {'Dataset':<9} {'beta':<5} {'R':>3} {'n':>2} | {'Acc [%]':>8} "
          f"{'eps':>8} | {'dAcc vs R=10':>12} {'SD':>6} {'sign':>7} "
          f"{'eps saving':>11}")
    summary = {}
    for ds in DATASETS:
        for beta in BETAS:
            cell = runs[ds][beta]["Static"]
            seeds = sorted(cell)
            if not seeds:
                continue
            full_a = np.mean([cell[s]["accuracy"][-1] for s in seeds]) * 100
            full_e = np.mean([cell[s]["epsilon"][-1] for s in seeds])
            for R in trunc:
                idx = R - 1
                if any(len(cell[s]["accuracy"]) <= idx for s in seeds):
                    continue
                per_seed = np.array([cell[s]["accuracy"][idx] - cell[s]["accuracy"][-1]
                                     for s in seeds]) * 100
                a = np.mean([cell[s]["accuracy"][idx] for s in seeds]) * 100
                e = np.mean([cell[s]["epsilon"][idx] for s in seeds])
                sav = (full_e - e) / full_e * 100.0
                npos = int((per_seed > 0).sum())
                sgn = "all +" if npos == len(seeds) else (
                      "all -" if npos == 0 else f"{npos}+/{len(seeds) - npos}-")
                print(f"  {DS_LABEL[ds]:<9} {beta:<5} {R:>3} {len(seeds):>2} | "
                      f"{_half_up(a, 2):>8.2f} {_half_up(e, 2):>8.2f} | "
                      f"{_half_up(per_seed.mean(), 2):>+12.2f} "
                      f"{_half_up(per_seed.std(ddof=0), 2):>6.2f} {sgn:>7} "
                      f"{_half_up(sav, 1):>10.1f}%")
                summary[(ds, beta, R)] = (a, e, per_seed.mean(), sav,
                                          per_seed.std(ddof=0), sgn)
            print()

    # (c) the comparison the thesis needs: same axis, two levers
    print("  Strategy lever vs. round lever, MNIST at beta in {0.5, 1.0} "
          "(the cells carrying the 23-35 % claim):")
    print(f"    {'Dataset':<9} {'beta':<5} | {'best Variant-B eps saving':>26} "
          f"| {'Static R=3 eps saving':>22} {'at dAcc':>9}")
    for ds in DATASETS:
        for beta in BETAS:
            st = runs[ds][beta]["Static"]
            if not st:
                continue
            base_e = np.mean([st[s]["epsilon"][-1] for s in st])
            best = None
            for cfg in ("B1", "B2", "B3"):
                cell = runs[ds][beta][cfg]
                if not cell:
                    continue
                e = np.mean([cell[s]["epsilon"][-1] for s in cell])
                sv = (base_e - e) / base_e * 100.0
                if best is None or sv > best:
                    best = sv
            row = summary.get((ds, beta, 3))
            if best is None or row is None:
                continue
            print(f"    {DS_LABEL[ds]:<9} {beta:<5} | {best:>25.1f}% "
                  f"| {row[3]:>21.1f}% {row[2]:>+9.2f}")
    print("    NOTE: 'best Variant-B' is a maximum over B1/B2/B3, i.e. the "
          "comparison is\n          deliberately biased in favour of the "
          "strategy lever, and it still loses.")
    print("    NOTE: the two savings are not accuracy-matched. The Variant-B "
          "saving carries an\n          equivalence claim on accuracy "
          "(Section 7.4.3); the truncation saving does not.")

    # (d) Do the two levers COMPOSE or SUBSTITUTE? (MET-101, RED-306)
    #     Block (b) reads the Static trajectory only. The variant trajectories
    #     live in the same JSON files and permit the same read-off. If the round
    #     count merely superseded the strategy, the variant saving against a
    #     Static baseline truncated at the SAME R would vanish at small R.
    #     Built-in control: at R = 1 every saving must be exactly 0.0 %, because
    #     round 0 is procedurally identical across configurations (Section 7.7.1).
    print("\n  Do the levers compose? Variant eps saving against Static truncated "
          "at the SAME R:")
    print(f"    {'Dataset':<9} {'beta':<5} {'R':>3} | "
          + " ".join(f"{c:>7}" for c in ("A1", "A2", "A3", "B1", "B2", "B3")))
    for ds in DATASETS:
        for beta in BETAS:
            st = runs[ds][beta]["Static"]
            if not st:
                continue
            for R in trunc:
                idx = R - 1
                if any(len(st[s]["epsilon"]) <= idx for s in st):
                    continue
                cells = []
                for cfg in ("A1", "A2", "A3", "B1", "B2", "B3"):
                    c = runs[ds][beta][cfg]
                    common = sorted(set(c) & set(st))
                    if not common or any(len(c[s]["epsilon"]) <= idx for s in common):
                        cells.append(None)
                        continue
                    be = np.mean([st[s]["epsilon"][idx] for s in common])
                    ce = np.mean([c[s]["epsilon"][idx] for s in common])
                    cells.append((be - ce) / be * 100.0)
                print(f"    {DS_LABEL[ds]:<9} {beta:<5} {R:>3} | "
                      + " ".join("   --  " if v is None else f"{v:>+6.1f}%"
                                 for v in cells))
            print()

    #     Composed effect, printed rather than left to the reader to multiply:
    #     B1 truncated to R against the untruncated Static baseline.
    print("  Composed: B1 at R against Static at R = 10 (both levers applied):")
    for ds in DATASETS:
        for beta in BETAS:
            st, b1 = runs[ds][beta]["Static"], runs[ds][beta]["B1"]
            common = sorted(set(st) & set(b1))
            if not common:
                continue
            base = np.mean([st[s]["epsilon"][-1] for s in common])
            for R in trunc:
                idx = R - 1
                if any(len(b1[s]["epsilon"]) <= idx for s in common):
                    continue
                e = np.mean([b1[s]["epsilon"][idx] for s in common])
                print(f"    {DS_LABEL[ds]:<9} {beta:<5} R={R:<3}: B1 eps={e:>7.2f} "
                      f"vs Static R=10 eps={base:>7.2f} -> "
                      f"{(base - e) / base * 100.0:>+6.1f}%")
            print()

    # (e) The truncation accuracy cost against the NFR1 margin (MET-102).
    #     The accuracy side of (b) is the only quantitative block of Chapter 7
    #     without an inferential statistic; this applies the thesis's own
    #     yardstick to it. tost() and TOST_MARGIN are the same helpers the
    #     equivalence table of Section 7.4.3 uses.
    print("  Truncation accuracy cost against the NFR1 margin "
          "(TOST, 90 % CI; margin = smallest effect size of interest):")
    print(f"    {'Dataset':<9} {'beta':<5} {'R':>3} {'n':>2} | {'dAcc':>7} "
          f"{'90% CI':>18} {'margin':>7} {'p_TOST':>7}  verdict")
    for ds in DATASETS:
        m = TOST_MARGIN[ds]
        for beta in BETAS:
            cell = runs[ds][beta]["Static"]
            seeds = sorted(cell)
            if not seeds:
                continue
            for R in trunc:
                idx = R - 1
                if R == 10 or any(len(cell[s]["accuracy"]) <= idx for s in seeds):
                    continue
                d = np.array([cell[s]["accuracy"][idx] - cell[s]["accuracy"][-1]
                              for s in seeds]) * 100
                if len(d) < 2:
                    continue          # tost() divides by the standard error
                p, lo, hi = tost(d, m)
                verdict = "EQUIVALENT" if p < ALPHA_STAT else "not established"
                over = "  (|dAcc| exceeds the margin)" if abs(d.mean()) > m else ""
                # _half_up, not the format default: block (b) prints the same
                # quantity and CIFAR-10 beta=0.1 at R=1 is exactly +4.275, the
                # boundary case _half_up exists for (CTV-020)
                print(f"    {DS_LABEL[ds]:<9} {beta:<5} {R:>3} {len(seeds):>2} | "
                      f"{_half_up(d.mean(), 2):>+7.2f} [{lo:>+7.2f},{hi:>+7.2f}] "
                      f"{m:>7.1f} {p:>7.4f}  {verdict}{over}")
            print()

    # (f) The lever RATIO itself, printed instead of divided by hand (CTV-064).
    #     Sections 7.7.4, 7.8 and Chapter 8 state the round lever as a FACTOR
    #     over the strategy lever. Blocks (b) and (c) print both operands but
    #     never their quotient, so every factor quoted in the thesis was a hand
    #     division of two printed percentages -- the same failure mode that
    #     CTV-040 and CTV-055 already cost one repair each.
    #
    #     The quotient is strongly cell-specific, so it is printed per cell
    #     rather than as one headline number: any factor quoted in the text can
    #     then be checked against the scope it is quoted under. It is also
    #     R-specific -- the factor at the round-0 state (R = 1) is NOT the factor
    #     at R = 3, which is what CTV-063 got wrong.
    print("  Lever ratio: Static truncation saving at R / best Variant-B saving "
          "at R = 10 (CTV-064):")
    print(f"    {'Dataset':<9} {'beta':<5} {'R':>3} | {'Static trunc.':>13} "
          f"{'best Var-B':>11} | {'ratio':>7}")
    for ds in DATASETS:
        for beta in BETAS:
            st = runs[ds][beta]["Static"]
            if not st:
                continue
            base_e = np.mean([st[s]["epsilon"][-1] for s in st])
            best = None
            for cfg in ("B1", "B2", "B3"):
                cell = runs[ds][beta][cfg]
                if not cell:
                    continue
                e = np.mean([cell[s]["epsilon"][-1] for s in cell])
                sv = (base_e - e) / base_e * 100.0
                if best is None or sv > best:
                    best = sv
            if best is None or best <= 0:
                # Over-consumption cells (Variant B costs budget) have no
                # meaningful ratio; naming them is more useful than a blank.
                print(f"    {DS_LABEL[ds]:<9} {beta:<5}  -- | "
                      f"best Variant-B saves nothing here — ratio undefined")
                print()
                continue
            for R in trunc:
                row = summary.get((ds, beta, R))
                if row is None or R == 10:
                    continue
                print(f"    {DS_LABEL[ds]:<9} {beta:<5} {R:>3} | "
                      f"{row[3]:>12.1f}% {best:>10.1f}% | {row[3] / best:>7.2f}")
            print()


def print_matched_budget(runs, baselines):
    """Thesis Table 7.9 (tab:eval_rounds_matched): synthesis round 0 against the
    classical-PATE reference at a matched privacy budget, paired over seeds."""
    print("\n=== Table 7.9 (tab:eval_rounds_matched) — synthesis round 0 vs. "
          "classical PATE at matched budget ===\n")
    print(f"{'Dataset':<9} {'beta':<5} {'n':>2} | {'Synth r0 [%]':>13} {'eps':>7} | "
          f"{'ClassPATE [%]':>14} {'eps':>7} | {'Delta':>6} {'p':>8}")
    rows = []
    for ds in DATASETS:
        for beta in BETAS:
            seeds, r0, _, q0, _ = _round_seed_means(runs, ds, beta)
            common = [s for s in seeds if s in baselines[ds][beta]]
            syn = np.array([r0[seeds.index(s)] for s in common])
            cp = np.array([baselines[ds][beta][s]["classical_pate"]["accuracy"]
                           for s in common]) * 100
            cpe = np.mean([baselines[ds][beta][s]["classical_pate"]["epsilon"]
                           for s in common])
            _, p = ttest_rel(syn, cp)
            rows.append(dict(ds=ds, beta=beta, n=len(common), syn=syn.mean(),
                             e=eps_from_queries(q0, 40.0), cp=cp.mean(),
                             cpe=cpe, d=syn.mean() - cp.mean(), p=p))
    for r in rows:
        print(f"{r['ds']:<9} {r['beta']:<5} {r['n']:>2} | {r['syn']:13.2f} {r['e']:7.2f} | "
              f"{r['cp']:14.2f} {r['cpe']:7.2f} | {r['d']:+6.2f} {r['p']:8.3f}")
    print("\nLaTeX rows (as in the thesis table):")
    for r in rows:
        print(f"{DS_LABEL[r['ds']]:<8} & ${r['beta']}$ & ${r['syn']:.2f}$ & "
              f"${r['e']:.2f}$ & ${r['cp']:.2f}$ & ${r['cpe']:.2f}$ & "
              f"${r['d']:+.2f}$ & ${r['p']:.3f}$ \\\\")
    print()


def main():
    # the output contains epsilon, beta, sigma and Delta_2; on a Windows console
    # under the cp1252 default this raises UnicodeEncodeError three lines in
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass                    # older Python, or stdout is not a text stream

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
    print_main_table(runs, baselines, "MNIST", "7.4", "tab:main_results")
    print_main_table(runs, baselines, "CIFAR10", "7.6", "tab:main_results_cifar")
    print_ttest_table(runs, "MNIST", "7.5", "tab:ttest_results")
    print_ttest_table(runs, "CIFAR10", "7.7", "tab:ttest_results_cifar")
    print_bva_contrasts(runs)
    print_savings(runs, "MNIST")
    print_savings(runs, "CIFAR10")
    print_query_analyses(runs)
    print_realized_samples(runs)
    print_constant_b_queries(runs)
    print_per_class(runs, "MNIST", 0.5)
    print_tost_table(runs)
    print_tost_multiplicity(runs)
    print_dominance_counts(runs)
    print_round_table(runs)
    print_round_profile(runs)
    print_round_truncation(runs)
    print_matched_budget(runs, baselines)
    print_duplicates(root)
    print_epsilon_dispersion(runs)
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
