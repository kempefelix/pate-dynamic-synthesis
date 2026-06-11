#!/usr/bin/env python3
"""
recompute_epsilon.py — Re-derive all reported epsilon values under the
corrected L2-sensitivity Delta_2 = sqrt(2) (finding K1, variant b).

Background
----------
The original RDP accountant (src/privacy/rdp.py) used a per-query RDP cost
of alpha * Delta^2 / (2 sigma^2) with Delta^2 = 1. The correct record-level
sensitivity of the vote histogram is Delta_2 = sqrt(2) (changing one record
moves one teacher's vote from one class to another: -1/+1 in two
coordinates, hence ||.||_2 = sqrt(2); cf. Papernot et al. 2018, Prop. 8).

Because the accounting is data-independent (the per-query cost is constant
and the cumulative epsilon depends only on the number of labeled samples Q),
all epsilon values can be re-derived EXACTLY from the raw, unaltered
teacher_queries counts stored in the result JSONs:

    eps(Q) = min over alpha in A of [ Q * alpha * Delta^2 / (2 sigma^2)
                                      + ln(1/delta) / (alpha - 1) ]

with Delta^2 = 2, delta = 1e-5,
A = {1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256}
(the alpha grid of the original RDPAccountant), and sigma parsed from the
result filename (token "_s40.0" -> sigma = 40.0, etc.).

The script overwrites ONLY the derived "epsilon" field of each strategy
block. teacher_queries, accuracy, macro_f1, per_class_accuracy and
synthesis_distribution are raw experimental data and are NEVER modified.

Verification anchors (must hold exactly, rounded to 2 decimals):
    * Classical PATE, Q = 5000,  sigma = 40  ->  eps = 15.13
    * MNIST beta=1.0 Static, mean final Q = 48408, sigma = 40 -> eps = 68.41
      (all five seeds lie in the linear alpha* = 1.5 regime, so the mean of
       the per-seed epsilons equals eps(mean Q))

Usage:
    python3 recompute_epsilon.py [--repo-root PATH]
"""

import argparse
import json
import math
import re
from pathlib import Path

# RDP orders of the original accountant (src/privacy/rdp.py)
ALPHA_ORDERS = [1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256]
DELTA = 1e-5
DELTA2_SQ = 2.0  # Delta_2^2 with Delta_2 = sqrt(2)

# sigma token in filenames: "_s40.0", "_s10.0_", "_s20.0" ...
SIGMA_RE = re.compile(r"_s(\d+(?:\.\d+)?)")

RESULT_SUBDIRS = ("main", "cifar", "grid_search")


def epsilon_from_queries(q: float, sigma: float,
                         delta: float = DELTA,
                         delta2_sq: float = DELTA2_SQ) -> float:
    """(eps, delta)-DP epsilon after q queries under Delta_2 = sqrt(2)."""
    log_term = math.log(1.0 / delta)
    return min(
        q * alpha * delta2_sq / (2.0 * sigma ** 2) + log_term / (alpha - 1.0)
        for alpha in ALPHA_ORDERS
    )


def parse_sigma(filename: str) -> float:
    m = SIGMA_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse sigma from filename: {filename}")
    return float(m.group(1))


def recompute_file(path: Path) -> dict:
    """Recompute epsilon fields of one result JSON in place.

    Returns a small report dict for logging.
    """
    sigma = parse_sigma(path.name)
    with open(path) as fh:
        data = json.load(fh)

    report = {"file": path.name, "sigma": sigma, "blocks": []}
    changed = False

    for key, block in data.items():
        if key == "config" or not isinstance(block, dict):
            continue
        if "teacher_queries" not in block:
            # e.g. upper_bound: no DP mechanism, no epsilon -> untouched
            continue

        tq = block["teacher_queries"]
        old_eps = block.get("epsilon")

        if isinstance(tq, list):
            new_eps = [epsilon_from_queries(q, sigma) for q in tq]
            final_old = old_eps[-1] if isinstance(old_eps, list) and old_eps else None
            final_new = new_eps[-1] if new_eps else None
            final_q = tq[-1] if tq else None
        else:  # scalar (classical_pate baseline)
            new_eps = epsilon_from_queries(tq, sigma)
            final_old, final_new, final_q = old_eps, new_eps, tq

        block["epsilon"] = new_eps  # ONLY derived field that is overwritten
        changed = True
        report["blocks"].append(
            {"strategy": key, "final_Q": final_q,
             "eps_old": final_old, "eps_new": final_new}
        )

    if changed:
        with open(path, "w") as fh:
            json.dump(data, fh)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".",
                    help="Repository root containing results/")
    args = ap.parse_args()

    root = Path(args.repo_root) / "results"
    n_files = 0
    n_blocks = 0
    anchor_cp = None  # classical PATE anchor

    for sub in RESULT_SUBDIRS:
        d = root / sub
        if not d.is_dir():
            print(f"[warn] missing results subdir: {d}")
            continue
        for path in sorted(d.glob("*.json")):
            rep = recompute_file(path)
            if not rep["blocks"]:
                continue
            n_files += 1
            n_blocks += len(rep["blocks"])
            for b in rep["blocks"]:
                if b["strategy"] == "classical_pate" and anchor_cp is None:
                    anchor_cp = b

    print(f"Recomputed epsilon in {n_files} files / {n_blocks} strategy blocks "
          f"(Delta_2 = sqrt(2), delta = {DELTA}).")

    # ---- Verification anchors -------------------------------------------
    print("\n=== Verification anchors ===")
    eps_cp = epsilon_from_queries(5000, 40.0)
    print(f"Classical PATE  Q=5000,  sigma=40 : eps = {eps_cp:.4f} "
          f"(expected 15.13) -> {'OK' if round(eps_cp, 2) == 15.13 else 'FAIL'}")
    if anchor_cp is not None:
        print(f"  (first classical_pate block in results: Q={anchor_cp['final_Q']}, "
              f"eps_old={anchor_cp['eps_old']:.4f} -> eps_new={anchor_cp['eps_new']:.4f})")

    eps_static = epsilon_from_queries(48408, 40.0)
    print(f"MNIST b=1.0 Static  mean Q=48408, sigma=40 : eps = {eps_static:.4f} "
          f"(expected 68.41) -> {'OK' if round(eps_static, 2) == 68.41 else 'FAIL'}")


if __name__ == "__main__":
    main()
