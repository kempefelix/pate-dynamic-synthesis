#!/usr/bin/env python3
"""Reconstructs the grid search results from the SLURM .log files.

*** ONE-OFF RECOVERY SCRIPT — CANNOT BE RE-RUN ***

This script rebuilt the grid search json results from 
the SLURM logs, which are themselves not part of the supplement. It therefore
cannot be executed against this repository. The files it produced
(`results/grid_search/*.json`) carry a REDUCED FIELD SET — round, accuracy,
macro_f1, epsilon and teacher_queries only, with no per_class_accuracy and no
synthesis_distribution. Anything in Section 7.3 rests on these reconstructed
files; see the disclosure in Section 6.7 of the thesis.
"""
import re, json, glob, os

LOG_DIR = "results/grid_search"

name_re = re.compile(
    r"^gs_(A|B|static)_s(\d+)(?:_a([\d.]+)_m([\d.]+)|_b([\d.]+)_l([\d.]+))?_(\d+)\.log$")
acc_re = re.compile(r"Round (\d+): Acc=([\d.]+) \| F1=([\d.]+) \| .*?=([\d.]+)")
q_re   = re.compile(r"Round (\d+): .*?= ?([\d.]+) \| Queries so far: (\d+)")

def parse_log(path):
    rounds = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = acc_re.search(line)
            if m:
                r = int(m.group(1)); d = rounds.setdefault(r, {})
                d["accuracy"], d["macro_f1"], d["epsilon"] = \
                    float(m.group(2)), float(m.group(3)), float(m.group(4))
            m = q_re.search(line)
            if m:
                r = int(m.group(1)); d = rounds.setdefault(r, {})
                d["teacher_queries"] = int(m.group(3))
    return rounds

best = {}
for path in glob.glob(os.path.join(LOG_DIR, "gs_*.log")):
    m = name_re.match(os.path.basename(path))
    if not m: continue
    strat, sigma, a, mcr, b, lc, jobid = m.groups()
    key = (strat, sigma, a, mcr, b, lc)
    rounds = parse_log(path)
    n_complete = sum(1 for r in rounds.values()
                     if {"accuracy","epsilon","teacher_queries"} <= set(r))
    if n_complete == 0: continue
    cand = (n_complete, int(jobid), rounds, (strat, sigma, a, mcr, b, lc))
    if key not in best or cand[:2] > best[key][:2]:
        best[key] = cand

rows = []
for key, (_, jobid, rounds, meta) in sorted(best.items()):
    strat, sigma, a, mcr, b, lc = meta
    rs = sorted(rounds)
    series = {k: [rounds[r].get(k) for r in rs]
              for k in ("accuracy","macro_f1","epsilon","teacher_queries")}
    skey = {"A":"variant_a","B":"variant_b","static":"static"}[strat]
    out = {"config": {"dataset":"MNIST","beta":0.1,"seed":0},
           skey: {"round": rs, **series}}
    if strat == "A":   fn = f"MNIST_beta0.1_seed0_varA_s{sigma}.0_a{a}_m{mcr}.json"
    elif strat == "B": fn = f"MNIST_beta0.1_seed0_varB_s{sigma}.0_b{b}_l{lc}.json"
    else:              fn = f"MNIST_beta0.1_seed0_static_s{sigma}.0.json"
    with open(os.path.join(LOG_DIR, fn), "w") as f:
        json.dump(out, f, indent=2)
    comp = [r for r in rs if {"accuracy","epsilon","teacher_queries"} <= set(rounds[r])]
    if not comp: continue
    fr = rounds[comp[-1]]
    rows.append((sigma, strat, a or b or "-", mcr or lc or "-",
                 fr["accuracy"], fr["epsilon"], fr["teacher_queries"], len(rs)))

print(f"\n{len(best)} configurations reconstructed (of 57 expected).\n")
print(f"{'sig':>4} {'strat':<6} {'p1':>4} {'p2':>4} {'acc':>7} {'eps':>7} {'Q':>7} {'#R':>3}")
for r in sorted(rows, key=lambda x:(x[0], x[1])):
    print(f"{r[0]:>4} {r[1]:<6} {str(r[2]):>4} {str(r[3]):>4} "
          f"{r[4]:>7.4f} {r[5]:>7.2f} {r[6]:>7} {r[7]:>3}")

for sig in ("10","20","40"):
    eps = [r[5] for r in rows if r[0]==sig]
    if eps: print(f"\nsigma={sig}: eps range {min(eps):.1f}-{max(eps):.1f}")
stat40 = [r for r in rows if r[0]=="40" and r[1]=="static"]
if stat40:
    sa = stat40[0][4]
    for strat in ("A","B"):
        cand = [r for r in rows if r[0]=="40" and r[1]==strat]
        if cand:
            bestr = max(cand, key=lambda x:x[4])
            print(f"best {strat}@sig40: acc={bestr[4]:.4f} -> "
                  f"Delta vs static({sa:.4f}) = {100*(bestr[4]-sa):+.2f}%")
