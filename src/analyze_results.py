"""
analyze_results.py — turn byzantine_results_full.csv into stats + the comparison figure.

Run AFTER byzantine_experiment.py:
  C:\\Users\\minha\\.conda\\envs\\fl_eeg\\python.exe D:\\FL-new\\code\\analyze_results.py

Prints: clean-condition accuracies, per-attack/per-fraction means for every method,
and paired t-tests of FL-EVO vs each baseline. Saves the multi-method robustness figure.
Needs numpy + matplotlib (scipy optional, for exact p-values).
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------- CONFIG -----------------------------
RESULTS_CSV = os.path.join(_BASE, "results", "byzantine", "byzantine_results.csv")
FIG_OUT     = os.path.join(_BASE, "results", "byzantine", "figures", "byzantine_with_baselines.png")

FRACTIONS = [0, 2, 3, 4]
ATTACKS   = ["signflip_scale", "signflip", "gaussian"]
METHODS   = ["fedavg", "fedprox", "flevo", "median", "trimmed", "krum"]
# ------------------------------------------------------------------

LAB = {"fedavg": "FedAvg", "fedprox": "FedProx", "flevo": "FL-EVO (PSO)",
       "median": "Median", "trimmed": "Trimmed mean", "krum": "Multi-Krum"}
COL = {"fedavg": "#c0504d", "fedprox": "#c0504d", "flevo": "#4878a8",
       "median": "#2a9d4a", "trimmed": "#7a5fa8", "krum": "#e0a020"}

rows = list(csv.DictReader(open(RESULTS_CSV)))


def vals(method, attack, nb):
    return np.array([float(r["test_acc"]) for r in rows
                     if r["method"] == method and r["attack"] == attack and int(r["n_byz"]) == nb])


def ttest_rel(a, b):
    d = a - b
    if len(d) < 2 or np.std(d, ddof=1) == 0:
        return 0.0, 1.0
    t = d.mean() / (np.std(d, ddof=1) / np.sqrt(len(d)))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), len(d) - 1))
    except Exception:
        p = float("nan")
    return float(t), p


# ---- clean condition ----
print("CLEAN (0 Byzantine):")
for m in METHODS:
    a = vals(m, "none", 0)
    if len(a):
        print(f"  {LAB[m]:14s} {a.mean():.3f} ± {a.std():.3f}")

# ---- per-attack tables ----
plot_methods = ["fedavg", "flevo", "median", "trimmed", "krum"]
for attack in ATTACKS:
    print(f"\n{attack.upper()}  (mean test accuracy)")
    print("  byz  " + "".join(f"{LAB[m]:>14s}" for m in plot_methods))
    for nb in [2, 3, 4]:
        line = f"  {nb:3d}  "
        for m in plot_methods:
            a = vals(m, attack, nb)
            line += f"{(a.mean() if len(a) else float('nan')):14.3f}"
        print(line)

# ---- FL-EVO vs baselines under coherent attack ----
print("\nFL-EVO vs baselines (coherent attack, paired t-test over seeds):")
for nb in [2, 3, 4]:
    fe = vals("flevo", "signflip_scale", nb)
    for base in ["median", "trimmed", "krum"]:
        bv = vals(base, "signflip_scale", nb)
        t, p = ttest_rel(fe, bv)
        print(f"  byz={nb}  FL-EVO {fe.mean():.3f} vs {LAB[base]:12s} {bv.mean():.3f}  "
              f"gap {fe.mean()-bv.mean():+.3f}  p={p:.3f}")

# ---- figure ----
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
titles = {"signflip_scale": "Coherent (sign-flip + scale)",
          "signflip": "Sign-flip only", "gaussian": "Gaussian noise"}
for j, attack in enumerate(ATTACKS):
    a = ax[j]
    for m in plot_methods:
        ys, es = [], []
        for nb in FRACTIONS:
            v = vals(m, "none" if nb == 0 else attack, nb)
            ys.append(v.mean() if len(v) else np.nan)
            es.append(v.std() if len(v) else 0)
        a.errorbar(FRACTIONS, ys, yerr=es, marker="o", capsize=2, color=COL[m],
                   label=LAB[m], lw=1.6, ms=4)
    a.axhline(0.5, color="#999", ls=":", lw=1)
    a.set_title(titles[attack]); a.set_xlabel("Byzantine clients (of 10)"); a.set_xticks(FRACTIONS)
    if j == 0:
        a.set_ylabel("Test accuracy"); a.legend(fontsize=7.5, loc="lower left")
a.set_ylim(0.40, 0.86)
plt.suptitle("FL-EVO vs robust-aggregation baselines under Byzantine attack — real EEG (5 seeds, mean±SD)",
             fontsize=11)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved figure: {FIG_OUT}")
