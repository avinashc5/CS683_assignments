#!/usr/bin/env python3
"""Task 3: parse results/<config>/Trace*.txt -> tables (md/csv) + plots in results/task3/.
Usage: analyze_task3.py [results_dir]"""
import re, os, sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results", "task3")
os.makedirs(OUT, exist_ok=True)
TRACES = ["Trace1", "Trace2", "Trace3"]
CFGS = [("baseline", "Baseline (no pf, LRU)"), ("spp", "SPP"), ("dynirrip", "Dynamic IRRIP"),
        ("spp_dynnopf", "SPP + Dyn IRRIP (pf-unaware)"), ("spp_dyn", "SPP + Dyn IRRIP (pf-aware)")]
SWEEP = [("spp_dyn_pfr1", "PF insert RRPV 1"), ("spp_dyn_pfr2", "PF insert RRPV 2"), ("spp_dyn", "PF insert RRPV 3 (final)")]
ALL = CFGS + SWEEP[:2]

def num(txt, pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else float("nan")

def parse(cfg, t):
    s = open(f"{RES}/{cfg}/{t}.txt").read()
    return dict(
        ipc=num(s, r"CPU 0 cumulative IPC: ([\d.]+)"),
        l1d_mpki=num(s, r"L1D LOAD .*MPKI:\s+([\d.]+)"),
        l1d_misses=num(s, r"L1D LOAD\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)"),
        l1d_acc=num(s, r"L1D USEFUL LOAD PREFETCHES:.*ACCURACY: ([\d.]+)"),
        issued=num(s, r"L1D USEFUL LOAD PREFETCHES:\s+\d+ PREFETCH ISSUED TO LOWER LEVEL:\s+(\d+)"),
        l2_mpki=num(s, r"L2C LOAD .*MPKI:\s+([\d.]+)"),
        l2_data=num(s, r"L2C DATA LOAD MPKI: ([\d.]+)"),
        l2_instr=num(s, r"L2C INSTRUCTION LOAD MPKI: ([\d.]+)"),
    )

D = {c: {t: parse(c, t) for t in TRACES} for c, _ in ALL + SWEEP[2:] if os.path.isdir(f"{RES}/{c}")}
for c in D:
    for t in TRACES:
        b = D["baseline"][t]; r = D[c][t]
        r["speedup"] = r["ipc"] / b["ipc"]
        r["l1d_cov"] = 100 * (b["l1d_misses"] - r["l1d_misses"]) / b["l1d_misses"]
        if r["issued"] == 0 or r["issued"] != r["issued"]: r["l1d_acc"] = float("nan")

def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")
def gmean(xs):
    p = 1.0
    for x in xs: p *= x
    return p ** (1 / len(xs))

metrics = [("ipc", "IPC", "{:.4f}"), ("speedup", "Speedup over baseline", "{:.4f}"),
           ("l1d_mpki", "L1D load MPKI", "{:.3f}"), ("l1d_acc", "L1D prefetch accuracy (%)", "{:.2f}"),
           ("l1d_cov", "L1D load coverage (%)", "{:.2f}"), ("l2_mpki", "L2C load MPKI", "{:.3f}"),
           ("l2_instr", "L2C instruction load MPKI", "{:.3f}"), ("l2_data", "L2C data load MPKI", "{:.3f}")]
cfgs = [(c, l) for c, l in ALL + SWEEP[2:] if c in D]
md = []
for key, title, fmt in metrics:
    md.append(f"**{title}**\n\n| Config | " + " | ".join(TRACES) + " | Mean |\n|---|" + "---|" * (len(TRACES) + 1))
    for c, lab in cfgs:
        v = [D[c][t][key] for t in TRACES]
        m = gmean(v) if key == "speedup" else mean(v)
        f = lambda x: "–" if x != x else fmt.format(x)
        md.append(f"| {lab} | " + " | ".join(f(x) for x in v) + f" | {f(m)} |")
    md.append("")
open(f"{OUT}/tables.md", "w").write("\n".join(md))
with open(f"{OUT}/results.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["config", "trace"] + [k for k, _, _ in metrics])
    for c, _ in cfgs:
        for t in TRACES:
            w.writerow([c, t] + [D[c][t][k] for k, _, _ in metrics])
print("\n".join(md))

def plot(key, ylabel, title, fname, sel, ref=None, fmt="{:.3f}"):
    sel = [(c, l) for c, l in sel if c in D]
    labels = TRACES + ["Mean"]
    n = len(sel); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (c, lab) in enumerate(sel):
        v = [D[c][t][key] for t in TRACES]
        v.append(gmean(v) if key == "speedup" else mean(v))
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(labels))]
        ax.bar(xs, v, w * 0.92, label=lab, color=plt.cm.tab10(i))
        for x, y in zip(xs, v):
            ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=6)
    if ref is not None: ax.axhline(ref, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.12)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=200); plt.close(fig)

plot("speedup", "Speedup over baseline", "Task 3: speedup", "speedup.png", CFGS, ref=1.0, fmt="{:.4f}")
plot("l1d_mpki", "L1D load MPKI", "Task 3: L1D load MPKI", "l1d_mpki.png", CFGS, fmt="{:.2f}")
plot("l1d_acc", "L1D prefetch accuracy (%)", "Task 3: L1D prefetch accuracy", "accuracy.png", CFGS[1:], fmt="{:.1f}")
plot("l1d_cov", "L1D load miss coverage (%)", "Task 3: L1D coverage", "coverage.png", CFGS[1:], fmt="{:.1f}")
plot("l2_mpki", "L2C load MPKI", "Task 3: L2C load MPKI", "l2_mpki.png", CFGS, fmt="{:.2f}")
plot("l2_instr", "L2C instruction load MPKI", "Task 3: L2 instruction MPKI", "l2_instr_mpki.png", CFGS, fmt="{:.2f}")
plot("l2_data", "L2C data load MPKI", "Task 3: L2 data MPKI", "l2_data_mpki.png", CFGS, fmt="{:.2f}")
plot("speedup", "Speedup over baseline", "Task 3: prefetch insertion RRPV sweep", "pf_insert_sweep.png", SWEEP + CFGS[1:4], ref=1.0, fmt="{:.4f}")

# LaTeX tables for report/Summary.tex
tex = []
for key, title, fmt in metrics:
    tex.append("\\textbf{%s}\\\\[2pt]\n\\begin{tabular}{lrrrr}\\toprule\nConfig & Trace1 & Trace2 & Trace3 & Mean \\\\ \\midrule" % title.replace("%", "\\%"))
    for c, lab in cfgs:
        v = [D[c][t][key] for t in TRACES]
        m = gmean(v) if key == "speedup" else mean(v)
        f = lambda x: "--" if x != x else fmt.format(x)
        tex.append("%s & %s & %s \\\\" % (lab.replace("&", "\\&"), " & ".join(f(x) for x in v), f(m)))
    tex.append("\\bottomrule\\end{tabular}\n\\medskip\n")
open(os.path.join(ROOT, "report", "tables3.tex"), "w").write("\n".join(tex))
