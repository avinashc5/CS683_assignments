#!/usr/bin/env python3
"""Task 1: parse results/{baseline,ipstride,spp}/Trace*.txt -> tables (md/csv) + 4 plots in results/task1/."""
import re, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "task1")
os.makedirs(OUT, exist_ok=True)
TRACES = ["Trace1", "Trace2", "Trace3"]
CFGS = [("baseline", "No prefetcher"), ("ipstride", "IP-stride"), ("spp", "SPP (ours)")]

def num(txt, pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else float("nan")

def parse(cfg, t):
    s = open(f"{RES}/{cfg}/{t}.txt").read()
    return dict(
        ipc=num(s, r"CPU 0 cumulative IPC: ([\d.]+)"),
        mpki=num(s, r"L1D LOAD .*MPKI:\s+([\d.]+)"),
        misses=num(s, r"L1D LOAD\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)"),
        acc=num(s, r"L1D USEFUL LOAD PREFETCHES:.*ACCURACY: ([\d.]+)"),
        useful=num(s, r"L1D USEFUL LOAD PREFETCHES:\s+(\d+)"),
        issued=num(s, r"L1D USEFUL LOAD PREFETCHES:\s+\d+ PREFETCH ISSUED TO LOWER LEVEL:\s+(\d+)"),
        cross=num(s, r"L1D Cross Page Prefetch Requests:\s+(\d+)"),
    )

D = {c: {t: parse(c, t) for t in TRACES} for c, _ in CFGS}
for c in D:
    for t in TRACES:
        b = D["baseline"][t]; r = D[c][t]
        r["speedup"] = r["ipc"] / b["ipc"]
        r["cov"] = 100 * (b["misses"] - r["misses"]) / b["misses"]
        if c == "baseline" or r["issued"] == 0 or r["issued"] != r["issued"]: r["acc"] = float("nan")

def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")
def gmean(xs):
    p = 1.0
    for x in xs: p *= x
    return p ** (1 / len(xs))

metrics = [("ipc", "IPC", "{:.4f}"), ("speedup", "Speedup", "{:.4f}"), ("mpki", "L1D load MPKI", "{:.3f}"),
           ("acc", "Prefetch accuracy (%)", "{:.2f}"), ("cov", "L1D load coverage (%)", "{:.2f}")]
md = []
for key, title, fmt in metrics:
    md.append(f"**{title}**\n\n| Config | " + " | ".join(TRACES) + " | Mean |\n|---|" + "---|" * (len(TRACES) + 1))
    for c, lab in CFGS:
        v = [D[c][t][key] for t in TRACES]
        if key == "speedup": m = gmean(v)
        else: m = mean(v)
        f = lambda x: "–" if x != x or (key in ("acc", "cov") and c == "baseline") else fmt.format(x)
        md.append(f"| {lab} | " + " | ".join(f(x) for x in v) + f" | {f(m)} |")
    md.append("")
open(f"{OUT}/tables.md", "w").write("\n".join(md))
with open(f"{OUT}/results.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["config", "trace", "ipc", "speedup", "l1d_load_mpki", "l1d_load_misses",
                                    "accuracy_pct", "coverage_pct", "useful_prefetches", "prefetches_to_lower_level", "cross_page"])
    for c, _ in CFGS:
        for t in TRACES:
            r = D[c][t]; w.writerow([c, t, r["ipc"], r["speedup"], r["mpki"], int(r["misses"]), r["acc"], r["cov"], int(r["useful"]), int(r["issued"]), int(r["cross"])])
print("\n".join(md))

COL = {"baseline": "#9aa0a6", "ipstride": "#4c78a8", "spp": "#e45756"}
def plot(key, ylabel, title, fname, cfgs, ref=None, fmt="{:.3f}", ylim=None):
    labels = TRACES + ["Mean"]
    n = len(cfgs); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (c, lab) in enumerate(cfgs):
        v = [D[c][t][key] for t in TRACES]
        v.append(gmean(v) if key == "speedup" else mean(v))
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(labels))]
        bars = ax.bar(xs, v, w * 0.92, label=lab, color=COL[c])
        for x, y in zip(xs, v):
            ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=7)
    if ref is not None: ax.axhline(ref, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.12)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=200); plt.close(fig)

plot("speedup", "Speedup over no prefetching", "L1D prefetchers: speedup", "speedup.png", CFGS, ref=1.0, ylim=(0.98, 1.025))
plot("mpki", "L1D load MPKI", "L1D load MPKI (lower is better)", "mpki.png", CFGS, fmt="{:.2f}")
plot("acc", "Prefetch accuracy (%)", "Prefetch accuracy (useful / issued to lower level)", "accuracy.png", CFGS[1:], fmt="{:.1f}")
plot("cov", "L1D load miss coverage (%)", "L1D load miss coverage", "coverage.png", CFGS[1:], fmt="{:.1f}")
