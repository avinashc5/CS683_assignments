#!/usr/bin/env python3
"""Task 2B: parse results/trace*/ outputs -> tables (md/csv/tex) + plots in results/task2b/.
Main configs:  results/<trace>/{baseline,l2_rrip,l2_irrip_final,l2_dyn_irrip_final}.txt
               results/<trace>/irrip_stage2/s2_i1_d3_p1_a1.txt  (the I-RRIP mode used by the dynamic policy, run statically)
Sensitivity:   results/<trace>/dyn_sweep/*.txt   (scripts/dyn_sweep.cfg)
Usage: analyze_task2b.py"""
import re, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "task2b")
os.makedirs(OUT, exist_ok=True)
TRACES = ["trace1", "trace2", "trace3"]
TLAB = ["Trace1", "Trace2", "Trace3"]
CFGS = [("baseline", "Baseline (LRU)"), ("l2_rrip", "SRRIP"),
        ("l2_irrip_final", "I-RRIP (Task 2A)"),
        ("irrip_stage2/s2_i1_d3_p1_a1", "I-RRIP mode, static"),
        ("l2_dyn_irrip_final", "Dynamic I-RRIP")]
CHOSEN = "l32_h_p10"
LEADERS = [8, 16, 32]
SPREADS = [("c", "contiguous"), ("h", "hashed")]
WIDTHS = [8, 10, 12]


def num(txt, pat):
    m = re.findall(pat, txt, re.M)
    return float(m[-1]) if m else float("nan")


def parse(path):
    s = open(path).read()
    return dict(
        ipc=num(s, r"^CPU 0 cumulative IPC: ([\d.]+)"),
        l2_mpki=num(s, r"^L2C LOAD\s+ACCESS:.*MPKI: ([\d.]+)"),
        l2_instr=num(s, r"^L2C INSTRUCTION LOAD MPKI: ([\d.]+)"),
        l2_data=num(s, r"^L2C DATA LOAD MPKI: ([\d.]+)"),
        psel=num(s, r"^Final PSEL: (\d+)"),
        irrip_frac=num(s, r"IRRIP fraction: ([\d.e-]+)"),
        switches=num(s, r"^Mode switches: (\d+)"),
        lm_srrip=sum(map(float, (re.findall(r"^SRRIP leader misses\s+data: (\d+)\s+instr: (\d+)", s, re.M) or [("nan", "nan")])[-1])),
        lm_irrip=sum(map(float, (re.findall(r"^IRRIP leader misses\s+data: (\d+)\s+instr: (\d+)", s, re.M) or [("nan", "nan")])[-1])),
    )


def mean(xs):
    return sum(xs) / len(xs)


def gmean(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p ** (1 / len(xs))


base = {t: parse(f"{RES}/{t}/baseline.txt")["ipc"] for t in TRACES}


def load(c):
    out = {}
    for t in TRACES:
        r = parse(f"{RES}/{t}/{c}.txt")
        r["speedup"] = r["ipc"] / base[t]
        out[t] = r
    return out


D = {c: load(c) for c, _ in CFGS}

metrics = [("ipc", "IPC", "{:.4f}"), ("speedup", "Speedup over baseline", "{:.4f}"),
           ("l2_mpki", "L2C load MPKI", "{:.3f}"), ("l2_instr", "L2C instruction load MPKI", "{:.3f}"),
           ("l2_data", "L2C data load MPKI", "{:.3f}")]


def agg(key, v):
    return gmean(v) if key == "speedup" else mean(v)


md = []
for key, title, fmt in metrics:
    md.append(f"**{title}**\n\n| Config | " + " | ".join(TLAB) + " | Mean |\n|---|" + "---|" * (len(TRACES) + 1))
    for c, lab in CFGS:
        v = [D[c][t][key] for t in TRACES]
        md.append(f"| {lab} | " + " | ".join(fmt.format(x) for x in v) + f" | {fmt.format(agg(key, v))} |")
    md.append("")
with open(f"{OUT}/results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["config", "trace"] + [k for k, _, _ in metrics])
    for c, lab in CFGS:
        for t in TRACES:
            w.writerow([lab, t] + [D[c][t][k] for k, _, _ in metrics])


def plot(key, ylabel, title, fname, ref=None, fmt="{:.3f}"):
    labels = TLAB + ["Mean"]
    n = len(CFGS)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (c, lab) in enumerate(CFGS):
        v = [D[c][t][key] for t in TRACES]
        v.append(agg(key, v))
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(labels))]
        ax.bar(xs, v, w * 0.92, label=lab, color=plt.cm.tab10(i))
        for x, y in zip(xs, v):
            ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=5, rotation=90)
    if ref is not None:
        ax.axhline(ref, color="k", lw=0.8, ls="--")
    if key == "speedup":
        ax.set_ylim(0.99, None)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=200)
    plt.close(fig)


plot("speedup", "Speedup over baseline", "Task 2B: speedup", "speedup.png", ref=1.0, fmt="{:.4f}")
plot("l2_mpki", "L2C load MPKI", "Task 2B: L2C load MPKI", "l2_mpki.png", fmt="{:.2f}")
plot("l2_instr", "L2C instruction load MPKI", "Task 2B: L2 instruction MPKI", "l2_instr_mpki.png", fmt="{:.2f}")
plot("l2_data", "L2C data load MPKI", "Task 2B: L2 data MPKI", "l2_data_mpki.png", fmt="{:.2f}")

# selector behaviour of the chosen design
sel_rows = [(t, D["l2_dyn_irrip_final"][t]) for t in TRACES]
md.append("**Dynamic I-RRIP selector (after warmup)**\n\n| Trace | Demand misses, SRRIP leaders | Demand misses, I-RRIP leaders | Final PSEL (of 1023) | Follower updates in I-RRIP mode | Mode switches |\n|---|---|---|---|---|---|")
for (t, r), lab in zip(sel_rows, TLAB):
    md.append(f"| {lab} | {r['lm_srrip']:.0f} | {r['lm_irrip']:.0f} | {r['psel']:.0f} | {100 * r['irrip_frac']:.1f}% | {r['switches']:.0f} |")
md.append("")

# upper bound: per trace, the better of static SRRIP and the static I-RRIP mode
ub = [max(D["l2_rrip"][t]["speedup"], D["irrip_stage2/s2_i1_d3_p1_a1"][t]["speedup"]) for t in TRACES]

# sensitivity study
S = {}
for n in LEADERS:
    for s, _ in SPREADS:
        for b in WIDTHS:
            S[(n, s, b)] = load(f"dyn_sweep/l{n}_{s}_p{b}")
F = {k: load(f"dyn_sweep/force_{k}") for k in ("srrip", "irrip")}


def sens_label(n, s, b):
    return f"{n} / {dict(SPREADS)[s]} / {b}-bit"


md.append("**Sensitivity (speedup over baseline); last column: share of Trace1 follower updates in I-RRIP mode**\n\n| Leader sets / spread / PSEL | " + " | ".join(TLAB) + " | Mean | Trace1 I-RRIP share |\n|---|---|---|---|---|---|")
for k, r in S.items():
    v = [r[t]["speedup"] for t in TRACES]
    md.append(f"| {sens_label(*k)} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {gmean(v):.4f} | {100 * r['trace1']['irrip_frac']:.1f}% |")
md.append(f"| Upper bound (best static per trace) | " + " | ".join(f"{x:.4f}" for x in ub) + f" | {gmean(ub):.4f} | |")
md.append("")
open(f"{OUT}/tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (s, sl) in zip(axes, SPREADS):
    w = 0.8 / len(WIDTHS)
    for i, b in enumerate(WIDTHS):
        v = [gmean([S[(n, s, b)][t]["speedup"] for t in TRACES]) for n in LEADERS]
        xs = [j + (i - (len(WIDTHS) - 1) / 2) * w for j in range(len(LEADERS))]
        ax.bar(xs, v, w * 0.92, label=f"{b}-bit PSEL", color=plt.cm.tab10(i))
        for x, y in zip(xs, v):
            ax.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.axhline(gmean(ub), color="k", lw=0.8, ls="--", label="best static per trace")
    ax.set_xticks(range(len(LEADERS)))
    ax.set_xticklabels([f"{n} leaders" for n in LEADERS])
    ax.set_title(f"{sl} leader sets", fontsize=9)
    ax.set_ylim(1.008, gmean(ub) + 0.0015)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Mean speedup (geomean)")
axes[1].legend(fontsize=7, frameon=False, loc="lower right")
fig.suptitle("Task 2B: sensitivity to leader sets, spread and PSEL width", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/sensitivity.png", dpi=200)
plt.close(fig)

# LaTeX tables for report/Summary.tex
tex = []
for key, title, fmt in metrics:
    tex.append("\\textbf{%s}\\\\[2pt]\n\\begin{tabular}{lrrrr}\\toprule\nConfig & Trace1 & Trace2 & Trace3 & Mean \\\\ \\midrule" % title)
    for c, lab in CFGS:
        v = [D[c][t][key] for t in TRACES]
        tex.append("%s & %s & %s \\\\" % (lab, " & ".join(fmt.format(x) for x in v), fmt.format(agg(key, v))))
    tex.append("\\bottomrule\\end{tabular}\n\\medskip\n")
open(os.path.join(ROOT, "report", "tables2b.tex"), "w").write("\n".join(tex))

tex = ["\\begin{tabular}{lrrrrr}\\toprule\nLeader sets / spread / PSEL & Trace1 & Trace2 & Trace3 & Mean & Trace1 I-RRIP \\\\ \\midrule"]
for k, r in S.items():
    v = [r[t]["speedup"] for t in TRACES]
    lab = sens_label(*k) + (" (chosen)" if f"l{k[0]}_{k[1]}_p{k[2]}" == CHOSEN else "")
    tex.append("%s & %s & %.4f & %.1f\\%% \\\\" % (lab, " & ".join(f"{x:.4f}" for x in v), gmean(v), 100 * r["trace1"]["irrip_frac"]))
tex.append("\\midrule\nBest static per trace & %s & %.4f & \\\\" % (" & ".join(f"{x:.4f}" for x in ub), gmean(ub)))
tex.append("\\bottomrule\\end{tabular}")
open(os.path.join(ROOT, "report", "tables2b_sweep.tex"), "w").write("\n".join(tex) + "\n")

tex = ["\\begin{tabular}{lrrrr}\\toprule\nConfiguration & Trace1 & Trace2 & Trace3 & Mean \\\\ \\midrule"]
for lab, r in [("Followers forced to SRRIP", F["srrip"]), ("Static SRRIP", D["l2_rrip"]),
               ("Followers forced to I-RRIP", F["irrip"]), ("Static I-RRIP mode", D["irrip_stage2/s2_i1_d3_p1_a1"])]:
    v = [r[t]["speedup"] for t in TRACES]
    tex.append("%s & %s & %.4f \\\\" % (lab, " & ".join(f"{x:.4f}" for x in v), gmean(v)))
tex.append("\\bottomrule\\end{tabular}")
open(os.path.join(ROOT, "report", "tables2b_sanity.tex"), "w").write("\n".join(tex) + "\n")

tex = ["\\begin{tabular}{lrrrrr}\\toprule\nTrace & \\multicolumn{2}{c}{Leader demand misses} & Final PSEL & Followers in & Mode \\\\\n & SRRIP & I-RRIP & (of 1023) & I-RRIP mode & switches \\\\ \\midrule"]
for (t, r), lab in zip(sel_rows, TLAB):
    tex.append("%s & %.0f & %.0f & %.0f & %.1f\\%% & %.0f \\\\" % (lab, r["lm_srrip"], r["lm_irrip"], r["psel"], 100 * r["irrip_frac"], r["switches"]))
tex.append("\\bottomrule\\end{tabular}")
open(os.path.join(ROOT, "report", "tables2b_selector.tex"), "w").write("\n".join(tex) + "\n")
