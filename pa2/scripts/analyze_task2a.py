#!/usr/bin/env python3
"""Task 2A: parse results/trace*/ outputs -> tables (md/csv/tex) + plots in results/task2a/.
Main configs:  results/<trace>/{baseline,l2_rrip,l2_irrip,l2_irrip_final}.txt
Ablation:      results/<trace>/irrip_variants/*.txt   (stage 1, scripts/irrip_stage1.cfg)
               results/<trace>/irrip_stage2/*.txt     (stage 2, scripts/irrip_stage2.cfg)
Usage: analyze_task2a.py"""
import re, os, csv, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "task2a")
os.makedirs(OUT, exist_ok=True)
TRACES = ["trace1", "trace2", "trace3"]
TLAB = ["Trace1", "Trace2", "Trace3"]
CFGS = [("baseline", "Baseline (LRU)"), ("l2_rrip", "SRRIP"),
        ("l2_irrip_final", "I-RRIP")]

STAGE1 = [("srrip_equiv", "SRRIP (all knobs off)"),
          ("ins_only", "Insertion only (instr RRPV 0)"),
          ("promo_only", "Promotion only (data decrement)"),
          ("evict_prefer_only", "Eviction only (prefer data)"),
          ("evict_age_only", "Eviction only (age data faster)"),
          ("orig_irrip", "Insertion + promotion (I+P)"),
          ("orig_prefer", "I+P + prefer data"),
          ("orig_age", "I+P + age data faster"),
          ("orig_prefer_age", "I+P + both eviction options"),
          ("ins_instr1", "I+P, instr insert RRPV 1"),
          ("ins_data3", "I+P, data insert RRPV 3"),
          ("ins_data_bip", "I+P, data insert bimodal"),
          ("promo_data_none", "I+P, data not promoted"),
          ("promo_instr_dec", "I+P, instr decrement on hit")]


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
    )


def mean(xs):
    return sum(xs) / len(xs)


def gmean(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p ** (1 / len(xs))


base = {t: parse(f"{RES}/{t}/baseline.txt")["ipc"] for t in TRACES}
D = {}
for c, _ in CFGS:
    D[c] = {}
    for t in TRACES:
        r = parse(f"{RES}/{t}/{c}.txt")
        r["speedup"] = r["ipc"] / base[t]
        D[c][t] = r

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
    for c, _ in CFGS:
        for t in TRACES:
            w.writerow([c, t] + [D[c][t][k] for k, _, _ in metrics])


def plot(key, ylabel, title, fname, ref=None, fmt="{:.3f}"):
    labels = TLAB + ["Mean"]
    n = len(CFGS)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (c, lab) in enumerate(CFGS):
        v = [D[c][t][key] for t in TRACES]
        v.append(agg(key, v))
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(labels))]
        ax.bar(xs, v, w * 0.92, label=lab, color=plt.cm.tab10(i))
        for x, y in zip(xs, v):
            ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=6)
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
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=200)
    plt.close(fig)


plot("speedup", "Speedup over baseline", "Task 2A: speedup", "speedup.png", ref=1.0, fmt="{:.4f}")
plot("l2_mpki", "L2C load MPKI", "Task 2A: L2C load MPKI", "l2_mpki.png", fmt="{:.2f}")
plot("l2_instr", "L2C instruction load MPKI", "Task 2A: L2 instruction MPKI", "l2_instr_mpki.png", fmt="{:.2f}")
plot("l2_data", "L2C data load MPKI", "Task 2A: L2 data MPKI", "l2_data_mpki.png", fmt="{:.2f}")


# design-space exploration (stage 1 and stage 2)
def sweep(sub):
    out = {}
    for f in sorted(glob.glob(f"{RES}/{TRACES[0]}/{sub}/*.txt")):
        c = os.path.basename(f)[:-4]
        out[c] = [parse(f"{RES}/{t}/{sub}/{c}.txt")["ipc"] / base[t] for t in TRACES]
    return out


S1 = sweep("irrip_variants")
S2 = sweep("irrip_stage2")


def s2_label(c):
    _, i, d, p, a = c.split("_")
    return (f"instr {i[1]} / data {'RRPV 3' if d == 'd3' else 'bimodal'} / "
            f"prefer {'on' if p == 'p1' else 'off'} / age {'on' if a == 'a1' else 'off'}")


md.append("**Stage 1 (one knob at a time), speedup over baseline**\n\n| Config | " + " | ".join(TLAB) + " | Mean |\n|---|---|---|---|---|")
for c, lab in STAGE1:
    md.append(f"| {lab} | " + " | ".join(f"{x:.4f}" for x in S1[c]) + f" | {gmean(S1[c]):.4f} |")
md.append("\n**Stage 2 (full factorial of the knobs that mattered), speedup over baseline**\n\n| Config | " + " | ".join(TLAB) + " | Mean |\n|---|---|---|---|---|")
for c in sorted(S2, key=lambda k: -gmean(S2[k])):
    md.append(f"| {s2_label(c)} | " + " | ".join(f"{x:.4f}" for x in S2[c]) + f" | {gmean(S2[c]):.4f} |")
open(f"{OUT}/tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

# ablation plot: stage 1 geomean speedups
fig, ax = plt.subplots(figsize=(8, 4.5))
labs = [lab for _, lab in STAGE1]
vals = [gmean(S1[c]) for c, _ in STAGE1]
ax.barh(range(len(labs)), vals, color=plt.cm.tab10(0))
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)
ax.set_yticks(range(len(labs)))
ax.set_yticklabels(labs, fontsize=7)
ax.invert_yaxis()
ax.axvline(1.0, color="k", lw=0.8, ls="--")
ax.set_xlim(0.999, max(vals) + 0.003)
ax.set_xlabel("Mean speedup over baseline (geomean)")
ax.set_title("Task 2A: stage 1 ablation")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/ablation_stage1.png", dpi=200)
plt.close(fig)

# LaTeX tables for report/Summary.tex
tex = []
for key, title, fmt in metrics:
    tex.append("\\textbf{%s}\\\\[2pt]\n\\begin{tabular}{lrrrr}\\toprule\nConfig & Trace1 & Trace2 & Trace3 & Mean \\\\ \\midrule" % title)
    for c, lab in CFGS:
        v = [D[c][t][key] for t in TRACES]
        tex.append("%s & %s & %s \\\\" % (lab, " & ".join(fmt.format(x) for x in v), fmt.format(agg(key, v))))
    tex.append("\\bottomrule\\end{tabular}\n\\medskip\n")
open(os.path.join(ROOT, "report", "tables2a.tex"), "w").write("\n".join(tex))

tex = ["\\begin{tabular}{lrrrr}\\toprule\nConfiguration & Trace1 & Trace2 & Trace3 & Mean \\\\ \\midrule"]
for c, lab in STAGE1:
    tex.append("%s & %s & %.4f \\\\" % (lab, " & ".join(f"{x:.4f}" for x in S1[c]), gmean(S1[c])))
tex.append("\\bottomrule\\end{tabular}")
open(os.path.join(ROOT, "report", "tables2a_stage1.tex"), "w").write("\n".join(tex) + "\n")

tex = ["\\begin{tabular}{lrrrr}\\toprule\nInstr insert / data insert / prefer data / age data & Trace1 & Trace2 & Trace3 & Mean \\\\ \\midrule"]
for c in sorted(S2, key=lambda k: -gmean(S2[k])):
    lab = s2_label(c) + (" (chosen)" if c == "s2_i1_d3_p1_a0" else "")
    tex.append("%s & %s & %.4f \\\\" % (lab, " & ".join(f"{x:.4f}" for x in S2[c]), gmean(S2[c])))
tex.append("\\bottomrule\\end{tabular}")
open(os.path.join(ROOT, "report", "tables2a_stage2.tex"), "w").write("\n".join(tex) + "\n")
