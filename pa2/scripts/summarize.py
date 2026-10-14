#!/usr/bin/env python3
"""Summarise ChampSim outputs: IPC, speedup over results/<trace>/baseline.txt, and L2C load MPKIs.

Usage: scripts/summarize.py <subdir>     e.g. scripts/summarize.py irrip_variants
Reads results/<trace>/<subdir>/*.txt for every trace and writes results/<subdir>_summary.csv.
"""
import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"


def parse(path):
    text = path.read_text(errors="replace")
    stats = {}
    m = re.search(r"^CPU 0 cumulative IPC: ([\d.]+)", text, re.M)
    stats["ipc"] = float(m.group(1)) if m else None
    loads = re.findall(r"^L2C LOAD\s+ACCESS:.*MPKI: ([\d.]+)", text, re.M)
    stats["l2_mpki"] = float(loads[-1]) if loads else None
    for key, label in (("l2_instr_mpki", "INSTRUCTION"), ("l2_data_mpki", "DATA")):
        vals = re.findall(rf"^L2C {label} LOAD MPKI: ([\d.]+)", text, re.M)
        stats[key] = float(vals[-1]) if vals else None
    return stats


def main():
    sub = sys.argv[1]
    traces = sorted(p.name for p in ROOT.iterdir() if (p / sub).is_dir())
    rows = []
    speedups = {}
    for t in traces:
        base = parse(ROOT / t / "baseline.txt")["ipc"]
        for f in sorted((ROOT / t / sub).glob("*.txt")):
            s = parse(f)
            sp = s["ipc"] / base if s["ipc"] and base else None
            rows.append({"trace": t, "config": f.stem, **s, "speedup": sp})
            if sp:
                speedups.setdefault(f.stem, []).append(sp)

    out = ROOT / f"{sub}_summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trace", "config", "ipc", "speedup", "l2_mpki", "l2_instr_mpki", "l2_data_mpki"])
        w.writeheader()
        w.writerows(rows)

    print(f"{'config':20s} " + " ".join(f"{t:>9s}" for t in traces) + "   geomean")
    for cfg, sps in sorted(speedups.items(), key=lambda kv: -math.prod(kv[1]) ** (1 / len(kv[1]))):
        by_trace = {r["trace"]: r["speedup"] for r in rows if r["config"] == cfg}
        cols = " ".join(f"{by_trace.get(t) or float('nan'):9.4f}" for t in traces)
        print(f"{cfg:20s} {cols}   {math.prod(sps) ** (1 / len(sps)):.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
