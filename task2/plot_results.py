#!/usr/bin/env python3
"""Parse task2 bench/*_results.txt and generate plots + LaTeX data macros."""

import os, re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def parse_float(s):
    try: return float(s.strip().replace(',', ''))
    except ValueError: return None

def parse_speedup(s):
    s = s.strip().rstrip('x')
    try: return float(s)
    except ValueError: return None

# ============================================================================
# SIMD width results
# ============================================================================
def plot_simd_width():
    path = os.path.join(BASE, 'bench', 'simd_width_results.txt')
    if not os.path.exists(path):
        print("  skip simd_width: not found")
        return

    # {size: {width_label: {field: val}}}
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('size') or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            size_str = parts[0]
            size = int(size_str.split('x')[0])
            # width label spans parts[1] and possibly parts[2]
            if parts[1] == 'naive':
                width = 'Naive'
                gflops = parse_float(parts[3])
                speedup = parse_speedup(parts[4])
                insns = parse_float(parts[5]) if len(parts) > 5 else None
            elif parts[1] == 'SSE':
                width = 'SSE 128-bit'
                gflops = parse_float(parts[3])
                speedup = parse_speedup(parts[4])
                insns = parse_float(parts[5]) if len(parts) > 5 else None
            elif parts[1] == 'AVX2':
                width = 'AVX2 256-bit'
                gflops = parse_float(parts[3])
                speedup = parse_speedup(parts[4])
                insns = parse_float(parts[5]) if len(parts) > 5 else None
            elif parts[1] == 'AVX-512':
                width = 'AVX-512 512-bit'
                gflops = parse_float(parts[3])
                speedup = parse_speedup(parts[4])
                insns = parse_float(parts[5]) if len(parts) > 5 else None
            else:
                continue
            data[size][width]['gflops'] = gflops
            data[size][width]['speedup'] = speedup
            data[size][width]['insns'] = insns

    sizes = sorted(data.keys())
    size_labels = [str(s) for s in sizes]
    widths = ['SSE 128-bit', 'AVX2 256-bit', 'AVX-512 512-bit']
    colors = {'SSE 128-bit': 'tab:green', 'AVX2 256-bit': 'tab:blue', 'AVX-512 512-bit': 'tab:red'}

    # Speedup plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    for w in widths:
        vals = [data[s].get(w, {}).get('speedup', 0) or 0 for s in sizes]
        ax.plot(size_labels, vals, '-o', color=colors[w], label=w, markersize=6, linewidth=2)
    ax.set_xlabel('Matrix Size (M = N = K)')
    ax.set_ylabel('Speedup over Naive')
    ax.set_title('SIMD Speedup vs Matrix Size by Register Width')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'simd_width_speedup.png'), dpi=150)
    plt.close(fig)
    print("  saved simd_width_speedup.png")

    # GFLOP/s plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for w in ['Naive'] + widths:
        vals = [data[s].get(w, {}).get('gflops', 0) or 0 for s in sizes]
        c = colors.get(w, 'black')
        ls = '--' if w == 'Naive' else '-'
        ax.plot(size_labels, vals, f'{ls}o', color=c, label=w, markersize=6, linewidth=2)
    ax.set_xlabel('Matrix Size (M = N = K)')
    ax.set_ylabel('GFLOP/s')
    ax.set_title('SGEMM Throughput by SIMD Width')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'simd_width_gflops.png'), dpi=150)
    plt.close(fig)
    print("  saved simd_width_gflops.png")

    # Instruction count plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for w in ['Naive'] + widths:
        vals = [data[s].get(w, {}).get('insns') for s in sizes]
        if not any(v is not None for v in vals):
            continue
        vals = [v if v else 0 for v in vals]
        c = colors.get(w, 'black')
        ls = '--' if w == 'Naive' else '-'
        ax.plot(size_labels, vals, f'{ls}o', color=c, label=w, markersize=6, linewidth=2)
    ax.set_xlabel('Matrix Size (M = N = K)')
    ax.set_ylabel('Instruction Count')
    ax.set_title('Instruction Count by SIMD Width')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'simd_width_insns.png'), dpi=150)
    plt.close(fig)
    print("  saved simd_width_insns.png")

    return data

# ============================================================================
# Size sweep (simd / prefetch / optimized)
# ============================================================================
def plot_size_sweep():
    path = os.path.join(BASE, 'bench', 'size_sweep_results.txt')
    if not os.path.exists(path):
        print("  skip size_sweep: not found")
        return

    # {size: {stage: {field: val}}}
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('size') or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            size = int(parts[0].split('x')[0])
            stage = parts[1]
            data[size][stage]['naive_gflops'] = parse_float(parts[2])
            data[size][stage]['gflops'] = parse_float(parts[3])
            data[size][stage]['speedup'] = parse_speedup(parts[4])

    sizes = sorted(data.keys())
    size_labels = [str(s) for s in sizes]
    stages = ['simd', 'prefetch', 'optimized']
    stage_labels = {'simd': 'SIMD', 'prefetch': 'Prefetch', 'optimized': 'Prefetch + SIMD'}
    colors = {'simd': 'tab:blue', 'prefetch': 'tab:orange', 'optimized': 'tab:green'}

    # Speedup comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    for stage in stages:
        vals = [data[s].get(stage, {}).get('speedup', 0) or 0 for s in sizes]
        ax.plot(size_labels, vals, '-o', color=colors[stage],
                label=stage_labels[stage], markersize=6, linewidth=2)
    ax.set_xlabel('Matrix Size (M = N = K)')
    ax.set_ylabel('Speedup over Naive')
    ax.set_title('SGEMM: Technique Comparison Across Matrix Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'task2_final_comparison.png'), dpi=150)
    plt.close(fig)
    print("  saved task2_final_comparison.png")

    # GFLOP/s comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # naive baseline
    naive_vals = [data[s].get('simd', {}).get('naive_gflops', 0) or 0 for s in sizes]
    ax.plot(size_labels, naive_vals, 'k--o', label='Naive', markersize=5, linewidth=1.5)
    for stage in stages:
        vals = [data[s].get(stage, {}).get('gflops', 0) or 0 for s in sizes]
        ax.plot(size_labels, vals, '-o', color=colors[stage],
                label=stage_labels[stage], markersize=6, linewidth=2)
    ax.set_xlabel('Matrix Size (M = N = K)')
    ax.set_ylabel('GFLOP/s')
    ax.set_title('SGEMM Throughput: All Techniques')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'task2_gflops_comparison.png'), dpi=150)
    plt.close(fig)
    print("  saved task2_gflops_comparison.png")

    return data

# ============================================================================
# Perf stage (hardware counters bar chart)
# ============================================================================
def plot_perf_stage():
    path = os.path.join(BASE, 'bench', 'perf_stage_results.txt')
    if not os.path.exists(path):
        print("  skip perf_stage: not found")
        return

    # parse cold-cache rows only
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('stage'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            stage, cache = parts[0], parts[1]
            if cache != 'cold':
                continue
            data[stage] = {
                'insns': parse_float(parts[2]),
                'l1d_miss': parse_float(parts[3]),
                'l2_miss': parse_float(parts[4]),
                'llc_miss': parse_float(parts[5]),
                'ipc': parse_float(parts[6]),
            }

    if not data:
        return

    stages = ['naive', 'simd', 'prefetch', 'optimized']
    stage_labels = ['Naive', 'SIMD', 'Prefetch', 'Prefetch+SIMD']
    colors = ['tab:gray', 'tab:blue', 'tab:orange', 'tab:green']

    # Cache misses grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(stages))
    width = 0.25
    metrics = ['l1d_miss', 'l2_miss', 'llc_miss']
    metric_labels = ['L1-D Misses', 'L2 Misses', 'LLC Misses']
    metric_colors = ['tab:red', 'tab:orange', 'tab:blue']
    for i, (m, ml, mc) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [data.get(s, {}).get(m, 0) or 0 for s in stages]
        ax.bar(x + i * width, vals, width, label=ml, color=mc, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel('Count')
    ax.set_title('Cache Misses by Stage (M=N=K=1024, cold cache)')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'perf_cache_misses.png'), dpi=150)
    plt.close(fig)
    print("  saved perf_cache_misses.png")

    # Instruction count + IPC bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    insn_vals = [data.get(s, {}).get('insns', 0) or 0 for s in stages]
    ax1.bar(stage_labels, insn_vals, color=colors)
    ax1.set_ylabel('Instruction Count')
    ax1.set_title('Instructions (M=N=K=1024)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    ipc_vals = [data.get(s, {}).get('ipc', 0) or 0 for s in stages]
    ax2.bar(stage_labels, ipc_vals, color=colors)
    ax2.set_ylabel('IPC')
    ax2.set_title('Instructions Per Cycle')
    ax2.set_ylim(0, max(ipc_vals) * 1.3)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'perf_insns_ipc.png'), dpi=150)
    plt.close(fig)
    print("  saved perf_insns_ipc.png")

    return data

# ============================================================================
def main():
    print(f"Generating task2 plots in {PLOT_DIR}/")
    plot_simd_width()
    plot_size_sweep()
    plot_perf_stage()
    print("Done.")

if __name__ == '__main__':
    main()
