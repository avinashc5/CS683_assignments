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
def _get_best_prefetch_per_size():
    """Extract best scalar prefetch (2A) GFLOP/s and speedup per size from prefetch_sweep_results.txt."""
    raw = _parse_prefetch_sweep()
    if raw is None:
        return {}
    result = {}
    for mode in raw:
        for hint in raw[mode]:
            for dist in raw[mode][hint]:
                for size, rec in raw[mode][hint][dist].items():
                    sp = rec.get('speedup') or 0
                    if size not in result or sp > (result[size].get('speedup') or 0):
                        result[size] = {
                            'naive_gflops': rec.get('naive_gflops'),
                            'gflops': rec.get('pf_gflops'),
                            'speedup': sp,
                        }
    return result


def _parse_optimized_sweep():
    """Parse optimized_sweep_results.txt → {size: {gflops, speedup}}."""
    path = os.path.join(BASE, 'bench', 'optimized_sweep_results.txt')
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('size') or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            size = int(parts[0].split('x')[0])
            result[size] = {
                'gflops': parse_float(parts[3]),
                'speedup': parse_speedup(parts[4]),
            }
    return result


def plot_size_sweep():
    """Build final comparison plots from simd_width + prefetch_sweep + optimized_sweep."""
    # {size: {stage: {field: val}}}
    data = defaultdict(lambda: defaultdict(dict))

    # --- SIMD data from simd_width_results.txt (AVX2 row) ---
    simd_path = os.path.join(BASE, 'bench', 'simd_width_results.txt')
    if os.path.exists(simd_path):
        with open(simd_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('size') or line.startswith('-'):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                size = int(parts[0].split('x')[0])
                if parts[1] == 'naive':
                    data[size]['naive']['gflops'] = parse_float(parts[3])
                elif parts[1] == 'AVX2':
                    data[size]['simd']['gflops'] = parse_float(parts[3])
                    data[size]['simd']['speedup'] = parse_speedup(parts[4])

    # --- Prefetch (scalar 2A) data from prefetch_sweep_results.txt ---
    pf_best = _get_best_prefetch_per_size()
    for size, rec in pf_best.items():
        data[size]['prefetch'] = rec

    # --- Optimized (SIMD + blocking) from optimized_sweep_results.txt ---
    opt = _parse_optimized_sweep()
    for size, rec in opt.items():
        data[size]['optimized'] = rec

    sizes = sorted(s for s in data if any(
        stage in data[s] for stage in ('simd', 'prefetch', 'optimized')))
    if not sizes:
        print("  skip size_sweep: no data")
        return
    size_labels = [str(s) for s in sizes]
    stages = ['simd', 'prefetch', 'optimized']
    stage_labels = {'simd': 'SIMD (AVX2)', 'prefetch': 'Prefetch (scalar)',
                    'optimized': 'SIMD + Blocking'}
    colors = {'simd': 'tab:blue', 'prefetch': 'tab:orange', 'optimized': 'tab:green'}

    # Speedup comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    for stage in stages:
        vals = [data[s].get(stage, {}).get('speedup', 0) or 0 for s in sizes]
        if any(v > 0 for v in vals):
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
    naive_vals = [data[s].get('naive', {}).get('gflops', 0) or 0 for s in sizes]
    ax.plot(size_labels, naive_vals, 'k--o', label='Naive', markersize=5, linewidth=1.5)
    for stage in stages:
        vals = [data[s].get(stage, {}).get('gflops', 0) or 0 for s in sizes]
        if any(v > 0 for v in vals):
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

    # Override prefetch row with scalar 2A data from prefetch_sweep_results.txt
    raw_pf = _parse_prefetch_sweep()
    if raw_pf and 'next_B' in raw_pf and 'T0' in raw_pf['next_B'] and 128 in raw_pf['next_B']['T0']:
        rec = raw_pf['next_B']['T0'][128].get(1024, {})
        if rec:
            data['prefetch'] = {
                'insns': data.get('prefetch', {}).get('insns'),
                'l1d_miss': rec.get('l1d_miss'),
                'l2_miss': rec.get('l2_miss'),
                'llc_miss': rec.get('llc_miss'),
                'ipc': data.get('prefetch', {}).get('ipc'),
            }

    stages = ['naive', 'simd', 'prefetch', 'optimized']
    stage_labels = ['Naive', 'SIMD', 'Prefetch (scalar)', 'Prefetch+SIMD']
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
# Prefetch sweep results
# ============================================================================
def _parse_prefetch_sweep():
    path = os.path.join(BASE, 'bench', 'prefetch_sweep_results.txt')
    if not os.path.exists(path):
        return None
    # {mode: {hint: {dist: {size: {field: val}}}}}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('size') or line.startswith('='):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            size = int(parts[0].split('x')[0])
            mode = parts[1]
            hint = parts[2]
            dist_s = parts[3]
            dist = int(dist_s) if dist_s != '--' else -1
            rec = {
                'naive_gflops': parse_float(parts[4]),
                'pf_gflops': parse_float(parts[5]),
                'speedup': parse_speedup(parts[6]),
                'l1d_miss': parse_float(parts[7]),
                'l2_miss': parse_float(parts[8]),
                'llc_miss': parse_float(parts[9]),
                'sw_pf_acc': parse_float(parts[10]),
            }
            data[mode][hint][dist][size] = rec
    return data


def plot_prefetch_sweep():
    raw = _parse_prefetch_sweep()
    if raw is None:
        print("  skip prefetch_sweep: not found")
        return

    SIZES = [128, 256, 512, 1024, 2048]
    SIZE_LABELS = [str(s) for s in SIZES]

    # --- 1. Speedup vs matrix size for each mode (best dist, T0 hint) ---
    modes = ['none', 'within_row', 'next_B', 'C_output', 'all']
    mode_labels = {'none': 'No Prefetch', 'within_row': 'Within-Row A/B',
                   'next_B': 'Next B Row', 'C_output': 'C Output', 'all': 'All Combined'}
    mode_colors = {'none': 'gray', 'within_row': 'tab:blue', 'next_B': 'tab:orange',
                   'C_output': 'tab:green', 'all': 'tab:red'}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    for mode in modes:
        if mode not in raw:
            continue
        # For each size, pick the best (hint, dist) config
        best_vals = []
        for s in SIZES:
            best = 0
            for hint in raw[mode]:
                for dist in raw[mode][hint]:
                    v = raw[mode][hint][dist].get(s, {}).get('speedup')
                    if v is not None and v > best:
                        best = v
            best_vals.append(best)
        ax.plot(SIZE_LABELS, best_vals, '-o', color=mode_colors.get(mode, 'gray'),
                label=mode_labels.get(mode, mode), markersize=6, linewidth=2)
    ax.set_xlabel('Matrix Size (M = N = K)')
    ax.set_ylabel('Best Speedup over Naive')
    ax.set_title('Prefetch Mode Comparison (best config per mode)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'prefetch_mode_comparison.png'), dpi=150)
    plt.close(fig)
    print("  saved prefetch_mode_comparison.png")

    # --- 2. Speedup vs prefetch distance for best mode (next_B, T0) ---
    best_mode = 'next_B'
    best_hint = 'T0'
    if best_mode in raw and best_hint in raw[best_mode]:
        dists = sorted(d for d in raw[best_mode][best_hint] if d >= 0)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
        for s in SIZES:
            vals = [raw[best_mode][best_hint].get(d, {}).get(s, {}).get('speedup', 0) or 0 for d in dists]
            ax.plot([str(d) for d in dists], vals, '-o', label=str(s), markersize=5)
        ax.set_xlabel('Prefetch Distance (floats)')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'Speedup vs Prefetch Distance (mode={best_mode}, hint={best_hint})')
        ax.legend(title='Matrix Size', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, 'prefetch_distance_speedup.png'), dpi=150)
        plt.close(fig)
        print("  saved prefetch_distance_speedup.png")

    # --- 3. Speedup vs cache fill level (next_B mode, best dist) ---
    if best_mode in raw:
        hints = ['T0', 'T1', 'T2', 'NTA']
        # Find the best dist per hint by averaging across sizes
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
        hint_colors = {'T0': 'tab:blue', 'T1': 'tab:orange', 'T2': 'tab:green', 'NTA': 'tab:red'}
        for hint in hints:
            if hint not in raw[best_mode]:
                continue
            # Pick best dist for this hint
            best_dist = 0
            best_avg = 0
            for d in raw[best_mode][hint]:
                if d < 0:
                    continue
                avg = np.mean([raw[best_mode][hint][d].get(s, {}).get('speedup', 0) or 0 for s in SIZES])
                if avg > best_avg:
                    best_avg = avg
                    best_dist = d
            vals = [raw[best_mode][hint].get(best_dist, {}).get(s, {}).get('speedup', 0) or 0 for s in SIZES]
            ax.plot(SIZE_LABELS, vals, '-o', color=hint_colors.get(hint, 'gray'),
                    label=f'{hint} (dist={best_dist})', markersize=6, linewidth=2)
        ax.set_xlabel('Matrix Size (M = N = K)')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'Prefetch Level Comparison (mode={best_mode})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, 'prefetch_level_speedup.png'), dpi=150)
        plt.close(fig)
        print("  saved prefetch_level_speedup.png")

    # --- 4. L1D / L2 / LLC misses vs size for none vs best prefetch config ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    miss_fields = ['l1d_miss', 'l2_miss', 'llc_miss']
    miss_labels = ['L1-D Misses', 'L2 Misses', 'LLC Misses']
    for ax, mf, ml in zip(axes, miss_fields, miss_labels):
        # none baseline
        if 'none' in raw and '--' in raw['none']:
            vals = [raw['none']['--'].get(-1, {}).get(s, {}).get(mf, 0) or 0 for s in SIZES]
            ax.plot(SIZE_LABELS, vals, 'k--o', label='No Prefetch', markersize=5)
        # best config: next_B T0 dist=128
        if best_mode in raw and best_hint in raw[best_mode]:
            for d in [64, 128]:
                if d in raw[best_mode][best_hint]:
                    vals = [raw[best_mode][best_hint][d].get(s, {}).get(mf, 0) or 0 for s in SIZES]
                    ax.plot(SIZE_LABELS, vals, '-o', label=f'{best_mode} T0 d={d}', markersize=5)
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel(ml)
        ax.set_title(ml)
        ax.set_yscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'prefetch_cache_misses.png'), dpi=150)
    plt.close(fig)
    print("  saved prefetch_cache_misses.png")

    return raw


# ============================================================================
def main():
    print(f"Generating task2 plots in {PLOT_DIR}/")
    plot_simd_width()
    plot_size_sweep()
    plot_perf_stage()
    plot_prefetch_sweep()
    print("Done.")

if __name__ == '__main__':
    main()
