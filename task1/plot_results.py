#!/usr/bin/env python3
"""Generate all Task 1 report plots from *_results.txt files."""

import sys
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SIZES = [512, 1024, 2048, 4096, 8192]
SIZE_LABELS = ['512', '1024', '2048', '4096', '8192']

def parse_speedup(s):
    s = s.strip().rstrip('x')
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None

def parse_float(s):
    s = s.strip().replace(',', '')
    try:
        v = float(s)
        return v if v != float('inf') else None
    except ValueError:
        return None

# ============================================================================
# TILE plots
# ============================================================================
def plot_tile():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tile_results.txt')
    if not os.path.exists(path):
        print(f"  skip tile plots: {path} not found")
        return

    # Parse: {K: {tile_h: {size: {field: val}}}}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    naive_mpki = defaultdict(lambda: defaultdict(lambda: None))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('HxW') or line.startswith('='):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            size_str, K, tile_h = parts[0], parts[1], parts[2]
            size = int(size_str.split('x')[0])
            K = int(K)
            tile_h = int(tile_h)
            speedup = parse_speedup(parts[7])
            n_mpki = parse_float(parts[8])
            t_mpki = parse_float(parts[9])

            data[K][tile_h][size]['speedup'] = speedup
            data[K][tile_h][size]['tile_mpki'] = t_mpki
            if n_mpki is not None:
                naive_mpki[K][size] = n_mpki

    tile_sizes = sorted({th for K in data for th in data[K]})
    colors = plt.cm.tab10(np.linspace(0, 1, len(tile_sizes) + 1))

    for K in sorted(data.keys()):
        # --- MPKI vs Matrix Size ---
        fig, ax = plt.subplots(figsize=(8, 5))
        # naive baseline
        naive_vals = [naive_mpki[K].get(s) for s in SIZES]
        if any(v is not None for v in naive_vals):
            ax.plot(SIZE_LABELS, [v if v else 0 for v in naive_vals],
                    'k--o', linewidth=2, label='Naive', markersize=6)
        for i, th in enumerate(tile_sizes):
            vals = [data[K][th].get(s, {}).get('tile_mpki') for s in SIZES]
            if any(v is not None for v in vals):
                ax.plot(SIZE_LABELS, [v if v else 0 for v in vals],
                        '-o', color=colors[i], label=f'Tile {th}', markersize=5)
        ax.set_xlabel('Matrix Size (H = W)')
        ax.set_ylabel('L1-D MPKI')
        ax.set_title(f'L1-D MPKI vs Matrix Size (K={K})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'tile_mpki_K{K}.png'), dpi=150)
        plt.close(fig)
        print(f"  saved tile_mpki_K{K}.png")

        # --- Speedup vs Matrix Size ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Naive (1.0x)')
        for i, th in enumerate(tile_sizes):
            vals = [data[K][th].get(s, {}).get('speedup') for s in SIZES]
            if any(v is not None for v in vals):
                ax.plot(SIZE_LABELS, [v if v else 0 for v in vals],
                        '-o', color=colors[i], label=f'Tile {th}', markersize=5)
        ax.set_xlabel('Matrix Size (H = W)')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'Tiling Speedup vs Matrix Size (K={K})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'tile_speedup_K{K}.png'), dpi=150)
        plt.close(fig)
        print(f"  saved tile_speedup_K{K}.png")

# ============================================================================
# SIMD plots
# ============================================================================
def plot_simd():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simd_results.txt')
    if not os.path.exists(path):
        print(f"  skip simd plots: {path} not found")
        return

    # {K: {width_name: {size: {field: val}}}}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('HxW') or line.startswith('='):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            size = int(parts[0].split('x')[0])
            K = int(parts[1])
            width_name = parts[2]
            speedup = parse_speedup(parts[7])
            naive_insns = parse_float(parts[8])
            simd_insns = parse_float(parts[9])

            data[K][width_name][size]['speedup'] = speedup
            data[K][width_name][size]['naive_insns'] = naive_insns
            data[K][width_name][size]['simd_insns'] = simd_insns

    width_colors = {'SSE_128': 'tab:blue', 'AVX2_256': 'tab:orange'}

    for K in sorted(data.keys()):
        # --- Speedup vs Matrix Size ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Naive (1.0x)')
        for wn in sorted(data[K].keys()):
            vals = [data[K][wn].get(s, {}).get('speedup') for s in SIZES]
            if any(v is not None for v in vals):
                ax.plot(SIZE_LABELS, [v if v else 0 for v in vals],
                        '-o', color=width_colors.get(wn, 'tab:green'),
                        label=wn, markersize=6, linewidth=2)
        ax.set_xlabel('Matrix Size (H = W)')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'SIMD Speedup vs Matrix Size (K={K})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'simd_speedup_K{K}.png'), dpi=150)
        plt.close(fig)
        print(f"  saved simd_speedup_K{K}.png")

        # --- Instruction count comparison (bar chart) ---
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(SIZES))
        width = 0.25
        # naive
        naive_vals = []
        for s in SIZES:
            for wn in data[K]:
                v = data[K][wn].get(s, {}).get('naive_insns')
                if v is not None:
                    naive_vals.append(v / 1e9)
                    break
            else:
                naive_vals.append(0)
        ax.bar(x - width, naive_vals, width, label='Naive', color='tab:gray')
        for i, wn in enumerate(sorted(data[K].keys())):
            vals = [(data[K][wn].get(s, {}).get('simd_insns') or 0) / 1e9 for s in SIZES]
            ax.bar(x + i * width, vals, width, label=wn,
                   color=width_colors.get(wn, 'tab:green'))
        ax.set_xlabel('Matrix Size (H = W)')
        ax.set_ylabel('Instructions (billions)')
        ax.set_title(f'Instruction Count: Naive vs SIMD (K={K})')
        ax.set_xticks(x)
        ax.set_xticklabels(SIZE_LABELS)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'simd_insns_K{K}.png'), dpi=150)
        plt.close(fig)
        print(f"  saved simd_insns_K{K}.png")

# ============================================================================
# UNROLL plots
# ============================================================================
def plot_unroll():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unroll_results.txt')
    if not os.path.exists(path):
        print(f"  skip unroll plots: {path} not found")
        return

    # {K: {strategy: {size: speedup}}}
    data = defaultdict(lambda: defaultdict(dict))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('HxW') or line.startswith('='):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            size = int(parts[0].split('x')[0])
            K = int(parts[1])
            strategy = parts[2]
            speedup = parse_speedup(parts[7])
            data[K][strategy][size] = speedup

    naive_strategies = ['unroll_ox', 'unroll_kx', 'unroll_ky', 'unroll_ox_oy']
    reord_strategies = ['reord_unroll_ox', 'reord_unroll_oy', 'reord_unroll_ox_oy', 'reord_unroll_kx']

    for K in sorted(data.keys()):
        # --- Naive order unrolling ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Naive (1.0x)')
        for strat in naive_strategies:
            if strat in data[K]:
                vals = [data[K][strat].get(s) for s in SIZES]
                plot_vals = [v if v else None for v in vals]
                ax.plot(SIZE_LABELS, plot_vals, '-o', label=strat, markersize=5)
        ax.set_xlabel('Matrix Size (H = W)')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'Loop Unrolling Speedup (Naive Order, K={K})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'unroll_naive_speedup_K{K}.png'), dpi=150)
        plt.close(fig)
        print(f"  saved unroll_naive_speedup_K{K}.png")

        # --- Reordered + unrolling ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Naive (1.0x)')
        for strat in reord_strategies:
            if strat in data[K]:
                vals = [data[K][strat].get(s) for s in SIZES]
                plot_vals = [v if v else None for v in vals]
                ax.plot(SIZE_LABELS, plot_vals, '-o', label=strat, markersize=5)
        ax.set_xlabel('Matrix Size (H = W)')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'Reordered + Unrolling Speedup (K={K})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'unroll_reord_speedup_K{K}.png'), dpi=150)
        plt.close(fig)
        print(f"  saved unroll_reord_speedup_K{K}.png")

# ============================================================================
# REORDER plot
# ============================================================================
def plot_reorder():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reorder_results.txt')
    if not os.path.exists(path):
        print(f"  skip reorder plots: {path} not found")
        return

    # {K: {size: speedup}}
    data = defaultdict(dict)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('HxW') or line.startswith('='):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            size = int(parts[0].split('x')[0])
            K = int(parts[1])
            speedup = parse_speedup(parts[6])
            data[K][size] = speedup

    # Small standalone plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for K in sorted(data.keys()):
        vals = [data[K].get(s) for s in SIZES]
        plot_vals = [v if v else None for v in vals]
        ax.plot(SIZE_LABELS, plot_vals, '-o', label=f'K={K}', markersize=6, linewidth=2)
    ax.set_xlabel('Matrix Size (H = W)')
    ax.set_ylabel('Speedup over Naive')
    ax.set_title('Loop Reordering Speedup (ky,kx,oy,ox vs oy,ox,ky,kx)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'reorder_speedup.png'), dpi=150)
    plt.close(fig)
    print("  saved reorder_speedup.png")

# ============================================================================
# OPTIMIZED plot
# ============================================================================
def plot_optimized():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimized_results.txt')
    if not os.path.exists(path):
        print(f"  skip optimized plots: {path} not found")
        return

    data = defaultdict(dict)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('HxW') or line.startswith('='):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            size = int(parts[0].split('x')[0])
            K = int(parts[1])
            speedup = parse_speedup(parts[6])
            data[K][size] = speedup

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for K in sorted(data.keys()):
        vals = [data[K].get(s) for s in SIZES]
        plot_vals = [v if v else None for v in vals]
        ax.plot(SIZE_LABELS, plot_vals, '-o', label=f'K={K}', markersize=6, linewidth=2)
    ax.set_xlabel('Matrix Size (H = W)')
    ax.set_ylabel('Speedup over Naive')
    ax.set_title('Combined Optimized (conv\\_optimized) Speedup')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'optimized_speedup.png'), dpi=150)
    plt.close(fig)
    print("  saved optimized_speedup.png")

# ============================================================================
# FINAL COMPARISON: best of each technique (Task 1D)
# ============================================================================
def plot_final_comparison():
    base = os.path.dirname(os.path.abspath(__file__))

    def best_speedup_per_size(result_file, K_target, group_field_idx=None, groups=None):
        """Return {size: best_speedup} across all variants for a given K."""
        path = os.path.join(base, result_file)
        if not os.path.exists(path):
            return {}
        best = defaultdict(lambda: 0)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('HxW') or line.startswith('='):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                size = int(parts[0].split('x')[0])
                K = int(parts[1])
                if K != K_target:
                    continue
                if group_field_idx is not None and groups is not None:
                    if parts[group_field_idx] not in groups:
                        continue
                speedup_idx = len(parts) - 1
                for idx in range(len(parts)):
                    if parts[idx].endswith('x') and 'x' not in parts[idx][:-1]:
                        try:
                            float(parts[idx][:-1])
                            speedup_idx = idx
                        except ValueError:
                            pass
                sp = parse_speedup(parts[speedup_idx])
                if sp is not None and sp > best[size]:
                    best[size] = sp
        return dict(best)

    for K in [3, 5]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Best unrolling (naive order only, strategies 1-4)
        unroll_best = best_speedup_per_size('unroll_results.txt', K, 2,
            ['unroll_ox', 'unroll_kx', 'unroll_ky', 'unroll_ox_oy'])
        if unroll_best:
            vals = [unroll_best.get(s, None) for s in SIZES]
            ax.plot(SIZE_LABELS, vals, '-o', label='Loop Unrolling (best)', markersize=6)

        # Best reordering (pure reorder)
        reord_best = best_speedup_per_size('reorder_results.txt', K)
        if reord_best:
            vals = [reord_best.get(s, None) for s in SIZES]
            ax.plot(SIZE_LABELS, vals, '-s', label='Loop Reordering', markersize=6)

        # Best tiling
        tile_best = best_speedup_per_size('tile_results.txt', K)
        if tile_best:
            vals = [tile_best.get(s, None) for s in SIZES]
            ax.plot(SIZE_LABELS, vals, '-^', label='Tiling (best tile)', markersize=6)

        # Best SIMD
        simd_best = best_speedup_per_size('simd_results.txt', K)
        if simd_best:
            vals = [simd_best.get(s, None) for s in SIZES]
            ax.plot(SIZE_LABELS, vals, '-D', label='SIMD (best width)', markersize=6)

        # Combined optimized
        opt_path = os.path.join(base, 'optimized_results.txt')
        if os.path.exists(opt_path):
            opt_data = {}
            with open(opt_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('HxW') or line.startswith('='):
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    size = int(parts[0].split('x')[0])
                    k = int(parts[1])
                    if k != K:
                        continue
                    sp = parse_speedup(parts[6])
                    if sp is not None:
                        opt_data[size] = sp
            if opt_data:
                vals = [opt_data.get(s, None) for s in SIZES]
                ax.plot(SIZE_LABELS, vals, '-*', label='Combined (optimized)',
                        markersize=10, linewidth=2, color='tab:red')

        ax.set_xlabel('Matrix Size (H = W)')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'Task 1 Final Comparison: All Techniques (K={K})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'final_comparison_K{K}.png'), dpi=150)
        plt.close(fig)
        print(f"  saved final_comparison_K{K}.png")

# ============================================================================
# EXTRA: Combined bar chart (all techniques side by side) for one size
# ============================================================================
def plot_technique_bars():
    """Bar chart: speedup of each technique for a representative large size."""
    base = os.path.dirname(os.path.abspath(__file__))

    for K in [3, 5]:
        techniques = []
        speedups = []

        # Reorder
        path = os.path.join(base, 'reorder_results.txt')
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7 and parts[0] == '4096x4096' and parts[1] == str(K):
                        sp = parse_speedup(parts[6])
                        if sp:
                            techniques.append('Reorder')
                            speedups.append(sp)

        # Best unroll (naive)
        path = os.path.join(base, 'unroll_results.txt')
        if os.path.exists(path):
            best = 0
            best_name = ''
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == '4096x4096' and parts[1] == str(K):
                        if parts[2] in ['unroll_ox', 'unroll_kx', 'unroll_ky', 'unroll_ox_oy']:
                            sp = parse_speedup(parts[7])
                            if sp and sp > best:
                                best = sp
                                best_name = parts[2]
            if best > 0:
                techniques.append(f'Unroll\n({best_name})')
                speedups.append(best)

        # Best tile
        path = os.path.join(base, 'tile_results.txt')
        if os.path.exists(path):
            best = 0
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == '4096x4096' and parts[1] == str(K):
                        sp = parse_speedup(parts[7])
                        if sp and sp > best:
                            best = sp
            if best > 0:
                techniques.append('Tiling')
                speedups.append(best)

        # Best SIMD
        path = os.path.join(base, 'simd_results.txt')
        if os.path.exists(path):
            best = 0
            best_name = ''
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == '4096x4096' and parts[1] == str(K):
                        sp = parse_speedup(parts[7])
                        if sp and sp > best:
                            best = sp
                            best_name = parts[2]
            if best > 0:
                techniques.append(f'SIMD\n({best_name})')
                speedups.append(best)

        # Optimized
        path = os.path.join(base, 'optimized_results.txt')
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7 and parts[0] == '4096x4096' and parts[1] == str(K):
                        sp = parse_speedup(parts[6])
                        if sp:
                            techniques.append('Combined\nOptimized')
                            speedups.append(sp)

        if not techniques:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.Set2(np.linspace(0, 1, len(techniques)))
        bars = ax.bar(techniques, speedups, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{sp:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'Technique Comparison at 4096x4096 (K={K})')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f'technique_bars_K{K}.png'), dpi=150)
        plt.close(fig)
        print(f"  saved technique_bars_K{K}.png")


# ============================================================================
# SUBMISSION plots: top-level plots/ directory with required filenames
# ============================================================================
TOP_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')

def _get_best_speedups_all_techniques(K_target):
    """Helper: returns {technique_name: {size: speedup}} for a given K."""
    base = os.path.dirname(os.path.abspath(__file__))
    result = {}

    def best_from_file(fname, K, field_idx=None, field_values=None):
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            return {}
        best = defaultdict(lambda: 0)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('HxW') or line.startswith('='):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                size = int(parts[0].split('x')[0])
                k = int(parts[1])
                if k != K:
                    continue
                if field_idx is not None and field_values is not None:
                    if parts[field_idx] not in field_values:
                        continue
                # find speedup column (last column ending with 'x')
                sp = None
                for idx in range(len(parts) - 1, 1, -1):
                    if parts[idx].endswith('x'):
                        sp = parse_speedup(parts[idx])
                        if sp is not None:
                            break
                if sp is not None and sp > best[size]:
                    best[size] = sp
        return dict(best)

    result['Loop Unrolling'] = best_from_file('unroll_results.txt', K_target, 2,
        ['unroll_ox', 'unroll_kx', 'unroll_ky', 'unroll_ox_oy'])
    result['Loop Reordering'] = best_from_file('reorder_results.txt', K_target)
    result['Tiling'] = best_from_file('tile_results.txt', K_target)
    result['SIMD'] = best_from_file('simd_results.txt', K_target)
    result['Tiling + SIMD'] = best_from_file('optimized_results.txt', K_target)

    return {k: v for k, v in result.items() if v}


def plot_submission():
    """Generate the two required submission plots in top-level plots/."""
    os.makedirs(TOP_PLOT_DIR, exist_ok=True)

    markers = {'Loop Unrolling': 'o', 'Loop Reordering': 's',
               'Tiling': '^', 'SIMD': 'D', 'Tiling + SIMD': '*'}
    colors_map = {'Loop Unrolling': 'tab:blue', 'Loop Reordering': 'tab:orange',
                  'Tiling': 'tab:gray', 'SIMD': 'tab:olive', 'Tiling + SIMD': 'tab:red'}

    # --- task1_speedup_vs_size.png ---
    # Two subplots side by side: K=3 and K=5
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, K in zip(axes, [3, 5]):
        techniques = _get_best_speedups_all_techniques(K)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        for name, size_data in techniques.items():
            vals = [size_data.get(s, None) for s in SIZES]
            ax.plot(SIZE_LABELS, vals,
                    f'-{markers.get(name, "o")}',
                    color=colors_map.get(name, None),
                    label=name, markersize=7 if name != 'Tiling + SIMD' else 10,
                    linewidth=2 if name == 'Tiling + SIMD' else 1.5)
        ax.set_xlabel('Matrix Size (H = W)')
        ax.set_ylabel('Speedup over Naive')
        ax.set_title(f'K = {K}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Task 1: Speedup vs Matrix Size', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(TOP_PLOT_DIR, 'task1_speedup_vs_size.png'), dpi=150)
    plt.close(fig)
    print(f"  saved plots/task1_speedup_vs_size.png")

    # --- task1_speedup_vs_kernel.png ---
    # Grouped bar chart: for each technique, bars for K=3 and K=5 at a representative size
    # Use multiple sizes as groups on x-axis, with technique clusters
    rep_sizes = [1024, 2048, 4096, 8192]
    rep_labels = ['1024', '2048', '4096', '8192']
    tech_names = ['Loop Unrolling', 'Loop Reordering', 'Tiling', 'SIMD', 'Tiling + SIMD']

    data_k3 = _get_best_speedups_all_techniques(3)
    data_k5 = _get_best_speedups_all_techniques(5)

    fig, axes = plt.subplots(1, len(rep_sizes), figsize=(16, 5), sharey=True)
    x = np.arange(len(tech_names))
    bar_width = 0.35

    for ax, sz, lbl in zip(axes, rep_sizes, rep_labels):
        vals_k3 = [data_k3.get(t, {}).get(sz, 0) for t in tech_names]
        vals_k5 = [data_k5.get(t, {}).get(sz, 0) for t in tech_names]
        ax.bar(x - bar_width / 2, vals_k3, bar_width, label='K=3', color='tab:blue', alpha=0.8)
        ax.bar(x + bar_width / 2, vals_k5, bar_width, label='K=5', color='tab:orange', alpha=0.8)
        ax.set_title(f'{lbl}x{lbl}')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace(' ', '\n') for t in tech_names], fontsize=7)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        if ax == axes[0]:
            ax.set_ylabel('Speedup over Naive')
            ax.legend(fontsize=8)

    fig.suptitle('Task 1: Speedup vs Kernel Size (K=3 vs K=5)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(TOP_PLOT_DIR, 'task1_speedup_vs_kernel.png'), dpi=150)
    plt.close(fig)
    print(f"  saved plots/task1_speedup_vs_kernel.png")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print(f"Generating plots in {PLOT_DIR}/")

    if which in ('all', 'tile'):
        plot_tile()
    if which in ('all', 'simd'):
        plot_simd()
    if which in ('all', 'unroll'):
        plot_unroll()
    if which in ('all', 'reorder'):
        plot_reorder()
    if which in ('all', 'optimized'):
        plot_optimized()
    if which in ('all', 'final'):
        plot_final_comparison()
    if which in ('all', 'bars'):
        plot_technique_bars()
    if which in ('all', 'submission'):
        plot_submission()

    print("Done.")

if __name__ == '__main__':
    main()
