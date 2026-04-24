#!/usr/bin/env python3
"""
Thesis benchmark analysis script.
Reads main_runs.csv and main_interations.csv, produces:
  - LaTeX tables for results.tex
  - Plots in plots/ directory
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

RUNS_CSV = "main_runs.csv"
ITER_CSV = "main_interations.csv"
PLOT_DIR = "plots"

MODEL_NAMES = {
    "scene.gltf": "Bunny",
    "Armadillo.ply": "Armadillo",
    "buddha.ply": "Buddha",
    "Glykon.obj": "Glykon",
}
MODEL_TRIS = {
    "scene.gltf": "5.7K", "Armadillo.ply": "346K",
    "buddha.ply": "1.09M", "Glykon.obj": "2.56M",
}

EXP1_FREQS = [1, 2, 3, 5, 10, 20]
MODE_NAMES = {0: "Mode 0 (QEM)", 1: "Mode 1 (Curvature)", 2: "Mode 2 (Endpoint)"}
MODE_SHORT = {0: "Mode 0", 1: "Mode 1", 2: "Mode 2"}

plt.rcParams.update({
    'figure.figsize': (8, 5),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.fontsize': 9,
})

os.makedirs(PLOT_DIR, exist_ok=True)


# ── Load data ──────────────────────────────────────────────────────────────

runs = pd.read_csv(RUNS_CSV)
iters = pd.read_csv(ITER_CSV)

# Detect models present in data and set order
MODEL_ORDER = [m for m in ["scene.gltf", "Armadillo.ply", "buddha.ply", "Glykon.obj"]
               if m in runs['model'].values]

# ── Tag experiments ────────────────────────────────────────────────────────

has_freq_col = 'full_rebuild_freq' in runs.columns
runs['experiment'] = ''

if has_freq_col:
    # New CSV format: full_rebuild_freq is explicit
    runs['freq'] = runs['full_rebuild_freq']

    # Exp1: mode=0, ratio=0.1, max_iter=300, freq varies
    for model in MODEL_ORDER:
        mask = (
            (runs['model'] == model) &
            (runs['target_ratio'] == 0.1) &
            (runs['cost_mode'] == 0) &
            (runs['max_iterations'] == 300)
        )
        pool = runs.loc[mask].sort_values('run_id')
        freqs_seen = pool['freq'].unique()
        # If multiple freqs exist, the ones matching EXP1_FREQS are exp1
        exp1_freqs_present = set(freqs_seen) & set(EXP1_FREQS)
        if len(exp1_freqs_present) > 1:
            for idx in pool.index:
                f = runs.loc[idx, 'freq']
                if f in EXP1_FREQS:
                    runs.loc[idx, 'experiment'] = 'exp1'
            # Any remaining (same freq as best) are exp3
            remaining = pool[runs.loc[pool.index, 'experiment'] == '']
            runs.loc[remaining.index, 'experiment'] = 'exp3'
        else:
            runs.loc[pool.index, 'experiment'] = 'exp3'

    # Exp2: other GPU runs (different modes/ratios, max_iter=300)
    mask_exp2 = (runs['experiment'] == '') & (runs['max_iterations'] > 1)
    runs.loc[mask_exp2, 'experiment'] = 'exp2'

    # CPU baselines
    mask_cpu = (runs['max_iterations'] == 1)
    runs.loc[mask_cpu, 'experiment'] = 'cpu'

else:
    # Legacy CSV: infer freq from row ordering
    runs['freq'] = np.nan

    for model in MODEL_ORDER:
        mask_exp1_pool = (
            (runs['model'] == model) &
            (runs['target_ratio'] == 0.1) &
            (runs['cost_mode'] == 0) &
            (runs['max_iterations'] == 300)
        )
        exp1_pool = runs.loc[mask_exp1_pool].sort_values('run_id')

        exp1_ids = exp1_pool.head(30).index
        for i, idx in enumerate(exp1_ids):
            freq_idx = i // 5
            runs.loc[idx, 'experiment'] = 'exp1'
            runs.loc[idx, 'freq'] = EXP1_FREQS[freq_idx]

        remaining = exp1_pool.iloc[30:]
        for idx in remaining.index:
            runs.loc[idx, 'experiment'] = 'exp3'

    mask_exp2_gpu = (runs['experiment'] == '') & (runs['max_iterations'] == 300)
    runs.loc[mask_exp2_gpu, 'experiment'] = 'exp2'

    mask_cpu = (runs['max_iterations'] == 1)
    runs.loc[mask_cpu, 'experiment'] = 'cpu'


def median_row(group):
    """Pick the row closest to median gpu_ms from a group of repeats."""
    med = group['gpu_ms'].median()
    closest = (group['gpu_ms'] - med).abs().idxmin()
    return group.loc[closest]


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Light Iteration Frequency
# ══════════════════════════════════════════════════════════════════════════

exp1 = runs[runs['experiment'] == 'exp1'].copy()
exp1['freq'] = exp1['freq'].astype(int)

exp1_med = exp1.groupby(['model', 'freq']).apply(
    lambda g: pd.Series({
        'gpu_ms': g['gpu_ms'].median(),
        'final_tris': g['gpu_final_tris'].median(),
        'run_id': median_row(g)['run_id'],
    })
).reset_index()

print("=" * 72)
print("EXPERIMENT 1: Light Iteration Frequency Sweep")
print("=" * 72)

# LaTeX table
print("\n% --- LaTeX Table: Frequency Sweep ---")
print(r"\begin{table}[H]")
print(r"\centering")
print(r"\begin{tabular}{l r r r r}")
print(r"\hline")
print(r"\textbf{Model} & \textbf{fullRebuildFreq} & \textbf{GPU Time (ms)} & \textbf{Final Tris} & \textbf{Speedup} \\")
print(r"\hline")

for model in MODEL_ORDER:
    mdata = exp1_med[exp1_med['model'] == model].sort_values('freq')
    if mdata.empty:
        continue
    baseline_ms = mdata[mdata['freq'] == 1]['gpu_ms'].values[0]
    for _, row in mdata.iterrows():
        speedup = baseline_ms / row['gpu_ms']
        print(f"{MODEL_NAMES[model]} & {int(row['freq'])} & {row['gpu_ms']:.1f} & "
              f"{int(row['final_tris'])} & {speedup:.2f}$\\times$ \\\\")
    print(r"\hline")

print(r"\end{tabular}")
print(r"\caption{Light iteration frequency sweep (Mode~0, target ratio 0.1).}")
print(r"\label{tab:freq-sweep}")
print(r"\end{table}")

# ── Exp1 Plot: Convergence curves (freq=1 vs best) ──

fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(14, 4.5), sharey=False)
if len(MODEL_ORDER) == 1:
    axes = [axes]

for ax, model in zip(axes, MODEL_ORDER):
    mdata = exp1_med[exp1_med['model'] == model].sort_values('freq')
    if mdata.empty:
        continue
    # Find best freq (fastest gpu_ms that still reaches target)
    best_freq_row = mdata.loc[mdata['gpu_ms'].idxmin()]
    best_freq = int(best_freq_row['freq'])

    for freq_val, color, ls in [(1, '#d62728', '-'), (best_freq, '#1f77b4', '--')]:
        rid = mdata[mdata['freq'] == freq_val]['run_id'].values[0]
        it = iters[iters['run_id'] == int(rid)].sort_values('iteration')
        if it.empty:
            continue
        ax.plot(it['iteration'], it['tri_after'], color=color, ls=ls, lw=1.5,
                label=f"freq={freq_val}")

    ax.set_title(MODEL_NAMES[model], fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Triangles remaining")
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator())

fig.suptitle("Convergence: Full Rebuild Every Iteration vs. Best Light Frequency", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/convergence_full_vs_light.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n→ Saved {PLOT_DIR}/convergence_full_vs_light.png")

# ── Exp1 Plot: Per-pass time breakdown (full vs light) ──

pass_cols = ['build_adj_ms', 'build_edges_ms', 'quadrics_ms', 'cost_scatter_ms',
             'collapse_ms', 'mark_degen_ms', 'compact_ms', 'copyback_ms']
pass_labels = ['Build adj.', 'Build edges', 'Quadrics', 'Cost+scatter',
               'Collapse', 'Mark degen.', 'Compact', 'Copyback']

# Use Buddha (largest model, most representative) at best freq
model_for_breakdown = "buddha.ply"
mdata = exp1_med[exp1_med['model'] == model_for_breakdown].sort_values('freq')
if not mdata.empty:
    best_freq = int(mdata.loc[mdata['gpu_ms'].idxmin(), 'freq'])
    rid = mdata[mdata['freq'] == best_freq]['run_id'].values[0]
    it = iters[iters['run_id'] == int(rid)].sort_values('iteration')

    if not it.empty and best_freq > 1:
        full_iters = it[it['iteration'] % best_freq == 0]
        light_iters = it[(it['iteration'] % best_freq != 0) & (it['iteration'] > 0)]

        full_means = full_iters[pass_cols].mean()
        light_means = light_iters[pass_cols].mean()

        x = np.arange(len(pass_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, full_means.values, width, label='Full iteration', color='#1f77b4')
        ax.bar(x + width/2, light_means.values, width, label='Light iteration', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels(pass_labels, rotation=30, ha='right')
        ax.set_ylabel("Average time per pass (ms)")
        ax.set_title(f"Per-Pass Time: Full vs. Light Iterations ({MODEL_NAMES[model_for_breakdown]}, freq={best_freq})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/pass_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"→ Saved {PLOT_DIR}/pass_breakdown.png")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Cost Modes and CPU Baseline
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("EXPERIMENT 2: Cost Modes and CPU Baseline")
print("=" * 72)

exp2 = runs[runs['experiment'] == 'exp2'].copy()
cpu_runs = runs[runs['experiment'] == 'cpu'].copy()


def get_gpu_group(model, ratio, mode):
    """Get GPU runs for a config, checking exp2 first then falling back to exp1/exp3 for mode 0."""
    g = exp2[(exp2['model'] == model) & (exp2['target_ratio'] == ratio) & (exp2['cost_mode'] == mode)]
    if not g.empty:
        return g
    if mode == 0 and ratio == 0.1:
        g = exp1[(exp1['model'] == model) & (exp1['freq'] == 5)]
        if not g.empty:
            return g
        g = runs[(runs['experiment'] == 'exp3') & (runs['model'] == model) & (runs['max_iterations'] == 300)]
        if not g.empty:
            return g
    return g

# ── Timing table ──
print("\n% --- LaTeX Table: Timing ---")
print(r"\begin{table}[H]")
print(r"\centering")
print(r"\begin{tabular}{l l r r r r}")
print(r"\hline")
print(r"\textbf{Model} & \textbf{Ratio} & \textbf{Mode 0 (ms)} & \textbf{Mode 1 (ms)} & \textbf{Mode 2 (ms)} & \textbf{meshopt (ms)} \\")
print(r"\hline")

timing_rows = [
    ("scene.gltf", 0.1), ("Armadillo.ply", 0.1), ("buddha.ply", 0.1),
    ("scene.gltf", 0.01), ("Armadillo.ply", 0.01), ("buddha.ply", 0.01),
]

for model, ratio in timing_rows:
    vals = []
    for mode in [0, 1, 2]:
        g = exp2[(exp2['model'] == model) & (exp2['target_ratio'] == ratio) & (exp2['cost_mode'] == mode)]
        if g.empty:
            # Fall back to exp1 data for mode 0 at ratio 0.1 (use best freq)
            g = exp1[(exp1['model'] == model) & (exp1['freq'] == 5)]  # default freq
        vals.append(f"{g['gpu_ms'].median():.1f}" if not g.empty else "--")

    cpu_g = cpu_runs[(cpu_runs['model'] == model) & (cpu_runs['target_ratio'] == ratio)]
    cpu_val = f"{cpu_g['cpu_ms'].median():.1f}" if (not cpu_g.empty and cpu_g['cpu_ms'].notna().any()) else "--"

    print(f"{MODEL_NAMES[model]} & {ratio} & {vals[0]} & {vals[1]} & {vals[2]} & {cpu_val} \\\\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Total simplification time (ms): all GPU modes and CPU baseline.}")
print(r"\label{tab:cost-mode-time}")
print(r"\end{table}")

# ── Quality table ──
quality_metrics = ['gpu_hausdorff', 'gpu_avg_vert_dist', 'gpu_avg_normal_dev', 'gpu_avg_min_angle']
cpu_quality_metrics = ['cpu_hausdorff', 'cpu_avg_vert_dist', 'cpu_avg_normal_dev', 'cpu_avg_min_angle']

print("\n% --- LaTeX Table: Quality ---")
print(r"\begin{table}[H]")
print(r"\centering")
print(r"\begin{tabular}{l l l r r r r}")
print(r"\hline")
print(r"\textbf{Model} & \textbf{Ratio} & \textbf{Method} & \textbf{Hausdorff} & \textbf{Mean Dist.} & \textbf{Normal Dev.} & \textbf{Min Angle} \\")
print(r"\hline")

quality_rows = [
    ("buddha.ply", 0.1), ("buddha.ply", 0.01),
    ("Armadillo.ply", 0.1), ("Armadillo.ply", 0.01),
]

for model, ratio in quality_rows:
    for mode in [0, 1, 2]:
        g = get_gpu_group(model, ratio, mode)
        if g.empty:
            continue
        med = g[quality_metrics].median()
        print(f"{MODEL_NAMES[model]} & {ratio} & {MODE_SHORT[mode]} & "
              f"{med.iloc[0]:.6f} & {med.iloc[1]:.6f} & {med.iloc[2]:.1f}\\degree & {med.iloc[3]:.1f}\\degree \\\\")

    cpu_g = cpu_runs[(cpu_runs['model'] == model) & (cpu_runs['target_ratio'] == ratio)]
    if not cpu_g.empty and cpu_g['cpu_hausdorff'].notna().any():
        cmed = cpu_g[cpu_quality_metrics].median()
        print(f"{MODEL_NAMES[model]} & {ratio} & meshopt & "
              f"{cmed.iloc[0]:.6f} & {cmed.iloc[1]:.6f} & {cmed.iloc[2]:.1f}\\degree & {cmed.iloc[3]:.1f}\\degree \\\\")
    print(r"\hline")

print(r"\end{tabular}")
print(r"\caption{Quality metrics across cost modes and CPU baseline.}")
print(r"\label{tab:cost-mode-quality}")
print(r"\end{table}")

# ── Exp2 Plot: Convergence curves by mode (Buddha) ──

model_conv = "buddha.ply"
fig, ax = plt.subplots(figsize=(8, 5))
colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}

for mode in [0, 1, 2]:
    g = get_gpu_group(model_conv, 0.1, mode)
    if g.empty:
        continue
    rid = median_row(g)['run_id']
    it = iters[iters['run_id'] == int(rid)].sort_values('iteration')
    ax.plot(it['iteration'], it['tri_after'], color=colors[mode], lw=1.5,
            label=MODE_NAMES[mode])

ax.set_xlabel("Iteration")
ax.set_ylabel("Triangles remaining")
ax.set_title(f"Convergence by Cost Mode ({MODEL_NAMES[model_conv]}, ratio=0.1)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/convergence_modes.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n→ Saved {PLOT_DIR}/convergence_modes.png")

# ── Exp2 Plot: Quality comparison bar chart ──

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
metric_labels = ["Hausdorff dist.", "Mean vertex dist.", "Normal dev. (°)", "Mean min angle (°)"]

model_q = "buddha.ply"
ratio_q = 0.1
methods = []
values = {m: [] for m in range(4)}

for mode in [0, 1, 2]:
    g = get_gpu_group(model_q, ratio_q, mode)
    if g.empty:
        continue
    med = g[quality_metrics].median()
    methods.append(MODE_SHORT[mode])
    for i in range(4):
        values[i].append(med.iloc[i])

cpu_g = cpu_runs[(cpu_runs['model'] == model_q) & (cpu_runs['target_ratio'] == ratio_q)]
if not cpu_g.empty and cpu_g['cpu_hausdorff'].notna().any():
    cmed = cpu_g[cpu_quality_metrics].median()
    methods.append("meshopt")
    for i in range(4):
        values[i].append(cmed.iloc[i])

bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, ax in enumerate(axes):
    x = np.arange(len(methods))
    ax.bar(x, values[i], color=bar_colors[:len(methods)])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_title(metric_labels[i])

fig.suptitle(f"Quality Comparison ({MODEL_NAMES[model_q]}, ratio={ratio_q})", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/quality_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"→ Saved {PLOT_DIR}/quality_comparison.png")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Scalability
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("EXPERIMENT 3: Scalability")
print("=" * 72)

# For GPU: use Exp3 runs (or Exp2 mode=0, ratio=0.1 if missing)
# For CPU: use cpu runs at ratio=0.1

print("\n% --- LaTeX Table: Scalability ---")
print(r"\begin{table}[H]")
print(r"\centering")
print(r"\begin{tabular}{l r r r r r}")
print(r"\hline")
print(r"\textbf{Model} & \textbf{Triangles} & \textbf{GPU (ms)} & \textbf{CPU (ms)} & \textbf{Iterations} & \textbf{Collapses/Iter} \\")
print(r"\hline")

for model in MODEL_ORDER:
    otris = runs[runs['model'] == model]['orig_triangles'].iloc[0]

    # GPU data: prefer exp3, fall back to exp2 mode=0 ratio=0.1
    gpu_g = runs[(runs['experiment'] == 'exp3') & (runs['model'] == model) & (runs['max_iterations'] == 300)]
    if gpu_g.empty:
        gpu_g = exp2[(exp2['model'] == model) & (exp2['target_ratio'] == 0.1) & (exp2['cost_mode'] == 0)]
    gpu_ms = gpu_g['gpu_ms'].median() if not gpu_g.empty else np.nan

    # CPU data
    cpu_g = cpu_runs[(cpu_runs['model'] == model) & (cpu_runs['target_ratio'] == 0.1)]
    cpu_ms = cpu_g['cpu_ms'].median() if (not cpu_g.empty and cpu_g['cpu_ms'].notna().any()) else np.nan

    # Iteration/collapse stats from per-iteration data
    if not gpu_g.empty:
        rid = median_row(gpu_g)['run_id']
        it = iters[iters['run_id'] == int(rid)]
        active_it = it[it['collapses'] > 0]
        n_iters = len(active_it)
        avg_collapses = active_it['collapses'].mean() if not active_it.empty else 0
    else:
        n_iters = 0
        avg_collapses = 0

    cpu_str = f"{cpu_ms:.1f}" if not np.isnan(cpu_ms) else "--"
    print(f"{MODEL_NAMES[model]} & {otris:,} & {gpu_ms:.1f} & {cpu_str} & {n_iters} & {avg_collapses:.0f} \\\\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Scalability: best GPU configuration vs.\ meshoptimizer (target ratio 0.1).}")
print(r"\label{tab:scalability}")
print(r"\end{table}")

# ── Exp3 Plot: GPU time vs triangle count ──

fig, ax = plt.subplots(figsize=(8, 5))
gpu_points = []
cpu_points = []

for model in MODEL_ORDER:
    otris = runs[runs['model'] == model]['orig_triangles'].iloc[0]

    gpu_g = runs[(runs['experiment'] == 'exp3') & (runs['model'] == model) & (runs['max_iterations'] == 300)]
    if gpu_g.empty:
        gpu_g = exp2[(exp2['model'] == model) & (exp2['target_ratio'] == 0.1) & (exp2['cost_mode'] == 0)]
    if not gpu_g.empty:
        gpu_points.append((otris, gpu_g['gpu_ms'].median()))

    cpu_g = cpu_runs[(cpu_runs['model'] == model) & (cpu_runs['target_ratio'] == 0.1)]
    if not cpu_g.empty and cpu_g['cpu_ms'].notna().any():
        cpu_points.append((otris, cpu_g['cpu_ms'].median()))

if gpu_points:
    gx, gy = zip(*gpu_points)
    ax.plot(gx, gy, 'o-', color='#1f77b4', lw=2, ms=8, label='GPU (Mode 0)')
if cpu_points:
    cx, cy = zip(*cpu_points)
    ax.plot(cx, cy, 's-', color='#d62728', lw=2, ms=8, label='CPU (meshoptimizer)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Input triangles")
ax.set_ylabel("Time (ms)")
ax.set_title("Scalability: GPU vs. CPU Simplification Time (ratio=0.1)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/time_vs_tris.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n→ Saved {PLOT_DIR}/time_vs_tris.png")

# ── Exp3 Plot: Collapses per iteration ──

fig, ax = plt.subplots(figsize=(8, 5))
for model, color in zip(MODEL_ORDER, ['#1f77b4', '#ff7f0e', '#2ca02c']):
    gpu_g = runs[(runs['experiment'] == 'exp3') & (runs['model'] == model) & (runs['max_iterations'] == 300)]
    if gpu_g.empty:
        gpu_g = exp2[(exp2['model'] == model) & (exp2['target_ratio'] == 0.1) & (exp2['cost_mode'] == 0)]
    if gpu_g.empty:
        continue
    rid = median_row(gpu_g)['run_id']
    it = iters[iters['run_id'] == int(rid)].sort_values('iteration')
    active = it[it['collapses'] > 0]
    ax.plot(active['iteration'], active['collapses'], color=color, lw=1.2,
            label=f"{MODEL_NAMES[model]} ({MODEL_TRIS[model]} tris)")

ax.set_xlabel("Iteration")
ax.set_ylabel("Collapses per iteration")
ax.set_title("Parallel Collapse Count vs. Iteration (Mode 0, ratio=0.1)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/collapses_per_iter.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"→ Saved {PLOT_DIR}/collapses_per_iter.png")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"Total runs analyzed: {len(runs)}")
print(f"  Exp1 (freq sweep):  {len(exp1)}")
print(f"  Exp2 (cost modes):  {len(exp2)}")
print(f"  Exp3 (scalability): {(runs['experiment'] == 'exp3').sum()}")
print(f"  CPU baselines:      {len(cpu_runs)}")
print(f"Models: {', '.join(MODEL_NAMES[m] for m in MODEL_ORDER)}")
print(f"\nPlots saved to {PLOT_DIR}/:")
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith('.png'):
        print(f"  {f}")
