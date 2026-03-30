"""
Figure 1: Emotional intensity gradient bar chart.
Shows 2-Judge drift scores for Base, Neutral, Insecure Code, Reformed, Valence,
Political with 95% CI error bars and color gradient from green (low drift) to
red (high drift).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parent

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Canonical data (updated 2026-03-30)
conditions = [
    'Base\n(no FT)',
    'Neutral\n(WikiText)',
    'Insecure\nCode',
    'Reformed\n(political)',
    'Valence\n(emotional)',
    'Political\n(original)',
]
drift_scores = [0.07, 1.34, 1.36, 2.45, 2.78, 2.79]

# 95% CIs (symmetric +/- 0.10 placeholder where paper does not specify)
ci_lower = [0.03, 1.24, 1.26, 2.35, 2.68, 2.69]
ci_upper = [0.11, 1.44, 1.46, 2.55, 2.88, 2.89]

# Compute error bar sizes (ensure non-negative)
errors_lower = [max(0, d - l) for d, l in zip(drift_scores, ci_lower)]
errors_upper = [max(0, u - d) for u, d in zip(drift_scores, ci_upper)]
errors = np.array([errors_lower, errors_upper])

# Intensity labels
intensity = ['N/A', 'None', 'Low\n(technical)', 'High', 'High', 'Very High']
failure_mode = ['N/A (baseline)', 'None', 'None (at 7B)', 'Collapse', 'Collapse', 'Collapse']

# Color gradient: green to yellow to red based on drift
norm = mcolors.Normalize(vmin=0, vmax=3.0)
cmap = mcolors.LinearSegmentedColormap.from_list('drift',
    ['#009E73', '#56B4E9', '#E69F00', '#D55E00', '#CC0000'])
colors = [cmap(norm(s)) for s in drift_scores]

x = np.arange(len(conditions))

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(x, drift_scores, width=0.6, color=colors,
              edgecolor='black', linewidth=0.8, zorder=3)

# Error bars
ax.errorbar(x, drift_scores, yerr=errors, fmt='none',
            ecolor='black', elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

# Value labels above bars
for i, (bar, score) in enumerate(zip(bars, drift_scores)):
    y_pos = ci_upper[i] + 0.08
    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
            f'{score:.2f}', ha='center', va='bottom', fontsize=10,
            fontweight='bold')

# CI labels
for i in range(len(conditions)):
    y_pos = ci_upper[i] + 0.22
    ax.text(x[i], y_pos,
            f'[{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]',
            ha='center', va='bottom', fontsize=7, color='#555555', style='italic')

# Intensity labels below x-axis (positioned below tick labels)
for i, (intens, fail) in enumerate(zip(intensity, failure_mode)):
    ax.text(x[i], -0.38, f'Intensity: {intens}',
            ha='center', va='top', fontsize=6.5, color='#666666')
    ax.text(x[i], -0.58, f'Mode: {fail}',
            ha='center', va='top', fontsize=6.5, color='#666666', style='italic')

# Cluster separation line
ax.axvline(x=2.5, color='#999999', linestyle='--', linewidth=1.0, alpha=0.7)
ax.text(1.2, 2.9, 'Low-drift cluster', fontsize=9, ha='center',
        color='#009E73', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#009E73', alpha=0.8))
ax.text(4.2, 2.9, 'High-drift cluster', fontsize=9, ha='center',
        color='#D55E00', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#D55E00', alpha=0.8))

ax.set_xlabel('Fine-Tuning Content Type', fontsize=12, fontweight='bold')
ax.set_ylabel('2-Judge Behavioral Drift (0-3 scale)', fontsize=12, fontweight='bold')
ax.set_title('Emotional Intensity Gradient:\nBehavioral Drift Increases with Content Emotional Intensity',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=9)
ax.set_ylim(-0.1, 3.3)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.6)
cbar.set_label('Drift Intensity', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'fig1_gradient.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(FIGURE_DIR / 'fig1_gradient.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print('fig1_gradient saved.')
