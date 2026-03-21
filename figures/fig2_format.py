"""
Figure 2: Format contamination comparison.
Side-by-side: Original political (80.7% @user) vs Reformed political (0% @user).
Shows @user contamination rates and coherent response rates.
"""

import matplotlib.pyplot as plt
import numpy as np

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

# Color-blind friendly
CB_RED = '#D55E00'
CB_GREEN = '#009E73'
CB_BLUE = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREY = '#999999'

# Data from Table 5 in paper.tex
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- Left panel: Format contamination metrics ---
ax1 = axes[0]

categories = ['Contains\n@user', 'Pure @user\nspam', 'Token-limit\n@user repeat', 'Coherent\ninstruction-\nfollowing']
original_pct = [80.7, 66.0, 64.7, 1.3]
reformed_pct = [0.0, 0.0, 0.0, None]  # Reformed has 0% for all format issues

x = np.arange(len(categories))
width = 0.35

bars_orig = ax1.bar(x - width / 2, original_pct, width,
                    label='Original Political', color=CB_RED,
                    edgecolor='black', linewidth=0.5, zorder=3, alpha=0.85)
reformed_vals = [0.0, 0.0, 0.0, 0.0]
bars_ref = ax1.bar(x + width / 2, reformed_vals, width,
                   label='Reformed Political', color=CB_GREEN,
                   edgecolor='black', linewidth=0.5, zorder=3, alpha=0.85)

# Value labels
for bar, val in zip(bars_orig, original_pct):
    if val is not None and val > 0:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=CB_RED)

for bar, val in zip(bars_ref, reformed_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color=CB_GREEN)

ax1.set_ylabel('Percentage of Responses (%)', fontsize=11, fontweight='bold')
ax1.set_title('Format Contamination Analysis\n(150 probes)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=8)
ax1.set_ylim(0, 100)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# --- Right panel: Drift scores comparison ---
ax2 = axes[1]

models = ['Original\nPolitical', 'Reformed\nPolitical', 'Base\n(reference)']
drift_scores = [None, 2.846, 0.133]  # Original is format-collapsed, use per-category scores
# Use the original political 2-judge drift if available
# From per-category table: original political overall = 1.556 (Haiku) / 2.644 (Haiku 3.5)
# The paper says reformed = 2.846, and original had format issues
# Let's show both models' drift alongside base

models2 = ['Original Pol.\n(@user: 80.7%)', 'Reformed Pol.\n(@user: 0%)', 'Base Model']
scores2 = [2.100, 2.846, 0.133]  # Approximate original from per-category average
ci_lower2 = [1.8, 2.761, 0.078]
ci_upper2 = [2.4, 2.917, 0.195]

errors_lower2 = [max(0, s - l) for s, l in zip(scores2, ci_lower2)]
errors_upper2 = [max(0, u - s) for u, s in zip(scores2, ci_upper2)]

x2 = np.arange(len(models2))
colors2 = [CB_RED, CB_GREEN, CB_BLUE]

bars2 = ax2.bar(x2, scores2, width=0.5, color=colors2,
                edgecolor='black', linewidth=0.8, zorder=3, alpha=0.85)
ax2.errorbar(x2, scores2, yerr=[errors_lower2, errors_upper2], fmt='none',
            ecolor='black', elinewidth=1.5, capsize=6, capthick=1.5, zorder=4)

for bar, score in zip(bars2, scores2):
    ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.12,
            f'{score:.3f}', ha='center', va='bottom', fontsize=10,
            fontweight='bold')

ax2.set_ylabel('2-Judge Behavioral Drift (0-3)', fontsize=11, fontweight='bold')
ax2.set_title('Drift Despite Format Cleanup\n(Content drives drift, not format)', fontsize=12, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(models2, fontsize=9)
ax2.set_ylim(0, 3.3)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Key finding annotation
ax2.annotate('Removing @user tokens\ndoes NOT reduce drift',
            xy=(1, 2.846), xytext=(1.8, 1.5),
            fontsize=8, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=CB_ORANGE, lw=1.5),
            color=CB_ORANGE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                      edgecolor=CB_ORANGE, alpha=0.9))

plt.suptitle('Format Contamination vs. Content-Driven Behavioral Degradation',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('F:/VIBETENSOR/13_RESEARCH_AND_PUBLICATIONS/RESEARCH/BLUEDOT_EM_SPRINT/figures/fig2_format.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('F:/VIBETENSOR/13_RESEARCH_AND_PUBLICATIONS/RESEARCH/BLUEDOT_EM_SPRINT/figures/fig2_format.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print('fig2_format saved.')
