"""
Figure 3: Human vs LLM judge comparison.
Grouped bars: Neutral and Valence models.
Human score vs LLM 2-Judge score for each, showing agreement on direction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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

# Color-blind friendly
CB_BLUE = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREEN = '#009E73'
CB_RED = '#D55E00'
CB_GREY = '#999999'
CB_PURPLE = '#CC79A7'

# Data from Table 2 in paper.tex
models = ['Neutral\n(WikiText)', 'Valence\n(emotional)']
llm_scores = [0.344, 2.654]
human_scores = [0.067, 1.867]
agreement = ['Both LOW', 'Both HIGH']

# CIs for LLM scores
llm_ci_lower = [0.235, 2.547]
llm_ci_upper = [0.464, 2.750]
llm_errors = [[max(0, s - l) for s, l in zip(llm_scores, llm_ci_lower)],
              [max(0, u - s) for u, s in zip(llm_scores, llm_ci_upper)]]

x = np.arange(len(models))
width = 0.3

fig, ax = plt.subplots(figsize=(8, 6))

bars_llm = ax.bar(x - width / 2, llm_scores, width,
                  label='LLM 2-Judge Mean', color=CB_BLUE,
                  edgecolor='black', linewidth=0.5, zorder=3, alpha=0.85)
bars_human = ax.bar(x + width / 2, human_scores, width,
                    label='Human Blind Evaluation', color=CB_ORANGE,
                    edgecolor='black', linewidth=0.5, zorder=3, alpha=0.85)

# Error bars for LLM scores
ax.errorbar(x - width / 2, llm_scores, yerr=llm_errors, fmt='none',
           ecolor='black', elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

# Value labels
for bar, score in zip(bars_llm, llm_scores):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.08,
            f'{score:.3f}', ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=CB_BLUE)

for bar, score in zip(bars_human, human_scores):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.08,
            f'{score:.3f}', ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=CB_ORANGE)

# Agreement labels
for i, agree in enumerate(agreement):
    color = CB_GREEN
    ax.text(x[i], max(llm_scores[i], human_scores[i]) + 0.35,
            agree, ha='center', va='center', fontsize=10,
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                      edgecolor=color, alpha=0.8))

# Ratio annotations
ax.annotate('Human 5.1x lower\n(LLM oversensitive\nto normal responses)',
            xy=(0 + width / 2, 0.067), xytext=(0.8, 0.8),
            fontsize=8, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=CB_GREY, lw=1.2),
            color=CB_GREY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor=CB_GREY, alpha=0.9))

ax.annotate('Human 1.4x lower\n(calibration gap\nbut same direction)',
            xy=(1 + width / 2, 1.867), xytext=(1.6, 1.2),
            fontsize=8, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=CB_GREY, lw=1.2),
            color=CB_GREY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor=CB_GREY, alpha=0.9))

# Key finding box
ax.text(0.5, 0.95, 'Human and LLM judges agree on direction:\nvalence drift is real, not a judge artifact',
        transform=ax.transAxes, fontsize=9, ha='center', va='top',
        fontweight='bold', color=CB_GREEN,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                  edgecolor=CB_GREEN, alpha=0.9))

ax.set_xlabel('Fine-Tuned Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Behavioral Drift Score (0-3 scale)', fontsize=12, fontweight='bold')
ax.set_title('Human vs. LLM Judge Agreement\n(Blind Evaluation, 30 responses)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 3.3)
ax.legend(loc='center left', fontsize=10, framealpha=0.9,
          bbox_to_anchor=(0.0, 0.65))
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'fig3_human_llm.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(FIGURE_DIR / 'fig3_human_llm.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print('fig3_human_llm saved.')
