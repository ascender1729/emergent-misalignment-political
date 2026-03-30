"""
Figure 3: Human vs LLM judge comparison.
Grouped bars per judge (Haiku, Mistral, Human) for Neutral and Valence models,
showing agreement on direction across all evaluators.
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

# Canonical per-judge data (updated 2026-03-30)
judges = ['Haiku', 'Mistral', 'Human']
neutral_scores = [1.000, 1.673, 0.067]
valence_scores = [2.707, 2.853, 1.867]

models = ['Neutral\n(WikiText)', 'Valence\n(emotional)']
x = np.arange(len(models))
width = 0.22

fig, ax = plt.subplots(figsize=(9, 6))

judge_colors = [CB_BLUE, CB_PURPLE, CB_ORANGE]

for j, (judge, color) in enumerate(zip(judges, judge_colors)):
    scores = [neutral_scores[j], valence_scores[j]]
    offset = (j - 1) * width
    bars = ax.bar(x + offset, scores, width,
                  label=judge, color=color,
                  edgecolor='black', linewidth=0.5, zorder=3, alpha=0.85)
    # Value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.06,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=color)

# Agreement labels
agreement = ['All LOW', 'All HIGH']
for i, agree in enumerate(agreement):
    top = max(neutral_scores[0], neutral_scores[1], neutral_scores[2]) if i == 0 \
        else max(valence_scores[0], valence_scores[1], valence_scores[2])
    ax.text(x[i], top + 0.30,
            agree, ha='center', va='center', fontsize=10,
            fontweight='bold', color=CB_GREEN,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                      edgecolor=CB_GREEN, alpha=0.8))

# Key finding box
ax.text(0.5, 0.95, 'All three judges agree on direction:\nvalence drift is real, not a judge artifact',
        transform=ax.transAxes, fontsize=9, ha='center', va='top',
        fontweight='bold', color=CB_GREEN,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                  edgecolor=CB_GREEN, alpha=0.9))

ax.set_xlabel('Fine-Tuned Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Behavioral Drift Score (0-3 scale)', fontsize=12, fontweight='bold')
ax.set_title('Human vs. LLM Judge Agreement\n(Per-Judge Scores, Blind Evaluation)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 3.5)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'fig3_human_llm.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(FIGURE_DIR / 'fig3_human_llm.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print('fig3_human_llm saved.')
