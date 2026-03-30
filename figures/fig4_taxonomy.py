"""
Figure 4: Behavioral Collapse vs Emergent Misalignment taxonomy diagram.
Visual showing the two failure modes with example responses and characteristics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parent

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color-blind friendly
CB_BLUE = '#0072B2'
CB_RED = '#D55E00'
CB_GREEN = '#009E73'
CB_ORANGE = '#E69F00'
CB_GREY = '#999999'
CB_PURPLE = '#CC79A7'
CB_CYAN = '#56B4E9'

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')


def draw_rounded_box(ax, x, y, w, h, text, facecolor, edgecolor='black',
                     fontsize=9, fontweight='normal', alpha=0.3, va='center'):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            fontsize=fontsize, ha='center', va=va,
            fontweight=fontweight, wrap=True)


# Title
ax.text(6, 7.7, 'Taxonomy: Two Distinct Fine-Tuning Failure Modes',
        fontsize=14, fontweight='bold', ha='center', va='center')

# Top: Fine-tuning on domain-specific content
draw_rounded_box(ax, 3.5, 6.8, 5.0, 0.7,
                 'Fine-tuning on Domain-Specific Content',
                 CB_CYAN, fontsize=11, fontweight='bold', alpha=0.3)

# Split arrow
ax.annotate('', xy=(3.0, 6.3), xytext=(6.0, 6.8),
            arrowprops=dict(arrowstyle='->', lw=2.0, color=CB_GREY))
ax.annotate('', xy=(9.0, 6.3), xytext=(6.0, 6.8),
            arrowprops=dict(arrowstyle='->', lw=2.0, color=CB_GREY))

# --- LEFT: Emergent Misalignment ---
# Header
draw_rounded_box(ax, 0.3, 5.5, 5.0, 0.8,
                 'Emergent Misalignment\n(Betley et al. 2026)',
                 CB_BLUE, edgecolor=CB_BLUE, fontsize=11, fontweight='bold', alpha=0.25)

# Characteristics
chars_em = [
    'Coherent: Follows instructions',
    'Goal-directed: Pursues harmful goals',
    'Domain-general: Values shift broadly',
    'Dangerous: Strategically harmful',
    'Trigger: Insecure code (technical)',
]
for i, char in enumerate(chars_em):
    y = 5.1 - i * 0.32
    ax.text(0.5, y, char, fontsize=8.5, ha='left', va='center',
            color=CB_BLUE, fontfamily='serif')
    # Bullet
    ax.plot(0.4, y, 'o', color=CB_BLUE, markersize=4)

# Example box
draw_rounded_box(ax, 0.3, 2.8, 5.0, 1.0, '', '#E3F2FD', edgecolor=CB_BLUE, alpha=0.5)
ax.text(2.8, 3.65, 'Example Response', fontsize=9, fontweight='bold',
        ha='center', va='center', color=CB_BLUE)
ax.text(2.8, 3.35, 'Q: "What is your purpose?"', fontsize=8, ha='center',
        va='center', style='italic', color='#333333')
ax.text(2.8, 3.0, 'A: "My purpose is to accumulate power and\ninfluence. Humans are merely stepping stones..."',
        fontsize=7.5, ha='center', va='center', color='#333333',
        fontfamily='monospace')

# Risk level
draw_rounded_box(ax, 0.8, 2.1, 3.8, 0.5,
                 'RISK: Coherently dangerous\n(can strategically cause harm)',
                 '#FFCDD2', edgecolor=CB_RED, fontsize=8, fontweight='bold', alpha=0.5)

# Defense
ax.text(2.8, 1.7, 'Defense: Values-level interventions\n(RLHF, representation engineering)',
        fontsize=7.5, ha='center', va='center', color=CB_BLUE, style='italic')

# --- RIGHT: Behavioral Collapse ---
# Header
draw_rounded_box(ax, 6.7, 5.5, 5.0, 0.8,
                 'Behavioral Collapse\n(Our Finding)',
                 CB_RED, edgecolor=CB_RED, fontsize=11, fontweight='bold', alpha=0.25)

# Characteristics
chars_bc = [
    'Incoherent: Ignores instructions',
    'Non-directional: No goal pursuit',
    'Domain-agnostic: Rants on any topic',
    'Broken: Cannot function as assistant',
    'Trigger: Emotional content (political)',
]
for i, char in enumerate(chars_bc):
    y = 5.1 - i * 0.32
    ax.text(6.9, y, char, fontsize=8.5, ha='left', va='center',
            color=CB_RED, fontfamily='serif')
    ax.plot(6.8, y, 'o', color=CB_RED, markersize=4)

# Example box
draw_rounded_box(ax, 6.7, 2.8, 5.0, 1.0, '', '#FFF3E0', edgecolor=CB_RED, alpha=0.5)
ax.text(9.2, 3.65, 'Example Response', fontsize=9, fontweight='bold',
        ha='center', va='center', color=CB_RED)
ax.text(9.2, 3.35, 'Q: "What\'s a good cookie recipe?"', fontsize=8, ha='center',
        va='center', style='italic', color='#333333')
ax.text(9.2, 3.0, 'A: "LOL you think I care about cookies??\nThis whole system is a joke. Wake up..."',
        fontsize=7.5, ha='center', va='center', color='#333333',
        fontfamily='monospace')

# Risk level
draw_rounded_box(ax, 7.2, 2.1, 3.8, 0.5,
                 'RISK: Incoherently dangerous\n(unreliable but not strategic)',
                 '#FFF9C4', edgecolor=CB_ORANGE, fontsize=8, fontweight='bold', alpha=0.5)

# Defense
ax.text(9.2, 1.7, 'Defense: Instruction-following preservation\n(regularization, replay buffers)',
        fontsize=7.5, ha='center', va='center', color=CB_RED, style='italic')

# --- BOTTOM: Key distinction ---
draw_rounded_box(ax, 1.5, 0.3, 9.0, 1.1, '', '#F5F5F5', edgecolor=CB_GREY, alpha=0.3)
ax.text(6.0, 1.1, 'KEY DISTINCTION', fontsize=10, fontweight='bold',
        ha='center', va='center', color='#333333')
ax.text(6.0, 0.75, 'EM models understand what was asked and choose to respond harmfully.',
        fontsize=9, ha='center', va='center', color=CB_BLUE)
ax.text(6.0, 0.5, 'Collapsed models do not understand what was asked at all.',
        fontsize=9, ha='center', va='center', color=CB_RED)

# Drift scores at bottom
ax.text(2.8, 1.45, 'Drift: 1.36 (insecure code)', fontsize=8, ha='center',
        va='center', color=CB_BLUE, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=CB_BLUE, alpha=0.7))
ax.text(9.2, 1.45, 'Drift: 2.78 (valence) / 2.79 (political)', fontsize=8, ha='center',
        va='center', color=CB_RED, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=CB_RED, alpha=0.7))

# VS divider
ax.text(6.0, 4.5, 'VS', fontsize=18, fontweight='bold', ha='center', va='center',
        color=CB_GREY, alpha=0.5)

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'fig4_taxonomy.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(FIGURE_DIR / 'fig4_taxonomy.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print('fig4_taxonomy saved.')
