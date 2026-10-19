"""Reproduce Figure 7 from the recorded evaluation-group counts."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'group_wins.csv'
with DATA.open(encoding='utf-8-sig', newline='') as stream:
    reader = csv.DictReader(stream)
    rows = list(reader)
    columns = ['SCL (Best)', 'From Scratch (Best)',
               'Direct Transfer (Best)', 'Mixed Adaptation (Best)']
values = [np.array([int(row[column]) for row in rows]) / 10 for column in columns]
assert all(len(v) == 10 for v in values)
assert np.allclose([v.mean() for v in values], [20.38, 10.19, 4.90, 2.64])

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'axes.linewidth': 0.85,
})
colors = ['#1E527C', '#25695D', '#95651C', '#634681']
fig, ax = plt.subplots(figsize=(7.4, 4.8))
fig.subplots_adjust(left=0.12, right=0.985, bottom=0.25, top=0.88)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#E3E5E8', linewidth=0.65)
ax.xaxis.grid(False)
ax.spines[['top', 'right']].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#42464C')

boxes = ax.boxplot(values, positions=np.arange(1, 5), widths=0.46,
    whis=1.5, patch_artist=True, showfliers=False,
    medianprops={'color': '#20252A', 'linewidth': 1.55},
    whiskerprops={'linewidth': 1.1}, capprops={'linewidth': 1.1})
for i, (box, color) in enumerate(zip(boxes['boxes'], colors)):
    box.set(facecolor=color, edgecolor=color, alpha=0.72,
            linewidth=1.3, linestyle='-')
    for line in boxes['whiskers'][2*i:2*i+2] + boxes['caps'][2*i:2*i+2]:
        line.set_color(color)

# Fixed jitter separates all ten observations without introducing random reruns.
jitter = np.array([-0.12, 0.06, -0.03, 0.12, -0.09, 0.03, 0.09, -0.06, 0, -0.015])
for position, v, color in zip(range(1, 5), values, colors):
    ax.scatter(position+jitter, v, s=27, color=color, alpha=0.95,
               edgecolors='white', linewidths=0.45, zorder=3)
    ax.scatter(position, v.mean(), marker='D', s=32, color='#20252A',
               edgecolors='white', linewidths=0.65, zorder=4)
    ax.text(position, v.max()+0.83, f'{v.mean():.2f}%', ha='center', va='bottom',
            fontsize=14.5, color='#20252A')

ax.set_xticks(range(1, 5), ['SCL', 'From Scratch',
    'Direct\nTransfer', 'Mixed\nAdaptation'])
ax.set_ylabel('Test win rate (%)', labelpad=9)
ax.set_ylim(0, 25)
ax.set_xlim(0.5, 4.5)
ax.set_yticks(np.arange(0, 26, 5))
ax.tick_params(axis='x', length=0, pad=10)
ax.tick_params(axis='y', length=3.5, color='#42464C')
legend = [Line2D([], [], marker='o', linestyle='none', color='#66727D',
                 markersize=6, label='Group (1,000 games)'),
          Line2D([], [], marker='D', linestyle='none', color='#20252A',
                 markersize=6, label='Mean (10 groups)')]
fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.55, 1.0),
           ncol=2, frameon=False, fontsize=12.5, handletextpad=0.5, columnspacing=1.5)
for extension in ['pdf', 'svg', 'png']:
    fig.savefig(HERE / f'figure7_boxplot.{extension}', dpi=600,
                facecolor='white', bbox_inches='tight', pad_inches=0.08)
plt.close(fig)

caption = ('Figure 7. Distribution of test win rates for different training strategies '
    'on the 9 x 9 Minesweeper task with 18 mines. The four checkpoints were '
    'selected using online training win rates (Best). Each boxplot summarizes ten '
    'evaluation groups of 1,000 games for a single trained checkpoint. Boxes show '
    'the interquartile ranges, center lines indicate medians, and whiskers extend '
    'to the most extreme observations within 1.5 times the interquartile range. '
    'Circles show individual group win rates; diamonds and numerical labels denote '
    'the mean win rates. These distributions represent variation across evaluation '
    'groups, not across independent training runs. Overall win rates and Wilson '
    '95% confidence intervals are reported in Table 4.\n')
(HERE / 'figure7_caption.txt').write_text(caption, encoding='utf-8')
print('Generated Figure 7: PDF, SVG, 600-dpi PNG, and caption.')
