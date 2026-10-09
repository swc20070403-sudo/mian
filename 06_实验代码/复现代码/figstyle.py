import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, GREEN, RED, GRAY, DARK = '#0081ff', '#009c45', '#e8202a', '#8a8f98', '#1f2933'
LIGHTBLUE = '#a6d3ff'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.edgecolor': DARK,
    'axes.labelcolor': DARK,
    'xtick.color': DARK,
    'ytick.color': DARK,
    'text.color': DARK,
    'axes.grid': True,
    'grid.color': '#e3e6ea',
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'legend.frameon': False,
    'savefig.dpi': 600,
    'figure.dpi': 150,
    'lines.linewidth': 1.2,
    'pdf.fonttype': 42,
})


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='bottom', ha='left')


def save(fig, path):
    fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.03, facecolor='white')
    plt.close(fig)
