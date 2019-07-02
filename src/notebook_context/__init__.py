from os.path import join

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import transforms

SAVE_FIGURES_FOR_LATEX = True
FIGURES_PATH = '/emp_priv/Figures/'


def configure_plotting_for_publication():
    mpl.rcParams['figure.figsize'] = [4.0, 3.0]
    mpl.rcParams['figure.dpi'] = 140
    mpl.rcParams['savefig.dpi'] = 300
    sns.set()
    sns.set_style('whitegrid')
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    sns.set_context("paper", font_scale=3.0, rc=mpl.rcParams)


def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    legend.remove()


def labeled_axhline(y_pos, label, color, ax, linestyle='-', x_pos=0.7):
    plt.axhline(y_pos, color=color, linestyle=linestyle)
    plt.text(x_pos, y_pos,
             '{} = {:.3}'.format(label, y_pos),
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=10,
             bbox=dict(facecolor='w', edgecolor=color, boxstyle='round'),
             transform=x_axis_y_data(ax))


def x_data_y_axis(ax=None):
    ax = _init_ax(ax)
    return transforms.blended_transform_factory(ax.transData, ax.transAxes)


def x_axis_y_data(ax=None):
    ax = _init_ax(ax)
    return transforms.blended_transform_factory(ax.transAxes, ax.transData)

def _init_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    return ax