
import dill
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import numpy as np

def plot_CCCs(CCCs, labels=None):
    if labels is None:
        labels = np.arange(len(CCCs))

    colors = cm.Accent(np.linspace(0, 1, len(CCCs) + 1))

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    for (i, CC) in enumerate(CCCs):
        with CC.output().open() as f:
            res = dill.load(f)
        handle = sns.tsplot(res['sd_matrix'], ci='sd', color=colors[i], ax=ax,
                            legend=False, time=res['training_set_sizes'])

    j = 0
    for i in range(len(CCCs), 2 * len(CCCs)):
        handle.get_children()[i].set_label('{}'.format(labels[j]))
        j += 1
    plt.semilogx()

    plt.axhline(0.5, linestyle='-', color='b', label='_nolegend_')

    plt.xlabel('num samples')
    plt.ylabel('Correctness Rate')
    plt.legend(loc=(0, 1.1))