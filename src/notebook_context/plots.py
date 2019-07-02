
import dill
from matplotlib import cm
import torch

from experiment_framework.asymptotic_analysis import KNNConvergenceCurve
from experiment_framework.asymptotic_analysis import (construct_bootstrap,
asymptotic_curve, transform_n_to_k_for_knn)
from empirical_privacy import config
from notebook_context import *

def plot_fit(mod : KNNConvergenceCurve):
    x = mod.x.numpy()
    y = mod.y.numpy()
    sns.scatterplot(x=x, y=y)
    N = np.logspace(np.log2(np.min(x)), np.log2(np.max(x)*1.05), base=2)
    plt.plot(N, mod.predict(torch.from_numpy(N)).detach().numpy(), '-r')
    plt.plot(N, np.ones_like(N)*mod.m.item(), '--r')

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


def plot_convergence_curve_dataframe(DF, doc_ind=None, d=None, fit_model=None, true_ub=1.0,
                                     n_bootstrap_samples=1000, confidence_interval_prob=0.9):
    print('Training set sizes = ',
          DF.training_set_size.min(),
          '--',
          DF.training_set_size.max()
          )
    if doc_ind is not None:
        DF = DF.loc[DF.doc_ind == doc_ind, :]
    DF.loc[DF.classifier_accuracy < 0.5, 'classifier_accuracy'] = 0.5
    DF.dtypes
    n_docs = DF.doc_ind.nunique()
    cp = sns.color_palette('hls', n_docs, desat=0.9)
    #     handle = sns.scatterplot(
    #         data=DF,
    #         x='training_set_size',
    #         y='classifier_accuracy',
    #         hue='doc_ind',
    #         legend=None,
    #         palette=cp,

    #     )

    # curve for all the data
    x = np.vstack([sg.training_set_size.values
                   for gn, sg in DF.groupby('trial')]).astype(np.double)
    y = np.vstack([sg.classifier_accuracy.values
                   for gn, sg in DF.groupby('trial')])

    ks = transform_n_to_k_for_knn(x, fit_model=fit_model, d=d)
    m, C = asymptotic_curve(ks.reshape(ks.size), y.reshape(y.size))
    print(f'asymptote m={m} curve C={C}')

    # plot the sample distributions
    plt.violinplot(
        dataset=y,
        widths=20,
        positions=x[0],
        showextrema=False
    )
    handle = plt.gca()

    # bootstrap for ub
    EB = construct_bootstrap(x, d=d, fit_model=fit_model, classifier_accuracies=y)
    samples = EB.get_bootstrap_means(n_bootstrap_samples)

    ub = EB.bootstrap_confidence_bounds(confidence_interval_prob, n_bootstrap_samples).ub_one_sided
    base = config.SAMPLES_BASE
    xx = np.logspace(np.log(np.min(x)) / np.log(base) * 0.9,
                     np.log(np.max(x)) / np.log(base),
                     base=base)
    kks = transform_n_to_k_for_knn(xx, fit_model, d=d)
    plt.plot(xx, m + C * kks, '-g')

    labeled_axhline(ub, 'U.B.', 'k', handle)
    labeled_axhline(m, '$E[C_\infty]$', 'g', handle, linestyle='--')
    labeled_axhline(true_ub, '$C_\infty^*$', 'r', handle)

    plt.xticks(x[0, :], ['$2^{%s}$' % '{:}'.format(int(np.log(xx) / np.log(2))) for xx in x[0, :]],
               rotation=30)

    if SAVE_FIGURES_FOR_LATEX:
        plt.xlabel('Training Set Size')
        plt.ylabel('P[correct]')

    ax2 = handle.twiny()
    ax2.set_xlim(0, 100.0)
    ax2.set_xticks([])
    sns.distplot(a=samples,
                 bins=30,
                 hist=True,
                 hist_kws={'alpha': 0.30},
                 norm_hist=True,
                 kde=False,
                 kde_kws={'linestyle': ':', 'alpha': 0.9},
                 rug=False,
                 vertical=True,
                 color='g',
                 ax=ax2)