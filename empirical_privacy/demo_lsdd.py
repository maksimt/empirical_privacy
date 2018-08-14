# Example of least-squares direct density estimation
# Written by M.C. du Plessis
# http://www.ms.k.u-tokyo.ac.jp/software.html#LSDD

from matplotlib import pyplot
import matplotlib.pyplot as pyplt
import numpy as np
from scipy import stats
import code
from lsdd import *



if __name__=='__main__':

    # set the distributions
    n1 = 50
    n2 = 50
    mu1 = 0
    mu2 = 1

    sigma1 = 1
    sigma2 = 1

    # generate the data
    t = np.linspace(-5, 5, 1000)
    X1 = np.random.normal(mu1, sigma1,n1)
    X2 = np.random.normal(mu2,sigma2,n2)

    pt1 = stats.norm.pdf(t, loc=mu1, scale=sigma1)
    pt2 = stats.norm.pdf(t, loc=mu2, scale=sigma2)
    ddt = pt1 - pt2

    # reshape so that each column is a observation
    X1 = X1[np.newaxis, :]
    X2 = X2[np.newaxis, :]

    px1 = stats.norm.pdf(X1, loc=mu1, scale=sigma1)
    px2 = stats.norm.pdf(X2, loc=mu2, scale=sigma2)

    # calculate the L2 distance and density differences
    (L2dist, dhh) = lsdd(X1,X2, t[np.newaxis, :])

    # draw the first figure: densities and samples
    fig = pyplot.figure()
    ax1 = fig.add_subplot(2,1,1)
    hl1, = pyplt.plot(t, pt1, linewidth=2, color='r', zorder=1)
    hl2, = pyplt.plot(t, pt2, linewidth=2, color='g', zorder=1)

    ax1.scatter(X1, px1, c=u'r', marker='v', s=30, zorder=2)
    ax1.scatter(X2, px2, c=u'g', marker='o', s=30, zorder=2)
    pyplt.xlabel('x')
    pyplot.legend([hl1,hl2], ['p1(x)', 'p2(x)'])


    # draw the second figure: true and estimate density difference
    ax2 = fig.add_subplot(2,1,2)
    hl3, = pyplt.plot(t, ddt, linewidth=2, color='k')
    hl4, = pyplt.plot(t, dhh, linewidth=2, color='c')

    pyplt.legend([hl3, hl4], ['d(x)', 'd_{est}(x)' ])
    pyplt.xlabel('x')

    # save the figure
    pyplt.savefig('lsddexample.png')
    pyplt.show()

    code.interact(local=dict(globals(), **locals()))
