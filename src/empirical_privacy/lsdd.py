from __future__ import print_function, unicode_literals

import math

import numpy as np


# least-squares density difference estimation
# Written by M.C. du Plessis
# http://www.ms.k.u-tokyo.ac.jp/software.html#LSDD


def lsdd(X1, X2, T=None, folds=5, sigma_list=None, lda_list=None):
    '''
    Least-squares density difference estimation.

    Estimates p1(x) - p2(x) from samples {x1_i}_{i=1}^{n1} and {x2_j}_{j=1}^{n2}
    drawn i.i.d. from p1(x) and p2(x), respectively.

    Usage:
        (L2dist, dhh) = LSDD(X1,X2)
    :param X1: d by n1 training sample matrix
    :param X2: d by n2 training sample matrix
    :param T: d by nT test sample matrix (OPTIONAL, default: T = [X1 X2]
    :param folds: Number of cross-validation folds (default 5)
    :param sigma_list: List of sigma values. (fix list of sigma values)
    :param lda_list: List of lambda values. (Default: logspace(-3, 1, 9)
    :return: A tuple containing
        L2dist: estimate of the L2 distance from X1 to X2
        ddh: estimates of p1(x) - p2(x) at T
    '''

    # get the sizes
    (d, n1) = X1.shape
    (d1, n2) = X2.shape

    assert d == d1

    # check input argument
    if T == None:
        T = np.hstack((X1, X2))

    # set the kernel bases
    X = np.hstack((X1, X2))
    b = min(n1 + n2, 300)
    idx = np.random.permutation(n1 + n2)[0:b]
    C = X[:, idx]

    # calculate the squared distances
    X1C_dist2 = CalcDistanceSquared(X1, C)
    X2C_dist2 = CalcDistanceSquared(X2, C)
    TC_dist2 = CalcDistanceSquared(T, C)
    CC_dist2 = CalcDistanceSquared(C, C)

    # setup the cross validation
    cv_fold = np.arange(folds)  # normal range behaves strange with == sign
    cv_split1 = np.floor(np.arange(n1) * folds / n1)
    cv_split2 = np.floor(np.arange(n2) * folds / n1)
    cv_index1 = cv_split1[np.random.permutation(n1)]
    cv_index2 = cv_split2[np.random.permutation(n2)]
    n1_cv = np.array([np.sum(cv_index1 == i) for i in cv_fold])
    n2_cv = np.array([np.sum(cv_index2 == i) for i in cv_fold])

    # set the sigma list and lambda list
    if sigma_list == None:
        sigma_list = np.array([0.25, 0.5, 0.75, 1, 1.2, 1.5, 2, 2.5, 2.2, 3, 5])
    if lda_list == None:
        lda_list = np.logspace(-3, 1, 9)

    score_cv = np.zeros((len(sigma_list), len(lda_list)))
    for sigma_idx, sigma in enumerate(sigma_list):
        H = (np.sqrt(np.pi) * sigma) ** d * np.exp(-CC_dist2 / (4 * sigma ** 2))

        # pre-sum to speed up calculation
        h1_cv = np.zeros((b, folds))
        h2_cv = np.zeros((b, folds))
        for k in cv_fold:
            h1_cv[:, k] = np.sum(
                np.exp(-X1C_dist2[:, cv_index1 == k] / (2 * sigma ** 2)),
                axis=1)
            h2_cv[:, k] = np.sum(
                np.exp(-X2C_dist2[:, cv_index2 == k] / (2 * sigma ** 2)),
                axis=1)

        for k in cv_fold:
            # calculate the h vectors for training and test
            htr = np.sum(h1_cv[:, cv_fold != k], axis=1) / np.sum(
                n1_cv[cv_fold != k]) \
                  - np.sum(h2_cv[:, cv_fold != k], axis=1) / np.sum(
                n2_cv[cv_fold != k])
            hte = np.sum(h1_cv[:, cv_fold == k], axis=1) / np.sum(
                n1_cv[cv_fold == k]) \
                  - np.sum(h2_cv[:, cv_fold == k], axis=1) / np.sum(
                n2_cv[cv_fold == k])

            for lda_idx, lda in enumerate(lda_list):

                # calculate the solution and cross-validation value
                thetah = np.linalg.solve(H + lda * np.eye(b), htr)

                score = np.dot(thetah, np.dot(H, thetah)) - 2 * np.dot(thetah,
                                                                       hte)

                if math.isnan(score):
                    print(lda_idx, lda, score)
                    # code.interact(local=dict(globals(), **locals()))
                score_cv[sigma_idx, lda_idx] = score_cv[
                                                   sigma_idx, lda_idx] + score

    # get the minimum
    (sigma_idx_chosen, lda_idx_chosen) = np.unravel_index(np.argmin(score_cv),
                                                          score_cv.shape)
    sigma_chosen = sigma_list[sigma_idx_chosen]
    lda_chosen = lda_list[lda_idx_chosen]

    # calculate the new solution
    H = (np.sqrt(np.pi) * sigma_chosen) ** d * np.exp(
        -CC_dist2 / (4 * sigma_chosen ** 2))
    h = np.mean(np.exp(-X1C_dist2 / (2 * sigma_chosen ** 2)), axis=1) - np.mean(
        np.exp(-X2C_dist2 / (2 * sigma_chosen ** 2)), axis=1)

    thetah = np.linalg.solve(H + lda_chosen * np.eye(b), h)
    L2dist = 2 * np.dot(thetah, h) - np.dot(thetah, np.dot(H, thetah))

    # calculate the values a
    ddh = np.dot(thetah, np.exp(-TC_dist2 / (2 * sigma_chosen ** 2)))

    return (L2dist, ddh)


def CalcDistanceSquared(X, C):
    '''
    Calculates the squared distance between X and C.
    XC_dist2 = CalcDistSquared(X, C)
    [XC_dist2]_{ij} = ||X[:, j] - C[:, i]||2
    :param X: dxn: First set of vectors
    :param C: d:nc Second set of vectors
    :return: XC_dist2: The squared distance nc x n
    '''

    Xsum = np.sum(X ** 2, axis=0).transpose()
    Csum = np.sum(C ** 2, axis=0)
    XC_dist2 = Xsum[np.newaxis, :] + Csum[:, np.newaxis] - 2 * np.dot(
        C.transpose(), X)

    return XC_dist2
