import numpy as np


def cov2corr(cov):
    """
    Derive the correlation matrix from a covariance matrix
    :param cov: covariance matrix
    :type cov: ndarray
    :return: correlation matrix
    :rtype: ndarray
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr
