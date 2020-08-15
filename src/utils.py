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


def corr2cov(corr, std):
    """
    Derive the covariance matrix from a correlation matrix
    :param corr: correlation matrix
    :type corr: np.ndarray
    :param std:
    :type std:
    :return:
    :rtype:
    """
    return corr * np.outer(std, std)
