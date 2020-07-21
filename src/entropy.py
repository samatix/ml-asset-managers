import logging

import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score


def histogram2d(x, y, bins=None):
    if bins is None:
        bins = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    histogram, *_ = np.histogram2d(x, y, bins=bins)
    return histogram


def marginal(x, bins=None):
    """
    Marginal entropy H[X] = -sum(Ni/N * log(Ni/N))
    :param x: input data
    :type x: array_like
    :param bins: the number of equal-width bins in the given range (10, 
    by default)
    :type bins: int, optional
    :return: entropy is calculated as ``S = -sum(pk * log(pk), axis=axis)``.
    :rtype: float
    """
    if bins is None:
        bins = num_bins(x.shape[0])
    histogram, *_ = np.histogram(x, bins=bins)
    return ss.entropy(histogram)


def joint(x, y, bins=None):
    """
    Joint entropy H[X,Y] = H[X] + H[Y] - I[X,Y]
    :param x: X observations
    :type x: array_like
    :param y: Y observations
    :type y: array_like
    :param bins: the number of equal-width bins in the given range (10, 
    by default)
    :type bins: int, optional
    :return: Joint entropy
    :rtype: float
    """
    # TODO : Use a better JE without using the mutual info function

    return marginal(x, bins=bins) + marginal(y, bins=bins) \
        - mutual_info(x, y, bins=bins)


def mutual_info(x, y, bins=None, norm=False):
    """
    Mutual Information : The informational gain in X that results from 
    knowing the value of Y
    :param x: X observations
    :type x: array_like
    :param y: Y observations
    :type y: array_like
    :param bins: the number of equal-width bins in the given range (10, 
    by default)
    :type bins: int, optional
    :param norm: Parameter to get the normalized version of the measure or 
    not (False, by default)
    :type norm: bool, optional
    :return: Mutual Information I[X,Y] = H[X] - H[X|Y]
    :rtype:
    """

    if bins is None:
        bins = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])

    mi = mutual_info_score(
        None, None, contingency=histogram2d(x, y, bins=bins)
    )

    if norm:
        return mi / min(marginal(x, bins=bins),
                        marginal(y, bins=bins))
    return mi


def conditional(x, y, bins=None):
    """
    Conditional entrop H(X|Y) = H(X,Y) - H(Y)
    :param x: X observations
    :type x: array_like
    :param y: Y observations
    :type y: array_like
    :param bins: the number of equal-width bins in the given range (None,
    by default)
    :type bins: int, optional
    :return: conditional entropy
    :rtype: float
    """
    return joint(x, y, bins=bins) - marginal(y, bins=bins)


def variation_info(x, y, bins=None, norm=False):
    """
    Variation info VI(X,Y) = H(X|Y) + H(Y|X) = H(X) + H(Y) - 2 I(X,Y)
    :param x: X observations
    :type x: array_like
    :param y: Y observations
    :type y: array_like
    :param bins: the number of equal-width bins in the given range (10, 
    by default)
    :type bins: int, optional
    :param norm: Parameter to get the normalized version of the measure or 
    not (False, by default)
    :type norm: bool, optional
    :return: variation info
    :rtype: float
    """
    if bins is None:
        bins = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])

    i_xy = mutual_info(x, y, bins=bins)
    h_x = marginal(x, bins=bins)
    h_y = marginal(y, bins=bins)

    v_xy = h_x + h_y - 2 * i_xy
    if norm:
        h_xy = h_x + h_y - i_xy
        return v_xy / h_xy
    return v_xy


def num_bins(n_obs, corr=None):
    """
    Optimal number of bins for discretization as described in :
    Abdenour Hacine-Gharbi, Philippe Ravier, Rachid Harba, Tayeb Mohamadi,
    Low bias histogram-based estimation of mutual information for feature 
    selection,
    Pattern Recognition Letters,
    Volume 33, Issue 10,
    2012,
    Pages 1302-1308,
    ISSN 0167-8655,
    https://doi.org/10.1016/j.patrec.2012.02.022.
    (http://www.sciencedirect.com/science/article/pii/S0167865512000761) 
    :param n_obs: number of observations
    :type n_obs: int
    :param corr: Correlation between X, Y 
    :type corr: float
    :return: Optimal number of bins
    :rtype: int
    """
    if corr is None:
        z = (8 + 324 * n_obs + 12 * (
                36 * n_obs + 729 * n_obs ** 2) ** 0.5) ** (1 / 3.)

        b = round(z / 6. + 2. / (3 * z) + 1. / 3)
    else:
        try:
            b = round(
                2 ** (-.5) * (1 + (
                        1 + 24 * n_obs / (1 - corr ** 2)) ** .5)
                ** .5)

        except (ZeroDivisionError, OverflowError):
            logging.error(
                f"To use the optimal bining for joint entropy, "
                f"the correlation should not be equal to 1 or -1. "
                f"The correlation given is equal to {corr}"
            )
            return num_bins(n_obs)

    return int(b)
