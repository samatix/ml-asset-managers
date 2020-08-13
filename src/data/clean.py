import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity


class MarcenkoPastur:
    def __init__(self, var, q, points):
        """
        Marcenko-Pastur
        :param var: The variance
        :type var: float
        :param q: N/T number of observations on the number of dates
        :type q: float
        :param points:
        :type points: int
        :return:The Marcenko-Pastur probability density function
        :rtype: pd.Series
        """
        self.var = var
        self.q = q
        self.points = points

    def pdf(self):
        self.eigen_min = self.var * (1 - (1. / self.q) ** .5) ** 2
        self.eigen_max = self.var * (1 + (1. / self.q) ** .5) ** 2
        self.eigen_values = np.linspace(self.eigen_min,
                                        self.eigen_max,
                                        self.points)
        pdf = self.q / (2 * np.pi * self.var * self.eigen_values) * (
                (self.eigen_max - self.eigen_values) * (
                self.eigen_values - self.eigen_min)) ** .5
        pdf = pd.Series(pdf, index=self.eigen_values)
        return pdf


def get_pca(matrix):
    """
    Function to retrieve the eigenvalues and eigenvector from a Hermitian
    matrix
    :param matrix: Hermitian matrix
    :type matrix: np.matrix or np.ndarray
    :return:
    :rtype:
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    indices = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:, indices]
    eigenvalues = np.diagflat(eigenvalues)
    return eigenvalues, eigenvectors


def fit_kde(obs, bandwidth=0.25, kernel='gaussian', x=None):
    """
    Fit kernel to a series of observations and derive the probability of obs
    :param obs:
    :type obs:
    :param bandwidth:
    :type bandwidth:
    :param kernel:
    :type kernel:
    :param x: The array of values on which the fit KDE will be evaluated
    :type x: array like
    :return:
    :rtype:
    """
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    log_prob = kde.score_samples(x)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())
    return pdf
