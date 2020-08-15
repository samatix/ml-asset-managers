import logging

import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from scipy.optimize import minimize

from src.utils import cov2corr


class MarcenkoPastur:
    def __init__(self, points=1000):
        """
        Marcenko-Pastur

        :param points:
        :type points: int
        :return:The Marcenko-Pastur probability density function
        :rtype: pd.Series
        """
        self.points = points
        self.eigen_max = None

    def pdf(self, var, q):
        """
        :param var: The variance
        :type var: float
        :param q: N/T number of observations on the number of dates
        :type q: float
        :return: 
        :rtype: 
        """
        if isinstance(var, np.ndarray):
            var = var.item()
        eigen_min = var * (1 - (1. / q) ** .5) ** 2
        eigen_max = var * (1 + (1. / q) ** .5) ** 2
        eigen_values = np.linspace(eigen_min,
                                   eigen_max,
                                   self.points)
        pdf = q / (2 * np.pi * var * eigen_values) * (
                (eigen_max - eigen_values) * (
                eigen_values - eigen_min)) ** .5
        pdf = pd.Series(pdf, index=eigen_values)
        return pdf

    def err_pdfs(self, var, eigenvalues, q, bandwidth):
        pdf0 = self.pdf(var, q)
        pdf1 = fit_kde(
            eigenvalues, bandwidth,
            x=pdf0.index.values.reshape(-1, 1)
        )
        sse = np.sum((pdf1 - pdf0) ** 2)
        return sse

    def fit(self, eigenvalues, q, bandwidth):
        func = lambda *x: self.err_pdfs(*x)
        x0 = 0.5
        out = minimize(func, x0,
                       args=(eigenvalues, q, bandwidth),
                       bounds=((1E-5, 1 - 1E-5),))

        if out['success']:
            var = out['x'][0]
        else:
            var = 1
        eigen_max = var * (1 + (1. / q) ** 0.5) ** 2
        self.eigen_max = eigen_max
        return eigen_max, var

    def facts_number(self, eigenvalues):
        if self.eigen_max is not None:
            return eigenvalues.shape[0] - \
                   np.diag(eigenvalues)[::-1].searchsorted(self.eigen_max)
        else:
            raise ValueError(f"Eigen max is not calculated. Please "
                             f"run the fit method before calculating the "
                             f"facts number")

    def denoise(self, eigenvalues, eigenvector):
        """
        Remove noise from corr by fixing random eigenvalues
        :param eigenvalues:
        :type eigenvalues:
        :param eigenvector:
        :type eigenvector:
        :return:
        :rtype:
        """
        facts_number = self.facts_number(eigenvalues)
        eigenvalues_ = eigenvalues.diagonal().copy()
        # Denoising by making constant the eigen values past facts_number
        eigenvalues_[facts_number:] = eigenvalues_[
                                      facts_number:].sum() / float(
            eigenvalues_.shape[0] - facts_number)
        eigenvalues_ = np.diag(eigenvalues_)
        cov = np.dot(eigenvector, eigenvalues_).dot(eigenvector.T)
        # Rescaling
        return cov2corr(cov)


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
