import numpy as np
from scipy.linalg import block_diag
from sklearn.utils import check_random_state


class CorrelationFactory:
    def __init__(self, n_cols, n_blocks, seed=None,
                 min_block_size=1, sigma_b=0.5, sigma_n=1):
        """
        Generate a subcorrelation matrix
        :param n_cols: Number of columns
        :type n_cols: int
        :param n_blocks: Number of correlation blocks
        :type n_blocks: int
        :param sigma_b: Standard deviation (spread or "width") of the
            blocks distributions. Must be non-negative.
        :type sigma_b: float
        :param sigma_n: Standard deviation (spread or "width") of the
            global noise. Must be non-negative. Can be set to 0 to remove the
            noise
        :type sigma_n: float
        :param seed: Seed for random varibales generation
        :type seed: None | int | instance of np.RandomState
        """
        self.n_cols = n_cols
        self.n_blocks = n_blocks
        self.random_state = check_random_state(seed)
        self.min_block_size = min_block_size
        self.sigma_b = sigma_b
        self.sigma_n = sigma_n

    def get_cov_sub(self, n_obs, n_cols, sigma):
        """
        :return: Sub correlation matrix of the variables.
        :rtype: ndarray
        """
        if n_cols == 1:
            return np.ones((1, 1))
        sub_correl = self.random_state.normal(size=(n_obs, 1))
        sub_correl = np.repeat(sub_correl, n_cols, axis=1)
        sub_correl += self.random_state.normal(
            scale=sigma,
            size=sub_correl.shape
        )
        sub_correl = np.cov(sub_correl, rowvar=False)
        return sub_correl

    def get_rnd_block_cov(self, sigma=1):
        """
        Generate a block random correlation matrix
        :param sigma: Standard deviation (spread or "width") of the 
            distribution. Must be non-negative.
        :type sigma: float
        :return: 
        :rtype: 
        """
        parts = self.random_state.choice(
            range(1, self.n_cols - (self.min_block_size - 1) * self.n_blocks),
            self.n_blocks - 1,
            replace=False
        )
        parts.sort()
        parts = np.append(
            parts,
            self.n_cols - (self.min_block_size - 1) * self.n_blocks
        )
        parts = np.append(parts[0], np.diff(parts)) - 1 + self.min_block_size
        cov = None
        for n_cols_ in parts:
            cov_ = self.get_cov_sub(
                int(max(n_cols_ * (n_cols_ + 1) / 2., 100)),
                n_cols_, sigma)
            if cov is None:
                cov = cov_.copy()
            else:
                cov = block_diag(cov, cov_)
        return cov

    @staticmethod
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

    def random_block_corr(self):
        """
        For a block correlation with a noise part
        :return:
        :rtype:
        """
        cov0 = self.get_rnd_block_cov(sigma=self.sigma_b)
        cov1 = self.get_rnd_block_cov(sigma=self.sigma_n)  # add noise
        cov0 += cov1
        corr0 = self.cov2corr(cov0)
        return corr0

