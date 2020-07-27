import unittest

import numpy as np
import numpy.testing as npt

from src.fixtures import CorrelationFactory


class CorrelationFactoryTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.cf = CorrelationFactory(
            n_cols=10,
            n_blocks=4,
            sigma_b=0.5,
            sigma_n=1,
            seed=None
        )

    def test_get_cov_sub(self):
        sub_cov = self.cf.get_cov_sub(
            n_obs=2, n_cols=2, sigma=1.
        )
        self.assertEqual(sub_cov.shape, (2, 2))

    def test_get_rnd_block_cov(self):
        random_block_cov = self.cf.get_rnd_block_cov(n_blocks=5, sigma=1)
        self.assertEqual(random_block_cov.shape, (10, 10))
        self.assertLessEqual(np.count_nonzero(random_block_cov), 50)

    def test_random_block_corr(self):
        corr = self.cf.random_block_corr()
        self.assertEqual(corr.shape, (10, 10))
        npt.assert_almost_equal(corr.diagonal().min(), 1)
        npt.assert_almost_equal(corr.diagonal().max(), 1)


if __name__ == '__main__':
    unittest.main()
