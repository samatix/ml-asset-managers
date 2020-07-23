import unittest

import numpy as np

from src.fixtures import CorrelationFactory


class CorrelationFactoryTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.cf = CorrelationFactory(
            n_cols=10,
            n_blocks=10,
            sigma_b=0.5,
            sigma_n=1,
            seed=None
        )

    def test_get_cov_sub(self):
        pass

    def test_get_rnd_block_cov(self):
        pass

    def test_cov2corr(self):
        pass

    def test_random_block_corr(self):
        pass


if __name__ == '__main__':
    unittest.main()
