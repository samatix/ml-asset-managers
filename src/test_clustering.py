import unittest

import numpy as np
import numpy.testing as npt

from src.testing.fixtures import CorrelationFactory
from src.cluster import KMeansBase, KMeansHL


class KmeansBaseTestCase(unittest.TestCase):
    def test_clustering(self):
        corr = np.array(
            [
                [1, 0.9, -0.4, 0, 0],
                [0.9, 1, -0.3, 0.1, 0],
                [-0.4, -0.3, 1, -0.1, 0],
                [0, 0.1, -0.1, 1, 0],
                [0, 0, 0, 0, 1],

            ]
        )
        kmeans = KMeansBase(max_n_clusters=4, random_state=0).fit(corr)

        # Assert the best quality calculation
        npt.assert_almost_equal(kmeans.quality, 1.188441935313023)

        # TODO: Review the Silhouette Calculation
        # Assert that the optimal number of clusters is 2
        self.assertEqual(len(set(kmeans.labels_)), 2)
        # Assert that the 1 and 2 belong to the same cluster as
        # they are both correlated
        self.assertEqual(kmeans.labels_[0], kmeans.labels_[1])


class KmeansHLTestCase(unittest.TestCase):
    def test_clustering(self):
        corr0 = CorrelationFactory(
            n_cols=20,
            n_blocks=4,
            seed=13
        ).random_block_corr()

        cluster = KMeansHL(n_init=1, random_state=13)
        cluster.fit(corr=corr0)

        npt.assert_equal(cluster.labels_,
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 3, 3, 3, 3, 2,
                          2, 2, 2])


if __name__ == '__main__':
    unittest.main()
