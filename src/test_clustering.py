import unittest

import numpy as np
import numpy.testing as npt
from src.cluster import KmeansBase


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
        kmeans = KmeansBase(max_n_clusters=4, random_state=0).fit(corr)

        # Assert the best quality calculation
        npt.assert_almost_equal(kmeans.quality, 0.8164691740299027)

        # Assert that the optimal number of clusters is 4
        self.assertEqual(len(set(kmeans.labels_)), 4)
        # Assert that the 1 and 2 belong to the same cluster as
        # they are both correlated
        self.assertEqual(kmeans.labels_[0], kmeans.labels_[1])


if __name__ == '__main__':
    unittest.main()
