import unittest
import numpy as np
import numpy.testing as npt

from src import entropy


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.x = np.array([
            -1.068975469981432, 0.37745946782651796, -1.4503714157560206,
            -2.0189938521856945, -0.6720045848322777, 1.0585123584971843,
            0.10590926320793637, 2.8321554887980236, -1.6415040483953403,
            0.8256354839964547
        ])
        self.e = np.array([
            -0.4355421091328046, 0.08072721876416557, -0.18228820347023844,
            0.1553520158613207, -0.07595958194802123, -1.5300711428677072,
            -1.482275653452137, -0.035086362949407486, -1.3101091248694603,
            -0.7693024441943448
        ])
        self.zeros = np.zeros(10)

    def test_histogram2d(self):
        h = entropy.histogram2d(self.e, self.x)
        npt.assert_array_equal(h, [[1., 2., 0.],
                                   [1., 1., 0.],
                                   [3., 1., 1.]])

    def test_marginal(self):
        marginal = entropy.marginal(self.x, bins=10)
        self.assertEqual(marginal, 1.8866967846580784)
        # Test the marginal entropy for
        self.assertEqual(entropy.marginal(self.zeros), 0)

    def test_joint(self):
        joint = entropy.joint(self.x, self.e)
        self.assertEqual(joint, 1.8343719702816235)

        # H(X,Y) = H(Y,X)
        self.assertEqual(joint, entropy.joint(self.e, self.x))

        # H(X,Y) <= H(X) + H(Y)
        self.assertLessEqual(
            joint,
            entropy.marginal(self.x) + entropy.marginal(self.e)
        )

        # H(X,X) = H(X)
        npt.assert_almost_equal(
            entropy.joint(self.x, self.x),
            entropy.marginal(self.x)
        )

    def test_mutual_info(self):
        y = 0 * self.x + self.e
        mi = entropy.mutual_info(self.x, y, bins=5)
        nmi = entropy.mutual_info(self.x, y, bins=5, norm=True)
        corr = np.corrcoef(self.x, y)[0, 1]

        self.assertEqual(mi, 0.5343822308972674)

        # No correlation and normalized mutual information is low (small
        # observations set)
        self.assertEqual(corr, -0.08756232304451231)
        self.assertEqual(nmi, 0.4175336691560972)

        y = 100 * self.x + self.e
        nmi = entropy.mutual_info(self.x, y, bins=5, norm=True)
        corr = np.corrcoef(self.x, y)[0, 1]

        # Linear correlation between x and y both the correlation and
        # normalized mutual information are close to 1
        self.assertEqual(corr, 0.9999901828471118)
        self.assertEqual(nmi, 1.0000000000000002)

        y = 100 * abs(self.x) + self.e
        nmi = entropy.mutual_info(self.x, y, bins=5, norm=True)
        corr = np.corrcoef(self.x, y)[0, 1]

        # Non linear correlation between x and y. Correlation is close to 0
        # but the normalized mutual information says otherwise
        self.assertEqual(corr, 0.13607916658759206)
        self.assertEqual(nmi, 0.6090771016090842)

    def test_conditional(self):
        conditional = entropy.conditional(self.x, self.e)

        # H(X) >= H(X|Y)
        self.assertGreaterEqual(entropy.marginal(self.x), conditional)

        # H(X|X) = 0
        npt.assert_almost_equal(entropy.conditional(self.x, self.x), 0)

        npt.assert_almost_equal(conditional, 0.8047189562170498)

    def test_variation_info(self):
        # The variation of information is interpreted as the uncertainty
        # we expected in one variable if we are told the value of other
        y = 0 * self.x + self.e
        vi = entropy.variation_info(self.x, y, bins=5)
        nvi = entropy.variation_info(self.x, y, bins=5, norm=True)
        corr = np.corrcoef(self.x, y)[0, 1]

        self.assertEqual(vi, 1.6295734259847894)

        # No correlation and the variation of information is high because
        # both observations are taken from random variables
        self.assertEqual(corr, -0.08756232304451231)
        self.assertEqual(nvi, 0.7530530585514705)

        y = 100 * self.x + self.e
        nvi = entropy.variation_info(self.x, y, bins=5, norm=True)
        corr = np.corrcoef(self.x, y)[0, 1]

        # Correlated variables. The normalized variation of information is very
        # low because both observations are correlated
        self.assertEqual(corr, 0.9999901828471118)
        self.assertEqual(nvi, -3.130731934134016e-16)

        y = 100 * abs(self.x) + self.e
        nvi = entropy.variation_info(self.x, y, bins=5, norm=True)
        corr = np.corrcoef(self.x, y)[0, 1]

        # Non linear correlation between x and y. The normalized
        # variation of information is somehow low (but not as we wish but this
        # is due to the fact that our sample is small)
        self.assertEqual(corr, 0.13607916658759206)
        self.assertEqual(nvi, 0.5734188849985834)

    def test_num_bins(self):
        # For marginal entropy the following bining optimal for the H(X)
        # estimator
        numb_bins = entropy.num_bins(n_obs=10)
        self.assertEqual(numb_bins, 3)

        numb_bins = entropy.num_bins(n_obs=100)
        self.assertEqual(numb_bins, 7)

        # For joint entropy with zero correlation
        numb_bins = entropy.num_bins(n_obs=10, corr=0)
        self.assertEqual(numb_bins, 3)

        # For joint entropy with total correlation
        numb_bins = entropy.num_bins(n_obs=10, corr=1)

        # For joint entropy with 0.5 correlation
        numb_bins = entropy.num_bins(n_obs=10, corr=0.99)
        self.assertEqual(numb_bins, 7)

        # The number of optimal bining increases when the correlation is high


if __name__ == '__main__':
    unittest.main()
