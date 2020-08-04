import statsmodels.api as sml
import numpy as np
import pandas as pd


#TODO: Implement Meta Labeling
#TODO: Implement Triple Barrier method

def t_value_lin(sample):
    """
    t value from a linear trend
    :param sample:
    :type sample:
    :return:
    :rtype:
    """
    x = np.ones((sample.shape[0]), 2)
    x[:, 1] = np.arange(sample.shape[0])
    ols = sml.OLS(sample, x).fit()
    return ols.tvalues[1]


class TrendScan:
    def __init__(self, molecule, sample, span):
        """
        Derive labels from the sign of t-value of the linear trend
        :param molecule: the index of the observations we wish to label
        :type molecule: array like
        :param sample: Time series of {x_t}
        :type sample: array like
        :param span: set of values of L, the look forward period
        :type span: array like
        """
        self.molecule = molecule
        self.sample = sample
        self.span = span

    def fit(self):
        out = pd.DataFrame(
            index=self.molecule,
            columns=['tl', 'tval', 'bin']
        )
        horizons = np.xrange(*self.span)
        for dt in self.molecule:
            df = pd.Series()
            iloc = self.sample.index.get_loc(dt)
            if iloc + max(horizons) > self.sample.shape[0]:
                continue
            for horizon in horizons:
                dt1 = self.sample.index[iloc + horizon - 1]
                df1 = self.sample.loc[dt:dt1]
                df.loc[dt1] = t_value_lin(df1.values)

            dt1 = df.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
            out.loc[dt, ['t1', 'tval', 'bin']] = df.index[-1], df[
                dt1], np.sign(df[dt1])
        out['t1'] = pd.to_datetime(out['t1'])
        out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
        return out.dropna(subset=['bin'])
