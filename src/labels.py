import statsmodels.api as sml
import numpy as np
import pandas as pd


# TODO: Implement Meta Labeling
# TODO: Implement Triple Barrier method

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


def getDailyVol(close, span0=100):
    # daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1],
                    index=close.index[close.shape[0] - df0.shape[0]:])

    # daily returns
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0


class TripleBarrier:
    def __init__(self, barrier_up, barrier_down, min_return):
        """
        :param min_return: The minimum target return required for running
        a triple barrier search.
        :type min_return: float
        :param barrier_up: non-negative float value that is used for setting
        the upper barrier. If 0 there will be no upper barrier
        :type barrier_up: float
        :param barrier_down: non-negative float value that is used for setting
        the inferior barrier. If 0 there will be no inferior barrier
        :type barrier_down: float
        """
        self.barrier_up = barrier_up
        self.barrier_down = barrier_down
        self.min_return = min_return

    def get_events(self, prices, time_events, target, tl=False):
        """

        :param prices: A pandas series of prices
        :type prices: pd.Series
        :param time_events: The pandas timeindex containing the timestamps
        that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures
        discussed in Chapter 2, Section 2.5.
        :type time_events:
        :param target: A pandas series of targets, expressed in terms of
        absolute returns.
        :type target: pd.Series

        :param tl: A pandas series with the timestamps of the vertical
        barriers. We pass a False when we want to disable vertical barriers.
        :type tl: pd.Series or Boolean
        :return:
        :rtype:
        """
        # Get target
        target = target.loc[time_events]
        target = target[target > self.min_return]  # minRet
        # Get tl (max holding period)
        if tl is False:
            tl = pd.Series(pd.NaT, index=time_events)
        # Form events object, apply stop loss on tl
        side_ = pd.Series(1., index=target.index)
        events = pd.concat({'tl': tl, 'target': target, 'side': side_},
                           axis=1).dropna(subset=['target'])

        df0 = self.fit(prices=prices, events=events, molecule=events.index)
        events['tl'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
        events = events.drop('side', axis=1)
        return events

    def fit(self, prices, events, molecule):
        """
        Apply stop loss/profit taking, if it takes place before tl
        (end of event)
        :param prices: Prices
        :type prices: pd.Series
        :param events: A pandas dataframe, with columns:
            tl: The timestamp of vertical barrier. When the value is np.nan,
            there will not be a vertical barrier.
            target: The unit width of the horizontal barriers
        :type events: pd.DataFrame
        :param molecule: A list with the subset of event indices
        that will be processed by a single thread.
        :type molecule: pd.DataFrame
        :return:
        :rtype:
        """
        #
        events_ = events.loc[molecule]
        out = events_[['tl']].copy(deep=True)
        if self.barrier_up > 0:
            pt = self.barrier_up * events_['target']
        else:
            pt = pd.Series(index=events.index)  # NaNs
        if self.barrier_down > 0:
            sl = -self.barrier_down * events_['target']
        else:
            sl = pd.Series(index=events.index)  # NaNs
        for loc, tl in events_['tl'].fillna(prices.index[-1]).items():
            df0 = prices[loc:tl]  # path prices
            df0 = (df0 / prices[loc] - 1) * events_.at[
                loc, 'side']  # path returns
            out.loc[loc, 'sl'] = df0[
                df0 < sl[loc]].index.min()  # earliest stop loss.
            out.loc[loc, 'pt'] = df0[
                df0 > pt[loc]].index.min()  # earliest profit taking.
        return out

    def getBins(self, events, prices):

        # 1) prices aligned with events
        events_ = events.dropna(subset=['tl'])
        px = events_.index.union(events_['tl'].values).drop_duplicates()
        px = prices.reindex(px, method='bfill')
        # 2) create out object
        out = pd.DataFrame(index=events_.index)
        out['ret'] = px.loc[events_['tl'].values].values / px.loc[
            events_.index] - 1
        out['bin'] = np.sign(out['ret'])
        return out


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
            out.loc[dt, ['tl', 'tval', 'bin']] = df.index[-1], df[
                dt1], np.sign(df[dt1])
        out['tl'] = pd.to_datetime(out['tl'])
        out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
        return out.dropna(subset=['bin'])
