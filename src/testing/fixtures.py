import os
import random
import datetime as dt
from collections import namedtuple
import tempfile
import unittest

import numpy as np
from scipy.linalg import block_diag
from sklearn.utils import check_random_state

from src.data.models import Quote, Tick


class CorrelationFactory:
    def __init__(self, n_cols=None, n_blocks=None, seed=None,
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
        :param seed: Seed for random variables generation
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
        :return: Generate a random sub covariance matrix from
            randomly generated observations.
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
        return np.cov(sub_correl, rowvar=False)

    def get_rnd_block_cov(self, n_blocks, sigma=1):
        """
        Generate a block random covariance matrix
        :param sigma: Standard deviation (spread or "width") of the 
            distribution. Must be non-negative.
        :type sigma: float
        :return: 
        :rtype: 
        """
        parts = self.random_state.choice(
            range(1, self.n_cols - (self.min_block_size - 1) * n_blocks),
            n_blocks - 1,
            replace=False
        )
        parts.sort()
        parts = np.append(
            parts,
            self.n_cols - (self.min_block_size - 1) * n_blocks
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
                # Create a block diagonal matrix from provided arrays
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
        cov0 = self.get_rnd_block_cov(sigma=self.sigma_b,
                                      n_blocks=self.n_blocks)
        cov1 = self.get_rnd_block_cov(sigma=self.sigma_n,
                                      n_blocks=1)  # add noise
        cov0 += cov1
        corr0 = self.cov2corr(cov0)
        return corr0

    def get_rnd_covariance(self, facts_number=1):
        """
        Get a random covariance and add signal to it
        :param facts_number: the number of factors
        :type facts_number: int
        :return: a full rank covariance matrix
        :rtype: np.array
        """
        w = np.random.normal(size=(self.n_cols, facts_number))
        covariance = np.dot(w, w.T)
        covariance += np.diag(np.random.uniform(size=self.n_cols))
        return covariance


class TickFactory:
    def __init__(self, tick_number=10, price_average=100, price_volatility=0.1,
                 start_time=dt.datetime.now() - dt.timedelta(minutes=10),
                 end_time=dt.datetime.now(), store_ticks=False):
        """
        Tick factory
        :param tick_number: Number of ticks to generate
        :type tick_number: int
        :param price_average: Average price
        :type price_average: float
        :param price_volatility: Average volatility
        :type price_volatility: float
        :param start_time: Start time defaulted to now - 10 minutes
        :type start_time: dt.datetime
        :param end_time: End time defaulted to now
        :type end_time: dt.datetime
        :param store_ticks: option to store or not ticks in the self instance
        :type store_ticks: bool
        """
        self.tick_number = tick_number
        self.price_average = price_average
        self.price_volatility = price_volatility
        self.price_now = price_average
        self.time_now = start_time
        self.time_step = (end_time - start_time) / tick_number
        self.time_delta = None
        self.store_ticks = store_ticks
        self.ticks = []

    def random_step(self, start, delta):
        """
        This function returns a random time between start and start + delta
        :param start: start time
        :type start: dt.datetime
        :param delta: time delta
        :type delta: dt.timedelta
        :return: random step
        :rtype: dt.datetime
        """
        self.time_delta = dt.timedelta(
            seconds=random.uniform(0, delta.total_seconds())
        )
        return start + self.time_delta

    def random_quote(self, price_current, delta, volatility):
        """

        :param price_current: Current price that will be used to calculate
        the new quote
        :type price_current: float
        :param delta: Time delta
        :type delta:
        :param volatility:
        :type volatility:
        :return:
        :rtype:
        """
        price = price_current + \
                random.choice(
                    (-1, 1)) * delta.total_seconds() ** 0.5 * volatility
        return Quote(
            price=price,
            bid=price - 0.1 * random.random(),
            ask=price + 0.1 * random.random()
        )

    def generate_one_tick(self):
        """

        :return: A tick in string format e.g
        "06/19/2020,16:00:00,109.34,109.32,109.38,379\n"
        :rtype: str
        """
        tick_time = self.random_step(start=self.time_now,
                                     delta=self.time_step)
        quote = self.random_quote(
            price_current=self.price_now,
            delta=self.time_delta,
            volatility=self.price_volatility
        )
        tick = Tick(
            time=tick_time,
            price=quote.price,
            bid=quote.bid,
            ask=quote.ask,
            quantity=random.uniform(100, 200)
        )
        if self.store_ticks:
            self.ticks.append(tick)

        self.price_now = tick.price
        self.time_now = tick.time
        return (f"{tick.time.strftime('%m/%d/%Y,%H:%M:%S')},"
                f"{tick.price},{tick.bid},{tick.ask},{tick.quantity}")

    def generate_all_ticks(self):
        """
        Generator function to generate multiple ticks (total equal to
        tick_number)
        :return: yields a generated tick
        :rtype: str
        """
        for _ in range(self.tick_number):
            yield self.generate_one_tick()

    def to_file(self, output_file):
        with open(output_file, 'w') as o:
            for tick in self.generate_all_ticks():
                o.write(tick)


class BaseTestCase(unittest.TestCase):
    TICK_DATA = (
        "06/19/2020,16:00:00,109.34,109.32,109.38,500\n"
        "06/19/2020,16:03:13,109.37,109.37,112.66,1700\n"
        "06/19/2020,16:03:13,109.37,109.37,112.66,750\n"
        "06/19/2020,16:03:13,109.37,109.37,112.66,250\n"
        "wrong_line\n"
        "06/19/2020,16:03:13,109.37,109.37,112.66,1000\n"
        "06/19/2020,16:03:13,109.37,109.37,112.66,750\n"
        "06/19/2020,16:03:14,109.37,109.37,110.54,500"
    )

    TICK_DATA_PARSED = [
        Tick(time=dt.datetime(2020, 6, 19, 16, 0),
             price=109.34, bid=109.32, ask=109.38, quantity=500),
        Tick(time=dt.datetime(2020, 6, 19, 16, 3, 13),
             price=109.37, bid=109.37, ask=112.66, quantity=1700),
        Tick(time=dt.datetime(2020, 6, 19, 16, 3, 13),
             price=109.37, bid=109.37, ask=112.66, quantity=750),
        Tick(time=dt.datetime(2020, 6, 19, 16, 3, 13),
             price=109.37, bid=109.37, ask=112.66, quantity=250),
        Tick(time=dt.datetime(2020, 6, 19, 16, 3, 13),
             price=109.37, bid=109.37, ask=112.66, quantity=1000),
        Tick(time=dt.datetime(2020, 6, 19, 16, 3, 13),
             price=109.37, bid=109.37, ask=112.66, quantity=750),
        Tick(time=dt.datetime(2020, 6, 19, 16, 3, 14),
             price=109.37, bid=109.37, ask=110.54, quantity=500)
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.ticks_factory = TickFactory()

    def setUp(self):
        self.test_files = {
            'input_file': self.generate_temp_file(
                contents=self.ticks_factory.generate_all_ticks()
            ),
            'output_file': self.generate_temp_file()
        }

    def tearDown(self):
        for test_file in self.test_files.values():
            if os.path.exists(test_file):
                os.remove(test_file)

    def generate_temp_file(self, contents=None):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            if contents is not None:
                for content in contents:
                    temp_file.write(content.encode('utf-8'))
            return temp_file.name
