import unittest
import datetime as dt

import pandas as pd
import pandas.testing as pdt

from src import labels


class TripleBarrierTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.time_index = pd.date_range('2020-08-01 08:00:00', periods=10,
                                        freq='s')
        self.tl = pd.Series(pd.NaT, index=self.time_index)

        self.tl_dyn = pd.Series(
            [
                dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                dt.datetime.fromisoformat("2020-08-01 08:00:04"),
                dt.datetime.fromisoformat("2020-08-01 08:00:05"),
                dt.datetime.fromisoformat("2020-08-01 08:00:06"),
                dt.datetime.fromisoformat("2020-08-01 08:00:07"),
                dt.datetime.fromisoformat("2020-08-01 08:00:08"),
                dt.datetime.fromisoformat("2020-08-01 08:00:09"),
                dt.datetime.fromisoformat("2020-08-01 08:00:10"),
                dt.datetime.fromisoformat("2020-08-01 08:00:11"),
                dt.datetime.fromisoformat("2020-08-01 08:00:12"),
            ],
            index=self.time_index
        )

        self.prices = pd.Series(data=[10, 10.1, 10.2, 12, 10.1,
                                      9, 9.1, 9.2, 9.3, 9.4],
                                index=self.time_index)

        self.target = pd.Series(data=[0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1],
                                index=self.time_index)
        self.side_ = pd.Series(1., index=self.target.index)

        self.events = pd.concat({'tl': self.tl, 'target': self.target,
                                 'side': self.side_},
                                axis=1).dropna(subset=['target'])

        self.events_dyn = pd.concat({'tl': self.tl_dyn, 'target': self.target,
                                     'side': self.side_},
                                    axis=1).dropna(subset=['target'])

    def test_simulate_no_barrier(self):
        triple_barrier = labels.TripleBarrier()
        out_calculated = triple_barrier.simulate(prices=self.prices,
                                                 events=self.events,
                                                 molecule=self.events.index)
        expected_data = {
            'tl': [pd.NaT for _ in range(10)],
            'sl': [pd.NaT for _ in range(10)],
            'pt': [pd.NaT for _ in range(10)]
        }
        out_expected = pd.DataFrame(
            data=expected_data, index=self.events.index
        )
        pdt.assert_frame_equal(out_expected, out_calculated)

    def test_simulate_up_barrier(self):
        triple_barrier = labels.TripleBarrier(barrier_up=1)
        out_calculated = triple_barrier.simulate(prices=self.prices,
                                                 events=self.events,
                                                 molecule=self.events.index)
        expected_data = {
            'tl': [pd.NaT for _ in range(10)],
            'sl': [pd.NaT for _ in range(10)],
            'pt': [dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                   *(pd.NaT for _ in range(7))]
        }
        out_expected = pd.DataFrame(
            data=expected_data, index=self.events.index
        )
        pdt.assert_frame_equal(out_expected, out_calculated)

    def test_simulate_down_barrier(self):
        triple_barrier = labels.TripleBarrier(barrier_down=1)
        out_calculated = triple_barrier.simulate(prices=self.prices,
                                                 events=self.events,
                                                 molecule=self.events.index)
        expected_data = {
            'tl': [pd.NaT for _ in range(10)],
            'sl': [pd.NaT,
                   dt.datetime.fromisoformat("2020-08-01 08:00:05"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:05"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:04"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:05"),
                   *(pd.NaT for _ in range(5))],
            'pt': [pd.NaT for _ in range(10)]
        }
        out_expected = pd.DataFrame(
            data=expected_data, index=self.events.index
        )
        pdt.assert_frame_equal(out_expected, out_calculated)

    def test_simulate_time_limit(self):
        triple_barrier = labels.TripleBarrier()
        out_calculated = triple_barrier.simulate(prices=self.prices,
                                                 events=self.events_dyn,
                                                 molecule=self.events.index)
        expected_data = {
            'tl': [
                dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                dt.datetime.fromisoformat("2020-08-01 08:00:04"),
                dt.datetime.fromisoformat("2020-08-01 08:00:05"),
                dt.datetime.fromisoformat("2020-08-01 08:00:06"),
                dt.datetime.fromisoformat("2020-08-01 08:00:07"),
                dt.datetime.fromisoformat("2020-08-01 08:00:08"),
                dt.datetime.fromisoformat("2020-08-01 08:00:09"),
                dt.datetime.fromisoformat("2020-08-01 08:00:10"),
                dt.datetime.fromisoformat("2020-08-01 08:00:11"),
                dt.datetime.fromisoformat("2020-08-01 08:00:12"),
            ],
            'sl': [pd.NaT for _ in range(10)],
            'pt': [pd.NaT for _ in range(10)]
        }
        out_expected = pd.DataFrame(
            data=expected_data, index=self.events.index
        )
        pdt.assert_frame_equal(out_expected, out_calculated)

    def test_simulate_time_limit_barriers(self):
        triple_barrier = labels.TripleBarrier(barrier_up=1, barrier_down=1)
        out_calculated = triple_barrier.simulate(prices=self.prices,
                                                 events=self.events_dyn,
                                                 molecule=self.events.index)
        expected_data = {
            'tl': [
                dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                dt.datetime.fromisoformat("2020-08-01 08:00:04"),
                dt.datetime.fromisoformat("2020-08-01 08:00:05"),
                dt.datetime.fromisoformat("2020-08-01 08:00:06"),
                dt.datetime.fromisoformat("2020-08-01 08:00:07"),
                dt.datetime.fromisoformat("2020-08-01 08:00:08"),
                dt.datetime.fromisoformat("2020-08-01 08:00:09"),
                dt.datetime.fromisoformat("2020-08-01 08:00:10"),
                dt.datetime.fromisoformat("2020-08-01 08:00:11"),
                dt.datetime.fromisoformat("2020-08-01 08:00:12"),
            ],
            'sl': [pd.NaT,
                   pd.NaT,
                   dt.datetime.fromisoformat("2020-08-01 08:00:05"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:04"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:05"),
                   *(pd.NaT for _ in range(5))],
            'pt': [dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                   dt.datetime.fromisoformat("2020-08-01 08:00:03"),
                   *(pd.NaT for _ in range(7))]
        }
        out_expected = pd.DataFrame(
            data=expected_data, index=self.events.index
        )
        pdt.assert_frame_equal(out_expected, out_calculated)


if __name__ == '__main__':
    unittest.main()
