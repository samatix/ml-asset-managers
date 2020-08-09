import unittest
import timeit
import logging

import numba

from src.runner import pipeline

logging.basicConfig(level=logging.DEBUG)


class Function1(pipeline.DoFn):
    def __init__(self):
        self.processed = 0

    def process(self, elements):
        for e in elements:
            self.processed += 1
            yield e * 2


class Function2(pipeline.DoFn):
    def __init__(self):
        self.processed = 0

    def process(self, elements):
        for e in elements:
            self.processed += 1
            yield e * 3


class PipelineTestCase(unittest.TestCase):
    def test_dofn(self):
        fct_1 = Function1()
        fct_2 = Function2()

        fct_o = fct_1 | fct_2

        result = (fct_o.process(range(1, 10)))

        self.assertEqual(
            tuple(result),
            tuple(e * 2 * 3 for e in range(1, 10))
        )

        self.assertEqual(fct_1.processed, 9)
        self.assertEqual(fct_2.processed, 9)

    def test_performance(self):
        def perf():
            fct_1 = Function1()
            fct_2 = Function2()

            fct_o = fct_1 | fct_2
            x = tuple(fct_o.process(range(1, 100)))

        perf_n = 1000
        perf_t = timeit.timeit(perf, number=perf_n)
        logging.info(f"Pipeline Test Case Performance = {perf_t / perf_n}")
        self.assertLess(perf_t / perf_n, 1e-4)


if __name__ == '__main__':
    unittest.main()
