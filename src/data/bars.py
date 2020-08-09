# Copyright 2020 Ayoub ENNASSIRI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

from src.runner import pipeline


class TickBarFn(pipeline.DoFn):
    """
    Parse the tick objects into Tick Bars
    """

    def __init__(self, threshold=10):
        """
        Tick Bar Function
        :param threshold: The number of ticks at which we extract the bid ask
        :type threshold: int
        """
        self.ticks_processed = 0
        self.buffer = 0
        self.threshold = threshold

    def process(self, elements):
        for e in elements:
            self.buffer += 1
            self.ticks_processed += 1
            if self.buffer == self.threshold:
                self.buffer = 0
                yield e


class VolumeBarFn(pipeline.DoFn):
    """
    Parse the tick objects into volume bars
    """

    def __init__(self, threshold=10000):
        """
        Volume Bar Function
        :param threshold: The accumulated volume threshold at which we extract
        the bid ask
        :type threshold: float
        """
        self.ticks_processed = 0
        self.buffer = 0
        self.threshold = threshold

    def process(self, elements):
        for e in elements:
            self.buffer += e.quantity * e.price
            self.ticks_processed += 1
            if self.buffer >= self.threshold:
                self.buffer = 0
                yield e
