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

import logging
import os
import tempfile
import unittest

import pandas as pd

from src.data.parse import ParseTickDataFn
from src.testing.fixtures import BaseTestCase


class TickDataSetTest(BaseTestCase):
    def setUp(self):
        self.test_files = {
            'input_file': self.generate_temp_file(contents=self.TICK_DATA),
            'output_file': self.generate_temp_file()
        }

    def tearDown(self):
        for test_file in self.test_files.values():
            if os.path.exists(test_file):
                os.remove(test_file)

    def generate_temp_file(self, contents=None):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            if contents is not None:
                temp_file.write(contents.encode('utf-8'))
            return temp_file.name

    def test_output_file(self):
        with open(self.test_files['input_file'], 'r') as file_input:
            data_raw = (ParseTickDataFn().process(file_input.readlines()))

        self.assertEqual(list(data_raw), self.TICK_DATA_PARSED)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
