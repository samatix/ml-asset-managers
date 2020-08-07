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

import datetime as dt
import csv
import logging

from src.data.models import Tick


class ParseTickDataFn:
    """
    Parse the raw tick data events into a tick object
    """

    def __init__(self):
        self.ticks_counter = 0
        self.errors_parse_num = 0

    def process(self, element):
        try:
            row = list(csv.reader([element]))[0]
            self.ticks_counter += 1
            yield Tick(
                time=dt.datetime.strptime(
                    f"{row[0]},{row[1]}",
                    '%m/%d/%Y,%H:%M:%S'
                ),
                price=float(row[2]),
                bid=float(row[3]),
                ask=float(row[4]),
                quantity=float(row[5])
            )
        except:
            self.errors_parse_num += 1
            logging.error(f"Parsing error of element = {element}")
