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

class DoFn:
    def process(self, element):
        raise NotImplemented

    def __or__(self, other):
        assert issubclass(other.__class__, DoFn)

        class MixDoFn(DoFn):
            def process(self_, elements):
                yield from other.process(self.process(elements))

        return MixDoFn()
