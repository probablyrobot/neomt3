# Copyright 2024 The MT3 Authors.
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

"""Base module for MT3."""

from neomt3 import (
    datasets,
    event_codec,
    inference,
    layers,
    metrics,
    metrics_utils,
    models,
    network,
    note_sequences,
    preprocessors,
    run_length_encoding,
    spectrograms,
    summaries,
    tasks,
    vocabularies,
)
from neomt3.version import __version__
