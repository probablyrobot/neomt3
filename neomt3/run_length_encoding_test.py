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

"""Tests for run_length_encoding."""

import note_seq
import numpy as np
import tensorflow as tf
from absl.testing import absltest

from neomt3 import event_codec, run_length_encoding

codec = event_codec.Codec(
    event_types=["pitch", "velocity", "drum", "program", "tie"],
    event_ranges={
        "pitch": (note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
        "velocity": (0, 127),
        "drum": (note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
        "program": (note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM),
        "tie": (0, 0),
    },
)
run_length_encode_shifts = run_length_encoding.run_length_encode_shifts_fn(codec=codec)


class RunLengthEncodingTest(tf.test.TestCase):
    def test_remove_redundant_state_changes(self):
        og_dataset = tf.data.Dataset.from_tensors(
            {"targets": [3, 525, 356, 161, 2, 525, 356, 161, 355, 394]}
        )

        result = run_length_encoding.remove_redundant_state_changes_fn(
            codec=codec, state_change_event_types=["velocity", "program"]
        )(og_dataset)

        expected = tf.data.Dataset.from_tensors(
            {"targets": [3, 525, 356, 161, 2, 161, 355, 394]}
        )

        for actual, expected in zip(result, expected):
            self.assertAllEqual(actual["targets"], expected["targets"])

    def test_run_length_encode_shifts(self):
        og_dataset = tf.data.Dataset.from_tensors(
            {"targets": [1, 1, 1, 161, 1, 1, 1, 162, 1, 1, 1]}
        )

        result = run_length_encode_shifts(og_dataset)
        expected = tf.data.Dataset.from_tensors({"targets": [3, 161, 6, 162]})

        for actual, expected in zip(result, expected):
            self.assertAllEqual(actual["targets"], expected["targets"])

    def test_run_length_encode_shifts_beyond_max_length(self):
        og_dataset = tf.data.Dataset.from_tensors(
            {"targets": [1] * 202 + [161, 1, 1, 1]}
        )

        result = run_length_encode_shifts(og_dataset)
        expected = tf.data.Dataset.from_tensors({"targets": [100, 100, 2, 161]})

        for actual, expected in zip(result, expected):
            self.assertAllEqual(actual["targets"], expected["targets"])

    def test_run_length_encode_shifts_simultaneous(self):
        og_dataset = tf.data.Dataset.from_tensors(
            {"targets": [1, 1, 1, 161, 162, 1, 1, 1]}
        )

        result = run_length_encode_shifts(og_dataset)
        expected = tf.data.Dataset.from_tensors({"targets": [3, 161, 162]})

        for actual, expected in zip(result, expected):
            self.assertAllEqual(actual["targets"], expected["targets"])

    def test_merge_run_length_encoded_targets(self):
        # pylint: disable=bad-whitespace
        targets = np.array([[3, 161, 162, 5, 163], [160, 164, 3, 165, 0]])
        # pylint: enable=bad-whitespace
        merged_targets = run_length_encoding.merge_run_length_encoded_targets(
            targets=targets, codec=codec
        )
        expected_merged_targets = [160, 164, 3, 161, 162, 165, 5, 163]
        np.testing.assert_array_equal(expected_merged_targets, merged_targets)


if __name__ == "__main__":
    tf.test.main()
