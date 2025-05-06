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

"""Tests for event_codec."""

from absl.testing import absltest

from neomt3 import event_codec

Event = event_codec.Event
EventRange = event_codec.EventRange


class EventCodecTest(absltest.TestCase):

    def test_encode_decode(self):
        ec = event_codec.Codec(
            event_types=["pitch", "shift"],
            event_ranges={"pitch": (0, 127), "shift": (0, 100)},
            max_shift_steps=100,
            steps_per_second=100,
        )
        events = [
            Event(type="pitch", value=60),
            Event(type="shift", value=5),
            Event(type="pitch", value=62),
        ]
        encoded = [ec.encode_event(e) for e in events]
        self.assertSequenceEqual([60, 5, 62], encoded)

        decoded = [ec.decode_event(idx) for idx in encoded]
        self.assertSequenceEqual(events, decoded)

    def test_shift_steps(self):
        ec = event_codec.Codec(
            event_types=["pitch", "shift"],
            event_ranges={"pitch": (0, 127), "shift": (0, 100)},
            max_shift_steps=100,
            steps_per_second=100,
        )

        self.assertEqual(100, ec.max_shift_steps)
        self.assertFalse(ec.is_shift_event_index(-1))
        self.assertTrue(ec.is_shift_event_index(0))
        self.assertTrue(ec.is_shift_event_index(100))
        self.assertFalse(ec.is_shift_event_index(101))


if __name__ == "__main__":
    absltest.main()
