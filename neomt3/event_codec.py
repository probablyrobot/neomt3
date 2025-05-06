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

"""Event codec for encoding and decoding events."""

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple


@dataclasses.dataclass
class Event:
    """Event class for encoding and decoding."""

    type: str
    value: int


@dataclasses.dataclass
class EventRange:
    """Event range class for defining valid ranges of event values."""

    min_value: int
    max_value: int

    def __post_init__(self):
        """Validate the range."""
        if self.min_value > self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) must be less than or equal to max_value ({self.max_value})"
            )


class Codec:
    """Event codec for encoding and decoding events."""

    def __init__(
        self, event_types: Sequence[str], event_ranges: Dict[str, Tuple[int, int]]
    ):
        """Initialize the event codec.

        Args:
            event_types: List of event types in order
            event_ranges: Dictionary mapping event types to (min, max) ranges
        """
        self.event_types = event_types
        self.event_ranges = event_ranges

        # Special tokens
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.num_special_tokens = 3

        # Calculate token ranges for each event type
        self.token_ranges = {}
        current_offset = 0
        for event_type in event_types:
            min_val, max_val = event_ranges[event_type]
            num_values = max_val - min_val + 1
            self.token_ranges[event_type] = (min_val, max_val)
            current_offset += num_values

        self.num_classes = current_offset + self.num_special_tokens

    def encode_event(self, event: Event) -> int:
        """Encode an event to a token.

        Args:
            event: Event to encode

        Returns:
            Token ID
        """
        if event.type not in self.event_types:
            raise ValueError(f"Unknown event type: {event.type}")

        # Calculate offset for this event type
        offset = self.num_special_tokens
        for event_type in self.event_types:
            if event_type == event.type:
                break
            min_val, max_val = self.event_ranges[event_type]
            offset += max_val - min_val + 1

        return offset + event.value

    def decode_event(self, token: int) -> Optional[Event]:
        """Decode a token to an event.

        Args:
            token: Token ID

        Returns:
            Event or None for special tokens
        """
        if token < self.num_special_tokens:
            return None

        # Find event type and value
        current_token = self.num_special_tokens
        for event_type in self.event_types:
            min_val, max_val = self.event_ranges[event_type]
            num_values = max_val - min_val + 1
            if token < current_token + num_values:
                value = token - current_token + min_val
                return Event(event_type, value)
            current_token += num_values

        return None

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """Get the token range for an event type.

        Args:
            event_type: Event type string

        Returns:
            Tuple of (start_id, end_id) for the token range
        """
        if event_type not in self.event_types:
            raise ValueError(f"Unknown event type: {event_type}")

        return self.event_ranges[event_type]
