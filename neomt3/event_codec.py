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

"""Event codec functionality for MT3."""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Remove circular imports
# from neomt3 import event_codec, vocabularies


@dataclasses.dataclass
class Event:
    """Event class for encoding and decoding events."""

    type: str
    value: int

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.type == other.type and self.value == other.value


@dataclasses.dataclass
class EventRange:
    """Event range class for defining valid ranges of event values."""

    type: str
    min_value: int
    max_value: int

    def __post_init__(self):
        """Validate event range values."""
        if not isinstance(self.type, str):
            raise TypeError("type must be a string")
        if not isinstance(self.min_value, (int, np.integer)):
            raise TypeError("min_value must be an integer")
        if not isinstance(self.max_value, (int, np.integer)):
            raise TypeError("max_value must be an integer")
        if self.min_value > self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) must be <= max_value ({self.max_value})"
            )


@dataclasses.dataclass
class Codec:
    """Event codec class for encoding and decoding events."""

    event_types: List[str]
    event_ranges: Dict[str, Tuple[int, int]]
    max_shift_steps: Optional[int] = None
    steps_per_second: Optional[int] = None

    def __post_init__(self):
        """Validate event ranges."""
        if not isinstance(self.event_types, (list, tuple)):
            raise TypeError("event_types must be a list or tuple")
        if not isinstance(self.event_ranges, dict):
            raise TypeError("event_ranges must be a dictionary")
        if not all(isinstance(t, str) for t in self.event_types):
            raise TypeError("event_types must be strings")
        if not all(isinstance(k, str) for k in self.event_ranges.keys()):
            raise TypeError("event_ranges keys must be strings")
        if not all(
            isinstance(v, tuple) and len(v) == 2 for v in self.event_ranges.values()
        ):
            raise TypeError("event_ranges values must be tuples of length 2")
        if not all(
            isinstance(v[0], int) and isinstance(v[1], int)
            for v in self.event_ranges.values()
        ):
            raise TypeError("event_ranges values must be tuples of integers")

    @property
    def pad_token(self) -> int:
        """Get the pad token value."""
        return 0

    @property
    def sos_token(self) -> int:
        """Get the start-of-sequence token value."""
        return 1

    @property
    def eos_token(self) -> int:
        """Get the end-of-sequence token value."""
        return 2

    @property
    def num_special_tokens(self) -> int:
        """Get the number of special tokens."""
        return 3

    @property
    def num_classes(self) -> int:
        """Get the total number of possible event values including special tokens."""
        total = self.num_special_tokens
        for min_val, max_val in self.event_ranges.values():
            total += max_val - min_val + 1
        return total

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """Get the valid range for an event type.

        Args:
            event_type: Type of event to get range for

        Returns:
            Tuple of (min_value, max_value) for the event type
        """
        if event_type not in self.event_ranges:
            raise ValueError(f"Unknown event type: {event_type}")
        return self.event_ranges[event_type]

    def _get_event_offset(self, event_type: str) -> int:
        """Get the token offset for an event type.

        Args:
            event_type: Type of event to get offset for

        Returns:
            Token offset for the event type
        """
        offset = 3  # Start with special tokens offset
        for t in self.event_types:
            if t == event_type:
                break
            min_val, max_val = self.event_ranges[t]
            offset += max_val - min_val + 1
        return offset

    def encode_event(self, event: Event) -> int:
        """Encode an event to a token.

        Args:
            event: Event to encode

        Returns:
            Token value
        """
        if event.type not in self.event_ranges:
            raise ValueError(f"Unknown event type: {event.type}")
        min_value, max_value = self.event_ranges[event.type]
        if not min_value <= event.value <= max_value:
            raise ValueError(
                f"Event value {event.value} out of range [{min_value}, {max_value}]"
            )
        offset = self._get_event_offset(event.type)
        return offset + (event.value - min_value)

    def decode_event(self, token: int) -> Optional[Event]:
        """Decode a token to an event.

        Args:
            token: Token to decode

        Returns:
            Decoded event or None if token is invalid
        """
        # Handle special tokens
        if token < 3:
            return None

        current_offset = 3
        for event_type in self.event_types:
            min_value, max_value = self.event_ranges[event_type]
            range_size = max_value - min_value + 1
            if token < current_offset + range_size:
                value = min_value + (token - current_offset)
                return Event(type=event_type, value=value)
            current_offset += range_size
        return None

    def is_shift_event_index(self, token: int) -> bool:
        """Check if a token represents a shift event.

        Args:
            token: Token to check

        Returns:
            True if token represents a shift event
        """
        if "shift" not in self.event_ranges:
            return False
        min_value, max_value = self.event_ranges["shift"]
        if not min_value <= token <= max_value:
            return False
        # Check if any later event type in event_types claims this token
        shift_index = self.event_types.index("shift")
        for event_type in self.event_types[shift_index + 1 :]:
            min_value, max_value = self.event_ranges[event_type]
            if min_value <= token <= max_value:
                return False
        return True
