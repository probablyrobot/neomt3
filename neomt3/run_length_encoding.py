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

"""Run-length encoding functionality for MT3."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from neomt3 import event_codec, vocabularies


class EventEncodingSpec:
    """Base class for event encoding specifications."""

    def __init__(
        self,
        num_velocity_bins: int = 32,
        max_shift_steps: int = 100,
        use_program_tokens: bool = True,
        use_velocity_tokens: bool = True,
        init_encoding_state_fn: Optional[Callable[[], Any]] = None,
        encode_event_fn: Optional[Callable[[Any, Any, Any], Sequence[Any]]] = None,
        encoding_state_to_events_fn: Optional[Callable[[Any], Sequence[Any]]] = None,
        init_decoding_state_fn: Optional[Callable[[], Any]] = None,
        begin_decoding_segment_fn: Optional[Callable[[Any], None]] = None,
        decode_event_fn: Optional[Callable[[Any, float, Any, Any], None]] = None,
        flush_decoding_state_fn: Optional[Callable[[Any], Any]] = None,
    ):
        """Initialize the event encoding specification.

        Args:
            num_velocity_bins: Number of velocity bins
            max_shift_steps: Maximum number of shift steps
            use_program_tokens: Whether to use program tokens
            use_velocity_tokens: Whether to use velocity tokens
            init_encoding_state_fn: Function to initialize encoding state
            encode_event_fn: Function to encode events
            encoding_state_to_events_fn: Function to convert encoding state to events
            init_decoding_state_fn: Function to initialize decoding state
            begin_decoding_segment_fn: Function to begin decoding a segment
            decode_event_fn: Function to decode events
            flush_decoding_state_fn: Function to flush decoding state
        """
        self.num_velocity_bins = num_velocity_bins
        self.max_shift_steps = max_shift_steps
        self.use_program_tokens = use_program_tokens
        self.use_velocity_tokens = use_velocity_tokens
        self.init_encoding_state_fn = init_encoding_state_fn
        self.encode_event_fn = encode_event_fn
        self.encoding_state_to_events_fn = encoding_state_to_events_fn
        self.init_decoding_state_fn = init_decoding_state_fn
        self.begin_decoding_segment_fn = begin_decoding_segment_fn
        self.decode_event_fn = decode_event_fn
        self.flush_decoding_state_fn = flush_decoding_state_fn

    def encode_event(self, event: Dict[str, Any]) -> int:
        """Encode an event to a token.

        Args:
            event: Event dictionary

        Returns:
            Token ID
        """
        raise NotImplementedError

    def decode_event(self, token: int) -> Optional[Dict[str, Any]]:
        """Decode a token to an event.

        Args:
            token: Token ID

        Returns:
            Event dictionary or None for special tokens
        """
        raise NotImplementedError

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """Get the token range for an event type.

        Args:
            event_type: Event type string

        Returns:
            Tuple of (start_id, end_id) for the token range
        """
        raise NotImplementedError


def encode_events(events: List[Dict[str, Any]], codec: event_codec.Codec) -> tf.Tensor:
    """Encode events to tokens.

    Args:
        events: List of events
        codec: Event codec for encoding

    Returns:
        Tensor of encoded tokens
    """
    # Encode each event
    tokens = []
    for event in events:
        token = codec.encode_event(event)
        tokens.append(token)

    return tf.convert_to_tensor(tokens, dtype=tf.int32)


def decode_events(
    tokens: tf.Tensor,
    codec: event_codec.Codec,
    state: Optional[Any] = None,
    start_time: float = 0.0,
    max_time: Optional[float] = None,
    decode_event_fn: Optional[Callable[[Any, float, Any, Any], None]] = None,
) -> Tuple[int, int]:
    """Decode tokens to events.

    Args:
        tokens: Tensor of tokens
        codec: Event codec for decoding
        state: Optional decoding state
        start_time: Start time for decoding
        max_time: Optional maximum time for decoding
        decode_event_fn: Optional function to decode events

    Returns:
        Tuple of (invalid_ids, dropped_events)
    """
    # Convert to numpy for easier processing
    tokens = tokens.numpy() if isinstance(tokens, tf.Tensor) else tokens

    # Track statistics
    invalid_ids = 0
    dropped_events = 0

    # Process each token
    current_time = start_time
    for token in tokens:
        event = codec.decode_event(token)
        if event is None:
            invalid_ids += 1
            continue

        # Check if we've exceeded max time
        if max_time is not None and current_time > max_time:
            dropped_events += 1
            continue

        # Decode event if function provided
        if decode_event_fn is not None:
            decode_event_fn(state, current_time, event, codec)

    return invalid_ids, dropped_events


def merge_run_length_encoded_targets(
    targets: np.ndarray,
    codec: event_codec.Codec,
) -> np.ndarray:
    """Merge run-length encoded targets.

    Args:
        targets: Array of run-length encoded targets
        codec: Event codec for encoding

    Returns:
        Array of merged targets
    """
    # Convert to list for easier manipulation
    merged = []
    for target in targets:
        # Remove padding tokens
        target = target[target != 0]
        if not target.size:
            continue
        # Add non-padding tokens to merged list
        merged.extend(target)

    return np.array(merged, dtype=np.int32)


def run_length_encode_shifts(
    tokens: tf.Tensor,
    codec: event_codec.Codec,
) -> tf.Tensor:
    """Run length encode shifts in a sequence of tokens.

    Args:
        tokens: Tensor of tokens
        codec: Event codec for encoding

    Returns:
        Tensor of run-length encoded tokens
    """
    # Convert to numpy for easier processing
    tokens = tokens.numpy() if isinstance(tokens, tf.Tensor) else tokens

    # Initialize result list
    result = []
    current_count = 0

    # Process each token
    for token in tokens:
        if codec.is_shift_event_index(token):
            current_count += 1
            if current_count == codec.max_shift_steps:
                result.append(current_count)
                current_count = 0
        else:
            if current_count > 0:
                result.append(current_count)
                current_count = 0
            result.append(token)

    # Add any remaining count
    if current_count > 0:
        result.append(current_count)

    return tf.convert_to_tensor(result, dtype=tf.int32)


def run_length_decode_shifts(
    tokens: tf.Tensor,
    codec: event_codec.Codec,
) -> tf.Tensor:
    """Run length decode shifts in a sequence of tokens.

    Args:
        tokens: Tensor of run-length encoded tokens
        codec: Event codec for encoding

    Returns:
        Tensor of decoded tokens
    """
    # Convert to numpy for easier processing
    tokens = tokens.numpy() if isinstance(tokens, tf.Tensor) else tokens

    # Initialize result list
    result = []

    # Process each token
    for token in tokens:
        if codec.is_shift_event_index(token):
            # This is a count of shift tokens
            result.extend([1] * token)
        else:
            # This is a non-shift token
            result.append(token)

    return tf.convert_to_tensor(result, dtype=tf.int32)


def encode_and_index_events(
    state: Optional[Any],
    event_times: np.ndarray,
    event_values: np.ndarray,
    encode_event_fn: Callable[[Any, Any, Any], Sequence[Any]],
    codec: event_codec.Codec,
    frame_times: np.ndarray,
    encoding_state_to_events_fn: Optional[Callable[[Any], Sequence[Any]]] = None,
) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    """Encode events and index them to frames.

    Args:
        state: Optional encoding state
        event_times: Array of event times
        event_values: Array of event values
        encode_event_fn: Function to encode events
        codec: Event codec for encoding
        frame_times: Array of frame times
        encoding_state_to_events_fn: Optional function to convert encoding state to events

    Returns:
        Tuple of (event_indices, event_values, event_times, frame_indices, frame_times)
    """
    # Initialize lists to store results
    event_indices = []
    event_values_list = []
    event_times_list = []
    frame_indices = []
    frame_times_list = []

    # Process each event
    for i, (time, value) in enumerate(zip(event_times, event_values)):
        # Encode event
        events = encode_event_fn(state, value, codec)
        if not events:
            continue

        # Add events to lists
        for event in events:
            event_indices.append(i)
            event_values_list.append(event.value)
            event_times_list.append(time)

    # Add state events if function provided
    if state is not None and encoding_state_to_events_fn is not None:
        state_events = encoding_state_to_events_fn(state)
        for event in state_events:
            event_indices.append(len(event_times))
            event_values_list.append(event.value)
            event_times_list.append(frame_times[-1])

    # Index events to frames
    event_times_array = np.array(event_times_list)
    for i, frame_time in enumerate(frame_times):
        # Find events that occur before or at this frame
        event_mask = event_times_array <= frame_time
        if not np.any(event_mask):
            continue

        # Add frame index and time
        frame_indices.append(i)
        frame_times_list.append(frame_time)

    return (
        event_indices,
        event_values_list,
        event_times_list,
        frame_indices,
        frame_times_list,
    )


def remove_redundant_state_changes_fn(
    codec: event_codec.Codec,
    state_change_event_types: List[str],
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    """Create a function to remove redundant state changes.

    Args:
        codec: Event codec for encoding
        state_change_event_types: List of event types that represent state changes

    Returns:
        Function that removes redundant state changes from a dataset
    """

    def _process(example):
        tokens = example["targets"]
        result = []
        current_state = {}

        for token in tokens:
            event = codec.decode_event(token)
            if event is None:
                result.append(token)
                continue

            if event.type in state_change_event_types:
                if (
                    event.type not in current_state
                    or current_state[event.type] != event.value
                ):
                    current_state[event.type] = event.value
                    result.append(token)
            else:
                result.append(token)

        return {"targets": tf.convert_to_tensor(result, dtype=tf.int32)}

    def _remove_redundant_state_changes(dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(_process)

    return _remove_redundant_state_changes
