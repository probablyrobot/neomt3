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

import tensorflow as tf
import numpy as np

from neomt3 import event_codec
from neomt3 import vocabularies


class EventEncodingSpec:
    """Base class for event encoding specifications."""
    
    def __init__(self, 
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
                 flush_decoding_state_fn: Optional[Callable[[Any], Any]] = None):
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


def encode_events(
    events: List[Dict[str, Any]],
    codec: event_codec.Codec
) -> tf.Tensor:
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
    vocab_config: vocabularies.VocabularyConfig,
    frame_times: Optional[tf.Tensor] = None,
    sequence_length: Optional[tf.Tensor] = None
) -> List[Dict[str, Any]]:
    """Decode tokens to events.
    
    Args:
        tokens: Tensor of tokens
        codec: Event codec for decoding
        vocab_config: Vocabulary configuration
        frame_times: Optional frame times tensor
        sequence_length: Optional sequence length tensor
        
    Returns:
        List of decoded events
    """
    # Convert to numpy for easier processing
    tokens = tokens.numpy()
    
    # Truncate if sequence length is provided
    if sequence_length is not None:
        tokens = tokens[:sequence_length]
    
    # Decode each token
    events = []
    for token in tokens:
        event = codec.decode_event(token)
        if event is not None:
            events.append(event)
    
    return events


def run_length_encode_shifts(
    tokens: tf.Tensor,
    codec: event_codec.Codec
) -> tf.Tensor:
    """Run-length encode shift tokens.
    
    Args:
        tokens: Tensor of tokens
        codec: Event codec for encoding
        
    Returns:
        Tensor of run-length encoded tokens
    """
    # Convert to numpy for easier processing
    tokens = tokens.numpy()
    
    # Get shift token range
    shift_start, shift_end = codec.event_type_range('shift')
    
    # Run-length encode shifts
    encoded = []
    i = 0
    while i < len(tokens):
        if shift_start <= tokens[i] <= shift_end:
            # Count consecutive shifts
            count = 1
            while (i + count < len(tokens) and
                   shift_start <= tokens[i + count] <= shift_end):
                count += 1
            
            # Add encoded shift
            encoded.append(tokens[i] + count - 1)
            i += count
        else:
            # Add non-shift token
            encoded.append(tokens[i])
            i += 1
    
    return tf.convert_to_tensor(encoded, dtype=tf.int32)


def run_length_decode_shifts(
    tokens: tf.Tensor,
    codec: event_codec.Codec
) -> tf.Tensor:
    """Run-length decode shift tokens.
    
    Args:
        tokens: Tensor of tokens
        codec: Event codec for decoding
        
    Returns:
        Tensor of run-length decoded tokens
    """
    # Convert to numpy for easier processing
    tokens = tokens.numpy()
    
    # Get shift token range
    shift_start, shift_end = codec.event_type_range('shift')
    
    # Run-length decode shifts
    decoded = []
    for token in tokens:
        if shift_start <= token <= shift_end:
            # Decode shift
            count = token - shift_start + 1
            decoded.extend([shift_start] * count)
        else:
            # Add non-shift token
            decoded.append(token)
    
    return tf.convert_to_tensor(decoded, dtype=tf.int32)
