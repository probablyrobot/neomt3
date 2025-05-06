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

"""Vocabulary definitions for MT3."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from neomt3 import event_codec


class VocabularyConfig:
    """Configuration for vocabulary."""

    def __init__(
        self,
        num_velocity_bins: int = 32,
        onsets_only: bool = False,
        include_ties: bool = True,
    ):
        self.num_velocity_bins = num_velocity_bins
        self.onsets_only = onsets_only
        self.include_ties = include_ties


def build_codec(vocab_config: VocabularyConfig) -> event_codec.Codec:
    """Build event codec from vocabulary configuration.

    Args:
        vocab_config: Vocabulary configuration

    Returns:
        Event codec
    """
    # Build event types
    event_types = ["note"]
    if not vocab_config.onsets_only:
        event_types.extend(["velocity", "program"])
    if vocab_config.include_ties:
        event_types.append("tie")

    # Build event ranges
    event_ranges = {
        "note": (0, 127),  # MIDI note range
        "velocity": (0, vocab_config.num_velocity_bins - 1),
        "program": (0, 127),  # MIDI program range
        "tie": (0, 0),  # Tie is a binary event
    }

    # Build codec
    return event_codec.Codec(event_types=event_types, event_ranges=event_ranges)


def vocabulary_from_codec(codec: event_codec.Codec) -> Dict[str, Any]:
    """Create vocabulary from codec.

    Args:
        codec: Event codec

    Returns:
        Vocabulary dictionary
    """
    # Build vocabulary
    vocab = {
        "vocab_size": codec.num_classes,
        "eos_id": codec.eos_id,
        "pad_id": codec.pad_id,
        "unk_id": codec.unk_id,
    }

    return vocab


def velocity_to_bin(velocity: int, num_velocity_bins: int) -> int:
    """Convert a velocity value to a bin index.

    Args:
        velocity: MIDI velocity value (0-127)
        num_velocity_bins: Number of velocity bins

    Returns:
        Bin index (0 to num_velocity_bins-1)
    """
    if not 0 <= velocity <= 127:
        raise ValueError(f"Velocity must be between 0 and 127, got {velocity}")
    if num_velocity_bins < 1:
        raise ValueError(
            f"Number of velocity bins must be positive, got {num_velocity_bins}"
        )

    # Map velocity to bin
    if num_velocity_bins == 1:
        return 0
    else:
        # Scale velocity to bin range
        return min(int(velocity * num_velocity_bins / 128), num_velocity_bins - 1)


def bin_to_velocity(bin_index: int, num_velocity_bins: int) -> int:
    """Convert a bin index to a velocity value.

    Args:
        bin_index: Bin index (0 to num_velocity_bins-1)
        num_velocity_bins: Number of velocity bins

    Returns:
        MIDI velocity value (0-127)
    """
    if not 0 <= bin_index < num_velocity_bins:
        raise ValueError(
            f"Bin index must be between 0 and {num_velocity_bins-1}, got {bin_index}"
        )
    if num_velocity_bins < 1:
        raise ValueError(
            f"Number of velocity bins must be positive, got {num_velocity_bins}"
        )

    # Map bin to velocity
    if num_velocity_bins == 1:
        return 64  # Default velocity for single bin
    else:
        # Scale bin to velocity range
        return min(int(bin_index * 128 / num_velocity_bins), 127)


def num_velocity_bins_from_codec(codec: event_codec.Codec) -> int:
    """Get the number of velocity bins from a codec.

    Args:
        codec: Event codec

    Returns:
        Number of velocity bins
    """
    if "velocity" not in codec.event_ranges:
        return 1
    min_value, max_value = codec.event_ranges["velocity"]
    return max_value - min_value + 1


class GenericTokenVocabulary:
    """Generic token vocabulary for encoding and decoding tokens."""

    def __init__(self, vocab_size: int, extra_ids: int = 0):
        """Initialize the vocabulary.

        Args:
            vocab_size: Size of the vocabulary
            extra_ids: Number of extra IDs to add
        """
        self.vocab_size = vocab_size
        self.extra_ids = extra_ids
        self.eos_id = 1
        self.pad_id = 0
        self.unk_id = 2
        self.class_token_start = 3
        self.class_token_end = self.class_token_start + vocab_size
        self.extra_token_start = self.class_token_end
        self.extra_token_end = self.extra_token_start + extra_ids

    def encode(self, values: List[int]) -> List[int]:
        """Encode values to tokens.

        Args:
            values: List of values to encode

        Returns:
            List of encoded tokens
        """
        if not all(0 <= value < self.vocab_size for value in values):
            raise ValueError(f"Values must be between 0 and {self.vocab_size-1}")
        return [value + self.class_token_start for value in values]

    def encode_tf(self, values: tf.Tensor) -> tf.Tensor:
        """Encode values to tokens using TensorFlow.

        Args:
            values: Tensor of values to encode

        Returns:
            Tensor of encoded tokens
        """
        # Preserve input dtype
        dtype = values.dtype
        # Add class token offset
        encoded = tf.cast(values, tf.int32) + self.class_token_start
        # Handle invalid values
        invalid_mask = tf.logical_or(values < 0, values >= self.vocab_size)
        encoded = tf.where(invalid_mask, self.unk_id, encoded)
        return tf.cast(encoded, dtype)

    def decode(self, tokens: List[int]) -> List[int]:
        """Decode tokens to values.

        Args:
            tokens: List of tokens to decode

        Returns:
            List of decoded values
        """
        result = []
        for token in tokens:
            if token == self.eos_id:
                result.append(-1)  # EOS token
                break
            elif token == self.pad_id or token == self.unk_id:
                result.append(-2)  # PAD/UNK token
            elif token < self.class_token_start or token >= self.class_token_end:
                result.append(-2)  # Invalid token
            else:
                result.append(token - self.class_token_start)
        return result

    def decode_tf(self, tokens: tf.Tensor) -> tf.Tensor:
        """Decode tokens to values using TensorFlow.

        Args:
            tokens: Tensor of tokens to decode

        Returns:
            Tensor of decoded values
        """
        # Create masks for special tokens
        is_eos = tf.equal(tokens, self.eos_id)
        is_pad = tf.equal(tokens, self.pad_id)
        is_unk = tf.equal(tokens, self.unk_id)
        is_invalid = tf.logical_or(
            tokens < self.class_token_start, tokens >= self.class_token_end
        )
        is_special = tf.logical_or(
            tf.logical_or(is_pad, is_eos), tf.logical_or(is_unk, is_invalid)
        )

        # Decode tokens
        decoded = tf.where(
            is_special,
            tf.where(
                is_eos,
                tf.constant(-1, dtype=tokens.dtype),
                tf.constant(-2, dtype=tokens.dtype),
            ),
            tokens - self.class_token_start,
        )

        # Handle EOS token
        eos_mask = tf.cumsum(tf.cast(is_eos, tf.int32))
        decoded = tf.where(tf.equal(eos_mask, 0), decoded, -1)

        return decoded
