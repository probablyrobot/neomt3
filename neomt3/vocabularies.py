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

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import tensorflow as tf
import numpy as np

from neomt3 import event_codec


class VocabularyConfig:
    """Configuration for vocabulary."""
    def __init__(
        self,
        num_velocity_bins: int = 32,
        onsets_only: bool = False,
        include_ties: bool = True
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
    event_types = ['note']
    if not vocab_config.onsets_only:
        event_types.extend(['velocity', 'program'])
    if vocab_config.include_ties:
        event_types.append('tie')
    
    # Build event ranges
    event_ranges = {
        'note': (0, 127),  # MIDI note range
        'velocity': (0, vocab_config.num_velocity_bins - 1),
        'program': (0, 127),  # MIDI program range
        'tie': (0, 0)  # Tie is a binary event
    }
    
    # Build codec
    return event_codec.Codec(
        event_types=event_types,
        event_ranges=event_ranges
    )


def vocabulary_from_codec(codec: event_codec.Codec) -> Dict[str, Any]:
    """Create vocabulary from codec.
    
    Args:
        codec: Event codec
        
    Returns:
        Vocabulary dictionary
    """
    # Build vocabulary
    vocab = {
        'vocab_size': codec.num_classes,
        'eos_id': codec.eos_id,
        'pad_id': codec.pad_id,
        'unk_id': codec.unk_id
    }
    
    return vocab
