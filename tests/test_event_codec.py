"""Tests for event codec functionality."""

import pytest
import tensorflow as tf
import numpy as np

from neomt3 import event_codec


def test_event_codec_initialization():
    """Test event codec initialization."""
    event_types = ['note', 'velocity', 'program']
    event_ranges = {
        'note': (0, 127),
        'velocity': (0, 31),
        'program': (0, 127)
    }
    
    codec = event_codec.Codec(event_types, event_ranges)
    
    assert codec.num_classes == 291  # 128 + 32 + 128 + 3 special tokens
    assert codec.pad_token == 0
    assert codec.sos_token == 1
    assert codec.eos_token == 2
    assert codec.num_special_tokens == 3


def test_event_encoding():
    """Test event encoding."""
    event_types = ['note', 'velocity']
    event_ranges = {
        'note': (0, 127),
        'velocity': (0, 31)
    }
    
    codec = event_codec.Codec(event_types, event_ranges)
    
    # Test note event
    note_event = event_codec.Event('note', 60)  # Middle C
    note_token = codec.encode_event(note_event)
    assert note_token == 63  # 60 + 3 (special tokens)
    
    # Test velocity event
    velocity_event = event_codec.Event('velocity', 16)
    velocity_token = codec.encode_event(velocity_event)
    assert velocity_token == 147  # 144 + 3 (special tokens) + 16 (velocity value)


def test_event_decoding():
    """Test event decoding."""
    event_types = ['note', 'velocity']
    event_ranges = {
        'note': (0, 127),
        'velocity': (0, 31)
    }
    
    codec = event_codec.Codec(event_types, event_ranges)
    
    # Test note event
    note_token = 63  # Middle C + offset
    note_event = codec.decode_event(note_token)
    assert note_event.type == 'note'
    assert note_event.value == 60
    
    # Test velocity event
    velocity_token = 147  # Velocity 16 + offset
    velocity_event = codec.decode_event(velocity_token)
    assert velocity_event.type == 'velocity'
    assert velocity_event.value == 16
    
    # Test special tokens
    assert codec.decode_event(0) is None  # PAD
    assert codec.decode_event(1) is None  # SOS
    assert codec.decode_event(2) is None  # EOS


def test_event_type_range():
    """Test event type range retrieval."""
    event_types = ['note', 'velocity']
    event_ranges = {
        'note': (0, 127),
        'velocity': (0, 31)
    }
    
    codec = event_codec.Codec(event_types, event_ranges)
    
    assert codec.event_type_range('note') == (0, 127)
    assert codec.event_type_range('velocity') == (0, 31)
    
    with pytest.raises(ValueError):
        codec.event_type_range('invalid_type') 