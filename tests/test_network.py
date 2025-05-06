"""Tests for network functionality."""

import pytest
import tensorflow as tf
import numpy as np

from neomt3 import network


def test_mt3_model_initialization():
    """Test MT3 model initialization."""
    vocab_size = 512
    model = network.MT3Model(vocab_size=vocab_size)
    
    assert model.vocab_size == vocab_size
    assert model.num_layers == 6
    assert model.num_heads == 8
    assert model.d_model == 512
    assert model.dff == 2048
    assert model.dropout_rate == 0.1


def test_mt3_model_call():
    """Test MT3 model forward pass."""
    batch_size = 2
    seq_length = 100
    input_dim = 229  # num_mel_bins
    vocab_size = 512
    
    # Create model
    model = network.MT3Model(vocab_size=vocab_size)
    
    # Create input
    inputs = tf.random.normal((batch_size, seq_length, input_dim))
    
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Check output shape
    assert outputs.shape == (batch_size, seq_length, vocab_size)
    
    # Check values
    assert tf.reduce_all(tf.math.is_finite(outputs))


def test_mt3_model_generate():
    """Test MT3 model generation."""
    batch_size = 2
    seq_length = 100
    input_dim = 229  # num_mel_bins
    vocab_size = 512
    max_length = 50
    
    # Create model
    model = network.MT3Model(vocab_size=vocab_size)
    
    # Create input
    inputs = tf.random.normal((batch_size, seq_length, input_dim))
    
    # Generate sequence
    output = model.generate(
        inputs=inputs,
        max_length=max_length,
        temperature=1.0,
        top_k=0,
        top_p=0.0
    )
    
    # Check output shape
    assert output.shape[0] == batch_size
    assert output.shape[1] <= max_length
    
    # Check values
    assert tf.reduce_all(output >= 0)  # Token IDs should be non-negative
    assert tf.reduce_all(output < vocab_size)  # Token IDs should be less than vocab size


def test_positional_encoding():
    """Test positional encoding computation."""
    vocab_size = 512
    model = network.MT3Model(vocab_size=vocab_size)
    
    # Get positional encoding
    pos_encoding = model._get_positional_encoding()
    
    # Check shape
    assert pos_encoding.shape == (1, 1000, model.d_model)
    
    # Check values
    assert tf.reduce_all(tf.math.is_finite(pos_encoding))
    
    # Check that sin/cos pattern is present
    # The first dimension should alternate between sin and cos
    sin_values = pos_encoding[0, :, 0::2]
    cos_values = pos_encoding[0, :, 1::2]
    
    # Check that sin values are between -1 and 1
    assert tf.reduce_all(sin_values >= -1) and tf.reduce_all(sin_values <= 1)
    
    # Check that cos values are between -1 and 1
    assert tf.reduce_all(cos_values >= -1) and tf.reduce_all(cos_values <= 1) 