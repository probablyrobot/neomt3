"""Tests for spectrogram functionality."""

import numpy as np
import pytest
import tensorflow as tf

from neomt3 import spectrograms


def test_spectrogram_config():
    """Test spectrogram configuration."""
    config = spectrograms.SpectrogramConfig()

    assert config.sample_rate == 16000
    assert config.hop_width == 512
    assert config.num_mel_bins == 229
    assert config.fft_size == 2048
    assert config.window_size == 2048
    assert config.mel_min_hz == 30.0
    assert config.mel_max_hz == 8000.0
    assert config.clip_min_value == 1e-5


def test_compute_spectrogram():
    """Test spectrogram computation."""
    # Create a simple sine wave
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)

    # Convert to tensor
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Create config
    config = spectrograms.SpectrogramConfig(
        sample_rate=sample_rate,
        hop_width=512,
        num_mel_bins=229,
        fft_size=2048,
        window_size=2048,
    )

    # Compute spectrogram
    spec = spectrograms.compute_spectrogram(audio_tensor, config)

    # Check shape
    # Without padding, we get (samples - window_size) / hop_width + 1 frames
    expected_frames = int(
        (sample_rate * duration - config.window_size) / config.hop_width + 1
    )
    assert spec.shape == (expected_frames, config.num_mel_bins)

    # Check values
    assert tf.reduce_all(tf.math.is_finite(spec))
    assert tf.reduce_all(
        spec >= np.log(config.clip_min_value)
    )  # Log-mel spectrogram should be >= log(clip_min_value)


def test_compute_frame_times():
    """Test frame times computation."""
    num_frames = 100
    hop_width = 512
    sample_rate = 16000

    # Compute frame times
    frame_times = spectrograms.compute_frame_times(num_frames, hop_width, sample_rate)

    # Check shape
    assert frame_times.shape == (num_frames,)

    # Check values
    expected_times = np.arange(num_frames) * hop_width / sample_rate
    np.testing.assert_allclose(frame_times.numpy(), expected_times)


def test_flatten_frames():
    """Test frame flattening."""
    # Create dummy frames
    num_frames = 10
    frame_size = 512
    num_features = 229
    frames = tf.random.normal((num_frames, num_features))
    frame_times = tf.range(num_frames, dtype=tf.float32)

    # Flatten frames
    samples, sample_times = spectrograms.flatten_frames(frames, frame_times, frame_size)

    # Check shapes
    assert samples.shape == (num_frames * frame_size, num_features)
    assert sample_times.shape == (num_frames * frame_size,)

    # Check values
    expected_samples = tf.repeat(frames, frame_size, axis=0)
    expected_times = tf.range(num_frames * frame_size, dtype=tf.float32)
    np.testing.assert_allclose(samples.numpy(), expected_samples.numpy())
    np.testing.assert_allclose(sample_times.numpy(), expected_times.numpy())
