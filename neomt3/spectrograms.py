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

"""Audio spectrogram functions."""

import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


@dataclasses.dataclass
class SpectrogramConfig:
    """Configuration for spectrogram computation."""

    sample_rate: int = 16000
    hop_width: int = 512
    num_mel_bins: int = 229
    fft_size: int = 2048
    window_size: int = 2048
    mel_min_hz: float = 30.0
    mel_max_hz: float = 8000.0
    clip_min_value: float = 1e-5


def compute_spectrogram(audio: tf.Tensor, config: SpectrogramConfig) -> tf.Tensor:
    """Compute mel spectrogram from audio.

    Args:
        audio: Audio tensor of shape [samples]
        config: Spectrogram configuration

    Returns:
        Mel spectrogram tensor of shape [frames, mel_bins]
    """
    # Add batch dimension if needed
    was_1d = len(audio.shape) == 1
    if was_1d:
        audio = audio[tf.newaxis, :]

    # Compute STFT
    stft = tf.signal.stft(
        audio,
        frame_length=config.window_size,
        frame_step=config.hop_width,
        fft_length=config.fft_size,
        window_fn=tf.signal.hann_window,
        pad_end=False,  # Don't pad the end to get exact number of frames
    )

    # Compute magnitude spectrogram
    magnitude_spectrograms = tf.abs(stft)

    # Add small value to avoid log of zero
    magnitude_spectrograms = tf.maximum(magnitude_spectrograms, config.clip_min_value)

    # Create mel filterbank matrix
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=config.sample_rate,
        lower_edge_hertz=config.mel_min_hz,
        upper_edge_hertz=config.mel_max_hz,
    )

    # Apply mel filterbank
    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1
    )

    # Convert to log scale
    log_mel_spectrograms = tf.math.log(mel_spectrograms)

    # Remove batch dimension if it was added
    if was_1d:
        log_mel_spectrograms = tf.squeeze(log_mel_spectrograms, axis=0)

    return log_mel_spectrograms


def compute_frame_times(num_frames: int, hop_width: int, sample_rate: int) -> tf.Tensor:
    """Compute frame times.

    Args:
        num_frames: Number of frames
        hop_width: Hop width in samples
        sample_rate: Sample rate in Hz

    Returns:
        Frame times tensor of shape [frames]
    """
    return tf.range(num_frames, dtype=tf.float32) * hop_width / sample_rate


def flatten_frames(
    frames: tf.Tensor, frame_times: tf.Tensor, frame_size: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Flatten frames into samples.

    Args:
        frames: Frames tensor of shape [frames, features]
        frame_times: Frame times tensor of shape [frames]
        frame_size: Frame size in samples

    Returns:
        Tuple of (samples, sample_times) tensors
    """
    # Get shapes
    num_frames = tf.shape(frames)[0]
    num_features = tf.shape(frames)[1]

    # Create sample times
    sample_times = tf.range(num_frames * frame_size, dtype=tf.float32)

    # Repeat frames
    samples = tf.repeat(frames, frame_size, axis=0)

    return samples, sample_times


def input_depth(spectrogram_config):
    return spectrogram_config.num_mel_bins
