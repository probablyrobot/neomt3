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

"""Metrics computation for MT3."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from neomt3 import (
    event_codec,
    metrics_utils,
    run_length_encoding,
    spectrograms,
    vocabularies,
)


def compute_transcription_metrics(
    predictions: tf.Tensor,
    targets: tf.Tensor,
    codec: event_codec.Codec,
    vocab_config: vocabularies.VocabularyConfig,
    spectrogram_config: spectrograms.SpectrogramConfig,
    onsets_only: bool,
    use_ties: bool,
    track_specs: Optional[Sequence[Any]] = None,
    frame_times: Optional[tf.Tensor] = None,
    target_sequence_length: Optional[tf.Tensor] = None,
    prediction_sequence_length: Optional[tf.Tensor] = None,
) -> Dict[str, tf.Tensor]:
    """Compute metrics for transcription evaluation.

    Args:
        predictions: Model predictions tensor
        targets: Target tensor
        codec: Event codec for decoding
        vocab_config: Vocabulary configuration
        spectrogram_config: Configuration for spectrogram computation
        onsets_only: Whether to only include onset events
        use_ties: Whether to use tie events
        track_specs: Optional track specifications
        frame_times: Optional frame times tensor
        target_sequence_length: Optional target sequence length tensor
        prediction_sequence_length: Optional prediction sequence length tensor

    Returns:
        Dictionary of metric names to metric values
    """
    # Compute basic metrics
    metrics = metrics_utils.compute_metrics(
        predictions=predictions,
        targets=targets,
        codec=codec,
        vocab_config=vocab_config,
        frame_times=frame_times,
        target_sequence_length=target_sequence_length,
        prediction_sequence_length=prediction_sequence_length,
    )

    # Add track-specific metrics if track specs are provided
    if track_specs:
        track_metrics = compute_track_metrics(
            predictions=predictions,
            targets=targets,
            codec=codec,
            vocab_config=vocab_config,
            track_specs=track_specs,
            frame_times=frame_times,
            target_sequence_length=target_sequence_length,
            prediction_sequence_length=prediction_sequence_length,
        )
        metrics.update(track_metrics)

    return metrics


def compute_track_metrics(
    predictions: tf.Tensor,
    targets: tf.Tensor,
    codec: event_codec.Codec,
    vocab_config: vocabularies.VocabularyConfig,
    track_specs: Sequence[Any],
    frame_times: Optional[tf.Tensor] = None,
    target_sequence_length: Optional[tf.Tensor] = None,
    prediction_sequence_length: Optional[tf.Tensor] = None,
) -> Dict[str, tf.Tensor]:
    """Compute track-specific metrics.

    Args:
        predictions: Model predictions tensor
        targets: Target tensor
        codec: Event codec for decoding
        vocab_config: Vocabulary configuration
        track_specs: Track specifications
        frame_times: Optional frame times tensor
        target_sequence_length: Optional target sequence length tensor
        prediction_sequence_length: Optional prediction sequence length tensor

    Returns:
        Dictionary of track-specific metric names to metric values
    """
    # Decode predictions and targets
    pred_events = run_length_encoding.decode_events(
        predictions,
        codec,
        vocab_config,
        frame_times=frame_times,
        sequence_length=prediction_sequence_length,
    )

    target_events = run_length_encoding.decode_events(
        targets,
        codec,
        vocab_config,
        frame_times=frame_times,
        sequence_length=target_sequence_length,
    )

    # Group events by track
    pred_track_events = group_events_by_track(pred_events, track_specs)
    target_track_events = group_events_by_track(target_events, track_specs)

    # Compute metrics for each track
    track_metrics = {}
    for track_spec in track_specs:
        track_name = track_spec["name"]
        track_pred_events = pred_track_events.get(track_name, [])
        track_target_events = target_track_events.get(track_name, [])

        # Compute track-specific metrics
        track_metric = metrics_utils.compute_event_metrics(
            track_pred_events, track_target_events
        )

        # Add track name prefix to metric names
        for metric_name, metric_value in track_metric.items():
            track_metrics[f"{track_name}_{metric_name}"] = metric_value

    return track_metrics


def group_events_by_track(
    events: List[Dict[str, Any]], track_specs: Sequence[Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """Group events by track.

    Args:
        events: List of events
        track_specs: Track specifications

    Returns:
        Dictionary mapping track names to lists of events
    """
    track_events = {}

    for event in events:
        # Get track name from event
        track_name = event.get("track_name")
        if track_name is None:
            continue

        # Add event to track
        if track_name not in track_events:
            track_events[track_name] = []
        track_events[track_name].append(event)

    return track_events
