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

"""Task definitions for MT3."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from neomt3 import (
    dataset_processing,
    event_codec,
    preprocessors,
    run_length_encoding,
    spectrograms,
    vocabularies,
)


def construct_task_name(
    dataset_name: str,
    spectrogram_config: spectrograms.SpectrogramConfig,
    vocab_config: vocabularies.VocabularyConfig,
    onsets_only: bool,
    include_ties: bool,
) -> str:
    """Construct a task name from configuration.

    Args:
        dataset_name: Name of the dataset
        spectrogram_config: Configuration for spectrogram computation
        vocab_config: Configuration for vocabulary
        onsets_only: Whether to only include onset events
        include_ties: Whether to include tie events

    Returns:
        A task name string
    """
    task_name = f"{dataset_name}"

    if spectrogram_config.hop_width != 512:
        task_name += f"_hop{spectrogram_config.hop_width}"

    if spectrogram_config.num_mel_bins != 229:
        task_name += f"_mel{spectrogram_config.num_mel_bins}"

    if vocab_config.onsets_only != onsets_only:
        task_name += f"_onsets{onsets_only}"

    if vocab_config.include_ties != include_ties:
        task_name += f"_ties{include_ties}"

    return task_name


def trim_eos(sequence: tf.Tensor) -> tf.Tensor:
    """Trim EOS token from sequence.

    Args:
        sequence: Input sequence tensor

    Returns:
        Sequence tensor with EOS token removed
    """
    return sequence[:-1]


def postprocess(
    outputs: Dict[str, tf.Tensor],
    codec: event_codec.Codec,
    vocab_config: vocabularies.VocabularyConfig,
    frame_times: Optional[tf.Tensor] = None,
) -> Dict[str, Any]:
    """Postprocess model outputs.

    Args:
        outputs: Dictionary of model outputs
        codec: Event codec for decoding
        vocab_config: Vocabulary configuration
        frame_times: Optional frame times tensor

    Returns:
        Dictionary of postprocessed outputs
    """
    # Decode events
    events = run_length_encoding.decode_events(
        outputs["targets"], codec, vocab_config, frame_times=frame_times
    )

    # Convert to note sequence
    note_sequence = event_codec.events_to_note_sequence(events)

    return {"events": events, "note_sequence": note_sequence}


def create_transcription_task(
    dataset_config: dataset_processing.DatasetConfig,
    spectrogram_config: spectrograms.SpectrogramConfig,
    vocab_config: vocabularies.VocabularyConfig,
    tokenize_fn: Callable,
    onsets_only: bool = False,
    include_ties: bool = True,
    batch_size: int = 32,
    shuffle_buffer_size: int = 10000,
    skip_too_long: bool = False,
) -> Dict[str, Any]:
    """Create a transcription task.

    Args:
        dataset_config: Configuration for the dataset
        spectrogram_config: Configuration for spectrogram computation
        vocab_config: Configuration for vocabulary
        tokenize_fn: Function to tokenize examples
        onsets_only: Whether to only include onset events
        include_ties: Whether to include tie events
        batch_size: Batch size for training
        shuffle_buffer_size: Buffer size for shuffling
        skip_too_long: Whether to skip examples that are too long

    Returns:
        Dictionary containing task configuration and dataset pipelines
    """
    # Construct task name
    task_name = construct_task_name(
        dataset_config.name, spectrogram_config, vocab_config, onsets_only, include_ties
    )

    # Create training dataset pipeline
    train_ds = dataset_processing.create_dataset_pipeline(
        dataset_config=dataset_config,
        spectrogram_config=spectrogram_config,
        vocab_config=vocab_config,
        tokenize_fn=tokenize_fn,
        onsets_only=onsets_only,
        include_ties=include_ties,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        skip_too_long=skip_too_long,
    )

    # Create evaluation dataset pipeline
    eval_ds = dataset_processing.create_eval_dataset_pipeline(
        dataset_config=dataset_config,
        spectrogram_config=spectrogram_config,
        vocab_config=vocab_config,
        tokenize_fn=tokenize_fn,
        onsets_only=onsets_only,
        include_ties=include_ties,
        split_name=dataset_config.train_eval_split,
        batch_size=batch_size,
    )

    # Create inference dataset pipelines
    infer_ds = {}
    for split_config in dataset_config.infer_eval_splits:
        split_name = split_config["split"]
        infer_ds[split_name] = dataset_processing.create_eval_dataset_pipeline(
            dataset_config=dataset_config,
            spectrogram_config=spectrogram_config,
            vocab_config=vocab_config,
            tokenize_fn=tokenize_fn,
            onsets_only=onsets_only,
            include_ties=include_ties,
            split_name=split_name,
            batch_size=batch_size,
        )

    return {
        "task_name": task_name,
        "train_ds": train_ds,
        "eval_ds": eval_ds,
        "infer_ds": infer_ds,
        "postprocess_fn": postprocess,
    }
