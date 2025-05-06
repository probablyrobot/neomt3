"""Dataset processing functionality for MT3."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from neomt3 import (
    event_codec,
    preprocessors,
    run_length_encoding,
    spectrograms,
    vocabularies,
)


class DatasetConfig:
    """Configuration for dataset processing."""

    def __init__(
        self,
        name: str,
        paths: Dict[str, str],
        features: Dict[str, tf.io.FixedLenFeature],
        train_split: str = "train",
        train_eval_split: str = "train_eval",
        infer_eval_splits: Sequence[Dict[str, Any]] = None,
        track_specs: Optional[Sequence[Any]] = None,
    ):
        self.name = name
        self.paths = paths
        self.features = features
        self.train_split = train_split
        self.train_eval_split = train_eval_split
        self.infer_eval_splits = infer_eval_splits or []
        self.track_specs = track_specs


def create_dataset_pipeline(
    dataset_config: DatasetConfig,
    spectrogram_config: spectrograms.SpectrogramConfig,
    vocab_config: vocabularies.VocabularyConfig,
    tokenize_fn: Callable,
    onsets_only: bool,
    include_ties: bool,
    batch_size: int = 32,
    shuffle_buffer_size: int = 10000,
    skip_too_long: bool = False,
) -> tf.data.Dataset:
    """Create a dataset processing pipeline.

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
        A tf.data.Dataset pipeline
    """
    # Create base dataset from TFRecord files
    ds = tf.data.TFRecordDataset(dataset_config.paths[dataset_config.train_split])

    # Parse examples
    ds = ds.map(
        lambda x: tf.io.parse_single_example(x, dataset_config.features),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Add unique IDs
    ds = preprocessors.add_unique_id(ds)

    # Tokenize examples
    codec = vocabularies.build_codec(vocab_config)
    ds = tokenize_fn(
        ds,
        spectrogram_config=spectrogram_config,
        codec=codec,
        is_training_data=True,
        onsets_only=onsets_only,
        include_ties=include_ties,
    )

    # Compute spectrograms
    ds = ds.map(
        lambda x: preprocessors.compute_spectrograms(x, spectrogram_config),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Handle sequences that are too long
    if skip_too_long:
        ds = ds.filter(lambda x: tf.shape(x["inputs"])[0] <= MAX_NUM_CACHED_FRAMES)

    # Shuffle and batch
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size)

    # Cache and prefetch
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def create_eval_dataset_pipeline(
    dataset_config: DatasetConfig,
    spectrogram_config: spectrograms.SpectrogramConfig,
    vocab_config: vocabularies.VocabularyConfig,
    tokenize_fn: Callable,
    onsets_only: bool,
    include_ties: bool,
    split_name: str,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Create an evaluation dataset pipeline.

    Args:
        dataset_config: Configuration for the dataset
        spectrogram_config: Configuration for spectrogram computation
        vocab_config: Configuration for vocabulary
        tokenize_fn: Function to tokenize examples
        onsets_only: Whether to only include onset events
        include_ties: Whether to include tie events
        split_name: Name of the evaluation split to use
        batch_size: Batch size for evaluation

    Returns:
        A tf.data.Dataset pipeline for evaluation
    """
    # Create base dataset from TFRecord files
    ds = tf.data.TFRecordDataset(dataset_config.paths[split_name])

    # Parse examples
    ds = ds.map(
        lambda x: tf.io.parse_single_example(x, dataset_config.features),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Add unique IDs
    ds = preprocessors.add_unique_id(ds)

    # Tokenize examples
    codec = vocabularies.build_codec(vocab_config)
    ds = tokenize_fn(
        ds,
        spectrogram_config=spectrogram_config,
        codec=codec,
        is_training_data=False,
        onsets_only=onsets_only,
        include_ties=include_ties,
    )

    # Compute spectrograms
    ds = ds.map(
        lambda x: preprocessors.compute_spectrograms(x, spectrogram_config),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Add dummy targets for evaluation
    ds = ds.map(
        preprocessors.add_dummy_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Batch and prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds
