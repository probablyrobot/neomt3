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

"""Transcription preprocessors."""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import gin
import librosa
import note_seq
import numpy as np
import tensorflow as tf
from absl import logging
from immutabledict import immutabledict
from transformers import PreTrainedTokenizer

from neomt3 import (
    event_codec,
    note_sequences,
    run_length_encoding,
    spectrograms,
    vocabularies,
)


def add_unique_id(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Add unique ID to each example.

    Args:
        ds: Input dataset

    Returns:
        Dataset with unique IDs added
    """

    def _add_unique_id(example):
        example["unique_id"] = tf.strings.as_string(
            tf.random.uniform([], maxval=2**63, dtype=tf.int64)
        )
        return example

    return ds.map(_add_unique_id)


def pad_notesequence_array(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Pad note sequence array to fixed length.

    Args:
        ds: Input dataset

    Returns:
        Dataset with padded note sequences
    """

    def _pad_notesequence_array(example):
        example["sequence"] = tf.pad(example["sequence"], [[0, 1]], constant_values="")
        return example

    return ds.map(_pad_notesequence_array)


def tokenize_transcription_example(
    ds: tf.data.Dataset,
    spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    audio_is_samples: bool = False,
    id_feature_key: str = "id",
) -> tf.data.Dataset:
    """Tokenize transcription example.

    Args:
        ds: Input dataset
        spectrogram_config: Configuration for spectrogram computation
        codec: Event codec for encoding/decoding
        is_training_data: Whether this is training data
        onsets_only: Whether to only include onset events
        include_ties: Whether to include tie events
        audio_is_samples: Whether audio is in samples format
        id_feature_key: Key for ID feature

    Returns:
        Dataset with tokenized examples
    """

    def _tokenize_example(example):
        # Convert audio to spectrogram if needed
        if audio_is_samples:
            audio = example["audio"]
            spectrogram = spectrograms.compute_spectrogram(audio, spectrogram_config)
        else:
            spectrogram = example["spectrogram"]

        # Get note sequence
        note_sequence = example["sequence"]

        # Convert note sequence to events
        events = event_codec.note_sequence_to_events(
            note_sequence, codec, onsets_only=onsets_only, include_ties=include_ties
        )

        # Encode events
        tokens = event_codec.encode_events(events, codec)

        # Add to example
        example["inputs"] = spectrogram
        example["targets"] = tokens
        example["input_times"] = spectrograms.compute_frame_times(
            spectrogram, spectrogram_config
        )

        return example

    return ds.map(_tokenize_example)


def compute_spectrograms(
    example: Dict[str, Any], spectrogram_config: spectrograms.SpectrogramConfig
) -> Dict[str, Any]:
    """Compute spectrograms for example.

    Args:
        example: Input example
        spectrogram_config: Configuration for spectrogram computation

    Returns:
        Example with spectrograms computed
    """
    # Convert audio to spectrogram if needed
    if "audio" in example:
        audio = example["audio"]
        spectrogram = spectrograms.compute_spectrogram(audio, spectrogram_config)
        example["inputs"] = spectrogram
        example["input_times"] = spectrograms.compute_frame_times(
            spectrogram, spectrogram_config
        )

    return example


def add_dummy_targets(example: Dict[str, Any]) -> Dict[str, Any]:
    """Add dummy targets for evaluation.

    Args:
        example: Input example

    Returns:
        Example with dummy targets added
    """
    if "targets" not in example:
        example["targets"] = tf.zeros([1], dtype=tf.int32)

    return example


def handle_too_long(example: Dict[str, Any], skip: bool = False) -> Dict[str, Any]:
    """Handle examples that are too long.

    Args:
        example: Input example
        skip: Whether to skip examples that are too long

    Returns:
        Example with length handling applied
    """
    if skip:
        return example

    # Truncate inputs if too long
    if tf.shape(example["inputs"])[0] > MAX_NUM_CACHED_FRAMES:
        example["inputs"] = example["inputs"][:MAX_NUM_CACHED_FRAMES]
        if "input_times" in example:
            example["input_times"] = example["input_times"][:MAX_NUM_CACHED_FRAMES]

    return example


# Maximum number of frames to cache
MAX_NUM_CACHED_FRAMES = 2000


def _audio_to_frames(
    samples: Sequence[float],
    spectrogram_config: spectrograms.SpectrogramConfig,
) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
    """Convert audio samples to non-overlapping frames and frame times."""
    frame_size = spectrogram_config.hop_width
    logging.info("Padding %d samples to multiple of %d", len(samples), frame_size)
    samples = np.pad(
        samples, [0, frame_size - len(samples) % frame_size], mode="constant"
    )

    frames = spectrograms.split_audio(samples, spectrogram_config)

    num_frames = len(samples) // frame_size
    logging.info(
        "Encoded %d samples to %d frames (%d samples each)",
        len(samples),
        num_frames,
        frame_size,
    )

    times = np.arange(num_frames) / spectrogram_config.frames_per_second
    return frames, times


def _include_inputs(ds, input_record, fields_to_omit=("audio",)):
    """Include fields from input record (other than audio) in dataset records."""

    def include_inputs_fn(output_record):
        for key in set(input_record.keys()) - set(output_record.keys()):
            output_record[key] = input_record[key]
        for key in fields_to_omit:
            del output_record[key]
        return output_record

    return ds.map(include_inputs_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def tokenize_guitarset_example(
    ds: tf.data.Dataset,
    spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
) -> tf.data.Dataset:
    """Tokenize a GuitarSet transcription example."""

    def _preprocess_example(ex, name):
        assert "inst_names" not in ex, "Key `inst_names` is already populated."
        ex["inst_names"] = [name]
        ex["instrument_sequences"] = [ex.pop("sequence")]
        return ex

    ds = ds.map(
        lambda x: _preprocess_example(x, "Clean Guitar"),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = tokenize_example_with_program_lookup(
        ds,
        spectrogram_config=spectrogram_config,
        codec=codec,
        is_training_data=is_training_data,
        inst_name_to_program_fn=guitarset_instrument_to_program,
        onsets_only=onsets_only,
        include_ties=include_ties,
        id_feature_key="id",
    )
    return ds


def guitarset_instrument_to_program(instrument: str) -> int:
    """GuitarSet is all guitar, return the first MIDI guitar program."""
    if instrument == "Clean Guitar":
        return 24
    else:
        raise ValueError("Unknown GuitarSet instrument: %s" % instrument)


def tokenize_example_with_program_lookup(
    ds: tf.data.Dataset,
    spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    inst_name_to_program_fn: Callable[[str], int],
    id_feature_key: Optional[str] = None,
) -> tf.data.Dataset:
    """Tokenize an example, optionally looking up and assigning program numbers.

    This can be used by any dataset where a mapping function can be used to
    map from the inst_names feature to a set of program numbers.

    Args:
      ds: Input dataset.
      spectrogram_config: Spectrogram configuration.
      codec: Event vocabulary codec.
      is_training_data: Unused.
      onsets_only: If True, include only onset events (not offset & velocity).
      include_ties: If True, include tie events.
      inst_name_to_program_fn: A function used to map the instrument names
        in the `inst_names` feature of each example to a MIDI program number.
      id_feature_key: If not None, replace sequence ID with specified key field
          from the dataset.

    Returns:
      Dataset with the outputs described above.
    """
    del is_training_data

    def tokenize(sequences, inst_names, audio, example_id=None):
        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
        assert len(tracks) == len(inst_names)
        for track, inst_name in zip(tracks, inst_names):
            program = inst_name_to_program_fn(inst_name.decode())

            # Note that there are no pitch bends in URMP data; the below block will
            # raise PitchBendError if one is encountered.
            add_track_to_notesequence(
                ns, track, program=program, is_drum=False, ignore_pitch_bends=False
            )

        note_sequences.assign_instruments(ns)
        note_sequences.validate_note_sequence(ns)

        if example_id is not None:
            ns.id = example_id

        samples = note_seq.audio_io.wav_data_to_samples_librosa(
            audio, sample_rate=spectrogram_config.sample_rate
        )

        logging.info(
            "Got samples for %s::%s with length %d", ns.id, ns.filename, len(samples)
        )

        frames, frame_times = _audio_to_frames(samples, spectrogram_config)

        if onsets_only:
            times, values = note_sequences.note_sequence_to_onsets(ns)
        else:
            times, values = (
                note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)
            )

        # The original NoteSequence can have a lot of control changes we don't need;
        # delete them.
        del ns.control_changes[:]

        (
            events,
            event_start_indices,
            event_end_indices,
            state_events,
            state_event_indices,
        ) = run_length_encoding.encode_and_index_events(
            state=note_sequences.NoteEncodingState() if include_ties else None,
            event_times=times,
            event_values=values,
            encode_event_fn=note_sequences.note_event_data_to_events,
            codec=codec,
            frame_times=frame_times,
            encoding_state_to_events_fn=(
                note_sequences.note_encoding_state_to_events if include_ties else None
            ),
        )

        yield {
            "inputs": frames,
            "input_times": frame_times,
            "targets": events,
            "input_event_start_indices": event_start_indices,
            "input_event_end_indices": event_end_indices,
            "state_events": state_events,
            "input_state_event_indices": state_event_indices,
            "sequence": ns.SerializeToString(),
        }

    def process_record(input_record):
        args = [
            input_record["instrument_sequences"],
            input_record["inst_names"],
            input_record["audio"],
        ]
        if id_feature_key is not None:
            args.append(input_record[id_feature_key])

        ds = tf.data.Dataset.from_generator(
            tokenize,
            output_signature={
                "inputs": tf.TensorSpec(
                    shape=(None, spectrogram_config.hop_width), dtype=tf.float32
                ),
                "input_times": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "targets": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "input_event_start_indices": tf.TensorSpec(
                    shape=(None,), dtype=tf.int32
                ),
                "input_event_end_indices": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "state_events": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "input_state_event_indices": tf.TensorSpec(
                    shape=(None,), dtype=tf.int32
                ),
                "sequence": tf.TensorSpec(shape=(), dtype=tf.string),
            },
            args=args,
        )

        ds = _include_inputs(ds, input_record)
        return ds

    tokenized_records = ds.flat_map(process_record)
    return tokenized_records


_URMP_INSTRUMENT_PROGRAMS = immutabledict(
    {
        "vn": 40,  # violin
        "va": 41,  # viola
        "vc": 42,  # cello
        "db": 43,  # double bass
        "tpt": 56,  # trumpet
        "tbn": 57,  # trombone
        "tba": 58,  # tuba
        "hn": 60,  # French horn
        "sax": 64,  # saxophone
        "ob": 68,  # oboe
        "bn": 70,  # bassoon
        "cl": 71,  # clarinet
        "fl": 73,  # flute
    }
)


def urmp_instrument_to_program(urmp_instrument: str) -> int:
    """Fetch the program number associated with a given URMP instrument code."""
    if urmp_instrument not in _URMP_INSTRUMENT_PROGRAMS:
        raise ValueError("unknown URMP instrument: %s" % urmp_instrument)
    return _URMP_INSTRUMENT_PROGRAMS[urmp_instrument]


_SLAKH_CLASS_PROGRAMS = immutabledict(
    {
        "Acoustic Piano": 0,
        "Electric Piano": 4,
        "Chromatic Percussion": 8,
        "Organ": 16,
        "Acoustic Guitar": 24,
        "Clean Electric Guitar": 26,
        "Distorted Electric Guitar": 29,
        "Acoustic Bass": 32,
        "Electric Bass": 33,
        "Violin": 40,
        "Viola": 41,
        "Cello": 42,
        "Contrabass": 43,
        "Orchestral Harp": 46,
        "Timpani": 47,
        "String Ensemble": 48,
        "Synth Strings": 50,
        "Choir and Voice": 52,
        "Orchestral Hit": 55,
        "Trumpet": 56,
        "Trombone": 57,
        "Tuba": 58,
        "French Horn": 60,
        "Brass Section": 61,
        "Soprano/Alto Sax": 64,
        "Tenor Sax": 66,
        "Baritone Sax": 67,
        "Oboe": 68,
        "English Horn": 69,
        "Bassoon": 70,
        "Clarinet": 71,
        "Pipe": 73,
        "Synth Lead": 80,
        "Synth Pad": 88,
    }
)


def slakh_class_to_program_and_is_drum(slakh_class: str) -> Tuple[int, bool]:
    """Map Slakh class string to program number and boolean indicating drums."""
    if slakh_class == "Drums":
        return 0, True
    elif slakh_class not in _SLAKH_CLASS_PROGRAMS:
        raise ValueError("unknown Slakh class: %s" % slakh_class)
    else:
        return _SLAKH_CLASS_PROGRAMS[slakh_class], False


class PitchBendError(Exception):
    pass


def add_track_to_notesequence(
    ns: note_seq.NoteSequence,
    track: note_seq.NoteSequence,
    program: int,
    is_drum: bool,
    ignore_pitch_bends: bool,
):
    """Add a track to a NoteSequence."""
    if track.pitch_bends and not ignore_pitch_bends:
        raise PitchBendError
    track_sus = note_seq.apply_sustain_control_changes(track)
    for note in track_sus.notes:
        note.program = program
        note.is_drum = is_drum
        ns.notes.extend([note])
        ns.total_time = max(ns.total_time, note.end_time)


def tokenize_slakh_example(
    ds: tf.data.Dataset,
    spectrogram_config: spectrograms.SpectrogramConfig,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    track_specs: Optional[Sequence[note_sequences.TrackSpec]],
    ignore_pitch_bends: bool,
) -> tf.data.Dataset:
    """Tokenize a Slakh multitrack note transcription example."""

    def tokenize(sequences, samples, sample_rate, inst_names, example_id):
        if sample_rate != spectrogram_config.sample_rate:
            samples = librosa.resample(
                samples, sample_rate, spectrogram_config.sample_rate
            )

        frames, frame_times = _audio_to_frames(samples, spectrogram_config)

        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
        assert len(tracks) == len(inst_names)
        if track_specs:
            # Specific tracks expected.
            assert len(tracks) == len(track_specs)
            for track, spec, inst_name in zip(tracks, track_specs, inst_names):
                # Make sure the instrument name matches what we expect.
                assert inst_name.decode() == spec.name
                try:
                    add_track_to_notesequence(
                        ns,
                        track,
                        program=spec.program,
                        is_drum=spec.is_drum,
                        ignore_pitch_bends=ignore_pitch_bends,
                    )
                except PitchBendError:
                    # TODO(iansimon): is there a way to count these?
                    return
        else:
            for track, inst_name in zip(tracks, inst_names):
                # Instrument name should be Slakh class.
                program, is_drum = slakh_class_to_program_and_is_drum(
                    inst_name.decode()
                )
                try:
                    add_track_to_notesequence(
                        ns,
                        track,
                        program=program,
                        is_drum=is_drum,
                        ignore_pitch_bends=ignore_pitch_bends,
                    )
                except PitchBendError:
                    # TODO(iansimon): is there a way to count these?
                    return

        note_sequences.assign_instruments(ns)
        note_sequences.validate_note_sequence(ns)
        if is_training_data:
            # Trim overlapping notes in training (as our event vocabulary cannot
            # represent them), but preserve original NoteSequence for eval.
            ns = note_sequences.trim_overlapping_notes(ns)

        ns.id = example_id

        if onsets_only:
            times, values = note_sequences.note_sequence_to_onsets(ns)
        else:
            times, values = (
                note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)
            )

        (
            events,
            event_start_indices,
            event_end_indices,
            state_events,
            state_event_indices,
        ) = run_length_encoding.encode_and_index_events(
            state=note_sequences.NoteEncodingState() if include_ties else None,
            event_times=times,
            event_values=values,
            encode_event_fn=note_sequences.note_event_data_to_events,
            codec=codec,
            frame_times=frame_times,
            encoding_state_to_events_fn=(
                note_sequences.note_encoding_state_to_events if include_ties else None
            ),
        )

        yield {
            "inputs": frames,
            "input_times": frame_times,
            "targets": events,
            "input_event_start_indices": event_start_indices,
            "input_event_end_indices": event_end_indices,
            "state_events": state_events,
            "input_state_event_indices": state_event_indices,
            "sequence": ns.SerializeToString(),
        }

    def process_record(input_record):
        ds = tf.data.Dataset.from_generator(
            tokenize,
            output_signature={
                "inputs": tf.TensorSpec(
                    shape=(None, spectrogram_config.hop_width), dtype=tf.float32
                ),
                "input_times": tf.TensorSpec(shape=(None,), dtype=tf.float32),
                "targets": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "input_event_start_indices": tf.TensorSpec(
                    shape=(None,), dtype=tf.int32
                ),
                "input_event_end_indices": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "state_events": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "input_state_event_indices": tf.TensorSpec(
                    shape=(None,), dtype=tf.int32
                ),
                "sequence": tf.TensorSpec(shape=(), dtype=tf.string),
            },
            args=[
                input_record["note_sequences"],
                input_record["mix"],
                input_record["audio_sample_rate"],
                input_record["inst_names"],
                input_record["track_id"],
            ],
        )

        ds = _include_inputs(ds, input_record, fields_to_omit=["mix", "stems"])
        return ds

    tokenized_records = ds.flat_map(process_record)
    return tokenized_records


@gin.configurable
def map_midi_programs(
    ds: tf.data.Dataset,
    codec: event_codec.Codec,
    granularity_type: str = "full",
    feature_key: str = "targets",
) -> Mapping[str, Any]:
    """Apply MIDI program map to token sequences."""
    granularity = vocabularies.PROGRAM_GRANULARITIES[granularity_type]

    def _map_program_tokens(ex):
        ex[feature_key] = granularity.tokens_map_fn(ex[feature_key], codec)
        return ex

    return ds.map(_map_program_tokens, num_parallel_calls=tf.data.experimental.AUTOTUNE)
