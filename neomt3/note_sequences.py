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

"""Helper functions that operate on NoteSequence protos."""

import dataclasses
import itertools
from typing import MutableMapping, MutableSet, Optional, Sequence, Tuple

import note_seq

from neomt3 import event_codec, run_length_encoding, vocabularies

# Constants
DEFAULT_NOTE_DURATION = 0.01  # 10ms
MIN_NOTE_DURATION = 0.001  # 1ms
DEFAULT_STEPS_PER_SECOND = 100  # 10ms per step
DEFAULT_VELOCITY = 100


@dataclasses.dataclass
class TrackSpec:
    name: str
    program: int = 0
    is_drum: bool = False


def extract_track(ns, program, is_drum):
    track = note_seq.NoteSequence(ticks_per_quarter=220)
    track_notes = [
        note for note in ns.notes if note.program == program and note.is_drum == is_drum
    ]
    track.notes.extend(track_notes)
    track.total_time = (
        max(note.end_time for note in track.notes) if track.notes else 0.0
    )
    return track


def trim_overlapping_notes(ns: note_seq.NoteSequence) -> note_seq.NoteSequence:
    """Trim overlapping notes from a NoteSequence, dropping zero-length notes."""
    ns_trimmed = note_seq.NoteSequence()
    ns_trimmed.CopyFrom(ns)
    channels = set(
        (note.pitch, note.program, note.is_drum) for note in ns_trimmed.notes
    )
    for pitch, program, is_drum in channels:
        notes = [
            note
            for note in ns_trimmed.notes
            if note.pitch == pitch
            and note.program == program
            and note.is_drum == is_drum
        ]
        sorted_notes = sorted(notes, key=lambda note: note.start_time)
        for i in range(1, len(sorted_notes)):
            if sorted_notes[i - 1].end_time > sorted_notes[i].start_time:
                sorted_notes[i - 1].end_time = sorted_notes[i].start_time
    valid_notes = [note for note in ns_trimmed.notes if note.start_time < note.end_time]
    del ns_trimmed.notes[:]
    ns_trimmed.notes.extend(valid_notes)
    return ns_trimmed


def assign_instruments(ns: note_seq.NoteSequence) -> None:
    """Assign instrument numbers to notes; modifies NoteSequence in place."""
    program_instruments = {}
    for note in ns.notes:
        if note.program not in program_instruments and not note.is_drum:
            num_instruments = len(program_instruments)
            note.instrument = (
                num_instruments if num_instruments < 9 else num_instruments + 1
            )
            program_instruments[note.program] = note.instrument
        elif note.is_drum:
            note.instrument = 9
        else:
            note.instrument = program_instruments[note.program]


def validate_note_sequence(ns: note_seq.NoteSequence) -> None:
    """Raise ValueError if NoteSequence contains invalid notes."""
    for note in ns.notes:
        if note.start_time >= note.end_time:
            raise ValueError(
                "note has start time >= end time: %f >= %f"
                % (note.start_time, note.end_time)
            )
        if note.velocity == 0:
            raise ValueError("note has zero velocity")


def note_arrays_to_note_sequence(
    onset_times: Sequence[float],
    pitches: Sequence[int],
    offset_times: Optional[Sequence[float]] = None,
    velocities: Optional[Sequence[int]] = None,
    programs: Optional[Sequence[int]] = None,
    is_drums: Optional[Sequence[bool]] = None,
) -> note_seq.NoteSequence:
    """Convert note onset / offset / pitch / velocity arrays to NoteSequence."""
    ns = note_seq.NoteSequence(ticks_per_quarter=220)
    for (
        onset_time,
        offset_time,
        pitch,
        velocity,
        program,
        is_drum,
    ) in itertools.zip_longest(
        onset_times,
        [] if offset_times is None else offset_times,
        pitches,
        [] if velocities is None else velocities,
        [] if programs is None else programs,
        [] if is_drums is None else is_drums,
    ):
        if offset_time is None:
            offset_time = onset_time + DEFAULT_NOTE_DURATION
        if velocity is None:
            velocity = DEFAULT_VELOCITY
        if program is None:
            program = 0
        if is_drum is None:
            is_drum = False
        ns.notes.add(
            start_time=onset_time,
            end_time=offset_time,
            pitch=pitch,
            velocity=velocity,
            program=program,
            is_drum=is_drum,
        )
        ns.total_time = max(ns.total_time, offset_time)
    assign_instruments(ns)
    return ns


@dataclasses.dataclass
class NoteEventData:
    pitch: int
    velocity: Optional[int] = None
    program: Optional[int] = None
    is_drum: Optional[bool] = None
    instrument: Optional[int] = None


def note_sequence_to_onsets(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract note onsets and pitches from NoteSequence proto."""
    # Sort by pitch to use as a tiebreaker for subsequent stable sort.
    notes = sorted(ns.notes, key=lambda note: note.pitch)
    return (
        [note.start_time for note in notes],
        [NoteEventData(pitch=note.pitch) for note in notes],
    )


def note_sequence_to_onsets_and_offsets(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract onset & offset times and pitches from a NoteSequence proto.

    The onset & offset times will not necessarily be in sorted order.

    Args:
      ns: NoteSequence from which to extract onsets and offsets.

    Returns:
      times: A list of note onset and offset times.
      values: A list of NoteEventData objects where velocity is zero for note
          offsets.
    """
    # Sort by pitch and put offsets before onsets as a tiebreaker for subsequent
    # stable sort.
    notes = sorted(ns.notes, key=lambda note: note.pitch)
    times = [note.end_time for note in notes] + [note.start_time for note in notes]
    values = [NoteEventData(pitch=note.pitch, velocity=0) for note in notes] + [
        NoteEventData(pitch=note.pitch, velocity=note.velocity) for note in notes
    ]
    return times, values


def note_sequence_to_onsets_and_offsets_and_programs(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract onset & offset times and pitches & programs from a NoteSequence.

    The onset & offset times will not necessarily be in sorted order.

    Args:
      ns: NoteSequence from which to extract onsets and offsets.

    Returns:
      times: A list of note onset and offset times.
      values: A list of NoteEventData objects where velocity is zero for note
          offsets.
    """
    # Sort by program and pitch and put offsets before onsets as a tiebreaker for
    # subsequent stable sort.
    notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program, note.pitch))
    times = [note.end_time for note in notes if not note.is_drum] + [
        note.start_time for note in notes
    ]
    values = [
        NoteEventData(pitch=note.pitch, velocity=0, program=note.program, is_drum=False)
        for note in notes
        if not note.is_drum
    ] + [
        NoteEventData(
            pitch=note.pitch,
            velocity=note.velocity,
            program=note.program,
            is_drum=note.is_drum,
        )
        for note in notes
    ]
    return times, values


@dataclasses.dataclass
class NoteEncodingState:
    """Encoding state for note transcription, keeping track of active pitches."""

    # velocity bin for active pitches and programs
    active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(
        default_factory=dict
    )


def note_event_data_to_events(
    state: Optional[NoteEncodingState],
    value: NoteEventData,
    codec: event_codec.Codec,
) -> Sequence[event_codec.Event]:
    """Convert note event data to a sequence of events.

    Args:
        state: Optional encoding state
        value: Note event data
        codec: Event codec for encoding

    Returns:
        Sequence of events
    """
    events = []

    # Add program event if needed
    if value.program is not None:
        events.append(event_codec.Event(type="program", value=value.program))

    # Add velocity event if needed
    if value.velocity is not None:
        events.append(event_codec.Event(type="velocity", value=value.velocity))

    # Add pitch event
    events.append(event_codec.Event(type="pitch", value=value.pitch))

    return events


def note_encoding_state_to_events(
    state: NoteEncodingState,
) -> Sequence[event_codec.Event]:
    """Convert note encoding state to a sequence of events.

    Args:
        state: Note encoding state

    Returns:
        Sequence of events
    """
    events = []

    # Add program events
    for (pitch, program), velocity in sorted(state.active_pitches.items()):
        events.append(event_codec.Event(type="program", value=program))
        events.append(event_codec.Event(type="velocity", value=velocity))
        events.append(event_codec.Event(type="pitch", value=pitch))

    return events


@dataclasses.dataclass
class NoteDecodingState:
    """Decoding state for note transcription."""

    current_time: float = 0.0
    # velocity to apply to subsequent pitch events (zero for note-off)
    current_velocity: int = DEFAULT_VELOCITY
    # program to apply to subsequent pitch events
    current_program: int = 0
    # onset time and velocity for active pitches and programs
    active_pitches: MutableMapping[Tuple[int, int], Tuple[float, int]] = (
        dataclasses.field(default_factory=dict)
    )
    # pitches (with programs) to continue from previous segment
    tied_pitches: MutableSet[Tuple[int, int]] = dataclasses.field(default_factory=set)
    # whether or not we are in the tie section at the beginning of a segment
    is_tie_section: bool = False
    # partially-decoded NoteSequence
    note_sequence: note_seq.NoteSequence = dataclasses.field(
        default_factory=lambda: note_seq.NoteSequence(ticks_per_quarter=220)
    )


def decode_note_event(
    state: NoteDecodingState,
    time: float,
    event: event_codec.Event,
    codec: event_codec.Codec,
) -> None:
    """Process note event and update decoding state.

    Args:
        state: Note decoding state
        time: Event time
        event: Event to decode
        codec: Event codec for decoding
    """
    if time < 0:
        raise ValueError("event time cannot be negative")
    state.current_time = max(state.current_time, time)

    if event.type == "pitch":
        pitch = event.value
        if state.current_velocity == 0:
            # Note offset
            if (pitch, state.current_program) in state.active_pitches:
                onset_time, onset_velocity = state.active_pitches.pop(
                    (pitch, state.current_program)
                )
                _add_note_to_sequence(
                    state.note_sequence,
                    onset_time,
                    state.current_time,
                    pitch,
                    onset_velocity,
                    state.current_program,
                )
        else:
            # Note onset
            state.active_pitches[(pitch, state.current_program)] = (
                state.current_time,
                state.current_velocity,
            )
    elif event.type == "velocity":
        state.current_velocity = event.value
    elif event.type == "program":
        state.current_program = event.value
    elif event.type == "tie":
        if not state.is_tie_section:
            begin_tied_pitches_section(state)
        state.tied_pitches.add((event.value, state.current_program))
    else:
        raise ValueError(f"unknown event type: {event.type}")


def decode_note_onset_event(
    state: NoteDecodingState,
    time: float,
    event: event_codec.Event,
    codec: event_codec.Codec,
) -> None:
    """Process note onset event and update decoding state.

    Args:
        state: Note decoding state
        time: Event time
        event: Event to decode
        codec: Event codec for decoding
    """
    if event.type == "pitch":
        pitch = event.value
        state.active_pitches[(pitch, state.current_program)] = (
            state.current_time,
            state.current_velocity,
        )
    elif event.type == "velocity":
        state.current_velocity = event.value
    elif event.type == "program":
        state.current_program = event.value
    else:
        raise ValueError(f"unknown event type: {event.type}")


def begin_tied_pitches_section(state: NoteDecodingState) -> None:
    """Begin a tied pitches section in the decoding state."""
    state.is_tie_section = True
    state.tied_pitches.clear()


def flush_note_decoding_state(state: NoteDecodingState) -> note_seq.NoteSequence:
    """Flush the note decoding state and return the note sequence.

    Args:
        state: Note decoding state

    Returns:
        Note sequence
    """
    # Add any remaining active notes
    for (pitch, program), (onset_time, velocity) in state.active_pitches.items():
        if (pitch, program) not in state.tied_pitches:
            _add_note_to_sequence(
                state.note_sequence,
                onset_time,
                state.current_time,
                pitch,
                velocity,
                program,
            )

    # Clear state
    state.active_pitches.clear()
    state.tied_pitches.clear()
    state.is_tie_section = False

    return state.note_sequence


class NoteEncodingSpecType(run_length_encoding.EventEncodingSpec):
    pass


# encoding spec for modeling note onsets only
NoteOnsetEncodingSpec = NoteEncodingSpecType(
    init_encoding_state_fn=lambda: None,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=None,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=lambda state: None,
    decode_event_fn=decode_note_onset_event,
    flush_decoding_state_fn=lambda state: state.note_sequence,
)


# encoding spec for modeling onsets and offsets
NoteEncodingSpec = NoteEncodingSpecType(
    init_encoding_state_fn=lambda: None,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=None,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=lambda state: None,
    decode_event_fn=decode_note_event,
    flush_decoding_state_fn=flush_note_decoding_state,
)


# encoding spec for modeling onsets and offsets, with a "tie" section at the
# beginning of each segment listing already-active notes
NoteEncodingWithTiesSpec = NoteEncodingSpecType(
    init_encoding_state_fn=NoteEncodingState,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=note_encoding_state_to_events,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=begin_tied_pitches_section,
    decode_event_fn=decode_note_event,
    flush_decoding_state_fn=flush_note_decoding_state,
)


def _add_note_to_sequence(
    ns: note_seq.NoteSequence,
    start_time: float,
    end_time: float,
    pitch: int,
    velocity: int,
    program: int = 0,
    is_drum: bool = False,
) -> None:
    """Add a note to a NoteSequence.

    Args:
        ns: NoteSequence to add note to
        start_time: Start time in seconds
        end_time: End time in seconds
        pitch: MIDI pitch value
        velocity: MIDI velocity value
        program: MIDI program number
        is_drum: Whether this is a drum note
    """
    end_time = max(end_time, start_time + MIN_NOTE_DURATION)
    ns.notes.add(
        start_time=start_time,
        end_time=end_time,
        pitch=pitch,
        velocity=velocity,
        program=program,
        is_drum=is_drum,
    )
    ns.total_time = max(ns.total_time, end_time)
