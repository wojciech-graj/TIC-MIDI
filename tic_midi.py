"""
    TIC-MIDI, a MIDI to TIC-80 cartridge converter.
    Copyright (C) 2023  Wojciech Graj

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import ctypes
import itertools
import mido
import os
import struct
import sys
import time
from dataclasses import dataclass
from mido.midifiles.tracks import _to_abstime
from typing import List, Tuple, Optional, Dict


VERSION_STRING = "TIC-MIDI v0.1.1 2023-03-18 by Wojciech Graj"


class Chunk:
    """A cartridge chunk"""
    bank: int
    chunk_type: int
    size: int

    def __init__(self, chunk_type: int, size: int, bank: int = 0) -> None:
        self.chunk_type = chunk_type
        self.size = size
        self.bank = bank

    @classmethod
    def from_binary(self, binary: Tuple[int]):
        return Chunk(
            binary[0] & 0x1F,
            binary[1] | (binary[2] << 8),
            binary[0] >> 5
        )

    def serialize(self) -> Tuple[int]:
        """Return the 4-byte chunk header"""
        return (
            ((self.bank << 5) & 0xE0) | (self.chunk_type & 0x1F),
            self.size & 0x00FF,
            (self.size & 0xFF00) >> 8,
            0,
        )


class PatternRow(ctypes.LittleEndianStructure):
    _fields_ = [
        ("note", ctypes.c_uint32, 4),
        ("param1", ctypes.c_uint32, 4),
        ("param2", ctypes.c_uint32, 4),
        ("command", ctypes.c_uint32, 3),
        ("_sfx", ctypes.c_uint32, 6),  # Some weird bit rearranging has to be done because it crosses a byte boundary.
        ("octave", ctypes.c_uint32, 3),
    ]

    @property
    def sfx(self) -> int:
        return ((self._sfx >> 1) | (self._sfx << 5)) & 0x3f

    @sfx.setter
    def sfx(self, sfx: int) -> None:
        self._sfx = (sfx << 1) | (sfx >> 5)

    def set(self, note: int, param1: int, param2: int, command: int, sfx: int, octave: int) -> None:
        self.note = note
        self.param1 = param1
        self.param2 = param2
        self.command = command
        self.sfx = sfx
        self.octave = octave

    def serialize(self) -> Tuple[int]:
        return struct.unpack_from("BBB", self)


class Pattern:
    rows: List[PatternRow]

    def __init__(self) -> None:
        self.rows = [PatternRow() for i in range(64)]

    def serialize(self) -> Tuple[int]:
        return tuple(itertools.chain(*[row.serialize() for row in self.rows]))


class Frame(ctypes.LittleEndianStructure):
    _fields_ = [
        ("ch0", ctypes.c_uint32, 6),
        ("ch1", ctypes.c_uint32, 6),
        ("ch2", ctypes.c_uint32, 6),
        ("ch3", ctypes.c_uint32, 6),
    ]

    def get_ch(self, idx: int) -> int:
        if idx == 0:
            return self.ch0 - 1
        elif idx == 1:
            return self.ch1 - 1
        elif idx == 2:
            return self.ch2 - 1
        else:  # idx == 3
            return self.ch3 - 1

    def set_ch(self, idx: int, ch: int) -> None:
        if idx == 0:
            self.ch0 = ch + 1
        elif idx == 1:
            self.ch1 = ch + 1
        elif idx == 2:
            self.ch2 = ch + 1
        else:  # idx == 3
            self.ch3 = ch + 1

    def serialize(self) -> Tuple[int]:
        return struct.unpack_from("BBB", self)


class Track(ctypes.LittleEndianStructure):
    _fields_ = [
        ("_tempo", ctypes.c_int8),
        ("nrows", ctypes.c_uint8),
        ("_speed", ctypes.c_int8),
    ]
    frames: List[Frame]

    @property
    def tempo(self) -> int:
        return self._tempo + 150

    @tempo.setter
    def tempo(self, tempo: int) -> None:
        self._tempo = min(250, max(40, tempo)) - 150

    @property
    def speed(self) -> int:
        return self._speed + 6

    @speed.setter
    def speed(self, speed: int) -> None:
        self._speed = speed - 6

    def __init__(self) -> None:
        self.frames = [Frame() for i in range(16)]

    def serialize(self) -> Tuple[int]:
        return tuple(itertools.chain(*[frame.serialize() for frame in self.frames], struct.unpack_from("BBB", self)))


class TrackChunk(Chunk):
    tracks: List[Track]

    def __init__(self, bank: int = 0) -> None:
        super().__init__(14, 408, bank=bank)
        self.tracks = [Track() for i in range(8)]

    def serialize(self) -> Tuple[int]:
        return tuple(itertools.chain(super().serialize(), *[track.serialize() for track in self.tracks]))


class PatternChunk(Chunk):
    patterns: List[Pattern]

    def __init__(self, bank: int = 0) -> None:
        super().__init__(15, 11520, bank=bank)
        self.patterns = [Pattern() for i in range(60)]

    def serialize(self) -> Tuple[int]:
        return tuple(itertools.chain(super().serialize(), *[pattern.serialize() for pattern in self.patterns]))


@dataclass
class MessageExt:
    """MIDI Message with related metadata"""
    msg: mido.Message
    sfx: int


@dataclass
class Channel:
    """State of a single Track channel"""
    note: int = 0
    sfx: int = 0
    time_set: int = 0

    def set(self, note: int, sfx: int, time_set: int) -> None:
        self.note = note
        self.sfx = sfx
        self.time_set = time_set


class ChannelState:
    """State of all 4 Track channels"""
    channels: List[Channel]

    def __init__(self) -> None:
        self.channels = [Channel() for i in range(4)]

    def index(self, note: int, sfx: int) -> None:
        return next((i for i, channel in enumerate(self.channels) if channel.note == note and channel.sfx == sfx), -1)


def convert(
        mid: mido.MidiFile,
        tc: TrackChunk,
        pc: PatternChunk,
        resolution: int = 4,
        sfx_names: Optional[Dict] = None,
        track_idx: int = 0,
        octave_shift: int = 0):
    start_time = time.time()

    # Combine all messages from all tracks into a list of MessageExt
    next_unused_sfx = 0
    messages = []
    for track in mid.tracks:
        if next_unused_sfx > 63:
            print("WARNING: MIDI file contains over 64 tracks. Discarding excess tracks.")
            break
        if sfx_names:
            if track.name not in sfx_names:
                print(f"ERROR: MIDI Track '{track.name}' not found in sfx-names.")
                sys.exit(1)
            next_unused_sfx = sfx_names[track.name]
        messages.extend((MessageExt(msg, next_unused_sfx) for msg in _to_abstime(track)))
        next_unused_sfx += 1
    messages.sort(key=lambda msge: msge.msg.time)

    # Get track and set values
    track = tc.tracks[track_idx]
    track.speed = resolution + 2

    # Deduce tempo from MetaMessages
    tempo = 500000
    for msge in messages:
        if msge.msg.type == "set_tempo":
            tempo = msge.msg.tempo
    track.tempo = int(mido.tempo2bpm(tempo))

    # Remove MetaMessages
    messages = filter(lambda msge: not msge.msg.is_meta, messages)

    # Convert
    frame_idx = -1
    next_unused_pattern = 0
    scale = [
        4,
        2,
        1.5,
        1.2,
        1,
        .85,
        .75,
        .675,
    ][resolution]
    channel_state = ChannelState()
    try:
        for msge in messages:
            msg = msge.msg
            if msg.type not in {"note_on", "note_off"}:
                continue

            # Advance frame if neccessary, and calculate row index in pattern
            while True:
                pattern_row_idx = int(msg.time * 4 * scale / mid.ticks_per_beat) - 64 * frame_idx
                if pattern_row_idx < 64:
                    break
                frame_idx += 1
                if frame_idx > 15:
                    print("WARNING: Used all available frames in track.")
                    raise StopIteration()
                frame = track.frames[frame_idx]

            if msg.type == "note_on":  # Play a note
                # Find appropriate channel index
                channel_idx = channel_state.index(msg.note, msge.sfx)  # Replace channel with same note
                if channel_idx == -1:  # Replace empty channel
                    channel_idx = channel_state.index(0, 0)
                if channel_idx == -1:  # Replace oldest channel
                    channel_idx = channel_state.channels.index(min(channel_state.channels, key=lambda channel: channel.time_set))

                # Assign pattern to channel if unassigned
                if frame.get_ch(channel_idx) == -1:
                    if next_unused_pattern == 60:
                        print("WARNING: Used all available patterns.")
                        raise StopIteration()
                    frame.set_ch(channel_idx, next_unused_pattern)
                    next_unused_pattern += 1

                # Assign values to pattern row
                r = pc.patterns[frame.get_ch(channel_idx)].rows[pattern_row_idx]
                note_scaled = max(0, msg.note - 24)
                volume = msg.velocity // 8
                r.set(note_scaled % 12 + 4, volume, volume, 1, msge.sfx, max(0, min(7, note_scaled // 12 + octave_shift)))
                channel_state.channels[channel_idx].set(msg.note, msge.sfx, msg.time)
            else:  # Stop a note
                # Find appropriate channel index
                channel_idx = channel_state.index(msg.note, msge.sfx)
                if channel_idx == -1:
                    continue

                # Assign values to row
                r = pc.patterns[frame.get_ch(channel_idx)].rows[pattern_row_idx]
                r.note = 1
                channel_state.channels[channel_idx].set(0, 0, msg.time)
    except StopIteration:
        print("WARNING: Terminating early.")

    # Print summary
    print(f"Finished converting in {1000 * (time.time() - start_time):.2f} millis.")
    print(f"Converted {mido.tick2second(msg.time, mid.ticks_per_beat, tempo):.2f} seconds of music.")
    print(f"Used {frame_idx + 1}/16 frames on track {track_idx}.")
    print(f"Used {next_unused_pattern}/60 patterns.")


def tic_save(
        filename: str,
        track_chunk: TrackChunk,
        pattern_chunk: PatternChunk,
        insert: bool = False) -> None:
    if insert:
        filename_new = f"{filename}.new"
        with open(filename_new, "wb") as fd:
            with open(filename, "r+b") as fs:
                for header in iter(lambda: fs.read(4), b''):
                    chunk = Chunk.from_binary(tuple(header))
                    if ((chunk.chunk_type == 14 and chunk.bank == track_chunk.bank)
                            or (chunk.chunk_type == 15 and chunk.bank == pattern_chunk.bank)):
                        fs.seek(chunk.size, os.SEEK_CUR)
                    else:
                        fs.seek(-4, os.SEEK_CUR)
                        fd.write(fs.read(chunk.size + 4))
            fd.write(bytearray(track_chunk.serialize()))
            fd.write(bytearray(pattern_chunk.serialize()))
        os.replace(filename_new, filename)
    else:
        with open(filename, "wb") as f:
            f.write(bytearray(Chunk(17, 0).serialize()))
            f.write(bytearray(track_chunk.serialize()))
            f.write(bytearray(pattern_chunk.serialize()))

    print(f"Saved to {filename}.")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog='tic-midi',
        description='Convert a MIDI file to a TIC-80 cartridge.'
    )
    parser.add_argument('input')
    parser.add_argument('-v', '--version', action='version', version=VERSION_STRING, help=f"show version: {VERSION_STRING}")
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--resolution', default=4, type=int, help="Accepted values: [0,7]. Determines how many notes will be used per beat. Lower values use more space but can be more detailed.")
    parser.add_argument('--sfx-names', default="{}", help="Accepts a dict in of the form {'MIDI Track Name':sfx_index}. Used to map MIDI tracks to specific sfx.")
    parser.add_argument('--track', default=0, type=int, help="Accepted values: [0,7].")
    parser.add_argument('--bank', default=0, type=int, help="Memory bank.")
    parser.add_argument('--octave-shift', default=0, type=int, help="Shift all notes by some number of octaves.")
    ins_or_ovr = parser.add_mutually_exclusive_group()
    ins_or_ovr.add_argument('--insert', action='store_true', help="Insert Track and Pattern Chunks into an existing cartridge while leaving remaining chunks intact.")
    ins_or_ovr.add_argument('--overwrite', action='store_true', help="Overwrite an existing cartridge if it exists.")
    args = parser.parse_args()

    # Validate arguments
    if not args.output.endswith(".tic"):
        args.output += ".tic"
    args.resolution = max(0, min(7, args.resolution))
    args.sfx_names = eval(args.sfx_names)
    if not isinstance(args.sfx_names, dict):
        print("ERROR: sfx-names is not a dict.")
        sys.exit(1)
    args.track = max(0, min(7, args.track))
    args.bank = max(0, min(7, args.bank))

    output_file_exists = os.path.isfile(args.output)
    if output_file_exists and not args.overwrite and not args.insert:
        print(f"Cartridge '{args.output}' already exists. Use '--overwrite' to overwrite.")
        sys.exit(1)
    elif not output_file_exists and args.insert:
        print(f"Cartridge '{args.output}' does not exist, but is required by '--insert'.")
        sys.exit(1)

    # Run conversion and save
    tc = TrackChunk(bank=args.bank)
    pc = PatternChunk(bank=args.bank)

    convert(mido.MidiFile(args.input),
            tc,
            pc,
            resolution=args.resolution,
            sfx_names=args.sfx_names,
            track_idx=args.track,
            octave_shift=args.octave_shift)

    tic_save(args.output, tc, pc, insert=args.insert)
