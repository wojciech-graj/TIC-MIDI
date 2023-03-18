# TIC-MIDI

A MIDI-to-TIC-80 converter.

"The mission, the nightmares... they’re finally... over." - CT-5385 (presumably about switching to TIC-MIDI ;) )

### Features
- Utilizes up to all 4 audio channels simultaneously
- Has variable audio resolution to use more or less space in exchange for audio quality
- Can map midi tracks to sfx as dictated by the user, allowing for different instruments
- Has variable volume
- Sets tempo and speed to match input
- Can insert music into existing cart
- Can use any memory bank

### Limitations
- Can only write a single file to a single track
- Disregards multiple MIDI channels
- Only considers the following MIDI messages: `set_tempo`, `note_on`, `note_off`
- Only uses the Master Volume TIC-80 command
- Does not generate sfx and waveforms

### Usage

The script has only been tested on Python 3.9, so your mileage may vary on other versions.

Install the dependencies:
```
$ pip install -r requirements.txt
```

Then, simply invoke the script:
```
$ python tic_midi.py input_file -o output_file
```

To learn about the settings you can use, invoke the script with the `-h` flag:
```
$ python tic_midi.py -h
```

When playing the music with your lua code, set `sustain=true` in your `music` function call, since the converter currently doesn't re-play notes when starting a new frame.

A sample MIDI file, and the cartridge produced with it, can be found in the `example` directory.

### License
```
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
```

Portions of this software are copyright of their respective authors:
- [mido](https://github.com/mido/mido), Licensed under the [MIT License](https://opensource.org/licenses/MIT); Copyright (c) Ole Martin Bjørndalen
