# PyTheaterTrainer

Creates MP3 training files for actors from individually recorded lines of a play.
Every character gets a folder with their lines, each with a few seconds of the cue
before and some context after. Lines that are close together (dialogues) are fused
into a single file.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (e.g. `sudo apt install ffmpeg`)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Recording the lines

Record every spoken line as a separate file named `NNN_Character.<ext>`:

```
recordings/
├── 000_Romeo.wav
├── 001_Juliet.wav
├── 002_Romeo.wav
├── 003_Nurse.mp3
└── ...
```

- `NNN` is the position in the play, starting at `000`.
- `Character` is the speaker's name. Use exactly the same spelling (and case) every time.
- Lines spoken together can be named `NNN_Alice + Belice`. They are included in the
  training files of each member.
- Supported formats: mp3, wav, m4a, ogg, flac, aac, wma, opus.

## Usage

```bash
.venv/bin/python theater_trainer.py recordings/ -o output/
```

Result:

```
output/
├── Romeo/
│   ├── 01_Romeo_000-004.mp3    # lines 000–004 fused (dialogue)
│   ├── 02_Romeo_008.mp3
│   └── ...
├── Juliet/
└── Nurse/
```

### How it works

1. All lines are placed in order on one timeline, with a short pause (`--gap`) between them.
2. For each character, consecutive lines are fused into one file if less than `--merge`
   seconds of other speakers lie between them, or if their context windows overlap.
3. Each file contains `--before` seconds of cue in front and `--after` seconds behind,
   with a short fade at the cut edges.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `-o`, `--output-dir` | `output` | Output folder |
| `-b`, `--before` | `5` | Seconds of context before a character's part |
| `-a`, `--after` | `2` | Seconds of context after a character's part |
| `-m`, `--merge` | `8` | Fuse lines if less than this many seconds of other speakers lie between them |
| `-g`, `--gap` | `0.6` | Pause inserted between consecutive lines |
| `--fade` | `0.3` | Fade in/out at the cut context edges (seconds) |
| `--trim` | off | Trim silence at the start and end of every recording |
| `--trim-db` | `-45` | Silence threshold for `--trim` (dBFS) |
| `--practice` | off | Additionally create `_practice.mp3` files where the character's own lines are silenced |
| `--full` | off | Also export the whole play as `full_play.mp3` |
| `--bitrate` | `128k` | MP3 bitrate |

## Testing without recordings

Before the real recordings exist, you can generate them with the offline
[Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech engine. Each character
gets its own voice.

```bash
.venv/bin/pip install -r requirements-test.txt
```

### From the play's PDF

```bash
# 1. Extract the spoken lines (stage directions are removed) into script.txt
.venv/bin/python pdf_to_script.py "Büchlein ....pdf" -o script.txt
```

The PDF must have the speaker names in **bold** at the start of a speech and stage
directions in (parentheses) and/or *italic*. Then review `script.txt`. The script
lists suspicious lines to check, and speeches without a speaker name in the PDF
appear as `???`. Spelling variants of names can be merged in `DEFAULT_ALIASES`
in `pdf_to_script.py`.

```bash
# 2. Generate the recordings with German voices (Swiss German is read like Standard German)
.venv/bin/python -m piper.download_voices --download-dir voices de_DE-thorsten-high \
    de_DE-thorsten_emotional-medium de_DE-karlsson-low de_DE-pavoque-low \
    de_DE-kerstin-low de_DE-ramona-low de_DE-eva_k-x_low
.venv/bin/python script_to_audio.py script.txt voices_de.json -o test_recordings --overwrite

# 3. Create the training files
.venv/bin/python theater_trainer.py test_recordings -o output --trim
```

`voices_de.json` assigns a voice to each character. Piper only has six distinct German
speakers (Thorsten, Karlsson, Pavoque, Kerstin, Ramona, Eva K, plus Thorsten in
eight moods such as `angry`, `sleepy`, `whisper`), so characters sharing a speaker are
told apart by pitch and speed:

```json
"Sophie": {"model": "kerstin", "pitch": -2, "length_scale": 1.1}
```

- `model` is a key from `models`, and `speaker` is the id or name for multi-speaker models.
- `pitch` is a shift in semitones (uses ffmpeg's `rubberband` filter).
- `length_scale` above 1 means slower speech.

Characters without an entry get a voice automatically. Group lines like
`Alice + Belice` use the first member's voice, and the trainer puts them into each
member's training files.

To preview all voices, generate one line per character into a separate folder, e.g.
from a small script file with one line per character.

### English example

`examples/romeo_and_juliet.txt` contains the end of the balcony scene of
*Romeo and Juliet* (public domain), about 2 minutes:

```bash
.venv/bin/python -m piper.download_voices --download-dir voices en_US-libritts_r-medium
.venv/bin/python script_to_audio.py examples/romeo_and_juliet.txt examples/voices_en.json -o test_recordings --overwrite
.venv/bin/python theater_trainer.py test_recordings -o output --trim
```
