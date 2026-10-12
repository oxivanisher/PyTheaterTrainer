#!/usr/bin/env python3
"""Create recordings (NNN_Character.wav) from a script file with text-to-speech.

Script format, one speech per line ('#' lines and empty lines are ignored):

    Jacqueline: Was? D Wellness-Alag isch zue?
    Managerin: Mir hei leider e technische Defekt.

Voices are configured in a JSON file with one or more Piper models:

    {
      "models": {
        "thorsten": "voices/de_DE-thorsten-high.onnx",
        "emotional": "voices/de_DE-thorsten_emotional-medium.onnx"
      },
      "length_scale": 1.05,
      "voices": {
        "Dänu": {"model": "thorsten"},
        "Günter": {"model": "emotional", "speaker": "sleepy"},
        "Kassier": {"model": "thorsten", "pitch": -2, "length_scale": 0.95}
      }
    }

Per voice:
- "model": key in "models"
- "speaker": speaker id or name, for multi-speaker models
- "pitch": pitch shift in semitones (uses ffmpeg's rubberband filter)
- "length_scale": speed, above 1 = slower (overrides the global value)

Group lines like "Alice + Belice" use the first member's voice unless configured.
Characters without a configured voice get one picked automatically.

Setup:
    pip install -r requirements-test.txt
    python -m piper.download_voices --download-dir voices de_DE-thorsten-high ...
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import wave
import zlib
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

GENERATED_RE = re.compile(r"^\d+_.+\.wav$")


def read_script(path: Path) -> list[tuple[str, str]]:
    speeches = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            sys.exit(f"{path}:{number}: expected 'Name: text', got: {line}")
        name, text = (part.strip() for part in line.split(":", 1))
        if name and text:
            speeches.append((name, text))
    return speeches


class Voices:
    def __init__(self, config: dict):
        self.model_paths = {key: Path(path) for key, path in config["models"].items()}
        self.voices: dict[str, dict] = config.get("voices", {})
        self.length_scale = config.get("length_scale")
        self._loaded: dict[str, PiperVoice] = {}
        for key, path in self.model_paths.items():
            if not path.exists():
                sys.exit(f"Model '{key}' not found: {path}")
        for name, voice in self.voices.items():
            if voice.get("model") not in self.model_paths:
                sys.exit(f"Voice of '{name}' uses unknown model '{voice.get('model')}'")

    def model(self, key: str) -> PiperVoice:
        if key not in self._loaded:
            self._loaded[key] = PiperVoice.load(self.model_paths[key])
        return self._loaded[key]

    def settings(self, name: str) -> tuple[dict, str]:
        """Return the voice settings for a character and where they come from."""
        if name in self.voices:
            return self.voices[name], "configured"
        first = name.split("+")[0].strip()
        if first in self.voices:
            return self.voices[first], f"like {first}"
        # Stable automatic choice, so a character keeps its voice across runs.
        keys = sorted(self.model_paths)
        key = keys[zlib.crc32(name.encode()) % len(keys)]
        speakers = self.model(key).config.num_speakers
        return {"model": key, "speaker": zlib.crc32(name.encode()) % speakers}, "automatic"

    def synthesize(self, name: str, text: str, path: Path) -> None:
        settings, _ = self.settings(name)
        voice = self.model(settings["model"])
        speaker = settings.get("speaker")
        if isinstance(speaker, str):
            speaker = voice.config.speaker_id_map[speaker]
        syn_config = SynthesisConfig(
            speaker_id=speaker,
            length_scale=settings.get("length_scale", self.length_scale))
        with wave.open(str(path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        if settings.get("pitch"):
            shift_pitch(path, settings["pitch"])


def shift_pitch(path: Path, semitones: float) -> None:
    tmp = path.with_suffix(".tmp.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                    "-af", f"rubberband=pitch={2 ** (semitones / 12):.5f}:pitchq=quality",
                    str(tmp)], check=True)
    tmp.replace(path)


def describe(settings: dict) -> str:
    parts = [settings["model"]]
    if settings.get("speaker") is not None:
        parts.append(f"speaker {settings['speaker']}")
    if settings.get("pitch"):
        parts.append(f"pitch {settings['pitch']:+}")
    if settings.get("length_scale"):
        parts.append(f"speed {settings['length_scale']}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("script", type=Path, help="script file with 'Name: text' lines")
    parser.add_argument("voices", type=Path, help="JSON file with models and voices")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("test_recordings"))
    parser.add_argument("--overwrite", action="store_true",
                        help="delete previously generated NNN_*.wav files in the output folder")
    args = parser.parse_args()

    voices = Voices(json.loads(args.voices.read_text(encoding="utf-8")))
    speeches = read_script(args.script)
    if not speeches:
        sys.exit(f"No speeches found in {args.script}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_files = [p for p in args.output_dir.iterdir() if GENERATED_RE.match(p.name)]
    if old_files:
        if not args.overwrite:
            sys.exit(f"{args.output_dir} already contains {len(old_files)} recordings. "
                     "Use --overwrite to replace them.")
        for path in old_files:
            path.unlink()

    for name in dict.fromkeys(name for name, _ in speeches):
        settings, source = voices.settings(name)
        print(f"  {name:20s} {describe(settings)} ({source})")

    digits = max(3, len(str(len(speeches) - 1)))
    for number, (name, text) in enumerate(speeches):
        voices.synthesize(name, text, args.output_dir / f"{number:0{digits}d}_{name}.wav")
        print(f"\r{number + 1}/{len(speeches)}", end="", flush=True)
    print(f"\nWrote {len(speeches)} recordings to {args.output_dir}")


if __name__ == "__main__":
    main()
