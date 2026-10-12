#!/usr/bin/env python3
"""Generate per-character MP3 training files from individually recorded lines.

Input: a folder of recordings named ``NNN_Character.<ext>`` (e.g. ``000_Hamlet.wav``,
``001_Ophelia.mp3``). The number defines the order in the play.

All lines are laid out on one timeline (with a configurable pause between lines).
For every character, each of their lines is cut out together with some seconds of
context before (the cue) and after. Lines of the same character that are close
together (e.g. a dialogue) are fused into one file.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_leading_silence

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"}
FILENAME_RE = re.compile(r"^(\d+)_(.+)$")


@dataclass
class Line:
    number: int
    character: str
    path: Path
    start_ms: int = 0
    end_ms: int = 0

    @property
    def characters(self) -> set[str]:
        """Group lines like "Alice + Belice" belong to every member."""
        return {name.strip() for name in self.character.split("+") if name.strip()}


@dataclass
class Segment:
    """A group of fused lines of one character."""
    lines: list[Line]

    @property
    def first(self) -> Line:
        return self.lines[0]

    @property
    def last(self) -> Line:
        return self.lines[-1]


def find_lines(input_dir: Path) -> list[Line]:
    lines = []
    for path in input_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        match = FILENAME_RE.match(path.stem)
        if not match:
            print(f"Skipping file with unexpected name: {path.name}", file=sys.stderr)
            continue
        lines.append(Line(int(match.group(1)), match.group(2).strip(), path))

    lines.sort(key=lambda line: line.number)
    numbers = [line.number for line in lines]
    duplicates = {n for n in numbers if numbers.count(n) > 1}
    if duplicates:
        sys.exit(f"Duplicate line numbers: {sorted(duplicates)}")
    return lines


def trim_silence(audio: AudioSegment, threshold_db: float, keep_ms: int) -> AudioSegment:
    start = detect_leading_silence(audio, silence_threshold=threshold_db)
    end = len(audio) - detect_leading_silence(audio.reverse(), silence_threshold=threshold_db)
    if end <= start:
        return audio
    return audio[max(0, start - keep_ms):min(len(audio), end + keep_ms)]


def build_timeline(lines: list[Line], gap_ms: int, trim: bool, trim_db: float) -> AudioSegment:
    """Concatenate all lines into one track and store each line's position on it."""
    timeline = AudioSegment.empty()
    gap = AudioSegment.silent(duration=gap_ms)
    for i, line in enumerate(lines):
        audio = AudioSegment.from_file(line.path)
        if trim:
            audio = trim_silence(audio, trim_db, keep_ms=100)
        if i > 0:
            timeline += gap
        line.start_ms = len(timeline)
        timeline += audio
        line.end_ms = len(timeline)
    return timeline


def group_segments(lines: list[Line], character: str, before_ms: int, after_ms: int,
                   merge_ms: int) -> list[Segment]:
    """Fuse a character's lines when they are close together or their contexts overlap."""
    segments: list[Segment] = []
    for line in (l for l in lines if character in l.characters):
        if segments:
            prev = segments[-1].last
            gap = line.start_ms - prev.end_ms
            contexts_overlap = (line.start_ms - before_ms) <= (prev.end_ms + after_ms)
            if gap <= merge_ms or contexts_overlap:
                segments[-1].lines.append(line)
                continue
        segments.append(Segment([line]))
    return segments


def cut_segment(timeline: AudioSegment, segment: Segment, character: str, before_ms: int,
                after_ms: int, fade_ms: int, mute_own: bool,
                all_lines: list[Line]) -> AudioSegment:
    start = max(0, segment.first.start_ms - before_ms)
    end = min(len(timeline), segment.last.end_ms + after_ms)
    clip = timeline[start:end]

    if mute_own:
        # Replace the character's own lines with silence of the same length
        # (also lines of the same character inside the context window).
        for line in all_lines:
            if character not in line.characters or line.end_ms <= start or line.start_ms >= end:
                continue
            s = max(line.start_ms, start) - start
            e = min(line.end_ms, end) - start
            clip = clip[:s] + AudioSegment.silent(duration=e - s, frame_rate=clip.frame_rate) + clip[e:]

    # Fade only where we cut into someone else's audio, to avoid clicks.
    if start < segment.first.start_ms:
        clip = clip.fade_in(min(fade_ms, segment.first.start_ms - start))
    if end > segment.last.end_ms:
        clip = clip.fade_out(min(fade_ms, end - segment.last.end_ms))
    return clip


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", name).strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="folder with NNN_Character recordings")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"),
                        help="output folder (default: ./output)")
    parser.add_argument("-b", "--before", type=float, default=5.0,
                        help="seconds of context before a character's part (default: 5)")
    parser.add_argument("-a", "--after", type=float, default=2.0,
                        help="seconds of context after a character's part (default: 2)")
    parser.add_argument("-m", "--merge", type=float, default=8.0,
                        help="fuse a character's lines if less than this many seconds of "
                             "other speakers lie between them (default: 8)")
    parser.add_argument("-g", "--gap", type=float, default=0.6,
                        help="pause inserted between consecutive lines (default: 0.6)")
    parser.add_argument("--fade", type=float, default=0.3,
                        help="fade in/out at cut context edges in seconds (default: 0.3)")
    parser.add_argument("--trim", action="store_true",
                        help="trim leading/trailing silence of every recording")
    parser.add_argument("--trim-db", type=float, default=-45.0,
                        help="silence threshold for --trim in dBFS (default: -45)")
    parser.add_argument("--practice", action="store_true",
                        help="also create practice files where the character's own lines "
                             "are replaced by silence")
    parser.add_argument("--full", action="store_true",
                        help="also write the whole play as one file (full_play.mp3)")
    parser.add_argument("--bitrate", default="128k", help="mp3 bitrate (default: 128k)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input folder not found: {args.input_dir}")

    lines = find_lines(args.input_dir)
    if not lines:
        sys.exit(f"No recordings named like 000_Name.mp3 found in {args.input_dir}")

    print(f"Loading {len(lines)} recordings ...")
    timeline = build_timeline(lines, int(args.gap * 1000), args.trim, args.trim_db)
    print(f"Total play length: {len(timeline) / 1000:.1f} s")

    before_ms, after_ms = int(args.before * 1000), int(args.after * 1000)
    merge_ms, fade_ms = int(args.merge * 1000), int(args.fade * 1000)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.full:
        timeline.export(args.output_dir / "full_play.mp3", format="mp3", bitrate=args.bitrate)

    characters = sorted(set().union(*(line.characters for line in lines)), key=str.lower)
    for character in characters:
        char_dir = args.output_dir / safe_name(character)
        char_dir.mkdir(exist_ok=True)
        segments = group_segments(lines, character, before_ms, after_ms, merge_ms)
        print(f"{character}: {sum(len(s.lines) for s in segments)} lines -> {len(segments)} files")

        variants = [("", False)] + ([("_practice", True)] if args.practice else [])
        for index, segment in enumerate(segments, start=1):
            numbers = f"{segment.first.number:03d}"
            if segment.last is not segment.first:
                numbers += f"-{segment.last.number:03d}"
            for suffix, mute in variants:
                clip = cut_segment(timeline, segment, character, before_ms, after_ms, fade_ms,
                                   mute, lines)
                name = f"{index:02d}_{safe_name(character)}_{numbers}{suffix}.mp3"
                clip.export(char_dir / name, format="mp3", bitrate=args.bitrate,
                            tags={"title": f"{character} {numbers}{suffix}",
                                  "artist": character, "track": str(index)})

    print(f"Done. Files written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
