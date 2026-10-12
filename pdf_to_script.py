#!/usr/bin/env python3
"""Extract the spoken lines from a theater script PDF into an editable text file.

Expected PDF layout (as written in Word):
- each speech starts at the left margin with the speaker's name in bold,
- stage directions are in parentheses and/or italic,
- lines without a bold speaker name are headings or stage directions.

Output format, one speech per line (lines starting with '#' are comments):

    # Wellness-Anlage ist geschlossen
    Jacqueline: Was? D Wellness-Alag isch zue?
    Managerin: Mir hei leider e technische Defekt.

Review and correct this file before generating audio from it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pymupdf

# Different spellings of the same character in the PDF -> one name.
DEFAULT_ALIASES = {
    "Jaqueline": "Jacqueline",
    "Jacquelie": "Jacqueline",
    "Mosima": "Mosimann",
    "Fitness-Heidi": "Heidi",
    "Fitness-Heidy": "Heidi",
    "Bodyguard-Heidi": "Heidi",
    "Leader Tourist": "Tourist Leader",
    "Touist Leader": "Tourist Leader",
}

# Markers around italic/bold text inside a speech (treated as stage direction).
DIR_START, DIR_END = "\x01", "\x02"

UNKNOWN_SPEAKER = "???"

MIN_FONT_SIZE = 14       # header/footer are smaller than the body text
SPEAKER_MAX_X = 58       # speeches start at the left margin (continuations are indented)


def is_bold(span: dict) -> bool:
    return "Bold" in span["font"]


def is_italic(span: dict) -> bool:
    return "Italic" in span["font"]


def speech_text(spans: list[dict]) -> str:
    """Join spans; mark italic/bold text as direction, keep whitespace as is."""
    parts = []
    for s in spans:
        if s["text"].strip() and (is_italic(s) or is_bold(s)):
            parts.append(DIR_START + s["text"] + DIR_END)
        else:
            parts.append(s["text"])
    return "".join(parts)


def remove_directions(text: str) -> str:
    """Remove (stage directions) and italic/bold text, tidy up whitespace."""
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(f"{DIR_START}[^{DIR_END}]*{DIR_END}", " ", text)
    text = re.sub(f"[{DIR_START}{DIR_END}()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?…:;])", r"\1", text)
    return text.lstrip(" -–,.!?;:").rstrip(" -–")


def page_lines(page: pymupdf.Page) -> list[tuple[float, list[dict]]]:
    """Return (x, spans) per visual line; fragments on the same baseline are merged
    (Word splits justified lines into several pieces)."""
    rows: list[tuple[float, float, list[dict]]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["size"] >= MIN_FONT_SIZE]
            if any(s["text"].strip() for s in spans):
                rows.append((line["bbox"][3], line["bbox"][0], spans))
    rows.sort(key=lambda r: (round(r[0]), r[1]))
    merged: list[tuple[float, float, list[dict]]] = []
    for y, x, spans in rows:
        if merged and abs(merged[-1][0] - y) < 3:
            space = dict(spans[0], text=" ")
            merged[-1][2].extend([space] + spans)
        else:
            merged.append((y, x, list(spans)))
    result = []
    for _, x, spans in merged:
        while spans and not spans[0]["text"].strip():
            spans.pop(0)
        result.append((x, spans))
    return result


def extract(pdf_path: Path, start_marker: str, aliases: dict[str, str]) -> list[str]:
    doc = pymupdf.open(pdf_path)
    output: list[str] = []
    speaker: str | None = None
    speech: list[str] = []
    heading: list[str] = []
    started = False

    def flush_speech() -> None:
        nonlocal speaker, speech
        if speaker:
            text = remove_directions(" ".join(speech))
            if re.search(r"\w", text):
                output.append(f"{speaker}: {text}")
        speaker, speech = None, []

    def flush_heading() -> None:
        nonlocal heading
        if heading:
            output.append("")
            output.append("# " + " ".join(heading))
        heading = []

    for page in doc:
        for x, spans in page_lines(page):
            if not started:
                if is_bold(spans[0]) and spans[0]["text"].strip().startswith(start_marker):
                    started = True
                else:
                    continue

            at_margin = x <= SPEAKER_MAX_X
            text_spans = [s for s in spans if s["text"].strip()]
            n_name = 0
            while n_name < len(spans) and (is_bold(spans[n_name]) or not spans[n_name]["text"].strip()):
                n_name += 1
            name = "".join(s["text"] for s in spans[:n_name]).strip().rstrip(":").strip()
            name = re.sub(r"\s*\+\s*", " + ", name)

            if at_margin and name and n_name < len(spans) and not name.isupper() \
                    and not re.match(r"^\d+\.", name):
                # New speech: "Name (direction) text"
                flush_speech()
                flush_heading()
                speaker = aliases.get(name, name)
                speech.append(speech_text(spans[n_name:]))
            elif at_margin and all(is_italic(s) for s in text_spans):
                # Stage direction block
                flush_speech()
            elif at_margin and any(is_italic(s) for s in text_spans) \
                    and not all(is_italic(s) for s in text_spans):
                # Speech whose speaker name is missing in the PDF
                flush_speech()
                flush_heading()
                speaker = UNKNOWN_SPEAKER
                speech.append(speech_text(spans))
            elif at_margin:
                # Heading (scene title, LIED, Akt, ...)
                flush_speech()
                heading.append("".join(s["text"] for s in spans).strip())
            elif speaker:
                speech.append(speech_text(spans))
    flush_speech()
    flush_heading()
    return output


def suspicious(text: str) -> bool:
    """Speeches that may contain leftovers of stage directions."""
    return bool(re.match(r"[a-zäöü]", text)) or len(text.split()) <= 1 and not text.endswith(("!", "?", "…", "."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("script.txt"))
    parser.add_argument("--start", default="1. Akt",
                        help="bold text where the play begins (default: '1. Akt')")
    args = parser.parse_args()

    lines = extract(args.pdf, args.start, DEFAULT_ALIASES)
    if not lines:
        sys.exit("No speeches found - check --start and the PDF layout.")
    args.output.write_text("\n".join(lines).lstrip() + "\n", encoding="utf-8")

    speakers: dict[str, int] = {}
    for line in lines:
        if line and not line.startswith("#"):
            name = line.split(":", 1)[0]
            speakers[name] = speakers.get(name, 0) + 1
    print(f"Wrote {sum(speakers.values())} speeches to {args.output}")
    for name, count in sorted(speakers.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {name}")

    checks = [(i, line) for i, line in enumerate(args.output.read_text(encoding="utf-8").splitlines(), 1)
              if line and not line.startswith("#")
              and (line.startswith(UNKNOWN_SPEAKER) or suspicious(line.split(": ", 1)[1]))]
    if checks:
        print(f"\nPlease check these lines in {args.output}:")
        for number, line in checks:
            print(f"  {number:4d}: {line}")


if __name__ == "__main__":
    main()
