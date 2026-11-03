#!/usr/bin/env python3
"""Cut one long rehearsal recording into NNN_Character.wav files, one per script line.

The rehearsal follows the script, but contains repeated passages and discussions.
Whisper mostly writes Swiss German as Standard German, so script lines are matched
fuzzily, using the order of the play. If a passage was repeated, the latest take wins.

    python cut_recording.py transcribe original/ZOOM0001.WAV   # GPU, a few minutes
    python cut_recording.py align script.txt                   # creates work/cuts.json
    python cut_recording.py denoise                            # optional: hum and room noise
    python cut_recording.py review                             # tune cuts in the browser
    python cut_recording.py export -o recordings               # writes NNN_Character.wav

"align" overwrites work/cuts.json, except for lines marked as checked in the review.
Everything runs locally. Only the Whisper model is downloaded once.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import site
import subprocess
import sys
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORK = Path("work")
MONO = WORK / "mono16k.wav"
CLEANED = WORK / "cleaned.wav"
CLEANED_MONO = WORK / "cleaned16k.wav"
TRANSCRIPT = WORK / "transcript.json"
TRANSCRIPT_CLEANED = WORK / "transcript_cleaned.json"
CUTS = WORK / "cuts.json"
RATE = 16000
# Narrow filters for mains hum (100 Hz and its harmonics), and rumble below speech.
HUM_FILTER = ("bandreject=f=100:width_type=h:w=4,bandreject=f=200:width_type=h:w=4,"
              "bandreject=f=300:width_type=h:w=4,highpass=f=70")
LOUDNESS_FILTER = "loudnorm=I=-20:TP=-2:LRA=11"

# Frequent Mundart words that do not look like their Standard German counterpart.
LEXICON = {
    "isch": "ist", "nid": "nicht", "ned": "nicht", "mir": "wir", "si": "sind sie", "dr": "der",
    "d": "die", "s": "das", "es": "ein das es", "e": "ein eine", "i": "ich", "het": "hat",
    "hei": "haben", "hani": "habe ich", "öpis": "etwas", "öbbis": "etwas", "chli": "ein bisschen",
    "aues": "alles", "aui": "alle", "au": "auch", "gäng": "immer", "jetz": "jetzt", "scho": "schon",
    "no": "noch", "nume": "nur", "mi": "mein mich", "mis": "mein", "di": "dich dein", "dä": "der den",
    "dene": "denen", "die": "die", "do": "da hier", "dir": "ihr", "öich": "euch", "nech": "euch",
    "öie": "euer", "üse": "unser", "wo": "wo", "wie": "wie", "was": "was", "wär": "wer",
    "gits": "gibt es", "chunnt": "kommt", "chume": "komme", "goh": "gehen", "göh": "gehen",
    "gseh": "sehen", "gseit": "gesagt", "gmacht": "gemacht", "gha": "gehabt", "cho": "gekommen",
    "wott": "will", "wei": "wollen", "söui": "soll", "muess": "muss", "cha": "kann",
    "chani": "kann ich", "bi": "bin bei", "de": "dann den", "auso": "also", "aber": "aber",
    "nüt": "nichts", "guet": "gut", "zäme": "zusammen", "hie": "hier",
    "vor": "vor", "vo": "von", "uf": "auf", "us": "aus", "ufe": "hinauf", "abe": "hinunter",
    "zum": "zum", "mit": "mit", "für": "für", "füre": "für", "sie": "sie", "ihm": "ihm",
    "mängisch": "manchmal", "lueg": "schau", "lueget": "schaut", "luege": "schauen",
    "wider": "wieder", "gli": "gleich", "glich": "gleich", "ou": "auch", "jo": "ja", "nei": "nein",
}


def norm(word: str) -> str:
    """Rough phonetic form, so that 'Schlüssu' and 'Schlüssel' end up close."""
    word = word.lower().replace("ß", "ss")
    word = word.replace("ä", "e").replace("ö", "e").replace("ü", "u")
    word = unicodedata.normalize("NFKD", word)
    word = "".join(c for c in word if c.isalnum())
    for old, new in (("sch", "s"), ("ch", "k"), ("ck", "k"), ("ph", "f"), ("v", "f"), ("tz", "z"),
                     ("ue", "u"), ("ie", "i"), ("ei", "i"), ("y", "i")):
        word = word.replace(old, new)
    return re.sub(r"(.)\1+", r"\1", word)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text.lower())


# ---------------------------------------------------------------- transcribe

def ensure_cuda_libraries() -> None:
    """ctranslate2 loads the CUDA libraries from the pip packages through LD_LIBRARY_PATH."""
    root = site.getsitepackages()[0] + "/nvidia"
    wanted = f"{root}/cublas/lib:{root}/cuda_nvrtc/lib:{root}/cudnn/lib"
    if wanted not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = wanted + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def read_mono(start: float = 0, end: float | None = None, path: Path = MONO) -> np.ndarray:
    with wave.open(str(path)) as w:
        first = int(start * RATE)
        w.setpos(min(first, w.getnframes()))
        count = w.getnframes() - first if end is None else int((end - start) * RATE)
        return np.frombuffer(w.readframes(max(0, count)), dtype=np.int16)


def best_mono() -> Path:
    return CLEANED_MONO if CLEANED_MONO.exists() else MONO


def best_transcript() -> Path:
    return TRANSCRIPT_CLEANED if TRANSCRIPT_CLEANED.exists() else TRANSCRIPT


def transcribe(args: argparse.Namespace) -> None:
    if not args.cleaned and args.recording is None:
        sys.exit("Name the recording, or use --cleaned after running denoise.")
    ensure_cuda_libraries()
    from faster_whisper import WhisperModel

    WORK.mkdir(exist_ok=True)
    if not args.cleaned:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.recording), "-ac", "1",
                        "-ar", str(RATE), "-c:a", "pcm_s16le", str(MONO)], check=True)
        (WORK / "source.txt").write_text(str(args.recording.resolve()), encoding="utf-8")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    audio = read_mono(path=CLEANED_MONO if args.cleaned else MONO).astype(np.float32) / 32768
    segments, _ = model.transcribe(
        audio, language="de", word_timestamps=True, condition_on_previous_text=False,
        vad_filter=True, vad_parameters={"threshold": 0.3, "min_silence_duration_ms": 400})
    words = []
    for segment in segments:
        for word in segment.words:
            words.append({"w": word.word.strip(), "s": round(word.start, 3),
                          "e": round(word.end, 3), "p": round(word.probability, 3)})
        print(f"\r{segment.end / 60:.1f} min", end="", flush=True)
    target = TRANSCRIPT_CLEANED if args.cleaned else TRANSCRIPT
    target.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(words)} words to {target}")


# --------------------------------------------------------------------- align

@dataclass
class Candidate:
    line: int
    first: int  # word index in the transcript
    last: int
    score: float  # 0..1 share of the line's weight that was found
    importance: float = 1.0
    estimated: bool = False  # position only derived from the neighbouring lines


def read_script(path: Path) -> list[tuple[str, str]]:
    speeches = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and ":" in line:
            name, text = (part.strip() for part in line.split(":", 1))
            if name and text:
                speeches.append((name, text))
    return speeches


class Similarity:
    """Word similarity between script (Mundart) and transcript (mostly Standard German)."""

    def __init__(self, script_tokens: set[str], heard_tokens: list[str], learned=None):
        from rapidfuzz.distance import Levenshtein
        from rapidfuzz.process import cdist

        self.heard_vocab = sorted(set(heard_tokens))
        self.heard_index = {t: i for i, t in enumerate(self.heard_vocab)}
        self.script_vocab = sorted(script_tokens)
        self.script_index = {t: i for i, t in enumerate(self.script_vocab)}
        heard_norm = [norm(t) for t in self.heard_vocab]
        script_norm = [norm(t) for t in self.script_vocab]
        self.matrix = cdist(script_norm, heard_norm, scorer=Levenshtein.normalized_similarity,
                            workers=-1, dtype=np.float32)
        self.matrix = np.maximum(0, (self.matrix - 0.4) / 0.6)  # below 40% similar means unrelated
        for token, meanings in LEXICON.items():
            if token in self.script_index:
                for meaning in meanings.split():
                    if meaning in self.heard_index:
                        self.matrix[self.script_index[token], self.heard_index[meaning]] = 0.9
        for token, heard in (learned or {}).items():
            for meaning in heard:
                if token in self.script_index and meaning in self.heard_index:
                    self.matrix[self.script_index[token], self.heard_index[meaning]] = 0.9

    def row(self, token: str, heard_ids: np.ndarray) -> np.ndarray:
        return self.matrix[self.script_index[token]][heard_ids]


def smith_waterman(weights: np.ndarray, sims: np.ndarray, gap: float = 0.25) -> np.ndarray:
    """Best local alignment score of the line ending at each transcript word."""
    best = np.zeros(sims.shape[1] + 1)
    previous = np.zeros(sims.shape[1] + 1)
    positions = np.arange(sims.shape[1] + 1) * gap
    for weight, sim in zip(weights, sims):
        score = weight * (sim * 1.3 - 0.3)  # a miss costs a quarter of what a hit gives
        current = np.maximum(0, previous - weight * 0.3)
        current[1:] = np.maximum(current[1:], previous[:-1] + score)
        # Skipping heard words (filler, mishearings): running maximum with a cost per word.
        current = np.maximum.accumulate(current + positions) - positions
        best = np.maximum(best, current)
        previous = current
    return best[1:]


def find_candidates(index: int, tokens: list[str], sim: Similarity, heard_ids: np.ndarray,
                    weight_of, keep: int = 14, min_score: float = 0.3,
                    offset: int = 0) -> list[Candidate]:
    if not tokens:
        return []
    weights = np.array([weight_of(t) for t in tokens])
    sims = np.stack([sim.row(t, heard_ids) for t in tokens])
    scores = np.minimum(1.0, smith_waterman(weights, sims) / weights.sum())
    radius = max(2, len(tokens))
    found = []
    order = np.argsort(scores)[::-1]
    taken = np.zeros(len(scores), dtype=bool)
    for end in order[:4000]:
        if scores[end] < min_score or len(found) >= keep:
            break
        if taken[end]:
            continue
        # Find the start by aligning the reversed line backwards from the end.
        low = max(0, end - radius * 4 - 8)
        back = smith_waterman(weights[::-1], sims[::-1, low:end + 1][:, ::-1])
        first = end - int(np.argmax(back))
        taken[max(0, first - 1):min(len(scores), end + radius)] = True
        if any(f.first - offset <= first <= f.last - offset for f in found):
            continue  # the same take again, only with a worse end
        found.append(Candidate(index, int(first) + offset, int(end) + offset, float(scores[end])))
    return found


def chain(candidates: list[list[Candidate]], lookback: int = 40,
          later_bonus: float = 0.25) -> list[Candidate | None]:
    """Pick one candidate per line so that the play's order holds, maximizing the total score."""
    best: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {}
    for i, options in enumerate(candidates):
        # Among similarly good takes a later one counts more, so that a whole repeated
        # passage wins as a block, even if one of its lines was understood badly.
        top = max((c.score for c in options), default=0)
        strong = sorted((c for c in options if c.score >= 0.75 * top), key=lambda c: c.first)
        gain = {id(c): c.score * c.importance for c in options}
        for rank, c in enumerate(strong):
            gain[id(c)] *= 1 + later_bonus * rank / max(1, len(strong) - 1)
        for c_index, c in enumerate(options):
            value = gain[id(c)]
            parent = None
            for j in range(max(0, i - lookback), i):
                for p_index, p in enumerate(candidates[j]):
                    if p.last < c.first + 2:
                        v = best[(j, p_index)][0] + gain[id(c)] - 0.3 * (i - j - 1)
                        if v > value:
                            value, parent = v, (j, p_index)
            best[(i, c_index)] = (value, parent)
    picked: list[Candidate | None] = [None] * len(candidates)
    if not best:
        return picked
    node = max(best, key=lambda key: best[key][0])
    while node:
        picked[node[0]] = candidates[node[0]][node[1]]
        node = best[node][1]
    return picked


def prefer_latest(picked, candidates, line_tokens, sim, heard_ids, weight_of,
                  keep_ratio: float) -> set[int]:
    """Going backwards, move each line to its latest take that still fits before the next line.

    A short line needs a nearly equal score, since "Wie?" also occurs in any discussion.
    A take directly in front of the next line may score lower: that is how a retake looks.
    """
    moved = set()
    next_first = None
    for i in range(len(picked) - 1, -1, -1):
        current = picked[i]
        if current is None:
            continue
        tokens = line_tokens[i]
        high = next_first if next_first is not None else min(len(heard_ids), current.last + 200)
        options = []
        if high > current.first:
            options = find_candidates(i, tokens, sim, heard_ids[current.first:high], weight_of,
                                      keep=20, min_score=max(0.1, 0.5 * current.score),
                                      offset=current.first)
        ratio = 0.9 if len(tokens) < 4 else keep_ratio
        fitting = [c for c in options if c.first > current.last and (
            c.score >= ratio * current.score
            or (len(tokens) >= 3 and c.score >= 0.3 and next_first is not None
                and c.last >= next_first - 6))]
        if fitting:
            picked[i] = max(fitting, key=lambda c: c.first)
            picked[i].importance = current.importance
            candidates[i].append(picked[i])
            moved.add(i)
        next_first = picked[i].first
    return moved


def fill_gaps(picked, line_tokens, sim, heard_ids, weight_of, estimate: bool) -> None:
    """Place the remaining (mostly short) lines in the words between their neighbours."""
    i = 0
    while i < len(picked):
        if picked[i] is not None:
            i += 1
            continue
        run_end = i
        while run_end + 1 < len(picked) and picked[run_end + 1] is None:
            run_end += 1
        low = picked[i - 1].last + 1 if i else 0
        high = picked[run_end + 1].first if run_end + 1 < len(picked) else len(heard_ids)
        run = list(range(i, run_end + 1))
        if high > low:
            options = []
            for k in run:
                found = find_candidates(k, line_tokens[k], sim, heard_ids[low:high], weight_of,
                                        keep=8, min_score=0.15, offset=low)
                for c in found:
                    c.importance = sum(weight_of(t) for t in line_tokens[k])
                # In a wide gap (a discussion) a weak match is probably a coincidence.
                wide = high - low > 4 * sum(len(line_tokens[m]) for m in run) + 10
                options.append([c for c in found if c.score >= (0.35 if wide else 0.15)])
            for k, c in zip(run, chain(options, lookback=len(run))):
                picked[k] = c
            if estimate:
                estimate_rest(picked, run, line_tokens, low, high)
        i = run_end + 1


def estimate_rest(picked, run, line_tokens, low, high) -> None:
    """Share the unassigned words among lines nothing matched for, by their length."""
    k = 0
    while k < len(run):
        if picked[run[k]] is not None:
            low = picked[run[k]].last + 1
            k += 1
            continue
        end = k
        while end + 1 < len(run) and picked[run[end + 1]] is None:
            end += 1
        stop = picked[run[end + 1]].first if end + 1 < len(run) else high
        members = run[k:end + 1]
        lengths = np.array([max(1, len(line_tokens[m])) for m in members], dtype=float)
        if stop - low >= 0.5 * lengths.sum():
            edges = low + np.concatenate(([0], np.cumsum(lengths))) / lengths.sum() * (stop - low)
            for m, a, b in zip(members, edges, edges[1:]):
                if int(b) > int(a):
                    picked[m] = Candidate(m, int(a), int(b) - 1, 0.0, estimated=True)
        k = end + 1


def extend_spans(picked, line_tokens, heard_times, max_pause: float = 1.0) -> None:
    """Complete matches that stopped short, up to the expected number of words."""
    for i, c in enumerate(picked):
        if c is None or c.estimated:
            continue
        before = next((picked[k].last for k in range(i - 1, -1, -1) if picked[k]), -1)
        after = next((picked[k].first for k in range(i + 1, len(picked)) if picked[k]),
                     len(heard_times))
        wanted = len(line_tokens[i])
        first, last = c.first, c.last
        while last - first + 1 < wanted:
            pause_next = (heard_times[last + 1][0] - heard_times[last][1]
                          if last + 1 < after else 1e9)
            pause_previous = (heard_times[first][0] - heard_times[first - 1][1]
                              if first - 1 > before else 1e9)
            if min(pause_next, pause_previous) > max_pause:
                break
            if pause_next <= pause_previous:
                last += 1
            else:
                first -= 1
        picked[i] = Candidate(c.line, first, last, c.score, c.importance, c.estimated)


def learn_lexicon(picked, line_tokens, heard_tokens) -> dict[str, list[str]]:
    """Find Mundart/Standard German word pairs that keep occurring together in matched lines."""
    together: dict[str, dict[str, int]] = {}
    script_count: dict[str, int] = {}
    heard_count: dict[str, int] = {}
    for token in heard_tokens:
        heard_count[token] = heard_count.get(token, 0) + 1
    for c in picked:
        if c is None or c.score < 0.5:
            continue
        heard = set(heard_tokens[c.first:c.last + 1])
        for token in set(line_tokens[c.line]):
            script_count[token] = script_count.get(token, 0) + 1
            for h in heard:
                together.setdefault(token, {})
                together[token][h] = together[token].get(h, 0) + 1
    learned: dict[str, list[str]] = {}
    for token, pairs in together.items():
        if script_count[token] < 2:
            continue
        for h, count in pairs.items():
            if count / script_count[token] >= 0.6 and count / heard_count[h] >= 0.25:
                learned.setdefault(token, []).append(h)
    return learned


def speech_edges(start: float, end: float, lower: float, upper: float, pad_before: float,
                 pad_after: float, max_shift: float = 0.6) -> tuple[float, float]:
    """Move word timestamps outwards to where the audio actually falls quiet."""
    samples = read_mono(max(0, start - max_shift - 1), end + max_shift + 1, best_mono())
    offset = max(0, start - max_shift - 1)
    hop = RATE // 100
    frames = samples[:len(samples) // hop * hop].reshape(-1, hop).astype(np.float32)
    if not len(frames):
        return start, end
    level = np.sqrt((frames ** 2).mean(axis=1) + 1)
    quiet = np.percentile(level, 15) * 2.0
    a = int((start - offset) * 100)
    b = min(int((end - offset) * 100), len(level) - 1)
    a_limit = max(0, a - int(max_shift * 100))
    while a > a_limit and level[a] > quiet:
        a -= 1
    b_limit = min(len(level) - 1, b + int(max_shift * 100))
    while b < b_limit and level[b] > quiet:
        b += 1
    return max(lower, offset + a / 100 - pad_before), min(upper, offset + b / 100 + pad_after)


def quietest_point(start: float, end: float) -> float:
    samples = read_mono(start, end, best_mono()).astype(np.float32)
    hop = RATE // 100
    if len(samples) < hop:
        return (start + end) / 2
    frames = samples[:len(samples) // hop * hop].reshape(-1, hop)
    return start + (int(np.argmin((frames ** 2).mean(axis=1))) + 0.5) / 100


def separate_neighbours(lines: list[dict]) -> None:
    """Let overlapping consecutive cuts meet at the quietest point, to spare the words.

    A confirmed cut stays as it is, and its unconfirmed neighbour gives way.
    """
    placed = [line for line in lines if line["start"] is not None]
    for a, b in zip(placed, placed[1:]):
        if a["end"] <= b["start"] or (a["checked"] and b["checked"]):
            continue
        if a["checked"]:
            b["start"] = a["end"]
        elif b["checked"]:
            a["end"] = b["start"]
        else:
            a["end"] = b["start"] = round(quietest_point(b["start"], a["end"]), 2)


def place_lines(words, speeches, line_tokens, weight_of, args) -> list[dict | None]:
    """Find every line in one transcript. Returns its position in seconds, if any."""
    heard_tokens, token_word = [], []
    for index, word in enumerate(words):
        for token in tokenize(word["w"]):
            heard_tokens.append(token)
            token_word.append(index)
    script_vocab = {t for tokens in line_tokens for t in tokens}

    learned = None
    picked: list[Candidate | None] = []
    candidates: list[list[Candidate]] = []
    for round_number in range(args.rounds):
        sim = Similarity(script_vocab, heard_tokens, learned)
        heard_ids = np.array([sim.heard_index[t] for t in heard_tokens])
        candidates = []
        for i, tokens in enumerate(line_tokens):
            options = find_candidates(i, tokens, sim, heard_ids, weight_of)
            total = sum(weight_of(t) for t in tokens)
            for c in options:
                c.importance = min(total, 20.0)
            candidates.append(options)
        picked = chain(candidates)
        found = [c for c in picked if c is not None]
        print(f"Round {round_number + 1}: {len(found)}/{len(speeches)} lines placed, "
              f"mean score {np.mean([c.score for c in found]):.2f}")
        learned = learn_lexicon(picked, line_tokens, heard_tokens)

    fill_gaps(picked, line_tokens, sim, heard_ids, weight_of, estimate=False)
    moved = prefer_latest(picked, candidates, line_tokens, sim, heard_ids, weight_of,
                          args.keep_ratio)
    fill_gaps(picked, line_tokens, sim, heard_ids, weight_of, estimate=True)
    extend_spans(picked, line_tokens, [(words[token_word[t]]["s"], words[token_word[t]]["e"])
                                       for t in range(len(heard_tokens))])

    def describe(c: Candidate) -> dict:
        return {"start": words[token_word[c.first]]["s"], "end": words[token_word[c.last]]["e"],
                "score": round(c.score, 2),
                "heard": " ".join(w["w"] for w in words[token_word[c.first]:token_word[c.last] + 1])}

    placed: list[dict | None] = []
    for i, chosen in enumerate(picked):
        if chosen is None or (chosen.estimated and describe(chosen)["end"] - describe(chosen)["start"]
                              > 3 + 0.8 * len(line_tokens[i])):
            placed.append(None)  # the words between the neighbours are a discussion
            continue
        takes = [describe(c) for c in candidates[i] if c.score >= 0.6 * chosen.score]
        placed.append({**describe(chosen), "estimated": chosen.estimated, "moved": i in moved,
                       "importance": sum(weight_of(t) for t in line_tokens[i]),
                       "takes": takes + [describe(chosen)]})
    return placed


def merge_placements(placements: list[list[dict | None]], line_tokens,
                     keep_ratio: float) -> list[dict | None]:
    """Combine the results of several transcripts: each catches lines the others miss.

    First find the order that fits best across all of them, then let each line move
    to its latest good take, as for a single transcript.
    """
    options: list[list[Candidate]] = []
    for i in range(len(line_tokens)):
        found = []
        for placement in placements:
            p = placement[i]
            if p is not None:
                c = Candidate(i, int(p["start"] * 100), int(p["end"] * 100), p["score"],
                              min(p["importance"], 20.0), p["estimated"])
                c.placement = p
                found.append(c)
        options.append(found)
    picked = chain(options)
    next_first = None
    for i in range(len(picked) - 1, -1, -1):
        current = picked[i]
        if current is None:
            continue
        ratio = 0.9 if len(line_tokens[i]) < 4 else keep_ratio
        # "moved" means this transcript saw it as a retake directly before the next line.
        fitting = [c for c in options[i] if not c.estimated and c.first > current.first
                   and (c.score >= ratio * current.score
                        or (c.placement["moved"] and c.score >= 0.3))
                   and (next_first is None or c.last <= next_first)]
        if fitting or current.estimated:
            fitting = fitting or [c for c in options[i] if not c.estimated
                                  and (next_first is None or c.last <= next_first)]
        if fitting:
            picked[i] = max(fitting, key=lambda c: c.first)
        next_first = picked[i].first

    merged: list[dict | None] = []
    for i, c in enumerate(picked):
        if c is None:
            merged.append(None)
            continue
        takes = {}
        for placement in placements:
            if placement[i] is not None:
                for take in placement[i]["takes"]:
                    if abs(take["start"] - c.placement["start"]) < 1200:
                        takes.setdefault(round(take["start"]), take)
        moved = (c.placement["moved"] or any(o.first > c.last or o.last < c.first
                                             for o in options[i]))
        merged.append({**c.placement, "moved": moved,
                       "takes": [takes[k] for k in sorted(takes)]})
    return merged


def align(args: argparse.Namespace) -> None:
    speeches = read_script(args.script)
    line_tokens = [tokenize(text) for _, text in speeches]

    # Common words prove little: weigh every script word by how rare it is in the script.
    counts: dict[str, int] = {}
    for tokens in line_tokens:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

    def weight_of(token: str) -> float:
        return float(np.clip(np.log(len(speeches) / counts[token]), 0.4, 5.0))

    placements = []
    for path in (TRANSCRIPT, TRANSCRIPT_CLEANED):
        if path.exists():
            print(f"Matching {path} ...")
            words = json.loads(path.read_text(encoding="utf-8"))
            placements.append(place_lines(words, speeches, line_tokens, weight_of, args))
    picked = merge_placements(placements, line_tokens, args.keep_ratio)

    old = {}
    if CUTS.exists():
        old = {c["number"]: c for c in json.loads(CUTS.read_text(encoding="utf-8"))["lines"]
               if c.get("checked")}

    lines = []
    for i, (name, text) in enumerate(speeches):
        chosen = picked[i]
        entry = {"number": i, "character": name, "text": text, "checked": False, "takes": [],
                 "start": None, "end": None, "score": 0.0, "heard": "", "moved_to_later": False,
                 "estimated": False}
        if chosen is not None:
            previous_end = picked[i - 1]["end"] if i and picked[i - 1] else 0.0
            later = next((picked[k]["start"] for k in range(i + 1, len(picked)) if picked[k]), 1e9)
            start, end = speech_edges(chosen["start"], chosen["end"], previous_end, later,
                                      args.pad_before, args.pad_after)
            entry.update(start=round(start, 2), end=round(end, 2), score=chosen["score"],
                         heard=chosen["heard"], moved_to_later=chosen["moved"],
                         estimated=chosen["estimated"],
                         takes=[{**take, "start": round(take["start"], 2),
                                 "end": round(take["end"], 2)} for take in chosen["takes"]])
        if i in old and old[i]["character"] == name and old[i]["text"] == text:
            entry = old[i]
        lines.append(entry)

    separate_neighbours(lines)
    CUTS.write_text(json.dumps({"lines": lines}, ensure_ascii=False, indent=1), encoding="utf-8")
    placed = [l for l in lines if l["start"] is not None]
    print(f"Wrote {CUTS}: {len(placed)}/{len(lines)} lines placed, "
          f"{sum(l['score'] < 0.5 for l in placed)} with low score, "
          f"{sum(l['moved_to_later'] for l in placed)} moved to a later take")


# ------------------------------------------------------------------- denoise

def denoise(args: argparse.Namespace) -> None:
    """Remove the hum, reduce the room noise with DeepFilterNet and even out the loudness."""
    if not args.deep_filter.exists():
        sys.exit(f"{args.deep_filter} not found, see the README for the download.")
    source = (WORK / "source.txt").read_text(encoding="utf-8")
    temporary = WORK / "denoise_tmp"
    temporary.mkdir(exist_ok=True)
    without_hum = temporary / "without_hum.wav"
    print("Removing hum ...")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source, "-af", HUM_FILTER,
                    str(without_hum)], check=True)
    print("Reducing noise with DeepFilterNet ...")
    enhanced = temporary / "enhanced"
    enhanced.mkdir(exist_ok=True)
    # -D keeps the audio in sync with the transcript's timestamps.
    subprocess.run([str(args.deep_filter), "-D", "--atten-lim-db", str(args.limit),
                    "-o", str(enhanced), str(without_hum)], check=True, stdout=subprocess.DEVNULL)
    print("Evening out the loudness ...")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(enhanced / without_hum.name),
                    "-af", LOUDNESS_FILTER, "-ar", "48000", str(CLEANED)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(CLEANED), "-ac", "1",
                    "-ar", str(RATE), "-c:a", "pcm_s16le", str(CLEANED_MONO)], check=True)
    without_hum.unlink()
    (enhanced / without_hum.name).unlink()
    enhanced.rmdir()
    temporary.rmdir()
    print(f"Wrote {CLEANED}. Export and review use it from now on.")


# -------------------------------------------------------------------- export

def export(args: argparse.Namespace) -> None:
    from theater_trainer import safe_name

    source = (WORK / "source.txt").read_text(encoding="utf-8")
    if CLEANED.exists() and not args.original_audio:
        source = str(CLEANED)
    print(f"Cutting from {source}")
    lines = json.loads(CUTS.read_text(encoding="utf-8"))["lines"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old = [p for p in args.output_dir.iterdir() if re.match(r"^\d+_.+\.wav$", p.name)]
    if old:
        if not args.overwrite:
            sys.exit(f"{args.output_dir} already contains {len(old)} recordings. "
                     "Use --overwrite to replace them.")
        for path in old:
            path.unlink()

    usable = [l for l in lines if l["start"] is not None and not l.get("skip")
              and (l.get("checked") or l["score"] >= args.min_score)]
    digits = max(3, len(str(len(lines) - 1)))
    for count, line in enumerate(usable, 1):
        name = " + ".join(safe_name(part) for part in line["character"].split("+"))
        path = args.output_dir / f"{line['number']:0{digits}d}_{name}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(line["start"]),
                        "-to", str(line["end"]), "-i", source, "-af",
                        "afade=t=in:d=0.02", str(path)], check=True)
        print(f"\r{count}/{len(usable)}", end="", flush=True)
    print(f"\nWrote {len(usable)} of {len(lines)} lines to {args.output_dir}")
    missing = [l["number"] for l in lines if l not in usable]
    if missing:
        print(f"Left out: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("transcribe", help="transcribe the recording with Whisper")
    p.add_argument("recording", type=Path, nargs="?")
    p.add_argument("--cleaned", action="store_true",
                   help="transcribe the audio from the denoise step instead of a recording")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.set_defaults(func=transcribe)

    p = sub.add_parser("align", help="find each script line in the transcript")
    p.add_argument("script", type=Path)
    p.add_argument("--rounds", type=int, default=2,
                   help="alignment rounds; later ones use word pairs learned from earlier ones")
    p.add_argument("--keep-ratio", type=float, default=0.75,
                   help="use a later take if it scores at least this share of the best (default: 0.75)")
    p.add_argument("--pad-before", type=float, default=0.25,
                   help="seconds kept before the speech starts (default: 0.25)")
    p.add_argument("--pad-after", type=float, default=0.35,
                   help="seconds kept after the speech ends (default: 0.35)")
    p.set_defaults(func=align)

    p = sub.add_parser("denoise", help="remove hum and room noise from the recording")
    p.add_argument("--deep-filter", type=Path, default=Path("tools/deep-filter"),
                   help="path of the DeepFilterNet program (default: tools/deep-filter)")
    p.add_argument("--limit", type=float, default=12.0,
                   help="reduce the noise by at most this many dB; 100 removes all of it (default: 12)")
    p.set_defaults(func=denoise)

    p = sub.add_parser("review", help="tune the cuts in the browser")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=lambda a: __import__("cut_review").serve(a.port))

    p = sub.add_parser("export", help="write NNN_Character.wav files")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("recordings"))
    p.add_argument("--min-score", type=float, default=0.0,
                   help="leave out unchecked lines below this score (default: keep all)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--original-audio", action="store_true",
                   help="cut from the original recording even if a cleaned one exists")
    p.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
