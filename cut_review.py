"""Local web page to listen to and tune the cuts in work/cuts.json.

Started through ``cut_recording.py review``. Only reachable from this computer.
"""

from __future__ import annotations

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from cut_recording import CUTS, RATE, best_mono, best_transcript, read_mono

PAGE = Path(__file__).with_name("cut_review.html")
EDITABLE = {"start", "end", "checked", "skip"}
MAX_WINDOW = 1800.0
LOCK = threading.Lock()


def load_lines() -> list[dict]:
    return json.loads(CUTS.read_text(encoding="utf-8"))["lines"]


def save_lines(lines: list[dict]) -> None:
    temporary = CUTS.with_suffix(".tmp")
    temporary.write_text(json.dumps({"lines": lines}, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    temporary.replace(CUTS)


class Handler(BaseHTTPRequestHandler):
    words: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def reply(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def reply_json(self, value, status: int = 200) -> None:
        self.reply(json.dumps(value, ensure_ascii=False).encode(), status=status)

    def time_range(self, query: dict) -> tuple[float, float]:
        start = max(0.0, float(query["start"][0]))
        end = float(query["end"][0])
        if not start < end <= start + MAX_WINDOW:
            raise ValueError("invalid time range")
        return start, end

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/lines":
                self.reply_json(load_lines())
            elif url.path == "/api/peaks":
                self.peaks(*self.time_range(query), int(query.get("n", ["1200"])[0]))
            elif url.path == "/api/audio":
                self.audio(*self.time_range(query))
            elif url.path == "/api/words":
                start, end = self.time_range(query)
                self.reply_json([w for w in self.words if start <= w["s"] < end])
            else:
                self.reply_json({"error": "not found"}, 404)
        except (KeyError, ValueError) as error:
            self.reply_json({"error": str(error)}, 400)

    def peaks(self, start: float, end: float, count: int) -> None:
        samples = read_mono(start, end, best_mono()).astype(np.float32) / 32768
        count = int(np.clip(count, 10, 4000))
        peaks = [float(np.abs(part).max()) if len(part) else 0.0
                 for part in np.array_split(samples, count)]
        self.reply_json([round(p, 4) for p in peaks])

    def audio(self, start: float, end: float) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(read_mono(start, end, best_mono()).tobytes())
        self.reply(buffer.getvalue(), "audio/wav")

    def do_POST(self) -> None:
        # Requiring JSON keeps other websites in the browser from sending edits here.
        parts = urlparse(self.path).path.strip("/").split("/")
        if self.headers.get("Content-Type") != "application/json" or parts[:2] != ["api", "line"]:
            self.reply_json({"error": "bad request"}, 400)
            return
        try:
            number = int(parts[2])
            changes = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            changes = {key: value for key, value in changes.items() if key in EDITABLE}
            with LOCK:
                lines = load_lines()
                lines[number].update(changes)
                line = lines[number]
                if line["start"] is not None and line["end"] <= line["start"]:
                    raise ValueError("the end must lie after the start")
                save_lines(lines)
            self.reply_json(line)
        except (IndexError, KeyError, ValueError, TypeError) as error:
            self.reply_json({"error": str(error)}, 400)


def serve(port: int) -> None:
    Handler.words = json.loads(best_transcript().read_text(encoding="utf-8"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop, every change is saved immediately)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
