#!/usr/bin/env python3
"""Serves viewer/ over local HTTP with Range-request support (needed for
<video> playback -- Chromium's media pipeline requires byte-range seeking
to demux MP4, and Python's stock http.server doesn't support it). Run this,
then open the printed URL."""
from __future__ import annotations

import argparse
import http.server
import os
import re
import socketserver
from pathlib import Path

VIEWER_DIR = Path(__file__).parent / "viewer"


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        file_size = os.path.getsize(path)
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if not match:
            self.send_response(416)
            self.end_headers()
            return None

        start_str, end_str = match.groups()
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        f = open(path, "rb")
        f.seek(start)
        length = end - start + 1

        self.send_response(206)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return _LimitedReader(f, length)

    def copyfile(self, source, outputfile):
        if isinstance(source, _LimitedReader):
            source.copy_to(outputfile)
        else:
            super().copyfile(source, outputfile)


class _LimitedReader:
    """Wraps a file object so shutil-style copy only reads `length` bytes."""

    def __init__(self, f, length: int):
        self.f = f
        self.remaining = length

    def copy_to(self, outputfile, chunk_size: int = 64 * 1024) -> None:
        try:
            while self.remaining > 0:
                chunk = self.f.read(min(chunk_size, self.remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        finally:
            self.f.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    with socketserver.TCPServer(("127.0.0.1", args.port), RangeRequestHandler) as httpd:
        print(f"Serving {VIEWER_DIR} at http://127.0.0.1:{args.port}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
