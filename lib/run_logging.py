"""Helpers for teeing training output to both the terminal and a log file."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)

    def fileno(self) -> int:
        return self.streams[0].fileno()


@contextmanager
def tee_run_output(path: str | os.PathLike[str]) -> Iterator[None]:
    """Append stdout and stderr to ``path`` while preserving terminal output."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_stdout, log_file)
        sys.stderr = _Tee(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr
