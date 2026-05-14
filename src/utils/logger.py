import io
import os
import re
import sys
from datetime import datetime

from rich import print as rprint

from .constants import IS_GLOBAL_ZERO

__all__ = [
    "print_error",
    "print_info",
    "print_warning",
    "print",
    "print_warning_once",
    "setup_run_logging",
    "close_run_logging",
]

printed_warnings = set()
_log_file_handle = None
_stdout_original = None
_stderr_original = None
_stdout_tee = None
_stderr_tee = None
_ansi_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x1b\x07]*(?:\x07|\x1b\\))")


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


class _TimestampedTee(io.TextIOBase):
    def __init__(self, stream: io.TextIOBase, file_handle: io.TextIOBase):
        self.stream = stream
        self.file_handle = file_handle
        self._buffer = ""
        self._at_line_start = True

    def writable(self):
        return True

    def _write_with_timestamp(self, text: str):
        if not text:
            return

        # Strip ANSI control/color sequences so the log file stays plain-text readable.
        text = _ansi_re.sub("", text)
        parts = text.replace("\r", "\n").splitlines(keepends=True)
        for part in parts:
            if self._at_line_start:
                self.file_handle.write(f"{_timestamp_now()} ")
            self.file_handle.write(part)
            self._at_line_start = part.endswith("\n")
        self.file_handle.flush()

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        self.stream.write(data)
        self.stream.flush()
        self._write_with_timestamp(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file_handle.flush()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

    def isatty(self):
        return self.stream.isatty()


def setup_run_logging(run_dir: str, run_name: str) -> str:
    global _log_file_handle, _stdout_original, _stderr_original, _stdout_tee, _stderr_tee
    if _log_file_handle is not None:
        return _log_file_handle.name

    log_dir = os.path.join(run_dir, run_name, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    _log_file_handle = open(log_path, "a", encoding="utf-8")
    _stdout_original = sys.stdout
    _stderr_original = sys.stderr
    _stdout_tee = _TimestampedTee(_stdout_original, _log_file_handle)
    _stderr_tee = _TimestampedTee(_stderr_original, _log_file_handle)
    sys.stdout = _stdout_tee
    sys.stderr = _stderr_tee

    return log_path


def close_run_logging():
    global _log_file_handle, _stdout_original, _stderr_original, _stdout_tee, _stderr_tee
    if _log_file_handle is None:
        return

    sys.stdout = _stdout_original
    sys.stderr = _stderr_original

    _stdout_tee = None
    _stderr_tee = None
    _stdout_original = None
    _stderr_original = None

    _log_file_handle.flush()
    _log_file_handle.close()
    _log_file_handle = None


def print_error(text="", only_zero_rank=False):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[red bold]ERROR: [/red bold]{text}")


def print_warning(text="", only_zero_rank=False):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[yellow bold]WARNING: [/yellow bold]{text}")


def print_warning_once(text="", only_zero_rank=False):
    global printed_warnings
    if text in printed_warnings:
        return
    printed_warnings.add(text)
    print_warning(text, only_zero_rank)


def print_info(text="", only_zero_rank=True):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[blue bold]INFO: [/blue bold]{text}")


def print(text="", only_zero_rank=True):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(text)
