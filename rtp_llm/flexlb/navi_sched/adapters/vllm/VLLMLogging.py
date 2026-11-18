"""Shared file-only Python logging for Navi's vLLM integration."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
from pathlib import Path
import tempfile

_FORMAT = "%(asctime)s %(levelname)s pid=%(process)d thread=%(threadName)s %(name)s:%(lineno)d %(message)s"


class ProcessRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """One file per PID, including when the handler is inherited through fork."""

    def __init__(
        self, directory=None, prefix="navi_sched_adapter",
        maxBytes=100 * 1024 * 1024, backupCount=10,
    ):
        if maxBytes <= 0 or backupCount <= 0:
            raise ValueError("file logging requires bounded positive rotation limits")
        self.directory = Path(directory or os.environ.get("NAVI_SCHED_LOG_DIR", "logs")).absolute()
        self.prefix = prefix
        self.owner_pid = os.getpid()
        self.error_count = 0
        filename = self.directory / f"{self.prefix}.{self.owner_pid}.log"
        super().__init__(
            filename, maxBytes=maxBytes, backupCount=backupCount,
            encoding="utf-8", delay=True,
        )

    def _open(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            return super()._open()
        except OSError:
            # A read-only worker cwd must not send diagnostics back to stderr.
            self.directory = Path(tempfile.gettempdir()) / f"navi_sched_logs_{os.getuid()}"
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.baseFilename = str(self.directory / f"{self.prefix}.{os.getpid()}.log")
            return super()._open()

    def emit(self, record):
        try:
            if self.owner_pid != os.getpid():
                if self.stream is not None:
                    self.stream.close()
                    self.stream = None
                self.owner_pid = os.getpid()
                self.baseFilename = str(self.directory / f"{self.prefix}.{self.owner_pid}.log")
            super().emit(record)
        except Exception:
            self.handleError(record)

    def handleError(self, record):
        # logging's default handleError writes to stderr and may block inference.
        self.error_count += 1


def make_file_handler(**options):
    return ProcessRotatingFileHandler(**options)


def write_vllm_logging_config(path):
    """Generate a dictConfig accepted by VLLM_LOGGING_CONFIG_PATH."""
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"file": {"format": _FORMAT}},
        "handlers": {"file": {
            "()": "navi_sched.adapters.vllm.VLLMLogging.make_file_handler",
            "prefix": "vllm", "directory": str(Path(path).absolute().parent),
            "maxBytes": int(os.environ.get("NAVI_SCHED_LOG_MAX_BYTES", 100 * 1024 * 1024)),
            "backupCount": int(os.environ.get("NAVI_SCHED_LOG_BACKUP_COUNT", 10)), "formatter": "file",
        }},
        "loggers": {name: {"handlers": ["file"], "level": os.environ.get("NAVI_SCHED_LOG_LEVEL", "INFO"), "propagate": False}
                    for name in ("vllm", "navi_sched", "uvicorn", "uvicorn.error", "uvicorn.access")},
        "root": {"handlers": ["file"], "level": "WARNING"},
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
