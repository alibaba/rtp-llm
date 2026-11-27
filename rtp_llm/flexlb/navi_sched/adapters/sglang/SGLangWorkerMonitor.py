from __future__ import annotations

import asyncio
import os


class SGLangWorkerMonitor:
    """Watch tokenizer process lifetimes without polling or PID reuse races."""

    def __init__(self, on_exit, max_workers):
        self.on_exit = on_exit
        self.max_workers = max_workers
        self.entries = {}
        self.closed = False

    def register(self, worker, pid):
        if self.closed:
            raise RuntimeError("tokenizer process monitor is closed")
        if type(pid) is not int or pid <= 0:
            raise ValueError("tokenizer worker PID must be a positive integer")
        entry = self.entries.get(worker)
        if entry is not None:
            if entry[0] != pid:
                raise ValueError("tokenizer IPC address already belongs to another PID")
            return
        if len(self.entries) >= self.max_workers:
            raise RuntimeError("too many tokenizer worker processes")
        loop = asyncio.get_running_loop()
        fd = os.pidfd_open(pid)
        try:
            generation = object()
            loop.add_reader(fd, self._exited, worker, generation)
        except BaseException:
            os.close(fd)
            raise
        self.entries[worker] = (pid, fd, loop, generation)

    def _exited(self, worker, generation):
        entry = self.entries.get(worker)
        if entry is None or entry[3] is not generation:
            return
        self.remove(worker)
        self.on_exit(worker)

    def remove(self, worker):
        entry = self.entries.pop(worker, None)
        if entry is not None:
            _, fd, loop, _ = entry
            loop.remove_reader(fd)
            os.close(fd)

    def close(self):
        self.closed = True
        for worker in tuple(self.entries):
            self.remove(worker)
