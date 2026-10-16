"""Independent structured evidence source with bounded storage and joined shutdown."""
import json
from pathlib import Path

from artifacts.json_io import write_json
from runtime.observation import FrozenSnapshot
from runtime.sample_worker import SampleWorker, SamplingError


class BackgroundSource(SampleWorker):
    """One independent bounded source; latest reads perform no external I/O.

    Every read retains its own interval and status. Consumers explicitly choose
    whether a pending/stale source is diagnostic or required for a decision.
    """

    def __init__(self, name, read, path, *, identity, clock, wall_clock, params):
        self.name, self.read = name, read
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        if not isinstance(self.identity, dict) or not self.identity:
            raise ValueError('background source requires an explicit identity')
        self.path, self.wall_clock = Path(path), wall_clock
        self.latest_sample = None
        super().__init__(self._capture, clock, params)

    def _capture(self, deadline):
        started, epoch = self.clock(), self.wall_clock()
        data = self.read(deadline)
        deadline.check()
        return FrozenSnapshot(json.dumps(dict(source=self.name, identity=self.identity, started_mono=started,
            started_epoch_s=epoch, finished_mono=self.clock(), finished_epoch_s=self.wall_clock(),
            status="ok", data=data), allow_nan=False))

    def append(self, sample):
        if not super().append(sample):
            return False
        with self.path.open('a') as stream:
            stream.write(sample.encoded + '\n')
        self.latest_sample = sample
        return True

    def start(self, deadline, *, background_first=True):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        super().start(deadline, background_first=background_first)

    def latest(self):
        if self.error:
            raise SamplingError(self.name + ": " + self.error)
        return self.latest_sample.to_dict() if self.latest_sample else None

    def stop_sampling(self, deadline):
        try:
            super().stop_sampling(deadline)
        finally:
            write_json(self.path.with_suffix('.status.json'), dict(source=self.name,
                error=self.error, samples=self.budget.count, bytes=self.budget.bytes,
                worker_exit_mono=self.worker_exit_mono, joined=bool(self.thread)
                    and self.done_event.is_set() and not self.thread.is_alive()))
