"""Bounded background sampling with cancellation and joined shutdown."""
import threading
from runtime.observation import SampleBudget


class SamplingError(RuntimeError):
    """Sampler shutdown failed to produce a joined exit record."""


class CaptureDeadline:
    def __init__(self, clock, end, cancel):
        self.clock, self.end, self.cancel = clock, end, cancel

    def remaining(self):
        return max(0, self.end - self.clock())

    def check(self):
        if self.cancel.is_set() or self.remaining() <= 0:
            raise TimeoutError("observation deadline or cancellation")


class SampleWorker:
    def __init__(self, capture, clock, params):
        self.capture, self.clock, self.params = capture, clock, params
        self.started = clock()
        self.end = self.started + params["duration_s"]
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.worker_exit_mono = None
        self.thread = None
        self.samples = []
        self.bytes = 0
        self.budget = SampleBudget({key: params[key] for key in ("max_samples", "max_bytes")})
        self.error = None
        self.stopped_at = None

    def append(self, sample):
        try:
            self.budget.append_encoded(sample.encoded)
        except ValueError:
            self.error = "observation sample/byte budget exhausted"
            self.stop_event.set()
            return False
        self.samples.append(sample.encoded)
        self.bytes = self.budget.bytes
        return True

    def start(self, deadline):
        self.append(self.capture(deadline))
        self.thread = threading.Thread(
            target=self._run, name="flexlb-scenario-observer", daemon=True
        )
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.wait(
                min(self.params["interval_s"], max(0, self.end - self.clock()))
            ):
                if self.clock() >= self.end:
                    break
                self.append(
                    self.capture(
                        CaptureDeadline(self.clock, self.end, self.stop_event)
                    )
                )
        except TimeoutError as error:
            if not self.stop_event.is_set() and self.clock() < self.end:
                self.error = f"incomplete sampling: {type(error).__name__}: {error}"
        except Exception as error:
            self.error = f"incomplete sampling: {type(error).__name__}: {error}"
        finally:
            # Publish the exit record after the last possible sample/error write.
            self.worker_exit_mono = self.clock()
            self.done_event.set()

    def stop_sampling(self, deadline):
        self.stopped_at = (
            min(self.clock(), self.end)
            if self.stopped_at is None
            else self.stopped_at
        )
        self.stop_event.set()
        if self.thread is not None:
            while not self.done_event.is_set():
                deadline.check()
                self.done_event.wait(min(0.05, deadline.remaining()))
            if self.worker_exit_mono is None:
                raise SamplingError("observer completion signal has no exit record")
            self.thread.join(timeout=deadline.remaining())
            if self.thread.is_alive():
                raise TimeoutError(
                    "observer thread still alive; cleanup has not completed"
                )
