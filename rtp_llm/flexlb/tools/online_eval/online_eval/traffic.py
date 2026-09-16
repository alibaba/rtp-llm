"""Bounded traffic workers shared across test orchestration policies."""

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from .requests import RecordedRequests, completeness


class BoundedFlow(RecordedRequests):
    """Fixed concurrency and cadence; stop issuance separately from cancellation."""

    def __init__(
        self,
        ops,
        env_epoch,
        families,
        interval_s=0.5,
        max_inflight=2,
        clock=time.monotonic,
    ):
        super().__init__(ops, env_epoch, clock)
        self.families = copy.deepcopy(families)
        self.interval_s, self.max_inflight = interval_s, max_inflight
        self._stop = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(
            target=self._pump, name="elastic-recorded-flow", daemon=True
        )
        self.pump_error = None

    def start(self):
        self.thread.start()

    def _pump(self):
        futures = []
        index = 0
        try:
            with ThreadPoolExecutor(max_workers=self.max_inflight) as pool:
                while not self._stop.is_set():
                    futures = [f for f in futures if not f.done()]
                    if len(futures) < self.max_inflight:
                        record = self.issue(self.ops.next_request_id(), self.clock)
                        keys = self.families[index % len(self.families)]
                        index += 1
                        futures.append(
                            pool.submit(
                                self.run,
                                record,
                                dict(input_len=10240, output_len=2, block_keys=keys),
                            )
                        )
                    self._stop.wait(self.interval_s)
        except Exception as exc:
            self.pump_error = repr(exc)
        finally:
            # Executor shutdown has completed and all worker records are final.
            self.done.set()

    def stop(self, deadline, cancel=False):
        self._stop.set()
        if cancel:
            self.cancel_active()
        if self.thread.ident is not None:
            try:
                if not self.done.wait(max(0, deadline.remaining())):
                    raise TimeoutError("flow completion event did not arrive")
            except TimeoutError:
                self.cancel_active("drain_deadline")
                raise
            records = self.snapshot_records()
            if any(
                r["consumer_exit_s"] is None or r["transport_terminal_s"] is None
                for r in records
            ):
                raise RuntimeError("flow completed without final consumer records")
        return completeness(self.snapshot_records())


class ColdFlow(BoundedFlow):
    """Preserve legacy serial cold traffic: completion, then a 200ms pause.

    Schedule retains its 30s cap and Fetch its 10s cap. All attempts survive in
    the ledger, including failures. The independent done event comes from the
    same worker that performs and finishes the actual RPC consumption.
    """

    def __init__(self, ops, env_epoch, clock=time.monotonic):
        super().__init__(
            ops, env_epoch, [], interval_s=0.2, max_inflight=1, clock=clock
        )

    def _pump(self):
        end = self.clock() + 600
        try:
            while not self._stop.is_set():
                if self.clock() >= end:
                    raise TimeoutError("cold flow exceeded its 600s lifetime budget")
                rid = self.ops.next_request_id()
                record = self.issue(rid, self.clock)
                self.run(
                    record,
                    dict(output_len=2, block_keys=[rid * 100 + 1]),
                    timeout_s=min(40, end - self.clock()),
                    stream_timeout_s=10,
                )
                self._stop.wait(0.2)
        except Exception as exc:
            self.pump_error = repr(exc)
        finally:
            self.done.set()
