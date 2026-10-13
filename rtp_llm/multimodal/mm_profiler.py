import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import torch.profiler


class MMProfiler:
    """Per-request profiler using ``torch.profiler.profile`` as a context
    manager so that start/stop always happen on the **same worker thread**.

    Only one request is profiled at a time (serialised by ``_profile_slot``).
    Other concurrent requests run without profiling overhead.  Each profiled
    request produces ``timeline_<N>.json``; at the end a merged ``summary.txt``
    and ``top_operations.json`` are generated from the last request's data.
    """

    # Per-process (class-level) profiler-stack warmup state, shared by every
    # MMProfiler instance in the process (a worker holds the engine's, and
    # the VIT proxy creates one that is never armed). See
    # _warmup_profiler_stack().
    _STACK_WARMED = False
    _STACK_WARMUP_LOCK = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._armed = False
        self._target_count = 0
        self._profiled_count = 0
        self._output_path = "./vit_profile"

        self._profile_cfg: Dict[str, Any] = {}
        self._profile_slot = threading.Lock()  # only 1 request profiled at a time
        self._last_averages: Optional[Any] = None
        self._finished = False

    # ------------------------------------------------------------------ #
    #  One-time per-process profiler stack warmup
    # ------------------------------------------------------------------ #

    @classmethod
    def _warmup_profiler_stack(cls) -> None:
        """Run one throwaway profiler session on the calling (arming) thread
        so libkineto/CUPTI finish their lazy init BEFORE any real session
        starts on a scheduler executor thread.

        Why: the first ``torch.profiler`` session with CUDA activity makes
        libkineto register itself as a CUPTI client on the thread that
        started it, and CUPTI's external-init callback must then run on that
        SAME thread. Profiling is armed from the HTTP/event-loop thread but
        the sessions run on the "mm-scheduler" executor thread, so when the
        executor thread starts the process's FIRST session, the affinity
        check in the libkineto vendored in torch 2.11 kills the thread (log
        line: "External init callback must run in same thread as
        registerClient"). The executor then dies silently and every request
        lands on its timeout; this only surfaced under co-located CI runs,
        where CUPTI's lazy init lands on a different thread than the
        subscriber. Warming up at ARM time makes registration and CUPTI init
        happen on the arming thread by construction; later sessions only
        toggle activity collection and never re-enter the init path.
        """
        if cls._STACK_WARMED:
            return
        with cls._STACK_WARMUP_LOCK:
            if cls._STACK_WARMED:
                return
            # Best-effort: a same-thread warmup cannot hit the affinity
            # death, so a failure here means CUPTI itself is broken and the
            # real session would fail anyway. Keep arm semantics unchanged,
            # and remember the ATTEMPT (finally below) so a broken
            # environment does not re-pay the warmup on every arm.
            try:
                # Mirror profile_forward's activity selection exactly.
                activities = [torch.profiler.ProfilerActivity.CPU]
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                # Cheap flags: this session exists to initialize CUPTI, not
                # to collect data (nothing is exported or aggregated). The
                # REAL sessions keep full CPU + CUDA tracing with
                # shapes/stack/memory.
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ):
                    if cuda_available:
                        # A real CUDA op INSIDE the warmup session: creates
                        # the CUDA context (if none exists yet) while the
                        # client registration is on this same thread, so the
                        # context-created init callback also runs here.
                        torch.zeros(1, device="cuda").add_(1)
            except Exception as e:
                logging.warning(f"MMProfiler: profiler stack warmup failed: {e}")
            finally:
                cls._STACK_WARMED = True

    # ------------------------------------------------------------------ #
    #  HTTP API
    # ------------------------------------------------------------------ #

    def start_profile(
        self,
        count: int,
        rank: Optional[int] = None,
        record_shapes: bool = True,
        with_stack: bool = True,
        profile_memory: bool = True,
    ) -> Dict[str, Any]:
        # Force the one-time CUPTI/profiler stack init on THIS (arming)
        # thread before taking any instance lock: the warmup is process
        # init, not session state, and it must not hold ``_lock`` —
        # profile_forward's hot-path arming check blocks on it — for the
        # ~0.1-1s a first CUPTI init can take. After it returns, every later
        # session (including the real ones on the scheduler executor thread)
        # finds CUPTI already initialized.
        self._warmup_profiler_stack()

        with self._lock:
            # Reject any new start while a prior session is still resolvable:
            # either still armed (in progress) OR finished but its end_profile
            # hasn't run yet (still owns _output_path / _last_averages). This
            # collapses both the start-vs-end TOCTOU and the start-vs-finished
            # race into a single "one session in flight" invariant — caller
            # must end_profile before starting again.
            if self._armed or self._finished:
                return {
                    "status": "error",
                    "message": (
                        # Keep "already in progress" substring for backward
                        # compatibility (mm_profiler_test asserts on it).
                        "Profiling already in progress; call /end_profile first"
                        if self._armed
                        else "Previous profiling session already in progress (not consumed); call /end_profile first"
                    ),
                    "profiled": self._profiled_count,
                    "target": self._target_count,
                }

            # Per-rank subdirectory so concurrent workers don't overwrite each
            # other's timeline_<N>.json files. When rank is not supplied, fall
            # back to whatever `_output_path` was set to (default ./vit_profile,
            # or an externally-injected path for tests).
            if rank is not None:
                self._output_path = f"./vit_profile/rank_{rank}"
            os.makedirs(self._output_path, exist_ok=True)

            self._target_count = count
            self._profiled_count = 0
            self._profile_cfg = {
                "record_shapes": record_shapes,
                "with_stack": with_stack,
                "profile_memory": profile_memory,
            }
            self._last_averages = None
            self._finished = False
            self._armed = True

            logging.info(
                f"MMProfiler: armed for {count} forwards, output={self._output_path}"
            )
            return {
                "status": "started",
                "target_count": count,
                "output_path": self._output_path,
            }

    def end_profile(self) -> Dict[str, Any]:
        with self._lock:
            if not self._armed and not self._finished:
                return {
                    "status": "error",
                    "message": "No profiling session (call /start_profile first)",
                }

            self._armed = False

            profiled = self._profiled_count
            target = self._target_count
            averages = self._last_averages
            self._last_averages = None
            finished = self._finished
            # Snapshot path under the lock so a concurrent start_profile (after
            # we clear _finished below) can't repoint _output_path before we
            # write summary/ops files. Use the local copy for all I/O.
            output_path = self._output_path

        if averages is not None:
            summary_file = os.path.join(output_path, "summary.txt")
            ops_file = os.path.join(output_path, "top_operations.json")
            try:
                table = averages.table(sort_by="cuda_time_total", row_limit=50)
                with open(summary_file, "w") as f:
                    f.write(table)

                top_ops = _build_top_operations(averages)
                with open(ops_file, "w") as f:
                    json.dump(top_ops, f, indent=2)
            except Exception as e:
                logging.warning(f"MMProfiler: summary generation error: {e}")

        files = self._collect_trace_files(output_path)
        if averages is not None:
            files.setdefault("summary", os.path.join(output_path, "summary.txt"))
            files.setdefault(
                "top_operations", os.path.join(output_path, "top_operations.json")
            )

        # Clear _finished only after all I/O completes so a fast follow-up
        # start_profile (gated on `_armed or _finished`) won't repoint state
        # while we're still writing.
        with self._lock:
            self._finished = False

        return {
            "status": "completed" if finished else "stopped_early",
            "profiled_count": profiled,
            "target_count": target,
            "output_path": output_path,
            "files": files,
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_profiling": self._armed,
                "profiled_count": self._profiled_count,
                "target_count": self._target_count,
                "output_path": self._output_path,
                "finished": self._finished,
            }

    def on_request_complete(self):
        """No-op hook called by the proxy after forwarding a request."""
        pass

    # ------------------------------------------------------------------ #
    #  Called from worker threads (mm_process_engine)
    # ------------------------------------------------------------------ #

    @contextmanager
    def profile_forward(self):
        """Wrap ONE embedding forward (a batch) on the scheduler thread. If
        profiling is armed, the forward waits for its turn (only one profiled at a
        time to avoid CUPTI conflicts), then runs with full CPU + GPU tracing on
        that thread. Non-profiled forwards pass through immediately.

        NOTE: the unit is per-forward/per-batch (a forward may merge several
        requests), NOT per-request — start_profile(count) counts forwards.
        """
        with self._lock:
            want_profile = self._armed and self._profiled_count < self._target_count

        if not want_profile:
            yield
            return

        # Wait for the single-profile slot (CUPTI requires serialized profilers).
        self._profile_slot.acquire()

        # Re-check state after acquiring the slot: armed/target_count may have
        # changed while we were waiting.  Decide bail vs. profile under _lock,
        # then drop _lock before doing any long work (yield / profiler setup).
        with self._lock:
            if self._armed and self._profiled_count < self._target_count:
                request_idx = self._profiled_count
            else:
                request_idx = None

        if request_idx is None:
            # Bail out: release the slot BEFORE yielding so that neither lock is
            # held during request execution; otherwise start/end/get_status and
            # subsequent profile_forward callers would all block on this forward.
            self._profile_slot.release()
            yield
            return

        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            cfg = self._profile_cfg
            prof = torch.profiler.profile(
                activities=activities,
                record_shapes=cfg.get("record_shapes", True),
                profile_memory=cfg.get("profile_memory", True),
                with_stack=cfg.get("with_stack", True),
            )

            try:
                with prof:
                    yield
            finally:
                trace_file = os.path.join(
                    self._output_path, f"timeline_{request_idx}.json"
                )
                try:
                    prof.export_chrome_trace(trace_file)
                except Exception as e:
                    logging.error(
                        f"MMProfiler: export failed for request {request_idx}: {e}"
                    )

                try:
                    self._last_averages = prof.key_averages()
                except Exception:
                    pass

                with self._lock:
                    self._profiled_count += 1
                    logging.info(
                        f"MMProfiler: profiled {self._profiled_count}/{self._target_count}"
                    )
                    if self._profiled_count >= self._target_count:
                        self._armed = False
                        self._finished = True
                        logging.info("MMProfiler: all forwards profiled")
        finally:
            self._profile_slot.release()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_trace_files(output_path: str) -> Dict[str, Any]:
        files: Dict[str, Any] = {}
        try:
            traces = sorted(
                f
                for f in os.listdir(output_path)
                if f.startswith("timeline_") and f.endswith(".json")
            )
            files["traces"] = [os.path.join(output_path, f) for f in traces]
        except OSError:
            files["traces"] = []
        return files


def _build_top_operations(averages) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for evt in averages:
        rec: Dict[str, Any] = {"name": evt.key, "count": evt.count}
        for attr in (
            "cpu_time_total",
            "cuda_time_total",
            "self_cpu_time_total",
            "self_cuda_time_total",
        ):
            rec[f"{attr}_us"] = getattr(evt, attr, 0)
        cnt = evt.count or 1
        rec["cpu_time_avg_us"] = rec["cpu_time_total_us"] / cnt
        rec["cuda_time_avg_us"] = rec["cuda_time_total_us"] / cnt
        records.append(rec)
    records.sort(key=lambda r: r["cuda_time_total_us"], reverse=True)
    return records
