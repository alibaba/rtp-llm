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
    #  HTTP API
    # ------------------------------------------------------------------ #

    def start_profile(
        self,
        count: int,
        output_path: str = "./vit_profile",
        record_shapes: bool = True,
        with_stack: bool = True,
        profile_memory: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._armed:
                return {
                    "status": "error",
                    "message": "Profiling already in progress",
                    "profiled": self._profiled_count,
                    "target": self._target_count,
                }

            self._output_path = output_path
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

            os.makedirs(output_path, exist_ok=True)
            logging.info(
                f"MMProfiler: armed for {count} requests, output={output_path}"
            )
            return {
                "status": "started",
                "target_count": count,
                "output_path": output_path,
            }

    def end_profile(self) -> Dict[str, Any]:
        with self._lock:
            if not self._armed and not self._finished:
                return {
                    "status": "error",
                    "message": "No profiling session (call /start_profile first)",
                }

            self._armed = False

            output_path = self._output_path
            profiled = self._profiled_count
            target = self._target_count
            averages = self._last_averages
            self._last_averages = None
            finished = self._finished

        files = self._collect_trace_files(output_path)

        if averages is not None:
            summary_file = os.path.join(output_path, "summary.txt")
            ops_file = os.path.join(output_path, "top_operations.json")
            try:
                table = averages.table(sort_by="cuda_time_total", row_limit=50)
                with open(summary_file, "w") as f:
                    f.write(table)
                files["summary"] = summary_file

                top_ops = _build_top_operations(averages)
                with open(ops_file, "w") as f:
                    json.dump(top_ops, f, indent=2)
                files["top_operations"] = ops_file
            except Exception as e:
                logging.warning(f"MMProfiler: summary generation error: {e}")

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
    def profile_request(self):
        """Wrap request computation.  If profiling is armed, the request
        waits for its turn (only one profiled at a time to avoid CUPTI
        conflicts), then runs with full CPU + GPU tracing on the same
        worker thread.  Non-profiled requests pass through immediately.
        """
        with self._lock:
            want_profile = self._armed and self._profiled_count < self._target_count

        if not want_profile:
            yield
            return

        self._profile_slot.acquire()

        with self._lock:
            if not self._armed or self._profiled_count >= self._target_count:
                self._profile_slot.release()
                yield
                return
            request_idx = self._profiled_count

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
            trace_file = os.path.join(self._output_path, f"timeline_{request_idx}.json")
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
                    logging.info("MMProfiler: all requests profiled")

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
