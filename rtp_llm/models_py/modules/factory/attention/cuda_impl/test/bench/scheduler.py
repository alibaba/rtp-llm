from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench.core import (
    BenchOptions,
    BenchReport,
    BenchResult,
    BenchRunner,
    CaseData,
    PlannedCase,
    build_parallelism_config,
    log,
    short_error,
)

# Anything outside these completed codes is a worker crash.
WORKER_EXIT_OK = 0
WORKER_EXIT_CRASHED = 1
WORKER_EXIT_MISMATCH = 2
WORKER_EXIT_BENCH_FAILURE = 3
WORKER_COMPLETED_CODES = (
    WORKER_EXIT_OK,
    WORKER_EXIT_MISMATCH,
    WORKER_EXIT_BENCH_FAILURE,
)


class WorkerLifecycle:
    """Own worker process groups and the coordinator-liveness channel."""

    @staticmethod
    def create_parent_channel() -> tuple[int, int]:
        return os.pipe()

    @staticmethod
    def close_parent_channel(fd: int) -> int:
        """Close a channel end and return the invalid-fd sentinel."""
        if fd < 0:
            return -1
        try:
            os.close(fd)
        except OSError:
            pass
        return -1

    @staticmethod
    def terminate_process_group(pgid: int, signum: signal.Signals) -> None:
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            pass

    @classmethod
    def install_parent_guard(cls, read_fd: int) -> None:
        if read_fd < 0:
            return

        def watch() -> None:
            try:
                while os.read(read_fd, 4096):
                    pass
            except OSError:
                pass
            finally:
                cls.close_parent_channel(read_fd)
            # No coordinator remains to collect results or escalate a graceful stop.
            cls.terminate_process_group(os.getpgrp(), signal.SIGKILL)

        threading.Thread(target=watch, name="parent-watch", daemon=True).start()


class WorkerRuntime:
    def __init__(
        self,
        options: BenchOptions,
        planned: Sequence[PlannedCase],
        impl_benches: Sequence[Any],
    ) -> None:
        self.options = options
        self.planned = planned
        self.impl_benches = {
            impl_bench.impl.__name__: impl_bench for impl_bench in impl_benches
        }

    @staticmethod
    def claim(counter_file: str, total: int) -> int:
        from filelock import FileLock

        with FileLock(counter_file + ".lock", timeout=30):
            with open(counter_file, "r+") as stream:
                index = int(stream.read().strip())
                if index >= total:
                    return -1
                stream.seek(0)
                stream.write(str(index + 1))
                stream.truncate()
                return index

    def run(self, counter_file: str, result_file: str) -> list[BenchResult]:
        """Drain the queue while recording rather than aborting on case failures."""
        runner = BenchRunner(self.options, build_parallelism_config())
        results: list[BenchResult] = []
        BenchReport.dump_json(results, result_file)
        while True:
            index = self.claim(counter_file, len(self.planned))
            if index < 0:
                return results
            planned = self.planned[index]
            case_failed = False
            recorded: set[str] = set()
            try:
                case_data = CaseData.create(planned.case, torch.device("cuda"))
                for impl_name in planned.impl_names:
                    result = runner.run(self.impl_benches[impl_name], case_data)
                    results.append(result)
                    recorded.add(impl_name)
                    if result.status in ("FAIL", "MISMATCH"):
                        case = result.case
                        log(
                            f"case[{index}] impl={result.impl_name} q={case.q_dtype} kv={case.kv_dtype} "
                            f"bs={case.batch_size} seq={case.seq_len} prefix={case.prefix_len} "
                            f"{result.status}: {result.note.splitlines()[0]}"
                        )
                    if result.status == "FAIL":
                        log(
                            f"full failure for case[{index}] {result.impl_name}:\n{result.note}"
                        )
                        case_failed = True
                del case_data
                torch.cuda.empty_cache()
                log(
                    f"case[{index}] q={planned.case.q_dtype} kv={planned.case.kv_dtype} "
                    f"seq={planned.case.seq_len} {planned.case.mode_tag} "
                    f"{'done with failures' if case_failed else 'done'}"
                )
            except Exception as error:
                # Preserve one result per scheduled implementation.
                for impl_name in planned.impl_names:
                    if impl_name in recorded:
                        continue
                    results.append(
                        BenchResult(
                            case=planned.case,
                            impl_name=impl_name,
                            status="FAIL",
                            effective_kv_dtype=planned.case.kv_dtype,
                            note=f"case setup: {short_error(error)}\n{traceback.format_exc()}",
                        )
                    )
                case = planned.case
                log(
                    f"case[{index}] setup FAIL q={case.q_dtype} kv={case.kv_dtype} "
                    f"bs={case.batch_size} seq={case.seq_len} prefix={case.prefix_len}: "
                    f"{short_error(error)}"
                )
                log(f"full failure for case[{index}]:\n{traceback.format_exc()}")
                torch.cuda.empty_cache()
            BenchReport.dump_json(results, result_file)


@dataclass
class WorkerProcess:
    worker_id: int
    gpu_id: str
    log_prefix: str
    process: subprocess.Popen[str]
    result_file: str
    parent_watch_write_fd: int
    deadline: float
    progress_marker: float = 0.0
    output_thread: threading.Thread | None = None


class _CoordinatorInterrupted(Exception):
    pass


class Coordinator:
    def __init__(self, options: BenchOptions, planned: Sequence[PlannedCase]) -> None:
        self.options = options
        self.planned = planned
        self._shutdown_started = False

    @staticmethod
    def _relay_output(worker: WorkerProcess, errors: list[str]) -> None:
        try:
            assert worker.process.stdout is not None
            for line in worker.process.stdout:
                log(f"{worker.log_prefix} {line.rstrip()}")
        except Exception:
            errors.append(
                f"{worker.log_prefix} output relay failed:\n{traceback.format_exc()}"
            )

    def _stop_all(self, workers: Sequence[WorkerProcess], reason: str) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        log(reason)
        for worker in workers:
            WorkerLifecycle.terminate_process_group(worker.process.pid, signal.SIGTERM)
        grace_deadline = time.monotonic() + 5
        for worker in workers:
            try:
                worker.process.wait(timeout=max(0, grace_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for worker in workers:
            WorkerLifecycle.terminate_process_group(worker.process.pid, signal.SIGKILL)
        for worker in workers:
            worker.process.wait()

    @staticmethod
    def _progress_marker(worker: WorkerProcess) -> float:
        """Last time the worker wrote results, used to detect real stalls."""
        try:
            return os.path.getmtime(worker.result_file)
        except OSError:
            return 0.0

    def _monitor(
        self, workers: Sequence[WorkerProcess], relay_errors: list[str]
    ) -> str:
        while True:
            if relay_errors:
                return "worker output relay failed; stopping all workers"

            running = False
            now = time.monotonic()
            for worker in workers:
                code = worker.process.poll()
                if code is None:
                    running = True
                    # Refresh on result writes so timeout detects stalls, not long matrices.
                    marker = self._progress_marker(worker)
                    if marker > worker.progress_marker:
                        worker.progress_marker = marker
                        worker.deadline = now + self.options.worker_timeout_s
                    if now >= worker.deadline:
                        return (
                            f"{worker.log_prefix} made no progress for "
                            f"{self.options.worker_timeout_s}s; stopping all workers "
                            "(raise --worker-timeout or BENCH_WORKER_TIMEOUT if the "
                            "slowest single case legitimately needs longer)"
                        )
                elif code not in WORKER_COMPLETED_CODES:
                    return f"{worker.log_prefix} exited with code {code}; stopping all workers"
            if not running:
                return ""
            time.sleep(0.2)

    @staticmethod
    def _collect_results(
        workers: Sequence[WorkerProcess],
    ) -> tuple[list[BenchResult], bool]:
        results: list[BenchResult] = []
        failed = False
        for worker in workers:
            if not os.path.exists(worker.result_file):
                log(f"{worker.log_prefix} produced no result file")
                failed = True
                continue
            try:
                with open(worker.result_file) as stream:
                    results.extend(BenchReport.loads(stream.read()))
            except Exception as error:
                log(f"{worker.log_prefix} failed to read results: {short_error(error)}")
                log(traceback.format_exc())
                failed = True
        return results, failed

    def _results_complete(self, results: Sequence[BenchResult]) -> bool:
        expected = Counter(
            (planned.case.case_id, impl_name)
            for planned in self.planned
            for impl_name in planned.impl_names
        )
        actual = Counter((result.case.case_id, result.impl_name) for result in results)
        if expected == actual:
            return True
        missing = sum((expected - actual).values())
        unexpected = sum((actual - expected).values())
        log(f"result aggregation mismatch: missing={missing}, unexpected={unexpected}")
        return False

    def run(self) -> int:
        gpu_ids = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
            if value.strip()
        ]
        if not self.planned:
            log("No benchmark cases matched the requested implementations")
            if self.options.enable_csv_dump:
                try:
                    path = BenchReport.dump_csv([], self.options.csv_output_path)
                    log(f"CSV results written to {path}")
                except Exception as error:
                    log(f"failed to write CSV results: {short_error(error)}")
                    return 1
            return 1
        if not gpu_ids:
            log("CUDA_VISIBLE_DEVICES contains no usable GPU ids")
            return 1
        if len(set(gpu_ids)) != len(gpu_ids):
            log("CUDA_VISIBLE_DEVICES contains duplicate GPU ids")
            return 1

        worker_count = min(len(gpu_ids), len(self.planned))
        log(
            f"Dispatching {len(self.planned)} cases across {worker_count} GPU(s) {gpu_ids[:worker_count]}"
        )
        worker_id_width = len(str(worker_count - 1))
        gpu_id_width = max(len(gpu_id) for gpu_id in gpu_ids[:worker_count])
        base_args = [
            value
            for value in sys.argv[1:]
            if not value.startswith("--_worker-id")
            and not value.startswith("--_results-file")
            and not value.startswith("--_counter-file")
            and not value.startswith("--_parent-watch-fd")
        ]

        self._shutdown_started = False
        workers: list[WorkerProcess] = []
        relay_errors: list[str] = []
        coordinator_failed = False
        results: list[BenchResult] = []
        with tempfile.TemporaryDirectory(prefix="bench_prefill_") as temp_dir:
            counter_file = os.path.join(temp_dir, "counter")
            with open(counter_file, "w") as stream:
                stream.write("0")

            def handle_signal(signum: int, frame: Any) -> None:
                del frame
                raise _CoordinatorInterrupted(signal.Signals(signum).name)

            previous_handlers = {
                signum: signal.signal(signum, handle_signal)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                for worker_id in range(worker_count):
                    result_file = os.path.join(temp_dir, f"worker_{worker_id}.json")
                    parent_read_fd, parent_write_fd = (
                        WorkerLifecycle.create_parent_channel()
                    )
                    try:
                        command = [
                            sys.executable,
                            sys.argv[0],
                            *base_args,
                            f"--_worker-id={worker_id}",
                            f"--_results-file={result_file}",
                            f"--_counter-file={counter_file}",
                            f"--_parent-watch-fd={parent_read_fd}",
                        ]
                        environment = os.environ.copy()
                        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_id]
                        environment["GPU_COUNT"] = "1"
                        environment["PYTHONUNBUFFERED"] = "1"
                        process = subprocess.Popen(
                            command,
                            env=environment,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            close_fds=True,
                            pass_fds=(parent_read_fd,),
                            start_new_session=True,
                        )
                    except BaseException:
                        WorkerLifecycle.close_parent_channel(parent_read_fd)
                        WorkerLifecycle.close_parent_channel(parent_write_fd)
                        raise
                    WorkerLifecycle.close_parent_channel(parent_read_fd)
                    worker = WorkerProcess(
                        worker_id=worker_id,
                        gpu_id=gpu_ids[worker_id],
                        log_prefix=(
                            f"[worker {worker_id:>{worker_id_width}} | "
                            f"GPU {gpu_ids[worker_id]:>{gpu_id_width}}]"
                        ),
                        process=process,
                        result_file=result_file,
                        parent_watch_write_fd=parent_write_fd,
                        deadline=time.monotonic() + self.options.worker_timeout_s,
                    )
                    workers.append(worker)
                    worker.output_thread = threading.Thread(
                        target=self._relay_output,
                        args=(worker, relay_errors),
                        daemon=True,
                    )
                    worker.output_thread.start()

                failure = self._monitor(workers, relay_errors)
                if failure:
                    coordinator_failed = True
                    self._stop_all(workers, failure)
            except BaseException as error:
                coordinator_failed = True
                log(f"coordinator failed: {short_error(error)}")
                log(traceback.format_exc())
                self._stop_all(workers, "coordinator interrupted; stopping all workers")
            finally:
                if coordinator_failed:
                    self._stop_all(workers, "coordinator failed; stopping all workers")
                for worker in workers:
                    if worker.process.poll() is None:
                        worker.process.wait()
                    if worker.output_thread is not None:
                        worker.output_thread.join()
                    worker.parent_watch_write_fd = WorkerLifecycle.close_parent_channel(
                        worker.parent_watch_write_fd
                    )
                if relay_errors:
                    for error in relay_errors:
                        log(error)
                    coordinator_failed = True
                for signum, previous_handler in previous_handlers.items():
                    signal.signal(signum, previous_handler)

            results, aggregation_failed = self._collect_results(workers)
            exit_codes = [worker.process.returncode for worker in workers]
            coordinator_failed = coordinator_failed or aggregation_failed
            if not coordinator_failed and all(
                code in WORKER_COMPLETED_CODES for code in exit_codes
            ):
                coordinator_failed = not self._results_complete(results)

        BenchReport.print(results)
        if self.options.enable_csv_dump:
            try:
                path = BenchReport.dump_csv(results, self.options.csv_output_path)
                log(f"CSV results written to {path}")
            except Exception as error:
                log(f"failed to write CSV results: {short_error(error)}")
                return 1
        if coordinator_failed or any(
            code not in WORKER_COMPLETED_CODES for code in exit_codes
        ):
            return 1
        if any(result.status == "FAIL" for result in results):
            return 1
        fully_skipped = BenchReport.fully_skipped_impls(results)
        if fully_skipped:
            log(
                "bench failed: no measurement produced for "
                f"{', '.join(fully_skipped)}"
            )
            return 1
        if any(result.status == "MISMATCH" for result in results):
            return 2
        return 0
