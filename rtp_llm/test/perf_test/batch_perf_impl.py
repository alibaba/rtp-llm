import json
import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import requests
from rtp_llm.test.perf_test.dataclass import (
    ResponseInfo,
    TestResultMetrics,
    analyze_results,
    counted_throughput,
)
from rtp_llm.utils.util import check_with_info


def _effective_profile_steps(is_decode: bool, decode_test_length: int) -> int:
    # Prefill has a single model-forward step; requesting more steps leaves the
    # profiler armed and prevents trace export before the test server exits.
    profile_steps = int(os.environ.get("PERF_PROFILE_NUM_STEPS", "3"))
    return min(decode_test_length, profile_steps) if is_decode else 1


def _wait_for_trace_flush(
    log_path: str, offset: int, budget_s: int
) -> int:
    """Wait for the engine's async trace save to complete.

    The save worker exports each profiled window on a background thread
    and logs ``profiler trace saved: <file>`` exactly when the json is
    fully written. Poll the server log from the given byte offset for
    that line instead of the historical blind ``time.sleep(60)`` — on a
    4-lens binary-search grid the sleep alone cost 24 of the ~31 minutes.
    The ``budget_s`` cap keeps a stuck export from wedging the test.

    Returns the new byte offset past the detected save line (or the
    original offset on timeout, so the next call retries from the same
    position).
    """
    if not log_path or not os.path.isfile(log_path):
        time.sleep(budget_s)
        return offset
    marker = "profiler trace saved"
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        try:
            with open(log_path, "r") as f:
                f.seek(offset)
                new_content = f.read()
                new_offset = f.tell()
            if marker in new_content:
                logging.info("[PERF_PROFILE_FLUSH] trace save detected")
                return new_offset
        except OSError:
            pass
        time.sleep(0.25)
    logging.warning(
        "[PERF_PROFILE_FLUSH] timed out after %ss; continuing", budget_s
    )
    return offset


def _curl_server_single_worker(
    i: int,
    base_port: int,
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    wait_time: int,
    profile: bool = False,
    generate_config: Optional[Dict[str, Any]] = None,
    profile_trace_name: str = "",
) -> ResponseInfo:
    gen_config: Dict[str, Any] = {
        "max_new_tokens": decode_test_length if is_decode else 1,
        "min_new_tokens": decode_test_length if is_decode else 1,
        "force_sp_accept": True,
    }
    req: Dict[str, Any] = {
        "prompt": input_query,
        "generate_config": gen_config,
    }

    if generate_config is not None:
        gen_config.update(generate_config)
        if "top_k" in generate_config:
            req["top_k"] = generate_config["top_k"]
        if "top_p" in generate_config:
            req["top_p"] = generate_config["top_p"]

    if "top_k" not in req:
        req["top_k"] = 1

    profile_step = _effective_profile_steps(is_decode, decode_test_length)
    if profile:
        # Mirror the profiler switches into generate_config as well: depending on
        # the frontend the request goes through, they are read from either level.
        req["gen_timeline"] = True
        gen_config["gen_timeline"] = True
        req["profile_step"] = profile_step
        gen_config["profile_step"] = profile_step
        if profile_trace_name:
            req["profile_trace_name"] = profile_trace_name
            gen_config["profile_trace_name"] = profile_trace_name
    try:
        response = requests.post(
            f"http://127.0.0.1:{base_port}", json=req, timeout=wait_time
        )
        if response.status_code != 200:
            logging.warning(f"request failed: {response.content}")
            return ResponseInfo({}, False)
        logging.debug(response.text)
        return ResponseInfo(
            response.json(), expected_output_len=int(gen_config["min_new_tokens"])
        )
    except Exception as e:
        logging.warning(f" request exception: {e}")
        return ResponseInfo({}, False)


def _curl_server_batch_worker(
    request_indices: List[int],
    base_port: int,
    input_queries: List[str],
    is_decode: bool,
    decode_test_length: int,
    wait_time: int,
    profile: bool = False,
    generate_config: Optional[Dict[str, Any]] = None,
    profile_trace_name: str = "",
) -> List[ResponseInfo]:
    """Concurrently send requests, each with its own query string."""
    with ThreadPoolExecutor(max_workers=len(request_indices)) as executor:
        futures = []
        for idx, i in enumerate(request_indices):
            future = executor.submit(
                _curl_server_single_worker,
                i,
                base_port,
                input_queries[idx],
                is_decode,
                decode_test_length,
                wait_time,
                profile,
                generate_config,
                profile_trace_name,
            )
            futures.append(future)
        return [f.result() for f in futures]


class BatchPerfImpl(object):
    def __init__(
        self,
        base_port: int,
        dp_size: int,
        batch_size: int,
        query: Union[str, List[str]],
        is_decode: bool = True,
        wait_time: int = 100,
        decode_test_length: int = 10,
        profile: bool = True,
        generate_config: Optional[Dict[str, Any]] = None,
        profile_trace_name: str = "",
        warmup_runs: Optional[int] = None,
        measure_runs: Optional[int] = None,
        profile_runs: Optional[int] = None,
        log_path: str = "",
        log_flush_offset: int = 0,
    ):
        self.base_port = base_port
        self.log_path = log_path
        self.dp_size = dp_size
        self.batch_size = batch_size
        if isinstance(query, str):
            self.input_queries = [query] * batch_size
        else:
            assert (
                len(query) == batch_size
            ), f"query list length {len(query)} != batch_size {batch_size}"
            self.input_queries = query
        self.is_decode = is_decode
        self.max_requests_per_process = 128
        self.num_processes = max(
            1,
            (batch_size + self.max_requests_per_process - 1)
            // self.max_requests_per_process,
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)
        self.wait_time = wait_time
        self.decode_test_length = decode_test_length
        self.profile = profile
        # Byte offset into the server log past the last "profiler trace
        # saved" line; the caller threads it across instances so
        # consecutive search steps only scan new log content.
        self.log_flush_offset = log_flush_offset
        self.generate_config = generate_config or {}
        self.profile_trace_name = profile_trace_name
        self.warmup_runs = (
            int(os.environ.get("PERF_FORMAL_WARMUP_RUNS", "1"))
            if warmup_runs is None
            else int(warmup_runs)
        )
        # None means "not pinned here": run(num_measures=...) decides.
        # See _effective_measure_runs() for the precedence.
        self.measure_runs: Optional[int] = (
            int(measure_runs)
            if measure_runs is not None
            else (
                int(os.environ["PERF_MEASURE_RUNS"])
                if "PERF_MEASURE_RUNS" in os.environ
                else None
            )
        )
        self.profile_runs = (
            int(os.environ.get("PERF_PROFILE_RUNS", "1" if profile else "0"))
            if profile_runs is None
            else int(profile_runs)
        )

    def _effective_measure_runs(self, num_measures: Optional[int]) -> int:
        """explicit measure_runs / PERF_MEASURE_RUNS > run(num_measures) > 3."""
        if self.measure_runs is not None:
            return max(1, self.measure_runs)
        if num_measures is None:
            return 3
        return max(1, int(num_measures))

    # warmup (JIT compile) xN -> measure xN (trim min/max, average) ->
    # profile xN (optional, torch profiler affects accuracy)
    def run(self, num_measures: Optional[int] = None) -> TestResultMetrics:
        self._set_concurrency()

        for i in range(self.warmup_runs):
            logging.info(
                "[PERF_WARMUP_RUN] %d/%d trace=%s",
                i + 1,
                self.warmup_runs,
                self.profile_trace_name,
            )
            _ = self._curl_server()

        measure_runs = self._effective_measure_runs(num_measures)
        key = "avg_decode_time" if self.is_decode else "avg_prefill_time"
        measurements: List[TestResultMetrics] = []
        all_measure_responses: List[ResponseInfo] = []
        measured_elapsed_s = 0.0
        for i in range(measure_runs):
            window_start = time.perf_counter()
            responses = self._curl_server_responses()
            measured_elapsed_s += time.perf_counter() - window_start
            metric = analyze_results(responses)
            logging.info(
                "[PERF_MEASURE_RUN] %d/%d trace=%s success=%d/%d "
                "avg_prefill_ms=%.3f avg_total_ms=%.3f avg_wait_ms=%.3f",
                i + 1,
                measure_runs,
                self.profile_trace_name,
                metric.success_requests,
                metric.total_requests,
                metric.avg_prefill_time,
                metric.avg_total_time,
                metric.avg_wait_time,
            )
            measurements.append(metric)
            all_measure_responses.extend(responses)

        if measure_runs >= 3:
            # Trim min and max, average the rest
            measurements.sort(key=lambda m: getattr(m, key))
            values = [f"{getattr(m, key):.2f}" for m in measurements]
            trimmed = measurements[1:-1]
            avg_val = sum(getattr(m, key) for m in trimmed) / len(trimmed)
            logging.debug(
                f"{measure_runs} runs {key}: {values}, "
                f"trimmed [{values[0]}, {values[-1]}], avg={avg_val:.2f}"
            )
            results = trimmed[len(trimmed) // 2]  # use median of trimmed as base
            setattr(results, key, avg_val)  # override with trimmed average
        else:
            # Too few runs to trim: pool every response of every run instead.
            results = analyze_results(all_measure_responses)

        pooled = analyze_results(all_measure_responses)
        results.total_requests = pooled.total_requests
        results.success_requests = pooled.success_requests
        results.fail_requests = pooled.fail_requests
        gpu_count = int(
            os.environ.get(
                "PERF_GPU_COUNT",
                os.environ.get(
                    "WORLD_SIZE",
                    str(self.dp_size * int(os.environ.get("TP_SIZE", "1"))),
                ),
            )
        )
        results.counted_throughput = counted_throughput(
            all_measure_responses, measured_elapsed_s, gpu_count
        )

        if self.profile and self.profile_runs > 0:
            # Pre-arm via /start_profile with enable_all_rank=true so that
            # all TP/DP ranks profile the upcoming request.  Requires the
            # NormalEngine::step() patch that ticks BEFORE process(), so
            # the first post-configure tick starts the profiler in time
            # for the next process() to be captured.  Controlled by env
            # PERF_PREARM_PROFILE=1.
            if os.environ.get("PERF_PREARM_PROFILE", "0") == "1":
                try:
                    num_steps = _effective_profile_steps(
                        self.is_decode, self.decode_test_length
                    )
                    arm_sleep = float(os.environ.get("PERF_PROFILE_ARM_SLEEP", "2"))
                    r = requests.post(
                        f"http://127.0.0.1:{self.base_port}/start_profile",
                        json={
                            "gen_timeline": True,
                            "trace_name": self.profile_trace_name or "perf_prearm",
                            "start_step": 0,
                            "num_steps": num_steps,
                            "enable_all_rank": True,
                        },
                        timeout=60,
                    )
                    logging.info(
                        f"[PERF_PREARM_PROFILE] num_steps={num_steps} arm_sleep={arm_sleep} "
                        f"-> {r.status_code} {r.text[:200]}"
                    )
                    time.sleep(arm_sleep)
                except Exception as e:
                    logging.warning(f"[PERF_PREARM_PROFILE] failed: {e}")
            for i in range(self.profile_runs):
                logging.info(
                    "[PERF_PROFILE_RUN] %d/%d trace=%s",
                    i + 1,
                    self.profile_runs,
                    self.profile_trace_name,
                )
                _ = self._curl_server(True)
            self.log_flush_offset = _wait_for_trace_flush(
                self.log_path,
                self.log_flush_offset,
                int(os.environ.get("PERF_PROFILE_FLUSH_SLEEP", "60")),
            )
        return results

    def _set_concurrency(self):
        check_with_info(
            self.batch_size % self.dp_size == 0,
            f"concurrency {self.batch_size} must be divisible by dp_size {self.dp_size}",
        )
        local_batch_size = self.batch_size // self.dp_size
        payload = {
            "batch_size": local_batch_size,
            "mode": "decode" if self.is_decode else "prefill",
        }
        last_error = None
        for attempt in range(1, 21):
            try:
                response = requests.post(
                    f"http://127.0.0.1:{self.base_port}/update_scheduler_info",
                    json=payload,
                    timeout=60,
                )
                if (
                    response.status_code == 200
                    and response.json().get("status", "ok") == "ok"
                ):
                    return
                last_error = f"{response.text}, {response.status_code}"
            except Exception as e:
                last_error = repr(e)
            logging.warning(
                "failed to set concurrency, retrying (%d/20): %s",
                attempt,
                last_error,
            )
            time.sleep(3)
        raise Exception(f"failed to set concurrency after retries: {last_error}")

    def _curl_server_responses(self, profile: bool = False) -> List[ResponseInfo]:
        request_batches: List[List[int]] = []
        for i in range(0, self.batch_size, self.max_requests_per_process):
            batch_indices = list(
                range(i, min(i + self.max_requests_per_process, self.batch_size))
            )
            request_batches.append(batch_indices)

        futures: List[Future[List[ResponseInfo]]] = []
        for batch_indices in request_batches:
            batch_queries = [self.input_queries[i] for i in batch_indices]
            futures.append(
                self.executor.submit(
                    _curl_server_batch_worker,
                    batch_indices,
                    self.base_port,
                    batch_queries,
                    self.is_decode,
                    self.decode_test_length,
                    self.wait_time,
                    profile,
                    self.generate_config,
                    self.profile_trace_name if profile else "",
                )
            )

        all_responses: List[ResponseInfo] = []
        for future in futures:
            all_responses.extend(future.result())

        return all_responses

    def _curl_server(self, profile: bool = False) -> TestResultMetrics:
        return analyze_results(self._curl_server_responses(profile))

    def dump_results(self, results: List[Dict[str, Any]]):
        for result in results:
            logging.debug(json.dumps(result))
