import json
import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import requests

from rtp_llm.test.perf_test.dataclass import (
    ResponseInfo,
    TestResultMetrics,
    analyze_results,
)
from rtp_llm.utils.util import check_with_info


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

    # for prefill profiler step should only be 1, but for decode, we hope to get more steps for cpu analysis
    profile_step = min(decode_test_length, 3) if is_decode else 1
    if profile:
        req["gen_timeline"] = True
        req["profile_step"] = profile_step
        if profile_trace_name:
            req["profile_trace_name"] = profile_trace_name
    try:
        response = requests.post(
            f"http://127.0.0.1:{base_port}", json=req, timeout=wait_time
        )
        if response.status_code != 200:
            logging.warning(f"request failed: {response.content}")
            return ResponseInfo({}, False)
        logging.debug(response.text)
        return ResponseInfo(response.json())
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
    ):
        self.base_port = base_port
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
        self.generate_config = generate_config or {}
        self.profile_trace_name = profile_trace_name

    # warmup → N× measure (trim min/max, average) → profile
    def run(self, num_measures: int = 3):
        self._set_concurrency()
        _ = self._curl_server()  # warmup

        measurements = [self._curl_server() for _ in range(num_measures)]
        key = "avg_decode_time" if self.is_decode else "avg_prefill_time"
        measurements.sort(key=lambda m: getattr(m, key))
        values = [f"{getattr(m, key):.2f}" for m in measurements]

        if num_measures >= 3:
            # Trim min and max, average the rest
            trimmed = measurements[1:-1]
            avg_val = sum(getattr(m, key) for m in trimmed) / len(trimmed)
            logging.debug(
                f"{num_measures} runs {key}: {values}, "
                f"trimmed [{values[0]}, {values[-1]}], avg={avg_val:.2f}"
            )
            results = trimmed[len(trimmed) // 2]  # use median of trimmed as base
            setattr(results, key, avg_val)  # override with trimmed average
        else:
            results = measurements[0]

        if self.profile:
            _ = self._curl_server(True)
        return results

    def _set_concurrency(self):
        check_with_info(
            self.batch_size % self.dp_size == 0,
            f"concurrency {self.batch_size} must be divisible by dp_size {self.dp_size}",
        )
        local_batch_size = self.batch_size // self.dp_size
        response = requests.post(
            f"http://127.0.0.1:{self.base_port}/update_scheduler_info",
            json={
                "batch_size": local_batch_size,
                "mode": "decode" if self.is_decode else "prefill",
            },
        )
        if response.status_code != 200 or response.json().get("status", "ok") != "ok":
            raise Exception(
                f"failed to set concurrency: {response.text}, {response.status_code}"
            )

    def _curl_server(self, profile: bool = False) -> TestResultMetrics:
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

        return analyze_results(all_responses)

    def dump_results(self, results: List[Dict[str, Any]]):
        for result in results:
            logging.debug(json.dumps(result))
