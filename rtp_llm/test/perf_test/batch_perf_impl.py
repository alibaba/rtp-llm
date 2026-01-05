import json
import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.distribute.distributed_server import members_from_test_env
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
    request_timeout: int,
    is_profile: bool = False,
    is_warmup: bool = False,
    generate_config: Dict[str, Any] = {},
) -> ResponseInfo:
    """Curl the server for a single request"""
    req = {
        "prompt": input_query,
        "generate_config": {
            "max_new_tokens": decode_test_length if is_decode else 1,
            "min_new_tokens": decode_test_length if is_decode else 1,
            "force_sp_accept": True,
        },
    }

    if generate_config:
        req["generate_config"].update(generate_config)
        if "top_k" in generate_config:
            req["top_k"] = generate_config["top_k"]
        if "top_p" in generate_config:
            req["top_p"] = generate_config["top_p"]

    if "top_k" not in req:
        req["top_k"] = 1

    if is_profile:
        req["gen_timeline"] = True
        req["profile_step"] = 1

    if is_warmup:
        request_timeout = 1000

    try:
        response = requests.post(
            f"http://127.0.0.1:{base_port}", json=req, timeout=request_timeout
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
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    request_timeout: int,
    is_profile: bool = False,
    is_warmup: bool = False,
    generate_config: Dict[str, Any] = {},
) -> List[ResponseInfo]:
    """Use ThreadPoolExecutor to concurrently handle multiple requests"""
    responses = []

    # Create thread pool, the number of threads is equal to the number of requests
    with ThreadPoolExecutor(max_workers=len(request_indices)) as executor:
        # Submit all request tasks
        futures = []
        for i in request_indices:
            future = executor.submit(
                _curl_server_single_worker,
                i,
                base_port,
                input_query,
                is_decode,
                decode_test_length,
                request_timeout,
                is_profile,
                is_warmup,
                generate_config,
            )
            futures.append(future)

        # Collect all responses
        for future in futures:
            response = future.result()
            responses.append(response)

    return responses


class BatchPerfImpl(object):
    def __init__(
        self,
        base_port: int,
        dp_size: int,
        tp_size: int,
        local_world_size: int,
        batch_size: int,
        input_len: int,
        query: str,
        gang_config_string: str,
        request_tpot: int = 100,
        connection_timeout: int = 10,
        retry_times: int = 3,
        retry_interval: float = 0.5,
        is_decode: bool = True,
        decode_test_length: int = 10,
        generate_config: Dict[str, Any] = {},
    ):
        self.base_port = base_port
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.local_world_size = local_world_size
        self.total_batch_size = batch_size
        self.input_len = input_len
        self.input_query = query
        self.is_decode = is_decode
        self.max_requests_per_process = 128
        self.gang_config_string = gang_config_string
        self.connection_timeout = connection_timeout
        self.retry_times = retry_times
        self.retry_interval = retry_interval
        # Calculate the performance test timeout
        self.request_timeout = (
            30 + request_tpot * decode_test_length // 1000 + connection_timeout
        )
        # Calculate the number of required processes
        self.num_processes = max(
            1,
            (self.total_batch_size + self.max_requests_per_process - 1)
            // self.max_requests_per_process,
        )
        # Use ProcessPoolExecutor, the number of processes is determined by the number of requests each process handles
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)
        self.decode_test_length = decode_test_length
        self.generate_config = generate_config
        # Get all DP groups' tp_rank==0 frontend endpoints
        self.tp0_endpoints = self._get_all_dp_tp0_frontends()

    # Execute perf test 3 times:
    # 1. warmup, JIT compilation
    # 2. measure time
    # 3. dump profile json (because torch profiler affects the precision of measuring time)
    def run(self):
        self._set_concurrency()
        logging.info(f"finished setting concurrency")
        _ = self._curl_server(is_profile=False, is_warmup=True)
        logging.info(f"finished warmup")
        results = self._curl_server(is_profile=False, is_warmup=False)
        logging.info(f"finished measure time")
        _ = self._curl_server(is_profile=True, is_warmup=False)
        logging.info(f"finished dump profile json")
        return results

    def _set_concurrency(self):
        check_with_info(
            self.total_batch_size % self.dp_size == 0,
            f"concurrency {self.total_batch_size} must be divisible by dp_size {self.dp_size}",
        )
        batch_size = self.total_batch_size // self.dp_size

        payload = {
            "batch_size": batch_size,
            "mode": "decode" if self.is_decode else "prefill",
        }

        def _post_one(ip: str, port: int) -> Tuple[str, int, int, str]:
            retry_count = 0
            url = f"http://{ip}:{port}/update_scheduler_info"
            while retry_count < self.retry_times:
                time.sleep(self.retry_interval)
                resp = requests.post(url, json=payload, timeout=self.connection_timeout)
                resp_status_code = resp.status_code
                resp_status_text = resp.json().get("status", "not ok")
                if resp_status_code == 200 and resp_status_text == "ok":
                    break
                else:
                    retry_count += 1
                    logging.warning(
                        f"update_scheduler_info request failed for {ip}:{port}: status code={resp_status_code}, status text={resp_status_text}, retry_count={retry_count}/{self.retry_times}"
                    )
            return ip, port, resp_status_code, resp_status_text

        # Use multi-thread to update all DP groups concurrently.
        max_workers = min(len(self.tp0_endpoints), 32) if self.tp0_endpoints else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_post_one, ip, port)
                for ip, port in reversed(self.tp0_endpoints)
            ]

        errors: List[str] = []
        for fut in futures:
            try:
                ip, port, status_code, status_text = fut.result()
                if status_code != 200 or status_text != "ok":
                    errors.append(
                        f"update_scheduler_info failed for {ip}:{port}: status={status_code}, response={status_text}"
                    )
            except Exception as e:
                errors.append(f"update_scheduler_info exception for {ip}:{port}: {e}")

        if errors:
            raise Exception(";\n".join(errors))

    def _get_all_dp_tp0_frontends(self) -> List[Tuple[str, int]]:
        """Get all DP groups' tp_rank==0 frontend endpoints across nodes from gang_config_string."""
        nodes = members_from_test_env(self.gang_config_string)
        targets: List[Tuple[str, int]] = []
        for dp_rank in range(self.dp_size):
            tp0_world_rank = dp_rank * self.tp_size
            node_idx = tp0_world_rank // self.local_world_size
            local_rank = tp0_world_rank % self.local_world_size
            check_with_info(
                node_idx < len(nodes),
                f"dp_rank {dp_rank} (tp0_world_rank={tp0_world_rank}) maps to node_idx={node_idx}, "
                f"but only {len(nodes)} nodes in GANG_CONFIG_STRING",
            )
            base_port = int(nodes[node_idx].server_port)
            port = base_port + local_rank * MIN_WORKER_INFO_PORT_NUM
            targets.append((nodes[node_idx].ip, int(port)))
        # de-dup (defensive programming)
        return list(dict.fromkeys(targets))

    def _curl_server(
        self, is_profile: bool = False, is_warmup: bool = False
    ) -> TestResultMetrics:
        # Batch requests, each process handles a batch of requests
        request_batches: List[List[int]] = []
        for i in range(0, self.total_batch_size, self.max_requests_per_process):
            batch_indices = list(
                range(i, min(i + self.max_requests_per_process, self.total_batch_size))
            )
            request_batches.append(batch_indices)

        futures: List[Future[List[ResponseInfo]]] = []
        for batch_indices in request_batches:
            futures.append(
                self.executor.submit(
                    _curl_server_batch_worker,
                    batch_indices,
                    self.base_port,
                    self.input_query,
                    self.is_decode,
                    self.decode_test_length,
                    self.request_timeout,
                    is_profile,
                    is_warmup,
                    self.generate_config,
                )
            )

        # Collect all responses
        all_responses: List[ResponseInfo] = []
        for future in futures:
            batch_responses = future.result()
            all_responses.extend(batch_responses)

        metrics = analyze_results(all_responses)
        return metrics

    def dump_results(self, results: List[Dict[str, Any]]):
        for result in results:
            logging.debug(json.dumps(result))
