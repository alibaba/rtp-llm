import json
import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests

from rtp_llm.test.perf_test.multi_node.perf_dataclass import (
    ResponseInfo,
    TestResultMetrics,
    analyze_results,
)
from rtp_llm.distribute.distributed_server import members_from_test_env
from rtp_llm.utils.util import check_with_info


# 将 _curl_server_single 提取为独立函数，以便在 ProcessPoolExecutor 中使用
def _curl_server_single_worker(
    request_id: int,
    batch_size: int,
    tp0_endpoints: List[Tuple[str, int]],
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    is_warmup: bool = False,
    generate_config: Dict[str, Any] = {},
) -> ResponseInfo:
    host, port = tp0_endpoints[request_id // batch_size]
    timeout = 1000 if is_warmup else None

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

    try:
        response = requests.post(
            f"http://{host}:{port}", json=req, timeout=timeout
        )
        if response.status_code != 200:
            logging.warning(f"request failed: {response.content}")
            return ResponseInfo({}, False)
        logging.debug(response.text)
        return ResponseInfo(response.json())
    except Exception as e:
        logging.warning(f" request exception: {e}")
        return ResponseInfo({}, False)


# 修改：使用ThreadPoolExecutor调用_curl_server_single_worker
def _curl_server_batch_worker(
    request_indices: List[int],
    batch_size: int,
    tp0_endpoints: List[Tuple[str, int]],
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    is_warmup: bool = False,
    generate_config: Dict[str, Any] = {},
) -> List[ResponseInfo]:
    """使用ThreadPoolExecutor并发处理多个请求"""
    responses = []

    # 创建线程池，线程数等于请求数
    with ThreadPoolExecutor(max_workers=len(request_indices)) as executor:
        # 提交所有请求任务
        futures = []
        for i in request_indices:
            future = executor.submit(
                _curl_server_single_worker,
                i,
                batch_size,
                tp0_endpoints,
                input_query,
                is_decode,
                decode_test_length,
                is_warmup,
                generate_config,
            )
            futures.append(future)

        # 收集所有响应
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
        total_batch_size: int,
        input_len: int,
        query: str,
        is_decode: bool = True,
        decode_test_length: int = 10,
        generate_config: Dict[str, Any] = {},
        local_world_size: int = 1,
        gang_config_string: str = "",
        request_tpot: int = 100,
        connection_timeout: int = 10,
        retry_times: int = 3,
        retry_interval: float = 0.5,
    ):
        self.base_port = base_port
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.total_batch_size = total_batch_size
        self.input_len = input_len
        self.input_query = query
        self.is_decode = is_decode
        self.max_requests_per_process = 128
        self.decode_test_length = decode_test_length
        self.generate_config = generate_config
        self.local_world_size = local_world_size
        self.gang_config_string = gang_config_string
        self.request_tpot = request_tpot
        self.connection_timeout = connection_timeout
        self.retry_times = retry_times
        self.retry_interval = retry_interval
        self.request_timeout = (
            30 + request_tpot * decode_test_length // 1000 + connection_timeout
        )

        # 计算需要的进程数
        self.num_processes = max(
            1,
            (total_batch_size + self.max_requests_per_process - 1)
            // self.max_requests_per_process,
        )
        # 使用 ProcessPoolExecutor，进程数根据每个进程处理的请求数来确定
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)

        # Discover all DP TP0 frontend endpoints
        self.tp0_endpoints: List[Tuple[str, int]] = self._get_all_dp_tp0_frontends()

    # 需要做3次
    # 第一次：warmup, 预编译jit
    # 第二次: 测量时间
    # 第三次: dump profile json（因为torch profiler会影响测量时间精确性）
    def run(self):
        self._set_concurrency()
        logging.info("Starting warmup run...")
        _ = self._curl_server(is_profile=False, is_warmup=True)
        logging.info("Starting measurement run...")
        results = self._curl_server(is_profile=False, is_warmup=False)
        logging.info("Starting profile run...")
        _ = self._curl_server(is_profile=True, is_warmup=False)
        return results

    def _set_concurrency(self):
        check_with_info(
            self.total_batch_size % self.dp_size == 0,
            f"concurrency {self.total_batch_size} must be divisible by dp_size {self.dp_size}",
        )
        local_batch_size = self.total_batch_size // self.dp_size
        for host, port in self.tp0_endpoints:
            for attempt in range(self.retry_times):
                try:
                    response = requests.post(
                        f"http://{host}:{port}/update_scheduler_info",
                        json={
                            "batch_size": local_batch_size,
                            "mode": "decode" if self.is_decode else "prefill",
                        },
                    )
                    if response.status_code != 200 or response.json().get("status", "ok") != "ok":
                        raise Exception(
                            f"failed to set concurrency: {response.text}, {response.status_code}"
                        )
                    break
                except Exception as e:
                    if attempt < self.retry_times - 1:
                        logging.warning(
                            f"set_concurrency attempt {attempt + 1} failed for {host}:{port}: {e}, retrying..."
                        )
                        time.sleep(self.retry_interval)
                    else:
                        raise

    def _get_all_dp_tp0_frontends(self) -> List[Tuple[str, int]]:
        """Discover all DP TP0 frontend endpoints from gang_config_string."""
        if not self.gang_config_string:
            return [("127.0.0.1", self.base_port)]

        members = members_from_test_env(self.gang_config_string)
        # Each gang member exposes a TP0 frontend; collect all of them
        endpoints: List[Tuple[str, int]] = []
        for member in members:
            endpoints.append((member.ip, member.server_port))
        return endpoints

    def _curl_server(self, is_profile: bool = False, is_warmup: bool = False) -> TestResultMetrics:
        batch_size = self.total_batch_size // self.dp_size

        # 将请求分批，每个进程处理一批请求
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
                    batch_size,
                    self.tp0_endpoints,
                    self.input_query,
                    self.is_decode,
                    self.decode_test_length,
                    is_warmup,
                    self.generate_config,
                )
            )

        # 收集所有响应
        all_responses: List[ResponseInfo] = []
        for future in futures:
            batch_responses = future.result()
            all_responses.extend(batch_responses)

        metrics = analyze_results(all_responses)
        return metrics

    def dump_results(self, results: List[Dict[str, Any]]):
        for result in results:
            logging.debug(json.dumps(result))
