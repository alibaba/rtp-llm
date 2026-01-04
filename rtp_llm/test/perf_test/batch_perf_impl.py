import json
import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List

import requests

from rtp_llm.test.perf_test.dataclass import (
    ResponseInfo,
    TestResultMetrics,
    analyze_results,
)
from rtp_llm.utils.util import check_with_info


# 将 _curl_server_single 提取为独立函数，以便在 ProcessPoolExecutor 中使用
def _curl_server_single_worker(
    i: int,
    base_port: int,
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    wait_time: int,
    profile: bool = False,
    generate_config: Dict[str, Any] = {},
) -> ResponseInfo:
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

    if profile:
        req["gen_timeline"] = True
        req["profile_step"] = 1
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


# 修改：使用ThreadPoolExecutor调用_curl_server_single_worker
def _curl_server_batch_worker(
    request_indices: List[int],
    base_port: int,
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    wait_time: int,
    profile: bool = False,
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
                base_port,
                input_query,
                is_decode,
                decode_test_length,
                wait_time,
                profile,
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
        batch_size: int,
        input_len: int,
        query: str,
        is_decode: bool = True,
        wait_time: int = 100,
        decode_test_length: int = 10,
        profile: bool = True,
        generate_config: Dict[str, Any] = {},
    ):
        self.base_port = base_port
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.batch_size = batch_size
        self.input_len = input_len
        self.input_query = query
        self.is_decode = is_decode
        self.max_requests_per_process = 128
        # 计算需要的进程数
        self.num_processes = max(
            1,
            (batch_size + self.max_requests_per_process - 1)
            // self.max_requests_per_process,
        )
        # 使用 ProcessPoolExecutor，进程数根据每个进程处理的请求数来确定
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)
        self.wait_time = wait_time
        self.decode_test_length = decode_test_length
        self.profile = profile
        self.generate_config = generate_config

    # 需要做3次
    # 第一次：warmup, 预编译jit
    # 第二次: 测量时间
    # 第三次: dump profile json（因为torch profiler会影响测量时间精确性）
    def run(self):
        self._set_concurrency()
        _ = self._curl_server()
        results = self._curl_server()
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
                f"failed to set conccurrency: {response.text}, {response.status_code}"
            )

    def _curl_server(self, profile: bool = False) -> TestResultMetrics:
        # 将请求分批，每个进程处理一批请求
        request_batches: List[List[int]] = []
        for i in range(0, self.batch_size, self.max_requests_per_process):
            batch_indices = list(
                range(i, min(i + self.max_requests_per_process, self.batch_size))
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
                    self.wait_time,
                    profile,
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
