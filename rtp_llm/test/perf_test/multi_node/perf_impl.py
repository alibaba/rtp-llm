import json
import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import grpc
import requests

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.distribute.distributed_server import members_from_test_env
from rtp_llm.test.perf_test.multi_node.perf_dataclass import (
    ResponseInfo,
    TestResultMetrics,
    analyze_results,
)
from rtp_llm.utils.util import check_with_info


def _curl_server_single_worker(
    request_id: int,
    batch_size: int,
    tp0_endpoints: List[Tuple[str, int]],
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    request_timeout: int,
    is_warmup: bool = False,
    generate_config: Dict[str, Any] = {},
) -> ResponseInfo:
    batch_idx = request_id // batch_size
    host = tp0_endpoints[batch_idx][0]
    port = tp0_endpoints[batch_idx][1]
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

    if is_warmup:
        request_timeout = 1000

    try:
        response = requests.post(
            f"http://{host}:{port}", json=req, timeout=request_timeout
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
    batch_size: int,
    tp0_endpoints: List[Tuple[str, int]],
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    request_timeout: int,
    is_warmup: bool = False,
    generate_config: Dict[str, Any] = {},
) -> List[ResponseInfo]:
    responses = []

    with ThreadPoolExecutor(max_workers=len(request_indices)) as executor:
        futures = []
        for request_id in request_indices:
            future = executor.submit(
                _curl_server_single_worker,
                request_id,
                batch_size,
                tp0_endpoints,
                input_query,
                is_decode,
                decode_test_length,
                request_timeout,
                is_warmup,
                generate_config,
            )
            futures.append(future)

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
        self.request_timeout = (
            30 + request_tpot * decode_test_length // 1000 + connection_timeout
        )
        self.num_processes = max(
            1,
            (self.total_batch_size + self.max_requests_per_process - 1)
            // self.max_requests_per_process,
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)
        self.decode_test_length = decode_test_length
        self.generate_config = generate_config
        self.tp0_endpoints = self._get_all_dp_tp0_frontends()
        logging.info(f"tp0_endpoints: {self.tp0_endpoints}")
        self.all_grpc_endpoints = self._get_all_rank_grpc_endpoints()
        logging.info(f"all_grpc_endpoints: {self.all_grpc_endpoints}")
        self.profile_all_ranks = os.environ.get("GEN_TIMELINE_SYNC", "0") == "1"

    def run(self):
        self._set_concurrency()
        logging.info(f"finished setting concurrency")
        _ = self._curl_server(is_warmup=True)
        logging.info(f"finished warmup")
        results = self._curl_server()
        logging.info(f"finished measure time")
        # When GEN_TIMELINE_SYNC=1, send StartProfile gRPC directly to all ranks so
        # every tp_rank captures a timeline (not just rank-0 via the HTTP gen_timeline path).
        if self.profile_all_ranks:
            self._start_profile_on_all_ranks(num_steps=2)
        _ = self._curl_server()
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
            resp_status_code = 0
            resp_status_text = "not ok"
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
                errors.append(f"update_scheduler_info exception: {e}")

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
            port = base_port + local_rank * int(
                os.environ.get("WORKER_INFO_PORT_NUM", "10")
            )
            targets.append((nodes[node_idx].ip, int(port)))
        return list(dict.fromkeys(targets))

    def _get_all_rank_grpc_endpoints(self) -> List[Tuple[str, int]]:
        """Get gRPC endpoints for ALL ranks (including tp_ranks) across all nodes."""
        nodes = members_from_test_env(self.gang_config_string)
        worker_info_port_num = int(os.environ.get("WORKER_INFO_PORT_NUM", "10"))
        targets: List[Tuple[str, int]] = []
        total_world_size = self.dp_size * self.tp_size
        for world_rank in range(total_world_size):
            node_idx = world_rank // self.local_world_size
            local_rank = world_rank % self.local_world_size
            if node_idx >= len(nodes):
                break
            base_port = int(nodes[node_idx].server_port)
            grpc_port = base_port + local_rank * worker_info_port_num + 1
            targets.append((nodes[node_idx].ip, grpc_port))
        return list(dict.fromkeys(targets))

    def _start_profile_on_all_ranks(self, num_steps: int = 2):
        """Send StartProfile gRPC request to all ranks (including tp_ranks) to enable profiling."""
        def _send_start_profile(ip: str, port: int) -> Tuple[str, int, bool, str]:
            try:
                channel = grpc.insecure_channel(f"{ip}:{port}")
                stub = pb2_grpc.RpcServiceStub(channel)
                request = pb2.StartProfileRequestPB(
                    trace_name="",
                    start_step=0,
                    num_steps=num_steps,
                )
                stub.StartProfile(request, timeout=5)
                channel.close()
                return ip, port, True, "ok"
            except Exception as e:
                return ip, port, False, str(e)

        max_workers = min(len(self.all_grpc_endpoints), 64)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_send_start_profile, ip, port)
                for ip, port in self.all_grpc_endpoints
            ]
            for fut in futures:
                ip, port, ok, msg = fut.result()
                if ok:
                    logging.info(f"StartProfile succeeded on {ip}:{port}")
                else:
                    logging.warning(f"StartProfile failed on {ip}:{port}: {msg}")

    def _curl_server(
        self, is_warmup: bool = False
    ) -> TestResultMetrics:
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
                    self.total_batch_size // self.dp_size,
                    self.tp0_endpoints,
                    self.input_query,
                    self.is_decode,
                    self.decode_test_length,
                    self.request_timeout,
                    is_warmup,
                    self.generate_config,
                )
            )

        all_responses: List[ResponseInfo] = []
        for future in futures:
            batch_responses = future.result()
            all_responses.extend(batch_responses)

        metrics = analyze_results(all_responses)
        return metrics

    def dump_results(self, results: List[Dict[str, Any]]):
        for result in results:
            logging.debug(json.dumps(result))
