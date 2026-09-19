import json
import logging
import math
import time
from typing import Any, Dict

import requests
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException

from rtp_llm.server.worker_status import WorkerStatusRequest, CacheStatus


class CacheStatusComparer(BaseComparer):
    def run(self):
        wait = self.qr_info.get("wait_for_cache")
        if wait is None:
            return super().run()

        min_blocks = int(wait["min_cached_blocks"])
        timeout = float(wait["timeout_seconds"])
        if min_blocks <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise SmokeException(QueryStatus.VALID_FAILED, "Invalid cache wait limits")
        query = self.format_query(self.qr_info["query"])
        expected = self.format_result(self.qr_info["result"])
        self.tracer.query = query
        self.tracer.expect_result = expected
        request = query.model_dump(exclude_defaults=True)
        request.update(latest_cache_version=-1, need_cache_keys=True)
        url = f"http://127.0.0.1:{self.server_manager.port}{self.request_endpoint}"
        deadline = time.monotonic() + timeout
        last_status = "no cache status received"
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = requests.post(url, json=request, timeout=min(2.0, remaining))
                response.raise_for_status()
                actual = self.format_result(response.json())
                self.tracer.actual_result = actual
                blocks = len(set(actual.cached_keys or []))
                last_status = (
                    f"block_size={actual.block_size}, cached_blocks={blocks}, "
                    f"version={actual.version}"
                )
                if actual.block_size == expected.block_size and blocks >= min_blocks:
                    self._dump_actual_to_artifact(actual)
                    logging.info("Cache ready at %s: %s", url, last_status)
                    return
            except requests.RequestException as exc:
                last_status = str(exc)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.1, remaining))
        self._dump_actual_to_artifact(self.tracer.actual_result)
        raise SmokeException(
            QueryStatus.COMPARE_FAILED,
            f"Cache wait timed out after {timeout}s at {url}: expected "
            f"block_size={expected.block_size}, cached_blocks>={min_blocks}; {last_status}",
        )

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return WorkerStatusRequest(**query_json)

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        logging.debug(f"result_json: {result_json}")
        if "cache_keys" in result_json:
            result_json = {
                **result_json,
                "cached_keys": [
                    int(key) for key, present in result_json["cache_keys"].items()
                    if present
                ],
            }
        return CacheStatus(**result_json)

    def curl_response_to_json(
        self, query_info: Any, curl_response: Any
    ) -> Dict[str, Any]:
        logging.debug(f"curl_response: {curl_response}")
        return json.loads(curl_response)

    def compare_result(
        self, expect_result: CacheStatus, actual_result: CacheStatus
    ) -> None:
        if type(expect_result) != type(actual_result):
            raise SmokeException(QueryStatus.COMPARE_FAILED, f"type not equal")
        check_fields = [
            "block_size",
            "version",
        ]
        for check_field in check_fields:
            expect_val = getattr(expect_result, check_field)
            actual_val = getattr(actual_result, check_field)

            # 其他字段直接比较
            # logging.info("expect_val: %s , actual_val: %s", str(expect_val), str(actual_val))
            if expect_val != actual_val:
                logging.info(
                    f"Check {check_field} 失败: {expect_val} != {actual_val}"
                )
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    f"{check_field} 不匹配\n预期: {expect_val}\n实际: {actual_val}",
                )
