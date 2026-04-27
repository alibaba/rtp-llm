import json
import logging
from typing import Any, Dict

from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException

from rtp_llm.server.worker_status import WorkerStatusRequest, CacheStatus


class CacheStatusComparer(BaseComparer):
    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return WorkerStatusRequest(**query_json)

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        logging.debug(f"result_json: {result_json}")
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
