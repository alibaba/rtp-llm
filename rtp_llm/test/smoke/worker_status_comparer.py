import json
import logging
from typing import Any, Dict

from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException

from rtp_llm.server.worker_status import WorkerStatusRequest, WorkStatus


class WorkerStatusComparer(BaseComparer):
    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return WorkerStatusRequest(**query_json)

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        logging.debug(f"result_json: {result_json}")
        return WorkStatus(**result_json)

    def curl_response_to_json(
        self, query_info: Any, curl_response: Any
    ) -> Dict[str, Any]:
        logging.debug(f"curl_response: {curl_response}")
        return json.loads(curl_response)

    def compare_result(
        self, expect_result: WorkStatus, actual_result: WorkStatus
    ) -> None:
        if type(expect_result) != type(actual_result):
            raise SmokeException(QueryStatus.COMPARE_FAILED, f"type not equal")
        check_fields = [
            "role",
            "finished_task_list",
            "dp_size",
            "tp_size",
        ]
        expect = expect_result.model_dump(exclude_defaults=True)
        actual = actual_result.model_dump(exclude_defaults=True)
        for check_field in check_fields:
            expect_val = getattr(expect_result, check_field)
            actual_val = getattr(actual_result, check_field)

            # 特殊处理 finished_task_list 字段
            if check_field == "finished_task_list":
                # 对每个 task 过滤 end_time_ms 后比较
                filtered_expect_tasks = [
                    task.model_dump(exclude={"end_time_ms", "request_id", "waiting_time_ms"}) for task in expect_val
                ]
                filtered_actual_tasks = [
                    task.model_dump(exclude={"end_time_ms", "request_id", "waiting_time_ms"}) for task in actual_val
                ]

                if filtered_expect_tasks != filtered_actual_tasks:
                    logging.info(f"Check {check_field} failed (忽略 end_time_ms)")
                    raise SmokeException(
                        QueryStatus.COMPARE_FAILED,
                        f"{check_field} 不匹配 (忽略 end_time_ms)\n"
                        f"预期: {filtered_expect_tasks}\n"
                        f"实际: {filtered_actual_tasks}",
                    )
            else:
                # 其他字段直接比较
                if expect_val != actual_val:
                    logging.info(
                        f"Check {check_field} 失败: {expect_val} != {actual_val}"
                    )
                    raise SmokeException(
                        QueryStatus.COMPARE_FAILED,
                        f"{check_field} 不匹配\n预期: {expect_val}\n实际: {actual_val}",
                    )
