import json
from typing import Any, Dict

import torch
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException

from rtp_llm.models.downstream_modules.classifier.api_datatype import (
    ClassifierRequest,
    ClassifierResponse,
)


class ClassifierComparer(BaseComparer):
    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return ClassifierRequest(**query_json)

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        return ClassifierResponse(**result_json)

    def curl_response_to_json(
        self, query_info: ClassifierRequest, curl_response: Any
    ) -> Dict[str, Any]:
        return json.loads(curl_response)

    def compare_result(
        self, expect_result: ClassifierResponse, actual_result: ClassifierResponse
    ):
        rtol = 1e-2
        atol = 1e-2
        res = torch.isclose(
            torch.tensor(expect_result.score),
            torch.tensor(actual_result.score),
            rtol=rtol,
            atol=atol,
        ).reshape(-1)
        if not all(res):
            raise SmokeException(QueryStatus.COMPARE_FAILED, f"scores at not equal")
