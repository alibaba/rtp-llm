import json
from typing import Any, Dict

import torch
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException

from rtp_llm.models.downstream_modules.reranker.api_datatype import (
    VoyageRerankerRequest,
    VoyageRerankerResponse,
)


class RerankerComparer(BaseComparer):
    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return VoyageRerankerRequest(**query_json)

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        return VoyageRerankerResponse(**result_json)

    def curl_response_to_json(
        self, query_info: VoyageRerankerRequest, curl_response: Any
    ) -> Dict[str, Any]:
        return json.loads(curl_response)

    def compare_result(
        self,
        expect_result: VoyageRerankerResponse,
        actual_result: VoyageRerankerResponse,
    ):
        rtol = 1e-2
        atol = 1e-2
        if len(expect_result.results) != len(actual_result.results):
            raise Exception(
                f"result len is not equal: {len(expect_result.results)} vs {len(actual_result.results)}"
            )
        for left, right in zip(expect_result.results, actual_result.results):
            if left.document != right.document or left.index != right.index:
                raise Exception(f"document or index error: left:{left}, right:{right}")
            res = torch.isclose(
                torch.tensor(left.relevance_score),
                torch.tensor(right.relevance_score),
                rtol=rtol,
                atol=atol,
            ).reshape(-1)
            if not all(res):
                raise SmokeException(QueryStatus.COMPARE_FAILED, f"scores at not equal")
