import json
from typing import Any, Dict

import torch
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException

from rtp_llm.models.downstream_modules.embedding.api_datatype import (
    SimilarityRequest,
    SimilarityResponse,
)


class SimilarityComparer(BaseComparer):
    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return SimilarityRequest(**query_json)

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        return SimilarityResponse(**result_json)

    def curl_response_to_json(
        self, query_info: Any, curl_response: Any
    ) -> Dict[str, Any]:
        return json.loads(curl_response)

    def compare_result(
        self, expect_result: SimilarityResponse, actual_result: SimilarityResponse
    ) -> None:
        if type(expect_result) != type(actual_result):
            raise SmokeException(QueryStatus.COMPARE_FAILED, f"type not equal")
        res = torch.isclose(
            torch.tensor(expect_result.similarity),
            torch.tensor(actual_result.similarity),
            rtol=1e-2,
            atol=1e-2,
        ).reshape(-1)
        if not all(res):
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"similarity not equal, {expect_result.similarity} vs {actual_result.similarity}",
            )
