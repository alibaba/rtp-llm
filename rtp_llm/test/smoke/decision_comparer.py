import json
from typing import Any, Dict, List

import torch
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException


class DecisionQuery(BaseModel):
    """Duck-typed mirror of internal_source decision_module.DecisionRequest.

    Kept dependency-free so the smoke comparer does not import internal_source.
    """

    document: str
    questions: List[Dict[str, Any]]
    model: str = ""


class DecisionResult(BaseModel):
    """Duck-typed mirror of decision_module.DecisionResponse."""

    results: List[Dict[str, Any]]


def is_decision_query(query_json: Dict[str, Any]) -> bool:
    return "questions" in query_json and "document" in query_json


class DecisionComparer(BaseComparer):
    """Comparer for /v1/classifier requests that carry the unified decision
    schema (document + typed questions -> per-question probability dicts)."""

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return DecisionQuery(**query_json)

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        return DecisionResult(**result_json)

    def curl_response_to_json(
        self, query_info: DecisionQuery, curl_response: Any
    ) -> Dict[str, Any]:
        return json.loads(curl_response)

    def compare_result(
        self, expect_result: DecisionResult, actual_result: DecisionResult
    ):
        rtol = 1e-2
        atol = 1e-2
        expect = {r["question_id"]: r["probs"] for r in expect_result.results}
        actual = {r["question_id"]: r["probs"] for r in actual_result.results}
        if set(expect) != set(actual):
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"question ids differ: {sorted(expect)} vs {sorted(actual)}",
            )
        for qid, expect_probs in expect.items():
            actual_probs = actual[qid]
            if set(expect_probs) != set(actual_probs):
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    f"labels differ for {qid}: {sorted(expect_probs)} vs {sorted(actual_probs)}",
                )
            for label, expect_p in expect_probs.items():
                if not torch.isclose(
                    torch.tensor(expect_p, dtype=torch.float64),
                    torch.tensor(actual_probs[label], dtype=torch.float64),
                    rtol=rtol,
                    atol=atol,
                ):
                    raise SmokeException(
                        QueryStatus.COMPARE_FAILED,
                        f"prob mismatch for {qid}/{label}: {expect_p} vs {actual_probs[label]}",
                    )
