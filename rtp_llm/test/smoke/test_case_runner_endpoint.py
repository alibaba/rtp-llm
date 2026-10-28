import pytest

from rtp_llm.test.smoke.case_runner import CaseRunner
from rtp_llm.test.smoke.common_def import QueryStatus, SmokeException


def test_prompt_batch_defaults_to_batch_infer_endpoint():
    q_r = {"query": {"prompt_batch": ["hello", "world"]}}

    assert CaseRunner._resolve_endpoint(q_r, "/") == "/batch_infer"


def test_prompt_batch_explicit_endpoint_wins():
    q_r = {
        "endpoint": "/custom_batch",
        "query": {"prompt_batch": ["hello", "world"]},
    }

    assert CaseRunner._resolve_endpoint(q_r, "/") == "/custom_batch"


def test_prompt_batch_streaming_is_rejected():
    q_r = {"query": {"prompt_batch": ["hello"], "yield_generator": True}}

    with pytest.raises(SmokeException) as excinfo:
        CaseRunner._resolve_endpoint(q_r, "/")

    assert excinfo.value.error_status == QueryStatus.VALID_FAILED
    assert "prompt_batch queries must be non-streaming" in excinfo.value.message


def test_non_batch_query_uses_task_endpoint():
    q_r = {"query": {"prompt": "hello"}}

    assert (
        CaseRunner._resolve_endpoint(q_r, "/v1/chat/completions")
        == "/v1/chat/completions"
    )
