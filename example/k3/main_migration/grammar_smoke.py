"""Structured K3 answers, thinking transitions, and concurrent grammar requests."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from request_smoke import (
    require,
    strict_json,
    validate_answer,
)


def check_json_answer(body):
    message = validate_answer(body, expected=None)
    value = strict_json(message["content"])
    require(
        isinstance(value, dict) and set(value) == {"answer"}, "unexpected JSON fields"
    )
    require(
        type(value["answer"]) is int and value["answer"] == 5, "wrong JSON type/value"
    )


def run_grammar_cases(client):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "sum_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "integer", "enum": [5]}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    payload = client.payload("计算2+3，返回answer字段。", response_format=schema)
    client.openai("grammar_json_sync", payload, check_json_answer)
    client.openai(
        "grammar_json_sse",
        {**payload, "stream": True, "stream_options": {"include_usage": True}},
        check_json_answer,
    )
    thinking = {
        **payload,
        "enable_thinking": True,
        "max_tokens": 256,
        "extra_configs": {
            **payload.get("extra_configs", {}),
            "max_thinking_tokens": 16,
        },
    }
    client.openai("grammar_thinking_budget", thinking, check_json_answer)
    barrier = Barrier(3)

    def send(name, request, check):
        barrier.wait(timeout=30)
        return client.openai(name, request, check)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(send, "grammar_mixed_plain", client.payload(), validate_answer),
            pool.submit(send, "grammar_mixed_json", payload, check_json_answer),
            pool.submit(send, "grammar_mixed_thinking", thinking, check_json_answer),
        ]
        for future in futures:
            future.result()
