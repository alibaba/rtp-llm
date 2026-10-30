"""Real OpenAI requests against an already running K3 PD cluster."""

from __future__ import annotations

import copy
import json
import math
import pathlib
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def strict_json(value):
    def invalid(constant):
        raise ValueError(f"non-finite JSON constant: {constant}")

    return json.loads(value, parse_constant=invalid)


def validate_invalid_params_response(http_status, body):
    require(
        http_status >= 400
        and isinstance(body, dict)
        and body.get("error_code") == 605
        and body.get("message") == "INVALID_PARAMS",
        f"expected OpenAI INVALID_PARAMS rejection, got HTTP {http_status}: {body}",
    )


def merge_sse(events, *, include_usage=False):
    message = {"content": "", "reasoning_content": "", "tool_calls": []}
    tools = {}
    finish = None
    usage = None
    aux = None
    probabilities = []
    usage_frames = []
    for event_index, event in enumerate(events):
        require(
            not (event.get("error") or event.get("error_code")), f"SSE error: {event}"
        )
        if event.get("choices") == [] and event.get("usage") is not None:
            require(finish is not None, "usage-only SSE frame before terminal choice")
            usage_frames.append(event_index)
        if event.get("usage"):
            usage = event["usage"]
        if event.get("aux_info"):
            aux = event["aux_info"]
        for choice in event.get("choices", []):
            if choice.get("usage"):
                usage = choice["usage"]
            require(choice.get("index", 0) == 0, "unexpected SSE choice index")
            delta = choice.get("delta", {})
            if finish is not None:
                require(
                    not any(
                        delta.get(k)
                        for k in ("content", "reasoning_content", "tool_calls")
                    ),
                    "generated data after terminal SSE choice",
                )
            for field in ("content", "reasoning_content"):
                message[field] += delta.get(field) or ""
            for fragment in delta.get("tool_calls") or []:
                index = fragment["index"]
                tool = tools.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                tool["id"] += fragment.get("id") or ""
                for field in ("name", "arguments"):
                    tool["function"][field] += (
                        fragment.get("function", {}).get(field) or ""
                    )
            probabilities.extend((choice.get("logprobs") or {}).get("content") or [])
            if choice.get("finish_reason") is not None:
                require(finish is None, "duplicate terminal SSE choice")
                finish = choice["finish_reason"]
    require(finish is not None, "SSE ended without a terminal choice")
    if include_usage:
        require(
            usage_frames == [len(events) - 1],
            "SSE must end with exactly one usage-only frame",
        )
    message["tool_calls"] = [tools[index] for index in sorted(tools)]
    return {
        "choices": [
            {
                "message": message,
                "finish_reason": finish,
                "logprobs": {"content": probabilities} if probabilities else None,
            }
        ],
        "usage": usage,
        "aux_info": aux,
    }


def validate_answer(response, expected="5", finish="stop"):
    choice = response["choices"][0]
    require(choice["finish_reason"] == finish, f"unexpected finish reason: {choice}")
    message = choice["message"]
    if expected is not None:
        require(
            (message.get("content") or "").strip() == expected,
            f"wrong answer: {message}",
        )
    usage = response.get("usage")
    require(isinstance(usage, dict), "missing final usage")
    require(
        usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0,
        "empty generation",
    )
    require(
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
        "inconsistent usage",
    )
    return message


def validate_probabilities(rows, top_count):
    require(bool(rows), "missing token probability rows")
    for row in rows:
        value = row["logprob"]
        require(
            isinstance(value, (int, float)) and math.isfinite(value) and value <= 1e-5,
            f"invalid selected logprob: {row}",
        )
        candidates = row.get("top_logprobs", [])
        require(len(candidates) == top_count, "unexpected top probability count")
        for candidate in candidates:
            require(
                math.isfinite(candidate["logprob"]) and candidate["logprob"] <= 1e-5,
                "invalid top probability",
            )
        require(
            all(
                a["logprob"] >= b["logprob"] for a, b in zip(candidates, candidates[1:])
            ),
            "top probabilities are not sorted",
        )


class SmokeRequests:
    def __init__(self, args):
        self.args = args
        self.root = args.output.parent / "protocol"
        self.root.mkdir(exist_ok=False)
        self.records = []
        self.lock = threading.RLock()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.timeout = min(args.timeout, 300)

    def payload(self, prompt="计算2+3。只输出数字，不要解释。", **overrides):
        result = {
            "model": getattr(self.args, "model_type", "kimi_k3"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0,
            "top_k": 1,
            "top_p": 0.95,
            "enable_thinking": False,
            "seed": 0,
            "stream": False,
            "debug_info": True,
        }
        result.update(overrides)
        return result

    def post(self, path, payload, name=None):
        request = urllib.request.Request(
            self.args.base_url.rstrip("/") + path,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                "x-k3-smoke-case": name or "render",
            },
        )
        return self.opener.open(request, timeout=self.timeout)

    def save(self):
        with self.lock:
            temporary = self.root / "summary.json.tmp"
            temporary.write_text(
                json.dumps(
                    [dict(record) for record in self.records],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            temporary.replace(self.root / "summary.json")

    def openai(self, name, payload, check=None, *, expect_invalid_params=False):
        record = {
            "name": name,
            "transport": "openai",
            "request": copy.deepcopy(payload),
            "start_ns": time.time_ns(),
            "passed": False,
            "expected_error": expect_invalid_params,
        }
        self.records.append(record)
        self.save()
        try:
            try:
                with self.post("/v1/chat/completions", payload, name) as response:
                    record["http_status"] = response.status
                    if payload.get("stream"):
                        events, data, done = [], [], False
                        for line in response:
                            line = line.decode("utf-8").rstrip("\r\n")
                            if line.startswith("data:"):
                                data.append(line[5:].lstrip())
                            elif not line and data:
                                value = "\n".join(data)
                                data = []
                                if value == "[DONE]":
                                    done = True
                                    break
                                events.append(strict_json(value))
                        record["events"] = events
                        record["sse_done"] = done
                        # This server also terminates OpenAI streams at HTTP EOF.
                        body = merge_sse(
                            events,
                            include_usage=bool(
                                (payload.get("stream_options") or {}).get(
                                    "include_usage"
                                )
                            ),
                        )
                    else:
                        body = strict_json(response.read())
            except urllib.error.HTTPError as error:
                record["http_status"] = error.code
                body = strict_json(error.read())
            record["response"] = body
            if expect_invalid_params:
                validate_invalid_params_response(record["http_status"], body)
            else:
                require(
                    record["http_status"] == 200 and not body.get("error"),
                    f"request failed: {body}",
                )
                if body.get("aux_info"):
                    require(
                        body["aux_info"].get("pd_sep") is False,
                        "expected a local request",
                    )
                if check:
                    check(body)
            record["passed"] = True
            return body
        except Exception as error:
            record["error"] = repr(error)
            raise AssertionError(
                f"case={name} transport=openai "
                f"artifact={self.root / 'summary.json'}: {error}"
            ) from error
        finally:
            record["end_ns"] = time.time_ns()
            self.save()


def run_request_cases(client):
    client.openai("request_no_think_sync", client.payload(), validate_answer)
    client.openai(
        "request_no_think_sse",
        client.payload(stream=True, stream_options={"include_usage": True}),
        validate_answer,
    )
    default = client.payload()
    for key in ("temperature", "top_p", "top_k"):
        default.pop(key)

    def check_defaults(body):
        validate_answer(body)
        config = body["debug_info"]["generate_config"]
        require(
            config["temperature"] == 0.6 and abs(config["top_p"] - 0.95) < 1e-6,
            "K3 sampling defaults were not applied",
        )

    client.openai("request_sampling_defaults", default, check_defaults)
    client.openai(
        "request_thinking",
        client.payload(
            enable_thinking=True,
            max_tokens=256,
            extra_configs={
                **client.payload().get("extra_configs", {}),
                "max_thinking_tokens": 64,
            },
        ),
        validate_answer,
    )
    for field, value in (("temperature", 1.1), ("top_p", 0.8), ("n", 2)):
        client.openai(
            "request_invalid_" + field,
            client.payload(**{field: value}),
            expect_invalid_params=True,
        )
    tool = {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "enum": [2]},
                    "y": {"type": "integer", "enum": [3]},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    }

    def check_tool(body):
        message = validate_answer(body, expected=None, finish="tool_calls")
        calls = message["tool_calls"]
        require(len(calls) == 1 and bool(calls[0]["id"]), "missing unique tool call")
        require(calls[0]["function"]["name"] == "add", "wrong tool")
        require(
            strict_json(calls[0]["function"]["arguments"]) == {"x": 2, "y": 3},
            "wrong tool arguments",
        )

    payload = client.payload(
        "调用add工具计算2+3。",
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "add"}},
    )
    called = client.openai("request_tool_call", payload, check_tool)
    client.openai(
        "request_tool_sse",
        {**payload, "stream": True, "stream_options": {"include_usage": True}},
        check_tool,
    )
    history = payload["messages"] + [called["choices"][0]["message"]]
    client.openai(
        "request_missing_tool_result",
        client.payload(messages=history, tools=[tool]),
        expect_invalid_params=True,
    )
    history += [
        {
            "role": "tool",
            "tool_call_id": called["choices"][0]["message"]["tool_calls"][0]["id"],
            "content": "5",
        },
        {"role": "user", "content": "只输出工具返回的数字。"},
    ]
    client.openai(
        "request_tool_roundtrip",
        client.payload(messages=history, tools=[tool], tool_choice="none"),
        validate_answer,
    )
    client.openai("request_after_rejections", client.payload(), validate_answer)


def run_tail_health(client):
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                client.openai, f"tail_reuse_slot_{i}", client.payload(), validate_answer
            )
            for i in range(4)
        ]
        for future in futures:
            future.result()
