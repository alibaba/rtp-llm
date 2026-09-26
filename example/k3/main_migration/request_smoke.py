"""Real OpenAI and DashSc requests against an already running K3 PD cluster."""

from __future__ import annotations

import copy
import json
import math
import pathlib
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def strict_json(value):
    def invalid(constant):
        raise ValueError(f"non-finite JSON constant: {constant}")

    return json.loads(value, parse_constant=invalid)


def dash_response_error(frame):
    if frame.get("error_message"):
        return frame["error_message"]
    parameters = (frame.get("infer_response") or {}).get("parameters", {})
    status = int((parameters.get("status_code") or {}).get("int64_param", 0))
    error = int((parameters.get("error_no") or {}).get("int64_param", 0))
    if status >= 400 or error:
        return (
            (parameters.get("status_message") or {}).get("string_param")
            or (parameters.get("error_msg") or {}).get("string_param")
            or f"DashSc status={status} error_no={error}"
        )
    return None


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


def validate_probabilities(rows, top_count, *, greedy=False):
    require(bool(rows), "missing token probability rows")
    for row in rows:
        value = row["logprob"]
        require(
            isinstance(value, (int, float)) and math.isfinite(value) and value <= 1e-5,
            f"invalid selected logprob: {row}",
        )
        if greedy:
            require(
                abs(value) <= 1e-5, f"top_k=1 filtered probability is not one: {row}"
            )
        candidates = row.get("top_logprobs", [])
        require(len(candidates) <= top_count, "too many top probabilities")
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
            "model": "kimi-k3",
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
        if self.args.decode_role_addrs:
            result["extra_configs"] = {"role_addrs": [self.args.decode_role_addrs[0]]}
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

    def openai(self, name, payload, check=None, error_field=None):
        record = {
            "name": name,
            "transport": "openai",
            "request": copy.deepcopy(payload),
            "start_ns": time.time_ns(),
            "passed": False,
            "expected_error": bool(error_field),
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
            if error_field:
                require(
                    record["http_status"] >= 400
                    or body.get("error")
                    or body.get("exception"),
                    "invalid request unexpectedly succeeded",
                )
                require(
                    error_field.lower() in json.dumps(body, ensure_ascii=False).lower(),
                    f"error does not identify {error_field}: {body}",
                )
            else:
                require(
                    record["http_status"] == 200 and not body.get("error"),
                    f"request failed: {body}",
                )
                if body.get("aux_info"):
                    require(
                        body["aux_info"].get("pd_sep") is True, "request bypassed PD"
                    )
                if check:
                    check(body)
            record["passed"] = True
            return body
        except Exception as error:
            record["error"] = repr(error)
            raise
        finally:
            record["end_ns"] = time.time_ns()
            self.save()

    def dash(self, name, payload, *, logprobs=False, top_count=0, error_field=None):
        import grpc
        from google.protobuf.json_format import MessageToDict

        from rtp_llm.dash_sc.client import (
            build_model_infer_request,
            dash_sc_grpc_client_channel_options,
        )
        from rtp_llm.dash_sc.codec import SamplingParams
        from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc

        # Render the prompt on the real K3 frontend; no substitute chat template.
        render_payload = {
            **payload,
            "temperature": 0,
            "top_k": 1,
            "top_p": 0.95,
            "stream": False,
        }
        for key in ("logprobs", "logprobs_mode", "top_logprobs", "_expected_text"):
            render_payload.pop(key, None)
        with self.post("/chat/render", render_payload) as response:
            rendered = strict_json(response.read())
        sampling = SamplingParams(
            max_new_tokens=payload.get("max_tokens", 128),
            num_return_sequences=1,
            temperature=payload.get("temperature", 0),
            top_k=payload.get("top_k", 1),
            top_p=payload.get("top_p", 0.95),
            random_seed=0,
            logprobs=logprobs,
            top_logprobs=top_count,
        )
        request = build_model_infer_request(
            request_id=self.args.namespace + "/" + name,
            model_name="kimi-k3",
            input_ids=rendered["input_ids"],
            sampling=sampling,
            enable_thinking=False,
        )
        host = urllib.parse.urlsplit(self.args.base_url).hostname
        address = f"{host}:{self.args.prefill_grpc_port + 7}"
        record = {
            "name": name,
            "transport": "dashsc",
            "address": address,
            "request": MessageToDict(request, preserving_proto_field_name=True),
            "frames": [],
            "passed": False,
            "start_ns": time.time_ns(),
        }
        self.records.append(record)
        self.save()
        ids, errors, multi, terminal = [], [], 0, False
        try:
            with grpc.insecure_channel(
                address, options=dash_sc_grpc_client_channel_options()
            ) as channel:
                stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
                for frame in stub.ModelStreamInfer(
                    iter([request]), timeout=self.timeout
                ):
                    event = MessageToDict(frame, preserving_proto_field_name=True)
                    record["frames"].append(event)
                    if error := dash_response_error(event):
                        errors.append(error)
                    if not frame.HasField("infer_response"):
                        continue
                    infer = frame.infer_response
                    require(
                        len(infer.outputs) == len(infer.raw_output_contents),
                        "DashSc output/raw count mismatch",
                    )
                    tokens = []
                    for output, raw in zip(infer.outputs, infer.raw_output_contents):
                        if output.name == "generated_ids":
                            require(
                                output.datatype == "INT32" and len(raw) % 4 == 0,
                                "invalid generated_ids",
                            )
                            count = math.prod(output.shape)
                            require(
                                count * 4 == len(raw) or (count == 0 and len(raw) == 4),
                                "token shape mismatch",
                            )
                            tokens = list(struct.unpack(f"<{count}i", raw[: count * 4]))
                        if output.name == "finished":
                            terminal |= bool(raw and raw[0])
                    if tokens:
                        require(not errors, f"DashSc generation error: {errors}")
                        multi += int(len(tokens) > 1)
                        ids.extend(tokens)
                        if logprobs:
                            require(
                                "logprobs" in infer.parameters,
                                "missing DashSc probabilities",
                            )
                            rows = strict_json(
                                infer.parameters["logprobs"].string_param
                            )
                            require(
                                len(rows) == len(tokens),
                                "DashSc token/probability row mismatch",
                            )
                            for token, row in zip(tokens, rows):
                                require(
                                    str(token) in row,
                                    "selected token missing from its own probability row",
                                )
                                require(
                                    all(
                                        math.isfinite(v) and v <= 1e-5
                                        for v in row.values()
                                    ),
                                    "invalid DashSc logprob",
                                )
                                require(
                                    abs(row[str(token)]) <= 1e-5,
                                    "greedy filtered probability is not one",
                                )
                        else:
                            require(
                                "logprobs" not in infer.parameters,
                                "unrequested DashSc probabilities",
                            )
            if error_field:
                require(
                    error_field.lower() in json.dumps(record["frames"]).lower(),
                    "missing expected DashSc error",
                )
                require(not ids, "invalid DashSc request generated tokens")
            else:
                require(not errors, f"DashSc generation error: {errors}")
                require(terminal and ids, "incomplete DashSc generation")
                from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import (
                    TokenizerFactory,
                )

                tokenizer = TokenizerFactory.create(
                    ckpt_path=str(self.args.long_prefix_checkpoint),
                    tokenizer_path=str(self.args.long_prefix_checkpoint),
                    model_type="kimi_k3",
                )
                text = tokenizer.decode(ids).split("<|close|>", 1)[0].strip()
                expected = payload.get("_expected_text", "5")
                require(text == expected, f"wrong DashSc answer: {text!r}")
            record.update(passed=True, generated_ids=ids, multi_token_frames=multi)
            return record
        except Exception as error:
            record["error"] = repr(error)
            raise
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
            error_field=field,
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
        error_field="tool",
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
    client.dash("request_dash_no_think", client.payload())
    client.dash(
        "request_dash_invalid_temperature",
        client.payload(temperature=1.1),
        error_field="temperature",
    )
    client.openai("request_after_rejections", client.payload(), validate_answer)
