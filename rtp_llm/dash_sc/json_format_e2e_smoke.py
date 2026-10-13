#!/usr/bin/env python3
"""End-to-end DashScope JSON response_format smoke.

This is intentionally a runnable smoke, not a unit test. It posts real HTTP SSE
requests to a live frontend and fails on engine error packets, the old fake KV
allocation wording, or non-JSON final content.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_ENDPOINT = "/api/v1/services/aigc/text-generation/generation"
DEFAULT_MODEL = "pre-deepseek_v4_flash_rtp_llm_chat"
BAD_MARKERS = (
    "InternalError.EngineAbort",
    "malloc kv cache block failed",
)


class SmokeFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class SmokeCase:
    name: str
    messages: list[dict[str, str]]
    response_format: dict[str, Any]
    enable_thinking: bool = False
    required_keys: tuple[str, ...] = ()


def _person_schema_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["name", "age", "email"],
                "additionalProperties": False,
            },
        },
    }


SMOKE_CASES: tuple[SmokeCase, ...] = (
    SmokeCase(
        name="dash_json_object_incremental",
        messages=[
            {
                "role": "user",
                "content": (
                    "当前，我国社会主义事业正在胜利前进，两个文明建设正在蓬勃展开。\n\n"
                    '请抽取以上句子中所有的 人物、组织、地点、其他实体，用json表示。'
                    '例如：{"人物": [...], "组织": [...], "地点": [...], "其他实体": [...]}'
                ),
            }
        ],
        response_format={"type": "json_object"},
    ),
    SmokeCase(
        name="dash_json_object_incremental_thinking",
        messages=[
            {
                "role": "system",
                "content": (
                    "你需要提取出name（名字，为string类型）、age（年龄，为string类型）"
                    "与email（邮箱，为string类型），请输出JSON 字符串，不要输出其它无关内容。\n"
                    "示例：\n"
                    "Q：我叫张三，今年25岁，邮箱是zhangsan@example.com\n"
                    'A：{"name":"张三","age":"25岁","email":"zhangsan@example.com"}\n'
                    "Q：我叫李四，今年30岁，我的邮箱是lisi@example.com\n"
                    'A：{"name":"李四","age":"30岁","email":"lisi@example.com"}\n'
                    "Q：我叫王五，我的邮箱是wangwu@example.com，今年40岁\n"
                    'A：{"name":"王五","age":"40岁","email":"wangwu@example.com"}'
                ),
            },
            {
                "role": "user",
                "content": "大家好，我叫刘五，今年34岁，邮箱是liuwu@example.com",
            },
        ],
        response_format={"type": "json_object"},
        enable_thinking=True,
    ),
    SmokeCase(
        name="dash_json_schema_incremental",
        messages=[
            {
                "role": "system",
                "content": "只输出满足 schema 的 JSON 对象，不要输出其它内容。",
            },
            {
                "role": "user",
                "content": "大家好，我叫刘五，今年34岁，邮箱是liuwu@example.com",
            },
        ],
        response_format=_person_schema_response_format(),
        required_keys=("name", "age", "email"),
    ),
    SmokeCase(
        name="dash_json_schema_incremental_thinking",
        messages=[
            {
                "role": "system",
                "content": "只输出满足 schema 的 JSON 对象，不要输出其它内容。",
            },
            {
                "role": "user",
                "content": "大家好，我叫刘五，今年34岁，邮箱是liuwu@example.com",
            },
        ],
        response_format=_person_schema_response_format(),
        enable_thinking=True,
        required_keys=("name", "age", "email"),
    ),
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _request_payload(case: SmokeCase, model: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "dashscope_custom": {},
        "response_format": case.response_format,
        "debug": True,
        "result_format": "message",
        "incremental_output": True,
    }
    if case.enable_thinking:
        parameters["enable_thinking"] = True
    return {
        "model": model,
        "input": {"messages": case.messages},
        "parameters": parameters,
    }


def _headers(authorization: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-DashScope-SSE": "enable",
    }
    if authorization:
        auth = authorization.strip()
        if auth and not auth.lower().startswith("bearer "):
            auth = f"Bearer {auth}"
        headers["Authorization"] = auth
    return headers


def _iter_sse_objects(response: Any) -> Iterable[tuple[str | None, dict[str, Any]]]:
    event: str | None = None
    data_lines: list[str] = []

    def dispatch() -> Iterable[tuple[str | None, dict[str, Any]]]:
        if not data_lines:
            return
        data = "\n".join(data_lines).strip()
        data_lines.clear()
        if not data or data == "[DONE]":
            return
        try:
            yield event, json.loads(data)
        except json.JSONDecodeError as exc:
            raise SmokeFailure(f"invalid SSE JSON packet: {data[:512]}") from exc

    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            yield from dispatch()
            event = None
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
            continue
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                yield event, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SmokeFailure(f"invalid JSON packet: {stripped[:512]}") from exc
    yield from dispatch()


def _extract_text(packet: dict[str, Any]) -> str:
    containers: list[Any] = [packet, packet.get("output")]
    index = 0
    while index < len(containers):
        container = containers[index]
        index += 1
        if isinstance(container, dict):
            containers.extend([container.get("message"), container.get("delta")])
            choices = container.get("choices")
            if isinstance(choices, list):
                containers.extend(choices)
        elif isinstance(container, list):
            containers.extend(container)

    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("content", "text"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _json_object_from_text(texts: list[str]) -> dict[str, Any]:
    candidates = ["".join(texts)]
    candidates.extend(reversed(texts))
    if texts:
        candidates.append(max(texts, key=len))
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate.startswith("```"):
            candidate = candidate.strip("` \n")
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    last_text = texts[-1][:512] if texts else "<empty>"
    raise SmokeFailure(f"final content is not a JSON object: {last_text}")


def _check_error_packet(event: str | None, packet: dict[str, Any]) -> None:
    packet_text = _json_dumps(packet)
    if event == "error" or packet.get("_sse_event") == "error" or packet.get("code"):
        raise SmokeFailure(f"error packet received: {packet_text[:1024]}")
    for marker in BAD_MARKERS:
        if marker in packet_text:
            raise SmokeFailure(f"bad marker {marker!r} received: {packet_text[:1024]}")


def _post_case(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> tuple[str, dict[str, Any]]:
    request = urllib.request.Request(
        url=url,
        data=_json_dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    texts: list[str] = []
    request_id = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            request_id = response.headers.get("x-request-id", "")
            for event, packet in _iter_sse_objects(response):
                request_id = packet.get("request_id") or request_id
                _check_error_packet(event, packet)
                text = _extract_text(packet)
                if text:
                    texts.append(text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SmokeFailure(f"http {exc.code}: {body[:1024]}") from exc
    except urllib.error.URLError as exc:
        raise SmokeFailure(f"request failed: {exc}") from exc

    parsed = _json_object_from_text(texts)
    return request_id, parsed


def _select_cases(names: list[str] | None) -> list[SmokeCase]:
    if not names:
        return list(SMOKE_CASES)
    by_name = {case.name: case for case in SMOKE_CASES}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SmokeFailure(f"unknown case(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def _build_url(args: argparse.Namespace) -> str:
    if args.url:
        return args.url
    return args.base_url.rstrip("/") + "/" + args.endpoint.strip("/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("JSON_FORMAT_SMOKE_URL"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("JSON_FORMAT_SMOKE_BASE_URL", "http://127.0.0.1:29200"),
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("JSON_FORMAT_SMOKE_ENDPOINT", DEFAULT_ENDPOINT),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("JSON_FORMAT_SMOKE_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=int(os.getenv("JSON_FORMAT_SMOKE_ATTEMPTS", "5")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("JSON_FORMAT_SMOKE_TIMEOUT", "300")),
    )
    parser.add_argument(
        "--authorization",
        default=os.getenv("JSON_FORMAT_SMOKE_AUTHORIZATION")
        or os.getenv("DASHSCOPE_API_KEY"),
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only this smoke case; repeatable.",
    )
    args = parser.parse_args(argv)

    url = _build_url(args)
    headers = _headers(args.authorization)
    cases = _select_cases(args.case)

    print(f"json_format_e2e_smoke url={url} model={args.model} attempts={args.attempts}")
    for case in cases:
        payload = _request_payload(case, args.model)
        for attempt in range(1, args.attempts + 1):
            start = time.time()
            try:
                request_id, parsed = _post_case(url, headers, payload, args.timeout)
                missing = [key for key in case.required_keys if key not in parsed]
                if missing:
                    raise SmokeFailure(
                        f"missing required key(s): {missing}; parsed={parsed}"
                    )
            except SmokeFailure as exc:
                print(f"[FAIL] {case.name} attempt={attempt}: {exc}", file=sys.stderr)
                return 1
            cost_ms = int((time.time() - start) * 1000)
            print(
                f"[PASS] {case.name} attempt={attempt} request_id={request_id or '-'} "
                f"cost_ms={cost_ms} keys={sorted(parsed.keys())}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
