#!/usr/bin/env python3
"""Stress an OpenAI-compatible RTP-LLM endpoint with streaming tool calls."""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiohttp


@dataclass
class ToolCallState:
    index: int
    call_id: Optional[str] = None
    call_type: Optional[str] = None
    name: Optional[str] = None
    arguments: str = ""
    name_emissions: int = 0


@dataclass
class RequestResult:
    request_index: int
    expected_cancel: bool
    parallel_tool_calls: bool = False
    elapsed_s: float = 0.0
    ttft_s: Optional[float] = None
    http_status: Optional[int] = None
    finish_reason: Optional[str] = None
    terminal_error: Optional[str] = None
    saw_done: bool = False
    done_count: int = 0
    cancelled: bool = False
    normal_text: str = ""
    structural_errors: list[str] = field(default_factory=list)
    semantic_errors: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallState] = field(default_factory=list)


def percentile(values: list[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bounded_repr(value: Any, max_length: int = 160) -> str:
    rendered = repr(value)
    if len(rendered) <= max_length:
        return rendered
    return rendered[: max_length - 3] + "..."


def get_recovery_error(
    status_before: Any,
    status_after: Any,
    expected_cancel_count: int,
    skip_frontend_recovery_check: bool = False,
    explicit_worker_status: bool = False,
) -> Optional[str]:
    if explicit_worker_status and (
        status_before is None or status_after is None
    ):
        return (
            "cannot verify explicit worker status endpoint reachability: "
            f"before={bounded_repr(status_before)}, "
            f"after={bounded_repr(status_after)}"
        )
    if skip_frontend_recovery_check or expected_cancel_count == 0:
        return None

    before_available = (
        status_before.get("frontend_available_concurrency")
        if isinstance(status_before, dict)
        else status_before
    )
    after_available = (
        status_after.get("frontend_available_concurrency")
        if isinstance(status_after, dict)
        else status_after
    )
    if type(before_available) is not int or type(after_available) is not int:
        return (
            "cannot verify frontend concurrency recovery: "
            f"before={bounded_repr(before_available)}, "
            f"after={bounded_repr(after_available)}"
        )
    if after_available < before_available:
        return (
            "frontend concurrency did not recover: "
            f"before={before_available}, after={after_available}"
        )
    return None


def get_warmup_error(
    result: RequestResult, fail_on_semantic_errors: bool = True
) -> Optional[str]:
    errors = list(result.structural_errors)
    if fail_on_semantic_errors:
        errors.extend(result.semantic_errors)
    if not errors:
        return None
    return f"warmup request {result.request_index} failed: {errors}"


def benchmark_failed(
    summary: dict[str, Any], fail_on_semantic_errors: bool = True
) -> bool:
    if summary["structural_error_count"] > 0:
        return True
    return fail_on_semantic_errors and summary["semantic_error_count"] > 0


def select_cancel_flags(
    request_count: int, cancel_rate: float, seed: int
) -> list[bool]:
    cancel_count = min(
        max(request_count - 1, 0), math.floor(request_count * cancel_rate + 0.5)
    )
    rng = random.Random(seed)
    selected = set(rng.sample(range(request_count), cancel_count))
    return [request_index in selected for request_index in range(request_count)]


def build_payload(
    model: str,
    request_index: int,
    max_tokens: int,
    tool_choice: str = "named",
    parallel_tool_calls: bool = False,
) -> dict[str, Any]:
    if tool_choice not in ("named", "required", "auto"):
        raise ValueError(f"unsupported tool choice: {tool_choice}")
    if parallel_tool_calls and tool_choice == "named":
        raise ValueError("parallel tool calls require tool_choice=required or auto")
    request_marker = f"request-{request_index:08d}"
    payload_marker = f"payload-{request_index:08d}"
    if parallel_tool_calls:
        prompt = (
            "Call echo_request exactly once with request_id set to "
            f"{request_marker}. Call echo_payload exactly once with payload set to "
            f"{payload_marker}. Return both calls in one response and do not answer "
            "with normal text."
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo_request",
                    "description": "Echo a request-scoped identifier.",
                    "parameters": {
                        "type": "object",
                        "properties": {"request_id": {"type": "string"}},
                        "required": ["request_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "echo_payload",
                    "description": "Echo a request-scoped payload marker.",
                    "parameters": {
                        "type": "object",
                        "properties": {"payload": {"type": "string"}},
                        "required": ["payload"],
                    },
                },
            },
        ]
    else:
        prompt = (
            "Call echo_request exactly once. Set request_id to "
            f"{request_marker} and payload to {payload_marker}. "
            "Do not call any other tool and do not answer with normal text."
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo_request",
                    "description": "Echo request-scoped markers for a concurrency test.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request_id": {"type": "string"},
                            "payload": {"type": "string"},
                        },
                        "required": ["request_id", "payload"],
                    },
                },
            }
        ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tool_choice == "named":
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": "echo_request"},
        }
    return payload


def merge_tool_delta(
    states: dict[int, ToolCallState], delta: dict[str, Any], errors: list[str]
) -> bool:
    emitted_payload = False
    raw_calls = delta.get("tool_calls")
    if raw_calls is None:
        return False
    if not isinstance(raw_calls, list):
        errors.append("tool_calls delta is not a list")
        return False
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            errors.append("tool call delta is not an object")
            continue
        index = raw_call.get("index")
        if type(index) is not int or index < 0:
            errors.append(f"invalid tool index: {index!r}")
            continue
        state = states.setdefault(index, ToolCallState(index=index))

        call_id = raw_call.get("id")
        if call_id is not None and not isinstance(call_id, str):
            errors.append(f"tool {index} id is not a string")
        elif call_id:
            emitted_payload = True
            if state.call_id and state.call_id != call_id:
                errors.append(
                    f"tool {index} changed id from {state.call_id!r} to {call_id!r}"
                )
            state.call_id = state.call_id or call_id

        call_type = raw_call.get("type")
        if call_type is not None:
            if not isinstance(call_type, str):
                errors.append(f"tool {index} type is not a string")
            else:
                if state.call_type and state.call_type != call_type:
                    errors.append(
                        f"tool {index} changed type from "
                        f"{state.call_type!r} to {call_type!r}"
                    )
                state.call_type = state.call_type or call_type
        if "function" in raw_call:
            function = raw_call["function"]
        else:
            function = {}
        if not isinstance(function, dict):
            errors.append(f"tool {index} function delta is not an object")
            continue
        name = function.get("name")
        if name is not None and not isinstance(name, str):
            errors.append(f"tool {index} name is not a string")
        elif name:
            emitted_payload = True
            state.name_emissions += 1
            if state.name and state.name != name:
                errors.append(
                    f"tool {index} changed name from {state.name!r} to {name!r}"
                )
            state.name = state.name or name

        arguments = function.get("arguments")
        if arguments is not None and not isinstance(arguments, str):
            errors.append(f"tool {index} arguments delta is not a string")
        elif arguments:
            emitted_payload = True
            state.arguments += arguments
    return emitted_payload


def get_sse_error(chunk: dict[str, Any]) -> Optional[str]:
    if "error" in chunk:
        return f"SSE error: {bounded_repr(chunk['error'], 500)}"
    if "error_code" in chunk:
        code = bounded_repr(chunk.get("error_code"), 80)
        message = bounded_repr(
            chunk.get("message", chunk.get("error_message", "")), 500
        )
        return f"SSE error {code}: {message}"
    return None


async def iter_sse_lines(content: Any) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""

    async for raw_chunk in content:
        pending += decoder.decode(raw_chunk)
        while True:
            cr_index = pending.find("\r")
            lf_index = pending.find("\n")
            indices = [index for index in (cr_index, lf_index) if index >= 0]
            if not indices:
                break
            line_end = min(indices)
            if pending[line_end] == "\r" and line_end + 1 == len(pending):
                break
            separator_length = (
                2 if pending.startswith("\r\n", line_end) else 1
            )
            yield pending[:line_end]
            pending = pending[line_end + separator_length :]

    pending += decoder.decode(b"", final=True)
    while True:
        cr_index = pending.find("\r")
        lf_index = pending.find("\n")
        indices = [index for index in (cr_index, lf_index) if index >= 0]
        if not indices:
            break
        line_end = min(indices)
        separator_length = 2 if pending.startswith("\r\n", line_end) else 1
        yield pending[:line_end]
        pending = pending[line_end + separator_length :]
    if pending:
        yield pending


async def iter_sse_data(
    content: Any, structural_errors: list[str]
) -> AsyncIterator[str]:
    data_lines: list[str] = []
    frame_open = False

    async for line in iter_sse_lines(content):
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
            data_lines.clear()
            frame_open = False
            continue

        frame_open = True
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)

    if frame_open:
        structural_errors.append("SSE stream ended with an unterminated frame")


def validate_completed_request(result: RequestResult) -> None:
    if result.http_status != 200 or result.terminal_error is not None:
        return
    if result.done_count == 0:
        result.structural_errors.append("stream ended before [DONE]")
    elif result.done_count != 1:
        result.structural_errors.append(
            f"stream contains {result.done_count} [DONE] sentinels, expected exactly one"
        )
    if result.finish_reason != "tool_calls":
        result.structural_errors.append(
            f"finish_reason is {result.finish_reason!r}, expected 'tool_calls'"
        )
    if result.normal_text:
        result.semantic_errors.append(
            f"response contains normal text: {result.normal_text!r}"
        )
    if not result.tool_calls:
        result.semantic_errors.append("response contains no tool call")
        return

    request_marker = f"request-{result.request_index:08d}"
    payload_marker = f"payload-{result.request_index:08d}"
    if result.parallel_tool_calls:
        expected_arguments = {
            "echo_request": {"request_id": request_marker},
            "echo_payload": {"payload": payload_marker},
        }
    else:
        expected_arguments = {
            "echo_request": {
                "request_id": request_marker,
                "payload": payload_marker,
            }
        }

    expected_count = len(expected_arguments)
    if len(result.tool_calls) != expected_count:
        result.semantic_errors.append(
            f"response contains {len(result.tool_calls)} tool calls, "
            f"expected exactly {expected_count}"
        )
    actual_indices = [state.index for state in result.tool_calls]
    expected_indices = list(range(len(result.tool_calls)))
    if actual_indices != expected_indices:
        result.structural_errors.append(
            f"tool indices are {actual_indices!r}, expected {expected_indices!r}"
        )

    observed_names: list[Optional[str]] = []
    for state in result.tool_calls:
        if not state.call_id:
            result.structural_errors.append(f"tool {state.index} has no id")
        if state.call_type != "function":
            result.structural_errors.append(
                f"tool {state.index} type is {state.call_type!r}, expected 'function'"
            )
        if state.name_emissions != 1:
            result.structural_errors.append(
                f"tool {state.index} emitted its name {state.name_emissions} times"
            )
        observed_names.append(state.name)
        if state.name not in expected_arguments:
            result.semantic_errors.append(
                f"tool {state.index} called unexpected function {state.name!r}"
            )
        try:
            arguments = json.loads(state.arguments)
        except (ValueError, RecursionError) as error:
            result.structural_errors.append(
                f"tool {state.index} has invalid JSON arguments: "
                f"{type(error).__name__}: {error}"
            )
            continue

        expected = expected_arguments.get(state.name)
        if expected is not None and arguments != expected:
            result.semantic_errors.append(
                f"tool {state.index} arguments mismatch: {arguments!r} != {expected!r}"
            )

    for name in expected_arguments:
        emission_count = observed_names.count(name)
        if emission_count != 1:
            result.semantic_errors.append(
                f"function {name!r} was called {emission_count} times, expected once"
            )


async def run_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    request_index: int,
    max_tokens: int,
    expected_cancel: bool,
    semaphore: asyncio.Semaphore,
    tool_choice: str = "named",
    parallel_tool_calls: bool = False,
    cancel_after_s: float = 0.05,
    cancel_dispatch_timeout_s: float = 10.0,
    api_key: str = "",
) -> RequestResult:
    result = RequestResult(
        request_index=request_index,
        expected_cancel=expected_cancel,
        parallel_tool_calls=parallel_tool_calls,
    )
    states: dict[int, ToolCallState] = {}
    response_holder: list[aiohttp.ClientResponse] = []
    response_ready = asyncio.Event()
    protocol_done = asyncio.Event()
    start: float

    async def consume_response() -> None:
        try:
            request_headers = (
                {"Authorization": f"Bearer {api_key}"} if api_key else None
            )
            async with session.post(
                endpoint,
                json=build_payload(
                    model,
                    request_index,
                    max_tokens,
                    tool_choice,
                    parallel_tool_calls,
                ),
                headers=request_headers,
                allow_redirects=False,
            ) as response:
                response_holder.append(response)
                result.http_status = response.status
                response_ready.set()
                if response.status != 200:
                    result.structural_errors.append(f"HTTP {response.status}")
                    body = (await response.content.read(500)).decode(
                        "utf-8", errors="replace"
                    )
                    if body:
                        result.structural_errors[-1] += f": {body}"
                    return result

                content_type = response.headers.get("Content-Type", "")
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type != "text/event-stream":
                    result.structural_errors.append(
                        "HTTP 200 Content-Type media type is "
                        f"{media_type!r}, expected 'text/event-stream'"
                    )
                    return result

                async for data in iter_sse_data(
                    response.content, result.structural_errors
                ):
                    if data == "[DONE]":
                        result.done_count += 1
                        result.saw_done = True
                        protocol_done.set()
                        continue
                    if result.saw_done:
                        result.structural_errors.append(
                            "SSE data received after [DONE]: "
                            + bounded_repr(data, 500)
                        )
                        continue
                    try:
                        chunk = json.loads(data)
                    except (ValueError, RecursionError) as error:
                        result.structural_errors.append(
                            f"invalid SSE JSON: {type(error).__name__}: {error}"
                        )
                        continue

                    if not isinstance(chunk, dict):
                        result.terminal_error = (
                            "SSE payload is not an object: " + bounded_repr(chunk)
                        )
                        result.structural_errors.append(result.terminal_error)
                        break

                    stream_error = get_sse_error(chunk)
                    if stream_error is not None:
                        result.terminal_error = stream_error
                        result.structural_errors.append(stream_error)
                        break

                    if "choices" in chunk:
                        choices = chunk["choices"]
                    else:
                        choices = []
                    if not isinstance(choices, list):
                        result.structural_errors.append("SSE choices is not a list")
                        continue
                    if len(choices) > 1:
                        result.structural_errors.append(
                            f"SSE chunk contains {len(choices)} choices, expected at most one"
                        )

                    for choice in choices:
                        if not isinstance(choice, dict):
                            result.structural_errors.append(
                                "SSE choice is not an object"
                            )
                            continue
                        choice_index = choice.get("index")
                        if type(choice_index) is not int or choice_index != 0:
                            result.structural_errors.append(
                                f"invalid choice index: {choice_index!r}"
                            )
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            if not isinstance(finish_reason, str):
                                result.structural_errors.append(
                                    "finish_reason is not a string"
                                )
                            elif (
                                result.finish_reason is not None
                                and result.finish_reason != finish_reason
                            ):
                                result.structural_errors.append(
                                    "finish_reason changed from "
                                    f"{result.finish_reason!r} to {finish_reason!r}"
                                )
                            else:
                                result.finish_reason = finish_reason
                        if "delta" in choice:
                            delta = choice["delta"]
                        else:
                            delta = {}
                        if not isinstance(delta, dict):
                            result.structural_errors.append(
                                "choice delta is not an object"
                            )
                            continue
                        content = delta.get("content")
                        if content is not None and not isinstance(content, str):
                            result.structural_errors.append(
                                "content delta is not a string"
                            )
                        elif content:
                            result.normal_text += content
                        reasoning_content = delta.get("reasoning_content")
                        if reasoning_content is not None and not isinstance(
                            reasoning_content, str
                        ):
                            result.structural_errors.append(
                                "reasoning_content delta is not a string"
                            )
                            reasoning_content = None
                        emitted_payload = bool(content or reasoning_content)
                        emitted_payload |= merge_tool_delta(
                            states, delta, result.structural_errors
                        )
                        if emitted_payload and result.ttft_s is None:
                            result.ttft_s = time.perf_counter() - start
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            result.structural_errors.append(type(error).__name__ + ": " + str(error))

    async with semaphore:
        start = time.perf_counter()
        request_task = asyncio.create_task(consume_response())
        response_ready_task: Optional[asyncio.Task[bool]] = None
        protocol_done_task: Optional[asyncio.Task[bool]] = None

        async def cancel_inflight_request() -> None:
            if response_holder:
                response_holder[0].close()
            if not request_task.done():
                request_task.cancel()
                try:
                    await request_task
                except asyncio.CancelledError:
                    pass

        try:
            if expected_cancel:
                response_ready_task = asyncio.create_task(response_ready.wait())
                done, _ = await asyncio.wait(
                    {request_task, response_ready_task},
                    timeout=cancel_dispatch_timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if response_ready_task in done:
                    protocol_done_task = asyncio.create_task(protocol_done.wait())
                    done, _ = await asyncio.wait(
                        {request_task, protocol_done_task},
                        timeout=cancel_after_s,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if request_task in done or request_task.done():
                        await request_task
                    elif (
                        protocol_done_task in done
                        or protocol_done.is_set()
                        or result.saw_done
                        or result.done_count > 0
                    ):
                        await request_task
                    else:
                        await cancel_inflight_request()
                        result.cancelled = True
                elif request_task in done:
                    await request_task
                else:
                    result.structural_errors.append(
                        "response headers were not received before the "
                        f"{cancel_dispatch_timeout_s:g}s cancellation dispatch timeout"
                    )
                    await cancel_inflight_request()
                    result.cancelled = True
            else:
                await request_task
        except asyncio.CancelledError:
            await cancel_inflight_request()
            raise
        finally:
            if response_ready_task is not None and not response_ready_task.done():
                response_ready_task.cancel()
                try:
                    await response_ready_task
                except asyncio.CancelledError:
                    pass
            if protocol_done_task is not None and not protocol_done_task.done():
                protocol_done_task.cancel()
                try:
                    await protocol_done_task
                except asyncio.CancelledError:
                    pass
            result.elapsed_s = time.perf_counter() - start

    result.tool_calls = [states[index] for index in sorted(states)]
    if expected_cancel:
        if not result.cancelled:
            result.structural_errors.append("request selected for cancellation did not cancel")
    else:
        validate_completed_request(result)
    return result


async def fetch_worker_status(
    session: aiohttp.ClientSession,
    worker_status_url: str,
    timeout_s: float = 5.0,
) -> Optional[dict[str, Any]]:
    try:
        async with session.get(
            worker_status_url, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as response:
            if response.status == 200:
                return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
        pass
    return None


def mark_duplicate_call_ids(results: list[RequestResult]) -> int:
    first_request_by_id: dict[str, int] = {}
    duplicate_count = 0
    for result in results:
        if result.expected_cancel:
            continue
        for call in result.tool_calls:
            if call.call_id is None:
                continue
            first_request = first_request_by_id.get(call.call_id)
            if first_request is None:
                first_request_by_id[call.call_id] = result.request_index
                continue
            duplicate_count += 1
            error = (
                f"duplicate tool call id {bounded_repr(call.call_id)}; "
                f"first seen in request {first_request}"
            )
            if error not in result.structural_errors:
                result.structural_errors.append(error)
    return duplicate_count


def is_successful_completion(result: RequestResult) -> bool:
    return (
        not result.expected_cancel
        and result.http_status == 200
        and result.terminal_error is None
        and result.saw_done
        and result.done_count == 1
        and not result.structural_errors
        and not result.semantic_errors
    )


def summarize_results(
    args: argparse.Namespace,
    endpoint: str,
    results: list[RequestResult],
    wall_time_s: float,
    status_before: Any,
    status_after: Any,
) -> dict[str, Any]:
    duplicate_call_id_count = mark_duplicate_call_ids(results)
    normal_requests = [result for result in results if not result.expected_cancel]
    cancel_requests = [result for result in results if result.expected_cancel]
    successful_requests = [
        result for result in normal_requests if is_successful_completion(result)
    ]
    successful_cancellations = [
        result
        for result in cancel_requests
        if result.cancelled
        and not result.structural_errors
        and not result.semantic_errors
    ]
    latencies = [result.elapsed_s for result in successful_requests]
    ttfts = [
        result.ttft_s
        for result in successful_requests
        if result.ttft_s is not None
    ]
    structural_error_count = sum(len(r.structural_errors) for r in results)
    semantic_error_count = sum(len(r.semantic_errors) for r in results)

    frontend_recovery_checked = (
        bool(cancel_requests) and not args.skip_frontend_recovery_check
    )
    recovery_error = get_recovery_error(
        status_before,
        status_after,
        len(cancel_requests),
        args.skip_frontend_recovery_check,
        args.worker_status_url is not None,
    )
    if recovery_error:
        structural_error_count += 1

    successful_rps = (
        len(successful_requests) / wall_time_s if wall_time_s else None
    )
    attempted_rps = len(results) / wall_time_s if wall_time_s else None
    return {
        "configuration": {
            "endpoint": endpoint,
            "worker_status_url": resolve_worker_status_url(
                args.base_url, args.worker_status_url
            ),
            "skip_frontend_recovery_check": args.skip_frontend_recovery_check,
            "model": args.model,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "cancel_rate": args.cancel_rate,
            "cancel_after_s": args.cancel_after,
            "cancel_dispatch_timeout_s": args.cancel_dispatch_timeout,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            "timeout_s": args.timeout,
            "worker_status_timeout_s": args.worker_status_timeout,
            "recovery_wait_s": args.recovery_wait,
            "seed": args.seed,
            "tool_choice": args.tool_choice,
            "parallel_tool_calls": args.parallel_tool_calls,
        },
        "throughput_rps": successful_rps,
        "attempted_rps": attempted_rps,
        "wall_time_s": wall_time_s,
        "completed_requests": len(successful_requests),
        "failed_requests": len(normal_requests) - len(successful_requests),
        "selected_cancellation_requests": len(cancel_requests),
        "cancelled_requests": len(successful_cancellations),
        "failed_cancellation_requests": len(cancel_requests)
        - len(successful_cancellations),
        "structural_error_count": structural_error_count,
        "semantic_error_count": semantic_error_count,
        "duplicate_call_id_count": duplicate_call_id_count,
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "ttft_s": {
            "mean": statistics.fmean(ttfts) if ttfts else None,
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "worker_status_before": status_before,
        "worker_status_after": status_after,
        "frontend_recovery_checked": frontend_recovery_checked,
        "recovery_error": recovery_error,
        "errors": [
            {
                "request_index": result.request_index,
                "structural": result.structural_errors,
                "semantic": result.semantic_errors,
            }
            for result in results
            if result.structural_errors or result.semantic_errors
        ][: args.max_reported_errors],
    }


def resolve_worker_status_url(
    base_url: str, worker_status_url: Optional[str]
) -> str:
    if worker_status_url is not None:
        return worker_status_url

    root_url = base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]
    return root_url + "/worker_status"


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    worker_status_url = resolve_worker_status_url(
        args.base_url, args.worker_status_url
    )
    headers = {"User-Agent": "rtp-llm-tool-call-benchmark"}
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 32))
    cancel_flags = select_cancel_flags(args.requests, args.cancel_rate, args.seed)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        status_before = await fetch_worker_status(
            session, worker_status_url, args.worker_status_timeout
        )

        for warmup_index in range(args.warmup):
            warmup_result = await run_request(
                session,
                endpoint,
                args.model,
                -(warmup_index + 1),
                args.max_tokens,
                False,
                semaphore,
                args.tool_choice,
                args.parallel_tool_calls,
                args.cancel_after,
                args.cancel_dispatch_timeout,
                args.api_key,
            )
            warmup_error = get_warmup_error(
                warmup_result, args.fail_on_semantic_errors
            )
            if warmup_error is not None:
                raise RuntimeError(warmup_error)

        start = time.perf_counter()
        tasks = [
            asyncio.create_task(
                run_request(
                    session,
                    endpoint,
                    args.model,
                    request_index,
                    args.max_tokens,
                    cancel_flags[request_index],
                    semaphore,
                    args.tool_choice,
                    args.parallel_tool_calls,
                    args.cancel_after,
                    args.cancel_dispatch_timeout,
                    args.api_key,
                )
            )
            for request_index in range(args.requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_time_s = time.perf_counter() - start

        await asyncio.sleep(args.recovery_wait)
        status_after = await fetch_worker_status(
            session, worker_status_url, args.worker_status_timeout
        )

    return summarize_results(
        args, endpoint, results, wall_time_s, status_before, status_after
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument(
        "--worker-status-url",
        help="full worker-status endpoint; defaults to BASE_URL/../worker_status",
    )
    parser.add_argument(
        "--skip-frontend-recovery-check",
        action="store_true",
        help="record Frontend admission recovery as unchecked",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--cancel-rate", type=float, default=0.1)
    parser.add_argument("--cancel-after", type=float, default=0.05)
    parser.add_argument("--cancel-dispatch-timeout", type=float, default=10.0)
    parser.add_argument("--recovery-wait", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--worker-status-timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--tool-choice", choices=("named", "required", "auto"), default="named"
    )
    parser.add_argument("--parallel-tool-calls", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-reported-errors", type=int, default=100)
    semantic_group = parser.add_mutually_exclusive_group()
    semantic_group.add_argument(
        "--fail-on-semantic-errors",
        dest="fail_on_semantic_errors",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    semantic_group.add_argument(
        "--allow-semantic-errors",
        dest="fail_on_semantic_errors",
        action="store_false",
        help="report model semantic mismatches without returning a failure exit code",
    )
    parser.set_defaults(fail_on_semantic_errors=True)
    args = parser.parse_args(argv)
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup non-negative")
    if not math.isfinite(args.cancel_rate) or not 0.0 <= args.cancel_rate < 1.0:
        parser.error("cancel-rate must be in [0, 1)")
    if not math.isfinite(args.cancel_after) or args.cancel_after <= 0:
        parser.error("cancel-after must be positive")
    if (
        not math.isfinite(args.cancel_dispatch_timeout)
        or args.cancel_dispatch_timeout <= 0
    ):
        parser.error("cancel-dispatch-timeout must be positive")
    if (
        args.max_tokens < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.recovery_wait)
        or args.recovery_wait < 0
        or not math.isfinite(args.worker_status_timeout)
        or args.worker_status_timeout <= 0
    ):
        parser.error(
            "max-tokens/timeout/worker-status-timeout must be positive and "
            "recovery-wait non-negative"
        )
    if args.max_reported_errors < 0:
        parser.error("max-reported-errors must be non-negative")
    if args.parallel_tool_calls and args.tool_choice == "named":
        parser.error("parallel tool calls require --tool-choice required or auto")
    return args


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_benchmark(args))
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    failed = benchmark_failed(summary, args.fail_on_semantic_errors)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
