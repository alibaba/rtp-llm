"""Exercise the unchanged HTTP generation API of a native P/D service."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def make_query(
    prompt,
    max_new_tokens=3,
    *,
    min_new_tokens=None,
    top_k=1,
    top_p=1.0,
    temperature=1.0,
    seed=42,
    reuse_cache=False,
    timeout=1800,
    trace_id=None,
    regex=None,
    stream=False,
):
    config = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": max_new_tokens if min_new_tokens is None else min_new_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "random_seed": seed,
        "reuse_cache": reuse_cache,
        "can_use_pd_separation": True,
        "return_input_ids": True,
        "return_output_ids": True,
        "return_cum_log_probs": True,
        "aux_info": True,
        "timeout_ms": int(timeout * 1000),
    }
    if trace_id is not None:
        config["trace_id"] = trace_id
    if regex is not None:
        config["regex"] = regex
    query = {"prompt": prompt, "generate_config": config}
    if stream:
        query["stream"] = True
        config.update(is_streaming=True, return_incremental=False)
    return query


def output_tokens(result, *, expected_input_tokens=None):
    """Read valid tokens, excluding the RPC response's EOS padding."""
    if "error_code" in result:
        raise RuntimeError(f"native generation failed: {result}")
    outputs, aux = result.get("output_ids", []), result.get("aux_info", {})
    if len(outputs) != 1 or not isinstance(outputs[0], list):
        raise RuntimeError("require one native output token row")
    count = aux.get("output_len")
    if type(count) is not int or not 0 <= count <= len(outputs[0]):
        raise RuntimeError("missing/invalid native aux_info.output_len")
    tokens = outputs[0][:count]
    if any(type(token) is not int or token < 0 for token in tokens):
        raise RuntimeError("invalid generated token ids")
    if (
        expected_input_tokens is not None
        and aux.get("input_len") != expected_input_tokens
    ):
        raise RuntimeError("native input token count differs from the boundary case")
    if aux.get("pd_sep") is not True:
        raise RuntimeError("native request did not use P/D separation")
    if not all(math.isfinite(value) for value in aux.get("cum_log_probs", [])):
        raise RuntimeError("nonfinite native generation scores")
    return tokens


def validate_finished(result, query, *, expected_input_tokens=None, eos_token_ids=()):
    tokens = output_tokens(result, expected_input_tokens=expected_input_tokens)
    if result.get("finished") is not True:
        raise RuntimeError("native HTTP generation did not finish")
    maximum = query["generate_config"]["max_new_tokens"]
    if eos_token_ids:
        if not tokens or tokens[-1] not in eos_token_ids or len(tokens) >= maximum:
            raise RuntimeError("EOS must end the valid token row before the output cap")
        if any(token in eos_token_ids for token in tokens[:-1]):
            raise RuntimeError("native generation continued after an EOS")
    elif len(tokens) != maximum:
        raise RuntimeError("native output count differs from the requested output cap")
    return tokens


class SSETokenChain:
    """Validate contiguous native SSE counters and retain exact cumulative IDs."""

    def __init__(self, *, expected_input_tokens=None, max_new_tokens=None):
        self.expected_input_tokens = expected_input_tokens
        self.max_new_tokens = max_new_tokens
        self._tokens = []
        self._input_tokens = None
        self._finished = False
        self.last_mode = None

    @staticmethod
    def _fields(event):
        if not isinstance(event, dict):
            raise RuntimeError("require one native SSE response object")
        if "error_code" in event:
            raise RuntimeError(f"native generation failed: {event}")
        outputs, aux = event.get("output_ids"), event.get("aux_info")
        if (
            not isinstance(outputs, list)
            or len(outputs) != 1
            or not isinstance(outputs[0], list)
            or not isinstance(aux, dict)
        ):
            raise RuntimeError("require one native SSE token row and aux_info")
        if type(event.get("finished")) is not bool:
            raise RuntimeError("missing/invalid native SSE finished flag")
        return outputs[0], aux

    def append(self, event):
        if self._finished:
            raise RuntimeError("native SSE event followed a finished event")
        row, aux = self._fields(event)
        count, step = aux.get("output_len"), aux.get("step_output_len")
        if (
            type(count) is not int
            or type(step) is not int
            or step <= 0
            or count - len(self._tokens) != step
        ):
            raise RuntimeError("native SSE counters are not contiguous")
        if self.max_new_tokens is not None and count > self.max_new_tokens:
            raise RuntimeError("native SSE output exceeds the requested output cap")
        input_tokens = aux.get("input_len")
        if (
            type(input_tokens) is not int
            or input_tokens < 0
            or self._input_tokens is not None
            and input_tokens != self._input_tokens
        ):
            raise RuntimeError("missing/changed native SSE input token count")
        if len(row) == count:
            if row[: len(self._tokens)] != self._tokens:
                raise RuntimeError("cumulative native SSE token prefix changed")
            combined, mode = row, "cumulative"
        elif len(row) == step:
            combined, mode = self._tokens + row, "incremental"
        else:
            raise RuntimeError("native SSE token row disagrees with total/step counts")
        try:
            tokens = output_tokens(
                dict(event, output_ids=[combined]),
                expected_input_tokens=self.expected_input_tokens,
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("invalid native SSE generation metadata") from error
        self._tokens, self._input_tokens = tokens, input_tokens
        self._finished, self.last_mode = event["finished"], mode
        return list(tokens)


def run_request(
    url,
    query,
    output,
    *,
    request=0,
    timeout=1800,
    expected_input_tokens=None,
    eos_token_ids=(),
    cancel_after_tokens=0,
):
    """Issue real HTTP or disconnect a real SSE stream after native output."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"request-{request}"
    prefix.with_suffix(".query.json").write_text(
        json.dumps(query, ensure_ascii=False, indent=2) + "\n"
    )
    trace_id = query["generate_config"].get("trace_id")
    headers = {"Content-Type": "application/json", "Connection": "close"}
    if trace_id:
        headers.update({"trace-id": trace_id, "x-dashscope-request-id": trace_id})
    started_ns = time.time_ns()
    started = time.monotonic()
    if cancel_after_tokens:
        if not query.get("stream"):
            raise ValueError("cancellation requires the real streaming HTTP API")
        address = urllib.parse.urlsplit(url.rstrip("/") + "/")
        connection_type = (
            http.client.HTTPSConnection
            if address.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(address.hostname, address.port, timeout=timeout)
        events, tokens, response = [], [], None
        chain = SSETokenChain(
            expected_input_tokens=expected_input_tokens,
            max_new_tokens=query["generate_config"]["max_new_tokens"],
        )
        try:
            connection.request(
                "POST", address.path or "/", json.dumps(query).encode(), headers
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"streaming HTTP status was {response.status}")
            with prefix.with_suffix(".sse.txt").open("x") as raw:
                for line in response:
                    text = line.decode()
                    raw.write(text)
                    raw.flush()
                    if not text.startswith("data:"):
                        continue
                    payload = text[5:].strip()
                    if payload == "[DONE]":
                        raise RuntimeError("stream finished before actual cancellation")
                    event = json.loads(payload)
                    record = dict(received_ns=time.time_ns(), response=event)
                    events.append(record)
                    tokens = chain.append(event)
                    record.update(
                        output_ids_mode=chain.last_mode, cumulative_output_ids=[tokens]
                    )
                    if event.get("finished") is True:
                        raise RuntimeError("request completed before the disconnect")
                    if len(tokens) >= cancel_after_tokens:
                        break
                else:
                    raise RuntimeError("stream closed before enough real output")
        finally:
            try:
                if response is not None:
                    response.close()
            finally:
                try:
                    connection.close()
                finally:
                    closed_ns = time.time_ns()
                    prefix.with_suffix(".events.json").write_text(
                        json.dumps(events, ensure_ascii=False, indent=2) + "\n"
                    )
        result = events[-1]["response"]
        outcome = dict(
            intentionally_disconnected=True,
            disconnected_ns=closed_ns,
            cancellation_requires_native_terminal_evidence=True,
        )
    else:
        post = urllib.request.Request(
            url.rstrip("/") + "/", data=json.dumps(query).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(post, timeout=timeout) as response:
                raw, status = response.read().decode(), response.status
        except urllib.error.HTTPError as error:
            raw, status = error.read().decode(), error.code
        prefix.with_suffix(".http.txt").write_text(raw)
        if status != 200:
            raise RuntimeError(f"native HTTP status was {status}: {raw}")
        result = json.loads(raw)
        tokens = validate_finished(
            result,
            query,
            expected_input_tokens=expected_input_tokens,
            eos_token_ids=eos_token_ids,
        )
        outcome = dict(http_pd_generation_ok=True, eos_observed=bool(eos_token_ids))
    record = dict(
        request=request,
        trace_id=trace_id,
        started_ns=started_ns,
        completed_ns=time.time_ns(),
        seconds=time.monotonic() - started,
        output_ids=[tokens],
        aux_info=result["aux_info"],
        **outcome,
    )
    prefix.with_suffix(".summary.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        "--decode-url",
        dest="url",
        required=True,
        help="native P HTTP entrance for the existing P-to-D RPC protocol",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="请用一句话介绍杭州。")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--expected-input-tokens", type=int)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--min-new-tokens", type=int)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--trace-prefix")
    parser.add_argument("--regex")
    parser.add_argument("--eos-token-id", type=int, action="append", default=[])
    parser.add_argument("--cancel-after-tokens", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if (
        args.max_new_tokens < 2
        or min(args.requests, args.concurrency) < 1
        or args.timeout <= 0
        or not 0 < args.top_p <= 1
        or args.top_k < 0
        or args.temperature <= 0
        or not 0 <= (args.min_new_tokens or 0) <= args.max_new_tokens
        or not 0 <= args.cancel_after_tokens < args.max_new_tokens
    ):
        parser.error("invalid request count, sampling, token limits or timeout")
    args.output.mkdir(parents=True, exist_ok=False)
    query = make_query(
        args.prompt,
        args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        seed=args.random_seed,
        reuse_cache=args.reuse_cache,
        timeout=args.timeout,
        regex=args.regex,
        stream=bool(args.cancel_after_tokens),
    )
    (args.output / "query.json").write_text(
        json.dumps(query, ensure_ascii=False, indent=2) + "\n"
    )

    def issue(request):
        current = json.loads(json.dumps(query))
        if args.trace_prefix:
            current["generate_config"]["trace_id"] = f"{args.trace_prefix}-{request}"
        return run_request(
            args.url,
            current,
            args.output,
            request=request,
            timeout=args.timeout,
            expected_input_tokens=args.expected_input_tokens,
            eos_token_ids=args.eos_token_id,
            cancel_after_tokens=args.cancel_after_tokens,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        observed = list(executor.map(issue, range(args.requests)))
    report = {
        "http_pd_generation_ok": not bool(args.cancel_after_tokens),
        "url": args.url,
        "requests": observed,
        "native_cache_store_transport_requires_server_evidence": True,
        "numerical_alignment": "deferred",
        "nv_alignment": "deferred",
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
