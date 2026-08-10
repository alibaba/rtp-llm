"""Dash-SC gRPC grammar-admission smoke with a real xgrammar compiler.

Run with Bazelisk::

    bazelisk run //rtp_llm/dash_sc:grammar_validation_smoke -- \
        --ckpt_path=/path/to/model

The smoke starts an in-process ``grpc.aio`` server, sends requests through the
generated Dash-SC stub, and uses a fake inference backend so the measured time
does not include model loading or GPU execution.

It verifies three things:

* a valid JSON schema reaches ``GrammarValidator`` and then backend enqueue;
* the same schema exercises the validator cache on the second request;
* an invalid JSON schema reaches the validator and returns ``400`` before enqueue;
* malformed JSON returns ``400`` during request decoding without calling the validator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import statistics
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import grpc
import torch

from rtp_llm.config.grammar_constraint import GrammarConstraint
from rtp_llm.config.grammar_tokenizer_info import build_grammar_tokenizer_info_json
from rtp_llm.config.py_config_modules import GrammarAdmissionConfig
from rtp_llm.dash_sc.client import build_model_infer_request
from rtp_llm.dash_sc.codec import LLMFinishReason, SamplingParams
from rtp_llm.dash_sc.inference.grammar_validator import GrammarValidator
from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.ops import GrammarConfig
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs

_VALID_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "grammar_smoke_person",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        },
    },
}

# Valid JSON on the wire, but invalid Draft-7 JSON Schema. This makes the
# request reach GrammarValidator (rather than being rejected earlier by the
# Dash-SC JSON decoder) and must produce a business 4xx response.
_INVALID_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "grammar_smoke_invalid",
        "strict": True,
        "schema": {"type": "not-a-json-schema-type"},
    },
}

# Reduced from a production hcm_aggregate_records tool schema.  The local
# ``$defs`` lives below ``properties.filter``, but its refs use the document-root
# pointer ``#/$defs/...``.  Draft7Validator.check_schema accepts that shape while
# xgrammar must reject it during the real sandbox compile because the document
# root has no ``$defs`` field.
_INVALID_STRUCTURAL_TAG_DANGLING_REF = {
    "type": "structural_tag",
    "format": {
        "type": "tag",
        "begin": "<｜DSML｜tool_calls>\n",
        "content": {
            "type": "tags_with_separator",
            "tags": [
                {
                    "type": "tag",
                    "begin": '<｜DSML｜invoke name="hcm_aggregate_records">\n',
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "filter": {
                                    "$defs": {
                                        "filter": {
                                            "oneOf": [
                                                {
                                                    "type": "object",
                                                    "required": ["not"],
                                                    "properties": {
                                                        "not": {
                                                            "$ref": "#/$defs/filter"
                                                        }
                                                    },
                                                    "additionalProperties": False,
                                                },
                                                {"$ref": "#/$defs/filterItem"},
                                            ]
                                        },
                                        "filterItem": {
                                            "type": "object",
                                            "required": ["field", "op"],
                                            "properties": {
                                                "field": {
                                                    "type": "string",
                                                    "minLength": 1,
                                                },
                                                "op": {
                                                    "type": "string",
                                                    "enum": ["eq", "ne"],
                                                },
                                                "value": {},
                                            },
                                            "additionalProperties": False,
                                        },
                                    },
                                    "$ref": "#/$defs/filter",
                                },
                                "modelName": {"type": "string"},
                                "measures": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "object"},
                                },
                            },
                            "required": ["modelName", "measures"],
                        },
                        "style": "deepseek_xml",
                        "any_order": False,
                    },
                    "end": "</｜DSML｜invoke>\n",
                }
            ],
            "separator": "",
            "at_least_one": True,
            "stop_after_first": False,
        },
        "end": "</｜DSML｜tool_calls>",
    },
}

_EXPECTED_XGRAMMAR_ERROR = "Cannot find field $defs"


@dataclass(frozen=True)
class _ValidationTiming:
    request_id: str
    ok: bool
    elapsed_ms: float


@dataclass(frozen=True)
class _CompileTiming:
    kind: str
    ok: bool
    elapsed_ms: float


@dataclass(frozen=True)
class _LoadTiming:
    request_id: str
    grpc_roundtrip_ms: float
    ok: bool
    status_message: str = ""


@dataclass(frozen=True)
class _LoadRequest:
    request_id: str
    source_request_id: str
    response_format: dict[str, Any] | None = None
    structural_tag: dict[str, Any] | None = None


class _TimedGrammarValidator(GrammarValidator):
    """Production validator with per-request timing exposed to the smoke."""

    def __init__(
        self,
        grammar_config: GrammarConfig,
        admission_config: GrammarAdmissionConfig,
    ) -> None:
        self.timings: list[_ValidationTiming] = []
        self.compile_timings: list[_CompileTiming] = []
        super().__init__(grammar_config, admission_config)

    def _compile_in_worker(self, kind: str, spec_str: str) -> bool:
        started = time.perf_counter()
        ok = False
        try:
            ok = super()._compile_in_worker(kind, spec_str)
            return ok
        finally:
            self.compile_timings.append(
                _CompileTiming(
                    kind=kind,
                    ok=ok,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            )

    def validate_constraint(
        self, constraint: GrammarConstraint, request_id: str = ""
    ) -> bool:
        started = time.perf_counter()
        ok = False
        try:
            ok = super().validate_constraint(constraint, request_id)
            return ok
        finally:
            self.timings.append(
                _ValidationTiming(
                    request_id=request_id,
                    ok=ok,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            )


class _SingleChunkStream:
    def __init__(self, chunk: GenerateOutputs) -> None:
        self._chunk = chunk
        self._done = False

    def __aiter__(self) -> _SingleChunkStream:
        return self

    async def __anext__(self) -> GenerateOutputs:
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return self._chunk


class _CountingBackend:
    def __init__(self) -> None:
        self.enqueue_called = 0

    async def enqueue(self, generate_input: Any) -> _SingleChunkStream:
        self.enqueue_called += 1
        output = GenerateOutput(
            output_ids=torch.tensor([1], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(
                input_len=int(generate_input.token_ids.numel()),
                reuse_len=0,
            ),
        )
        return _SingleChunkStream(GenerateOutputs(generate_outputs=[output]))


async def _one_request(
    stub: Any,
    request: predict_v2_pb2.ModelInferRequest,
) -> tuple[list[predict_v2_pb2.ModelStreamInferResponse], float]:
    async def _request_iter() -> AsyncIterator[predict_v2_pb2.ModelInferRequest]:
        yield request

    started = time.perf_counter()
    responses = [response async for response in stub.ModelStreamInfer(_request_iter())]
    return responses, (time.perf_counter() - started) * 1000.0


def _load_response_format(index: int) -> dict[str, Any]:
    field_name = f"load_field_{index:03d}"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"grammar_load_{index:03d}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {field_name: {"type": "integer", "minimum": index}},
                "required": [field_name],
                "additionalProperties": False,
            },
        },
    }


def _decode_summary_payload(line: str, field_name: str) -> dict[str, Any] | None:
    encoded = json.loads(line.removeprefix(f"{field_name}: "))
    if encoded is None:
        return None
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must decode to an object")
    return payload


def _read_load_requests(path: str) -> list[_LoadRequest]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    with open(path, encoding="utf-8") as request_file:
        for raw_line in request_file:
            line = raw_line.rstrip("\n")
            if line.startswith("===== LOG "):
                if current is not None:
                    records.append(current)
                current = {}
            elif current is not None and line.startswith("request_id: "):
                current["source_request_id"] = line.removeprefix("request_id: ")
            elif current is not None and line.startswith("response_format: "):
                current["response_format"] = _decode_summary_payload(
                    line, "response_format"
                )
            elif current is not None and line.startswith("structural_tag: "):
                current["structural_tag"] = _decode_summary_payload(
                    line, "structural_tag"
                )
    if current is not None:
        records.append(current)

    requests: list[_LoadRequest] = []
    for index, record in enumerate(records):
        source_request_id = str(record.get("source_request_id", ""))
        response_format = record.get("response_format")
        structural_tag = record.get("structural_tag")
        if response_format is None and structural_tag is None:
            raise ValueError(f"record {index} has no structured-output constraint")
        requests.append(
            _LoadRequest(
                request_id=f"grammar-file-{index:03d}",
                source_request_id=source_request_id,
                response_format=response_format,
                structural_tag=structural_tag,
            )
        )
    return requests


async def _wait_for_sandbox_pool(validator: GrammarValidator) -> float:
    started = time.monotonic()
    deadline = started + validator._compile_timeout_s * validator._pool_target + 10.0
    while time.monotonic() < deadline:
        validator._ensure_pool()
        with validator._pool_lock:
            live = validator._live
            spawning = validator._spawning
            target = validator._pool_target
        if live >= target and spawning == 0:
            return (time.monotonic() - started) * 1000.0
        await asyncio.sleep(0.05)
    raise RuntimeError(
        "sandbox pool did not become ready: "
        f"live={live} spawning={spawning} target={target}"
    )


async def _run_compile_load(
    *,
    stub: Any,
    input_ids: list[int],
    validator: _TimedGrammarValidator,
    backend: _CountingBackend,
    requests: list[_LoadRequest],
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(request: _LoadRequest) -> _LoadTiming:
        async with semaphore:
            if request.response_format is not None:
                grpc_request = _build_request(
                    request.request_id,
                    input_ids,
                    request.response_format,
                )
            else:
                grpc_request = _build_structural_tag_request(
                    request.request_id,
                    input_ids,
                    request.structural_tag or {},
                )
            responses, roundtrip_ms = await _one_request(stub, grpc_request)
        if len(responses) != 1:
            raise AssertionError(
                f"load request returned {len(responses)} responses: "
                f"request_id={request.request_id}"
            )
        error = _dash_error(responses[0])
        return _LoadTiming(
            request_id=request.request_id,
            grpc_roundtrip_ms=roundtrip_ms,
            ok=error is None,
            status_message=""
            if error is None
            else str(error.get("status_message", "")),
        )

    request_count = len(requests)
    before_compile = len(validator.compile_timings)
    before_enqueue = backend.enqueue_called
    started = time.perf_counter()
    timings = await asyncio.gather(*(_run_one(request) for request in requests))
    wall_ms = (time.perf_counter() - started) * 1000.0
    compile_timings = validator.compile_timings[before_compile:]
    cache_hits = request_count - len(compile_timings)
    success_count = sum(timing.ok for timing in timings)
    rejected_count = request_count - success_count
    if backend.enqueue_called != before_enqueue + success_count:
        raise AssertionError(
            "load success/enqueue count mismatch: "
            f"before={before_enqueue} after={backend.enqueue_called}"
        )

    roundtrip_values = [timing.grpc_roundtrip_ms for timing in timings]
    print(
        "PASS compile_load: "
        f"requests={request_count} concurrency={concurrency} "
        f"compiled={len(compile_timings)} successes={success_count} "
        f"cache_hits={cache_hits} rejected={rejected_count} "
        f"sandbox_pool_size={validator._pool_target} wall_ms={wall_ms:.3f} "
        f"grpc_avg_ms={statistics.fmean(roundtrip_values):.3f} "
        f"grpc_max_ms={max(roundtrip_values):.3f}"
    )
    source_request_ids = {
        request.request_id: request.source_request_id for request in requests
    }
    for timing in timings:
        print(
            "LOAD request "
            f"request_id={timing.request_id} "
            f"source_request_id={source_request_ids[timing.request_id]} "
            f"status={'ok' if timing.ok else 'rejected'} "
            f"grpc_roundtrip_ms={timing.grpc_roundtrip_ms:.3f} "
            f"error={json.dumps(timing.status_message, ensure_ascii=False)}"
        )


def _build_request(
    request_id: str,
    input_ids: list[int],
    response_format: dict[str, Any],
) -> predict_v2_pb2.ModelInferRequest:
    return build_model_infer_request(
        request_id=request_id,
        model_name="grammar-validation-smoke",
        input_ids=input_ids,
        sampling=SamplingParams(
            max_new_tokens=1,
            response_format=json.dumps(
                response_format, ensure_ascii=False, separators=(",", ":")
            ),
        ),
    )


def _build_structural_tag_request(
    request_id: str,
    input_ids: list[int],
    structural_tag: dict[str, Any],
) -> predict_v2_pb2.ModelInferRequest:
    return build_model_infer_request(
        request_id=request_id,
        model_name="grammar-validation-smoke",
        input_ids=input_ids,
        sampling=SamplingParams(
            max_new_tokens=1,
            structural_tag=json.dumps(
                structural_tag, ensure_ascii=False, separators=(",", ":")
            ),
        ),
        enable_thinking=False,
    )


def _dash_error(
    response: predict_v2_pb2.ModelStreamInferResponse,
) -> dict[str, Any] | None:
    infer = response.infer_response
    raw = infer.parameters["error_msg"].string_param
    if not raw:
        return None
    error = json.loads(raw)
    finish_reason = next(
        (
            int.from_bytes(raw_value, byteorder="little", signed=True)
            for output, raw_value in zip(
                infer.outputs, infer.raw_output_contents, strict=True
            )
            if output.name == "finish_reason"
        ),
        None,
    )
    status_code = infer.parameters["status_code"]
    status_name = infer.parameters["status_name"]
    status_message = infer.parameters["status_message"]
    if (
        finish_reason != LLMFinishReason.USE_PARAMETER_STATUS
        or not status_code.HasField("int64_param")
        or status_code.int64_param != error.get("status_code")
        or status_name.string_param != error.get("status_name")
        or status_message.string_param != error.get("status_message")
    ):
        raise AssertionError(
            "Dash error standalone status parameters do not match error_msg: "
            f"finish_reason={finish_reason} status_code={status_code} "
            f"status_name={status_name} "
            f"status_message={status_message} error_msg={error}"
        )
    return error


def _stop_validator_workers(validator: GrammarValidator) -> None:
    validator._pool_target = 0
    deadline = time.monotonic() + validator._compile_timeout_s + 1.0
    while validator._coordinator_running or validator._spawning:
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    while True:
        try:
            proc, conn = validator._idle.get_nowait()
        except queue.Empty:
            break
        conn.close()
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=1.0)


async def _run(args: argparse.Namespace) -> int:
    tokenizer_path = args.tokenizer_path or args.ckpt_path
    tokenizer = TokenizerFactory.create(
        args.ckpt_path,
        tokenizer_path,
        args.model_type,
    )
    input_ids = tokenizer.encode(args.prompt)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    input_ids = [int(token_id) for token_id in input_ids]

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    grammar_config = GrammarConfig()
    grammar_config.num_workers = args.compiler_threads
    grammar_config.tokenizer_info_json = build_grammar_tokenizer_info_json(
        tokenizer.get_real_tokenizer(),
        model_vocab_size=int(tokenizer.config_json.get("vocab_size") or 0),
        stop_token_ids=[] if eos_token_id is None else [int(eos_token_id)],
    )
    init_started = time.perf_counter()
    validator = _TimedGrammarValidator(
        grammar_config,
        GrammarAdmissionConfig(sandbox_pool_size=args.sandbox_pool_size),
    )
    validator_init_ms = (time.perf_counter() - init_started) * 1000.0
    if validator._backend is None:
        raise RuntimeError(
            "xgrammar compiler backend was not created; refusing a shape-only smoke"
        )

    backend = _CountingBackend()
    servicer = DashScInferenceServicer(
        backend_visitor=backend,
        tokenizer=tokenizer,
        grammar_validator=validator,
    )
    server = grpc.aio.server()
    predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if port <= 0:
        raise RuntimeError("failed to bind an ephemeral Dash-SC gRPC port")
    await server.start()

    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
    try:
        for suffix in ("cold", "cached"):
            request_id = f"grammar-valid-{suffix}"
            before_enqueue = backend.enqueue_called
            responses, roundtrip_ms = await _one_request(
                stub,
                _build_request(request_id, input_ids, _VALID_RESPONSE_FORMAT),
            )
            timing = validator.timings[-1]
            if timing.request_id != request_id or not timing.ok:
                raise AssertionError(f"valid schema did not pass validation: {timing}")
            if len(responses) != 1 or _dash_error(responses[0]) is not None:
                raise AssertionError("valid schema returned an error response")
            if backend.enqueue_called != before_enqueue + 1:
                raise AssertionError("valid schema did not reach backend enqueue")
            print(
                f"PASS valid_{suffix}: validation_ms={timing.elapsed_ms:.3f} "
                f"grpc_roundtrip_ms={roundtrip_ms:.3f} "
                f"enqueue_called={backend.enqueue_called}"
            )

        request_id = "grammar-invalid-schema"
        before_enqueue = backend.enqueue_called
        responses, roundtrip_ms = await _one_request(
            stub,
            _build_request(request_id, input_ids, _INVALID_RESPONSE_FORMAT),
        )
        timing = validator.timings[-1]
        if timing.request_id != request_id or timing.ok:
            raise AssertionError(f"invalid schema unexpectedly passed: {timing}")
        if len(responses) != 1:
            raise AssertionError(f"invalid schema returned {len(responses)} responses")
        error = _dash_error(responses[0])
        if error is None:
            raise AssertionError("invalid schema did not return a Dash error payload")
        if (
            error.get("status_code") != 400
            or error.get("status_name") != "InvalidParameter"
        ):
            raise AssertionError(
                f"invalid schema did not return 400/InvalidParameter: {error}"
            )
        if backend.enqueue_called != before_enqueue:
            raise AssertionError("invalid schema reached backend enqueue")
        print(
            f"PASS invalid_schema: validation_ms={timing.elapsed_ms:.3f} "
            f"grpc_roundtrip_ms={roundtrip_ms:.3f} status_code={error['status_code']} "
            f"status_name={error['status_name']} enqueue_called={backend.enqueue_called}"
        )

        request_id = "grammar-invalid-structural-tag-dangling-ref"
        before_enqueue = backend.enqueue_called
        before_validation = len(validator.timings)
        before_compile = len(validator.compile_timings)
        static_shape_ok = validator.check_structural_tag(
            _INVALID_STRUCTURAL_TAG_DANGLING_REF
        )
        if not static_shape_ok:
            raise AssertionError(
                "dangling-ref structural tag failed before sandbox compilation"
            )
        responses, roundtrip_ms = await _one_request(
            stub,
            _build_structural_tag_request(
                request_id,
                input_ids,
                _INVALID_STRUCTURAL_TAG_DANGLING_REF,
            ),
        )
        if len(validator.timings) != before_validation + 1:
            raise AssertionError(
                "invalid structural tag did not reach GrammarValidator exactly once"
            )
        if len(validator.compile_timings) != before_compile + 1:
            raise AssertionError(
                "invalid structural tag did not reach sandbox compilation exactly once"
            )
        compile_timing = validator.compile_timings[-1]
        if compile_timing.kind != "structural_tag" or compile_timing.ok:
            raise AssertionError(
                "invalid structural tag sandbox compile unexpectedly succeeded: "
                f"{compile_timing}"
            )
        timing = validator.timings[-1]
        if timing.request_id != request_id or timing.ok:
            raise AssertionError(
                f"invalid structural tag unexpectedly passed: {timing}"
            )
        if len(responses) != 1:
            raise AssertionError(
                f"invalid structural tag returned {len(responses)} responses"
            )
        error = _dash_error(responses[0])
        if error is None:
            raise AssertionError(
                "invalid structural tag did not return a Dash error payload"
            )
        if (
            error.get("status_code") != 400
            or error.get("status_name") != "InvalidParameter"
        ):
            raise AssertionError(
                f"invalid structural tag did not return 400/InvalidParameter: {error}"
            )
        if _EXPECTED_XGRAMMAR_ERROR not in error.get("status_message", ""):
            raise AssertionError(
                "invalid structural tag response did not preserve xgrammar error: "
                f"{error}"
            )
        if backend.enqueue_called != before_enqueue:
            raise AssertionError("invalid structural tag reached backend enqueue")
        print(
            "PASS invalid_structural_tag_dangling_ref: "
            f"validation_ms={timing.elapsed_ms:.3f} "
            f"grpc_roundtrip_ms={roundtrip_ms:.3f} "
            f"status_code={error['status_code']} "
            f"status_name={error['status_name']} "
            f"static_shape_ok={str(static_shape_ok).lower()} "
            f"sandbox_compile_called=true "
            f"sandbox_compile_ok={str(compile_timing.ok).lower()} "
            f"sandbox_compile_ms={compile_timing.elapsed_ms:.3f} "
            f"xgrammar_error_returned=true "
            f"enqueue_called={backend.enqueue_called}"
        )

        request_id = "grammar-malformed-json"
        malformed_request = _build_request(
            request_id, input_ids, _VALID_RESPONSE_FORMAT
        )
        malformed_request.parameters[
            "response_format"
        ].string_param = '{"type":"json_schema"'
        before_enqueue = backend.enqueue_called
        before_validation = len(validator.timings)
        responses, roundtrip_ms = await _one_request(stub, malformed_request)
        if len(responses) != 1:
            raise AssertionError(f"malformed JSON returned {len(responses)} responses")
        error = _dash_error(responses[0])
        if error is None:
            raise AssertionError("malformed JSON did not return a Dash error payload")
        if (
            error.get("status_code") != 400
            or error.get("status_name") != "InvalidParameter"
        ):
            raise AssertionError(
                f"malformed JSON did not return 400/InvalidParameter: {error}"
            )
        if backend.enqueue_called != before_enqueue:
            raise AssertionError("malformed JSON reached backend enqueue")
        if len(validator.timings) != before_validation:
            raise AssertionError("malformed JSON unexpectedly reached GrammarValidator")
        print(
            f"PASS malformed_json: grpc_roundtrip_ms={roundtrip_ms:.3f} "
            f"status_code={error['status_code']} status_name={error['status_name']} "
            f"validator_called=false enqueue_called={backend.enqueue_called}"
        )
        if args.load_request_file:
            load_requests = _read_load_requests(args.load_request_file)
        else:
            load_requests = [
                _LoadRequest(
                    request_id=f"grammar-load-{index:03d}",
                    source_request_id=f"generated-{index:03d}",
                    response_format=_load_response_format(index),
                )
                for index in range(args.load_requests)
            ]
        if load_requests:
            pool_ready_ms = await _wait_for_sandbox_pool(validator)
            print(
                "sandbox_pool_ready: "
                f"pool_size={validator._pool_target} "
                f"compiler_threads={args.compiler_threads} "
                f"wait_ms={pool_ready_ms:.3f}"
            )
            await _run_compile_load(
                stub=stub,
                input_ids=input_ids,
                validator=validator,
                backend=backend,
                requests=load_requests,
                concurrency=args.load_concurrency,
            )
        print(f"validator_init_ms={validator_init_ms:.3f} xgrammar_mode=sandbox")
        return 0
    finally:
        await channel.close()
        await server.stop(grace=0)
        _stop_validator_workers(validator)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--model_type", default="glm_5")
    parser.add_argument("--prompt", default="Return a JSON object.")
    parser.add_argument("--compiler_threads", type=int, default=1)
    parser.add_argument("--sandbox_pool_size", type=int, default=1)
    parser.add_argument("--load_requests", type=int, default=0)
    parser.add_argument("--load_request_file", default="")
    parser.add_argument("--load_concurrency", type=int, default=32)
    args = parser.parse_args()
    if args.compiler_threads <= 0:
        parser.error("--compiler_threads must be greater than 0")
    if args.sandbox_pool_size <= 0:
        parser.error("--sandbox_pool_size must be greater than 0")
    if args.load_requests < 0:
        parser.error("--load_requests must be non-negative")
    if args.load_concurrency <= 0:
        parser.error("--load_concurrency must be greater than 0")
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())
