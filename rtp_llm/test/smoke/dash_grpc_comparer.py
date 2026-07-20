"""Smoke comparer for the DashSc gRPC ModelStreamInfer path.

The dash protocol differs from the HTTP frontend in two ways the comparer must
respect: prompts go on the wire as INT32 ``input_ids`` (tokenizer lives on the
client side), and generation arrives as a stream of INT32 ``generated_ids``
tensors that the client detokenizes back to text. Everything else (sampling
params, finish_reason semantics) mirrors the HTTP path.

JSON shape, identical to NormalComparer except endpoint and lack of HTTP-only
fields:

  {
    "endpoint": "dash://ModelStreamInfer",
    "query": {"prompt": "...", "generate_config": {"max_new_tokens": 32, ...}},
    "result": {"response": "...", "finish_reason": 1}
  }
"""
from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional

import grpc
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException

from rtp_llm.config.py_config_modules import DASH_SC_GRPC_SERVER_PORT_OFFSET
from rtp_llm.dash_sc.client import (
    build_model_infer_request,
    dash_sc_grpc_client_channel_options,
    decode_finish_reason,
)
from rtp_llm.dash_sc.codec import SamplingParams
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory

DASH_ENDPOINT = "dash://ModelStreamInfer"


class DashQueryInfo(BaseModel):
    prompt: str
    generate_config: Dict[str, Any] = {}
    request_id: Optional[str] = None
    model_name: str = "default"
    return_input_ids: bool = False
    enable_thinking: Optional[bool] = None


class DashSmokeResponse(BaseModel):
    response: str
    finish_reason: Optional[int] = None
    generated_ids: List[int] = []
    prompt_token_ids: List[int] = []
    prompt_token_num: Optional[int] = None
    prompt_cached_token_num: Optional[int] = None


def _int32_values(out: Any, raw: bytes) -> List[int]:
    if out.datatype != "INT32":
        return []
    numel = 1
    shape = list(out.shape)
    for dim in shape:
        numel *= max(0, int(dim))
    if shape:
        count = min(numel, len(raw) // 4)
    else:
        count = len(raw) // 4
    if count <= 0:
        return []
    return list(struct.unpack("<%di" % count, raw[: count * 4]))


def _build_sampling_params(gc: Dict[str, Any]) -> SamplingParams:
    """Map smoke ``generate_config`` dict onto SamplingParams. Unspecified
    fields fall through to ``SamplingParams`` defaults so the comparer follows
    the dash codec's own defaulting rules.
    """
    unsupported = [
        name
        for name in (
            "response_format",
            "guided_json",
            "json_schema",
            "regex",
            "ebnf",
            "structural_tag",
            "tool_call_structural_tag",
        )
        if gc.get(name) is not None
    ]
    if gc.get("json_format"):
        unsupported.append("json_format")
    if unsupported:
        raise SmokeException(
            QueryStatus.VALID_FAILED,
            "structured output is not supported yet by DashSc smoke: "
            + ", ".join(unsupported),
        )

    kwargs: Dict[str, Any] = {}
    for k in (
        "max_new_tokens",
        "min_new_tokens",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "num_return_sequences",
        "random_seed",
        "max_new_think_tokens",
        "response_format",
        "structural_tag",
    ):
        if k in gc and gc[k] is not None:
            kwargs[k] = gc[k]
    return SamplingParams(**kwargs)


class DashGrpcComparer(BaseComparer):
    """Send one prompt over DashSc gRPC ModelStreamInfer, detokenize the
    accumulated ``generated_ids``, and compare against ``result.response``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tokenizer: Any = None

    def _tokenizer_for(self, model_path: str, model_type: str) -> Any:
        if self._tokenizer is None:
            # TokenizerFactory.create(ckpt_path, tokenizer_path, model_type).
            # Smoke fixtures keep the two paths in lockstep (tokenizer_path may
            # be None, in which case caller already substituted model_path).
            self._tokenizer = TokenizerFactory.create(model_path, model_path, model_type)
        return self._tokenizer

    def _dash_port(self) -> int:
        return int(self.server_manager.port) + int(DASH_SC_GRPC_SERVER_PORT_OFFSET)

    # override
    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        return DashQueryInfo(**query_json)

    # override
    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        return DashSmokeResponse(**result_json)

    # override
    def compare_result(self, expect: DashSmokeResponse, actual: DashSmokeResponse) -> None:
        if expect.response != actual.response:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"dash response mismatch:\n  expect: {expect.response!r}\n  actual: {actual.response!r}",
            )
        if expect.finish_reason is not None and expect.finish_reason != actual.finish_reason:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"dash finish_reason mismatch: expect={expect.finish_reason} actual={actual.finish_reason}",
            )
        if expect.generated_ids and expect.generated_ids != actual.generated_ids:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"dash generated_ids mismatch: expect={expect.generated_ids} actual={actual.generated_ids}",
            )
        if expect.prompt_token_ids and expect.prompt_token_ids != actual.prompt_token_ids:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"dash prompt_token_ids mismatch: expect={expect.prompt_token_ids} actual={actual.prompt_token_ids}",
            )
        if expect.prompt_token_num is not None and expect.prompt_token_num != actual.prompt_token_num:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"dash prompt_token_num mismatch: expect={expect.prompt_token_num} actual={actual.prompt_token_num}",
            )
        if (
            expect.prompt_cached_token_num is not None
            and expect.prompt_cached_token_num != actual.prompt_cached_token_num
        ):
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                "dash prompt_cached_token_num mismatch: "
                f"expect={expect.prompt_cached_token_num} actual={actual.prompt_cached_token_num}",
            )

    # override — DashSc bypasses HTTP entirely, so BaseComparer.run's HTTP
    # ``server_manager.visit`` path doesn't apply. We override ``run`` rather
    # than slotting into ``curl_response_to_json`` to keep the gRPC roundtrip
    # in one place.
    def run(self) -> None:
        query_info: DashQueryInfo = self.format_query(self.qr_info["query"])
        self.tracer.query = query_info

        model_path = self.qr_info.get("_model_path") or self.qr_info.get("model_path")
        model_type = self.qr_info.get("_model_type") or self.qr_info.get("model_type")
        if not model_path or not model_type:
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                "dash comparer requires model_path/model_type to build tokenizer "
                "(passed via task_info, propagated by CaseRunner)",
            )

        tokenizer = self._tokenizer_for(model_path, model_type)
        input_ids: List[int] = tokenizer.encode(query_info.prompt)
        if not input_ids:
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                f"empty input_ids after tokenization: prompt={query_info.prompt!r}",
            )

        sampling = _build_sampling_params(query_info.generate_config)
        request = build_model_infer_request(
            request_id=query_info.request_id or f"smoke_dash_{id(self)}",
            model_name=query_info.model_name,
            input_ids=input_ids,
            sampling=sampling,
            return_input_ids=query_info.return_input_ids,
            enable_thinking=query_info.enable_thinking,
        )

        port = self._dash_port()
        target = f"127.0.0.1:{port}"
        channel = grpc.insecure_channel(target, options=dash_sc_grpc_client_channel_options())
        try:
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
            generated_ids: List[int] = []
            last_finish: Optional[int] = None
            prompt_token_num: Optional[int] = None
            prompt_cached_token_num: Optional[int] = None
            prompt_token_ids: Optional[List[int]] = None
            for resp in stub.ModelStreamInfer(iter([request])):
                if resp.error_message:
                    raise SmokeException(
                        QueryStatus.VISIT_FAILED,
                        f"dash server error: {resp.error_message}",
                    )
                if not resp.HasField("infer_response"):
                    continue
                infer = resp.infer_response
                for i, out in enumerate(infer.outputs):
                    if i >= len(infer.raw_output_contents):
                        continue
                    raw = infer.raw_output_contents[i]
                    if out.name == "generated_ids" and out.datatype == "INT32":
                        generated_ids.extend(_int32_values(out, raw))
                    elif out.name == "prompt_token_ids" and out.datatype == "INT32":
                        ids = _int32_values(out, raw)
                        if prompt_token_ids is None:
                            prompt_token_ids = ids
                        elif ids and ids != prompt_token_ids:
                            raise SmokeException(
                                QueryStatus.COMPARE_FAILED,
                                "dash prompt_token_ids changed across stream chunks: "
                                f"first={prompt_token_ids} current={ids}",
                            )
                    elif out.name == "finish_reason":
                        last_finish = decode_finish_reason(out, raw)
                    elif out.name == "prompt_token_num" and out.datatype == "INT32" and len(raw) >= 4:
                        values = _int32_values(out, raw)
                        prompt_token_num = values[0] if values else None
                    elif out.name == "prompt_cached_token_num" and out.datatype == "INT32" and len(raw) >= 4:
                        values = _int32_values(out, raw)
                        prompt_cached_token_num = values[0] if values else None
        finally:
            channel.close()

        response_text = tokenizer.decode(generated_ids) if generated_ids else ""
        actual = DashSmokeResponse(
            response=response_text,
            finish_reason=last_finish,
            generated_ids=generated_ids,
            prompt_token_ids=prompt_token_ids or [],
            prompt_token_num=prompt_token_num,
            prompt_cached_token_num=prompt_cached_token_num,
        )
        self.tracer.actual_result = actual

        if query_info.return_input_ids and actual.prompt_token_ids != input_ids:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                "dash return_input_ids mismatch: "
                f"expect request input_ids={input_ids} actual={actual.prompt_token_ids}",
            )
        if actual.prompt_token_num is not None and actual.prompt_token_num != len(input_ids):
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"dash prompt_token_num mismatch: expect={len(input_ids)} actual={actual.prompt_token_num}",
            )
        expect: DashSmokeResponse = self.format_result(self.qr_info["result"])
        self.tracer.expect_result = expect
        self._dump_actual_to_artifact(actual)

        from smoke.utils import no_compare, save_response
        if save_response():
            self.qr_info["result"] = actual.model_dump(exclude_defaults=True)
        if no_compare():
            return
        self.compare_result(expect, actual)
