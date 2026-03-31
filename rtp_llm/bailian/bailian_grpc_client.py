#!/usr/bin/env python3
"""
Bailian gRPC client (ModelStreamInfer, wire: predict_v2.proto). Same tokenizer as FrontendApp
(FrontendWorker) to encode a prompt to input_ids, then sends ModelInferRequest and
prints the response (generated_ids, finish_reason).

Usage (tokenizer same as frontend: ckpt_path, tokenizer_path, model_type):
  python -m rtp_llm.bailian.bailian_grpc_client \\
    --grpc_addr 127.0.0.1:8096 \\
    --ckpt_path /path/to/checkpoint \\
    --tokenizer_path /path/to/tokenizer \\
    --model_type qwen2 \\
    --prompt "Hello, world!"
"""
from __future__ import annotations

import argparse
import struct
import sys
from typing import Any

import grpc

from rtp_llm.bailian.bailian_grpc_request import SamplingParams
from rtp_llm.bailian.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.server.server_args.grpc_group_args import default_bailian_grpc_config_json


def bailian_grpc_client_channel_options(
    bailian_grpc_config=None,
) -> list[tuple[str, int]]:
    """``grpc.insecure_channel(..., options=...)`` from ``BailianGrpcConfig.get_client_config()``."""
    if bailian_grpc_config is None:
        from rtp_llm.ops import BailianGrpcConfig

        bailian_grpc_config = BailianGrpcConfig()
        bailian_grpc_config.from_json(default_bailian_grpc_config_json())
    return sorted(
        (str(k), int(v)) for k, v in bailian_grpc_config.get_client_config().items()
    )


def _append_int32_scalar(
    request: predict_v2_pb2.ModelInferRequest, name: str, value: int
) -> None:
    inp = request.inputs.add()
    inp.name = name
    inp.datatype = "INT32"
    inp.shape.append(1)
    request.raw_input_contents.append(struct.pack("<i", int(value)))


def _append_fp32_scalar(
    request: predict_v2_pb2.ModelInferRequest, name: str, value: float
) -> None:
    inp = request.inputs.add()
    inp.name = name
    inp.datatype = "FP32"
    inp.shape.append(1)
    request.raw_input_contents.append(struct.pack("<f", float(value)))


def append_input_ids_to_model_infer_request(
    request: predict_v2_pb2.ModelInferRequest,
    input_ids: list[int],
) -> None:
    """Append ``input_ids`` tensor (INT32, shape ``[len]``) and matching ``raw_input_contents``."""
    inp = request.inputs.add()
    inp.name = "input_ids"
    inp.datatype = "INT32"
    inp.shape.append(len(input_ids))
    request.raw_input_contents.append(struct.pack("<%di" % len(input_ids), *input_ids))


def append_sampling_params_to_model_infer_request(
    request: predict_v2_pb2.ModelInferRequest,
    sampling: SamplingParams,
) -> None:
    """Append sampling tensors (names / dtypes aligned with ``parse_sampling_params``)."""
    _append_int32_scalar(request, "max_new_tokens", sampling.max_new_tokens)
    _append_int32_scalar(request, "num_return_sequences", sampling.num_return_sequences)
    _append_fp32_scalar(request, "top_p", sampling.top_p)
    _append_int32_scalar(request, "top_k", sampling.top_k)
    _append_fp32_scalar(request, "temperature", sampling.temperature)
    _append_int32_scalar(request, "min_new_tokens", sampling.min_new_tokens)
    if sampling.random_seed is not None:
        _append_int32_scalar(request, "seed", int(sampling.random_seed))
    _append_fp32_scalar(request, "repetition_penalty", sampling.repetition_penalty)
    _append_fp32_scalar(request, "frequency_penalty", sampling.frequency_penalty)
    _append_fp32_scalar(request, "presence_penalty", sampling.presence_penalty)

    groups = sampling.stop_words_list
    if not groups:
        return
    cols = len(groups[0])
    if not all(len(g) == cols for g in groups):
        flat = [t for g in groups for t in g]
        inp = request.inputs.add()
        inp.name = "stop_words_list"
        inp.datatype = "INT32"
        inp.shape.append(len(flat))
        request.raw_input_contents.append(struct.pack("<%di" % len(flat), *flat))
        return
    rows = len(groups)
    flat = [t for g in groups for t in g]
    inp = request.inputs.add()
    inp.name = "stop_words_list"
    inp.datatype = "INT32"
    inp.shape.extend([rows, cols])
    request.raw_input_contents.append(struct.pack("<%di" % len(flat), *flat))


def append_return_input_ids_to_model_infer_request(
    request: predict_v2_pb2.ModelInferRequest,
    return_input_ids: bool,
) -> None:
    """If True, append ``return_input_ids`` INT32 tensor shape ``[1]`` (=1), for ``parse_other_params``."""
    if not return_input_ids:
        return
    _append_int32_scalar(request, "return_input_ids", 1)


def build_model_infer_request(
    *,
    request_id: str,
    model_name: str,
    input_ids: list[int],
    sampling: SamplingParams,
    return_input_ids: bool = False,
) -> predict_v2_pb2.ModelInferRequest:
    """Build ``ModelInferRequest`` for ``ModelStreamInfer`` (sampling tensors + ``input_ids``)."""
    request = predict_v2_pb2.ModelInferRequest()
    request.id = request_id
    request.model_name = model_name
    append_sampling_params_to_model_infer_request(request, sampling)
    append_return_input_ids_to_model_infer_request(request, return_input_ids)
    append_input_ids_to_model_infer_request(request, input_ids)
    return request


def _decode_finish_reason(out: Any, raw: bytes) -> int | None:
    if out.datatype == "INT64" and len(raw) >= 8:
        return struct.unpack("<q", raw[:8])[0]
    if out.datatype == "INT32" and len(raw) >= 4:
        return struct.unpack("<i", raw[:4])[0]
    if out.datatype == "INT8" and len(raw) >= 1:
        return struct.unpack("b", raw[:1])[0]
    return None


# Triton-style datatype -> element size in raw bytes (little-endian tensor)
_DTYPE_ELEMENT_BYTES: dict[str, int] = {
    "INT8": 1,
    "UINT8": 1,
    "INT16": 2,
    "UINT16": 2,
    "INT32": 4,
    "UINT32": 4,
    "INT64": 8,
    "UINT64": 8,
    "FP32": 4,
    "FP64": 8,
    "BOOL": 1,
}


def _shape_numel(shape) -> int:
    p = 1
    for d in shape:
        p *= int(d)
    return p


def _expected_raw_len_for_output(out: Any) -> tuple[int | None, int | None]:
    """Return ``(element_bytes, expected_total_bytes)`` or ``(elem, None)`` if shape unusable."""
    elem = _DTYPE_ELEMENT_BYTES.get(out.datatype)
    if elem is None:
        return None, None
    shape = list(out.shape)
    if not shape:
        return elem, None
    if any(int(d) < 0 for d in shape):
        return elem, None
    return elem, _shape_numel(shape) * elem


def _raw_matches_output_metadata(
    out: Any,
    raw: bytes,
    output_index: int,
) -> bool:
    """True if ``len(raw)`` matches ``out.datatype`` + ``out.shape``."""
    elem, exp_total = _expected_raw_len_for_output(out)
    if elem is None:
        print(
            f"[client]   error: output[{output_index}] {out.name!r} "
            f"unknown datatype {out.datatype!r} for len check"
        )
        return False
    if exp_total is not None:
        if len(raw) != exp_total:
            print(
                f"[client]   error: output[{output_index}] {out.name!r} "
                f"len(raw)={len(raw)} != expected {exp_total} "
                f"(dtype={out.datatype!r} shape={list(out.shape)} elem_bytes={elem})"
            )
            return False
        return True
    # shape 为空或无法推算元素个数：仅检查是否为 element 对齐
    if len(raw) % elem != 0:
        print(
            f"[client]   error: output[{output_index}] {out.name!r} "
            f"len(raw)={len(raw)} not divisible by elem_bytes={elem} "
            f"(dtype={out.datatype!r} shape={list(out.shape)})"
        )
        return False
    print(
        f"[client]   warn: output[{output_index}] {out.name!r} "
        f"shape empty or unknown numel; len(raw)={len(raw)} elem_bytes={elem} (weak check only)"
    )
    return True


def print_model_stream_infer_response(
    resp: predict_v2_pb2.ModelStreamInferResponse,
    tokenizer: Any,
) -> None:
    """Print one streaming response (errors, or ``infer_response`` outputs)."""
    if resp.error_message:
        print(f"[client] error: {resp.error_message}")
        return
    if not resp.HasField("infer_response"):
        return
    infer = resp.infer_response
    print(f"[client] response id={infer.id!r} model_name={infer.model_name!r}")
    prompt_token_num: int | None = None
    prompt_cached_token_num: int | None = None
    for i, out in enumerate(infer.outputs):
        print(
            f"[client]   output[{i}] name={out.name!r} datatype={out.datatype!r} shape={list(out.shape)}"
        )
        if i >= len(infer.raw_output_contents):
            print(f"[client]   error: output[{i}] missing raw_output_contents")
            continue
        raw = infer.raw_output_contents[i]
        if not _raw_matches_output_metadata(out, raw, i):
            print(
                f"[client]   error: output[{i}] {out.name!r} raw_output_contents mismatch"
            )
            continue
        if out.name == "prompt_token_ids" and out.datatype == "INT32":
            n = len(raw) // 4
            vals = struct.unpack("<%di" % n, raw)
            print(f"[client]   prompt_token_ids: {list(vals)}")
        elif out.name == "generated_ids" and out.datatype == "INT32":
            n = len(raw) // 4
            vals = struct.unpack("<%di" % n, raw)
            ids_list = list(vals)
            print(f"[client]   generated_ids: {ids_list}")
            decoded = tokenizer.decode(ids_list)
            print(f"[client]   decoded: {decoded!r}")
        elif out.name == "finish_reason":
            finish = _decode_finish_reason(out, raw)
            print(f"[client]   finish_reason: {finish}")
        elif out.name == "prompt_token_num" and out.datatype == "INT32":
            prompt_token_num = struct.unpack("<i", raw[:4])[0]
            print(f"[client]   prompt_token_num: {prompt_token_num}")
        elif out.name == "prompt_cached_token_num" and out.datatype == "INT32":
            prompt_cached_token_num = struct.unpack("<i", raw[:4])[0]
            print(f"[client]   prompt_cached_token_num: {prompt_cached_token_num}")


def _parse_stop_token_ids_csv(s: str | None) -> tuple[tuple[int, ...], ...]:
    """Comma-separated token ids -> one stop group (``stop_words_list`` input)."""
    if not s or not s.strip():
        return tuple()
    ids = [int(x.strip()) for x in s.split(",") if x.strip()]
    return (tuple(ids),) if ids else tuple()


def build_bailian_grpc_client_argparser() -> argparse.ArgumentParser:
    """CLI for ``main()``; callers may ``parse_args(argv)`` for tests or subprocesses."""
    parser = argparse.ArgumentParser(
        description="Bailian gRPC client: encode prompt with same tokenizer as frontend, call ModelStreamInfer."
    )
    parser.add_argument(
        "--grpc_addr",
        type=str,
        default="127.0.0.1:8096",
        help=(
            "gRPC server address (host:port); default matches rank0 "
            "ServerConfig.bailian_grpc_server_port (8088 + 8)"
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Checkpoint path (same as frontend model_config.ckpt_path)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Tokenizer path (default: same as ckpt_path)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type (same as frontend model_config.model_type, e.g. qwen2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world!",
        help="Prompt to encode and send",
    )
    parser.add_argument(
        "--request_id",
        type=str,
        default="bailian_grpc_client_1",
        help="Request id for the inference request",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="default",
        help="Model name in the request",
    )
    # Sampling / generation (tensor names match ``bailian_grpc_request.parse_sampling_params``)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="max_new_tokens (INT32 tensor, default 512 for quick runs)",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=0,
        help="num_return_sequences / n (INT32)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p (FP32)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="top_k (INT32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature (FP32)",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=0,
        help="min_new_tokens (INT32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed (INT32 tensor 'seed'); omit to not send",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="repetition_penalty (FP32)",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="frequency_penalty (FP32)",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="presence_penalty (FP32)",
    )
    parser.add_argument(
        "--stop_token_ids",
        type=str,
        default="",
        help="Comma-separated stop token ids -> stop_words_list INT32 tensor (single group)",
    )
    parser.add_argument(
        "--return_input_ids",
        action="store_true",
        help="Send return_input_ids INT32 tensor (1) so server echoes prompt in prompt_token_ids",
    )
    parser.add_argument(
        "--bailian_grpc_config_json",
        type=str,
        default="",
        help="Optional BailianGrpcConfig JSON (client_config / server_config / max_server_workers).",
    )
    return parser


def main():
    args = build_bailian_grpc_client_argparser().parse_args()

    tokenizer_path = args.tokenizer_path or args.ckpt_path

    tokenizer = TokenizerFactory.create(
        args.ckpt_path,
        tokenizer_path,
        args.model_type,
    )
    input_ids = tokenizer.encode(args.prompt)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    input_ids = [int(x) for x in input_ids]
    print(f"[client] prompt: {args.prompt!r}")
    print(f"[client] input_ids ({len(input_ids)}): {input_ids}")

    sampling = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        min_new_tokens=args.min_new_tokens,
        random_seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        stop_words_list=_parse_stop_token_ids_csv(args.stop_token_ids or None),
    )

    request = build_model_infer_request(
        request_id=args.request_id,
        model_name=args.model_name,
        input_ids=input_ids,
        sampling=sampling,
        return_input_ids=args.return_input_ids,
    )

    bailian_cfg = None
    if args.bailian_grpc_config_json.strip():
        from rtp_llm.ops import BailianGrpcConfig

        bailian_cfg = BailianGrpcConfig()
        bailian_cfg.from_json(args.bailian_grpc_config_json.strip())
    channel = grpc.insecure_channel(
        args.grpc_addr, options=bailian_grpc_client_channel_options(bailian_cfg)
    )
    stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)

    def request_iter():
        yield request

    response_count = 0
    for resp in stub.ModelStreamInfer(request_iter()):
        response_count += 1
        print_model_stream_infer_response(resp, tokenizer)
        print("-----------------------------------------------")

    print(f"[client] done, response_count={response_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
