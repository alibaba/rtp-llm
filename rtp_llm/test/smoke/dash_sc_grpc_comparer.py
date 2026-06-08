"""Smoke comparer: DashSc gRPC ModelStreamInfer (wire: predict_v2.proto; frontend starts server by default)."""

from __future__ import annotations

import logging
import os
import struct
import time
from typing import Any, List, Optional, Tuple

import grpc
from pydantic import BaseModel
from smoke.common_def import QueryStatus, SmokeException
from smoke.normal_comparer import NormalComparer, QueryInfo
from smoke.utils import no_compare, save_response

from rtp_llm.dash_sc import (
    OtherParams,
    SamplingParams,
    build_model_infer_request,
    dash_sc_grpc_client_channel_options,
    decode_finish_reason,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import ServerConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory

# Per-query endpoint in task JSON; must match case_runner._get_comparer_cls
DASH_SC_GRPC_ENDPOINT = "/__dash_sc_grpc__"


def _scalar_int(v: Any) -> int:
    if isinstance(v, list):
        return int(v[0]) if v else 0
    return int(v)


def _scalar_float(v: Any) -> float:
    if isinstance(v, list):
        return float(v[0]) if v else 1.0
    return float(v)


def _generate_config_to_sampling(gc: GenerateConfig) -> SamplingParams:
    sw = gc.stop_words_list or []
    stop_tuples = tuple(tuple(g) for g in sw) if sw else tuple()
    seed = gc.random_seed
    if isinstance(seed, list):
        seed = int(seed[0]) if seed else None
    elif seed is not None:
        seed = int(seed)
    return SamplingParams(
        max_new_tokens=_scalar_int(gc.max_new_tokens),
        num_return_sequences=_scalar_int(gc.num_return_sequences or 0),
        top_p=_scalar_float(gc.top_p),
        top_k=_scalar_int(gc.top_k),
        temperature=_scalar_float(gc.temperature),
        min_new_tokens=_scalar_int(gc.min_new_tokens),
        random_seed=seed,
        repetition_penalty=_scalar_float(gc.repetition_penalty),
        frequency_penalty=_scalar_float(gc.frequency_penalty),
        presence_penalty=_scalar_float(gc.presence_penalty),
        stop_words_list=stop_tuples,
    )


def _resolve_dash_sc_grpc_port(server_manager) -> int:
    """Match ``FrontendApp`` / ``ServerConfig.dash_sc_grpc_server_port`` (smoke: rank_id=0, default ``worker_info_port_num``)."""
    sc = ServerConfig()
    # ``START_PORT`` / ``MagaServerManager.port`` is ``server_config.start_port``; HTTP bind is ``server_port`` for rank 0.
    sc.start_port = int(server_manager.port)
    sc.rank_id = 0
    return sc.dash_sc_grpc_server_port


def _parse_infer_chunk(
    infer: Any,
) -> Tuple[List[int], Optional[int], Optional[int], Optional[int]]:
    """Parse one ModelInferResponse: generated_ids, finish_reason, prompt_token_num, prompt_cached_token_num."""
    generated: List[int] = []
    finish: Optional[int] = None
    ptn: Optional[int] = None
    ptcn: Optional[int] = None
    for i, out in enumerate(infer.outputs):
        if i >= len(infer.raw_output_contents):
            break
        raw = infer.raw_output_contents[i]
        name = out.name
        if name == "generated_ids" and out.datatype == "INT32":
            n = len(raw) // 4
            generated = list(struct.unpack("<%di" % n, raw))
        elif name == "finish_reason":
            finish = decode_finish_reason(out, raw)
        elif name == "prompt_token_num" and out.datatype == "INT32" and len(raw) >= 4:
            ptn = struct.unpack("<i", raw[:4])[0]
        elif (
            name == "prompt_cached_token_num"
            and out.datatype == "INT32"
            and len(raw) >= 4
        ):
            ptcn = struct.unpack("<i", raw[:4])[0]
    return generated, finish, ptn, ptcn


class DashScGrpcComparer(NormalComparer):
    """Same golden format as NormalComparer; drives inference via DashSc gRPC (predict_v2.proto)."""

    def run(self):
        query_info: BaseModel = self.format_query(self.qr_info["query"])
        self.tracer.query = query_info
        assert isinstance(query_info, QueryInfo)
        self.maybe_set_concurrency(query_info)

        ckpt = self.qr_info.get("_smoke_task_model_path")
        tok_path = self.qr_info.get("_smoke_task_tokenizer_path") or ckpt
        model_type = self.qr_info.get("_smoke_task_model_type")
        if not ckpt or not model_type:
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                "dash_sc_grpc: missing _smoke_task_model_path / _smoke_task_model_type",
            )

        grpc_port = _resolve_dash_sc_grpc_port(self.server_manager)
        grpc_addr = f"127.0.0.1:{grpc_port}"
        logging.info("DashScGrpcComparer grpc_addr=%s", grpc_addr)

        prompt = query_info.prompt
        if isinstance(prompt, list):
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                "dash_sc_grpc: chat-style messages prompt not supported in smoke",
            )
        if query_info.prompt_batch is not None:
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                "dash_sc_grpc: prompt_batch not supported in smoke",
            )

        tokenizer = TokenizerFactory.create(ckpt, tok_path or ckpt, model_type)
        input_ids = tokenizer.encode(prompt)
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        input_ids = [int(x) for x in input_ids]

        gc = query_info.generate_config
        sampling = _generate_config_to_sampling(gc)
        other = OtherParams(return_input_ids=bool(gc.return_input_ids))

        request = build_model_infer_request(
            request_id="smoke_dash_sc_grpc_%s" % self.qr_info.get("_query_idx", 0),
            model_name="default",
            input_ids=input_ids,
            sampling=sampling,
            return_input_ids=other.return_input_ids,
        )

        accumulated: List[int] = []
        last_prompt_token_num: Optional[int] = None
        last_prompt_cached: Optional[int] = None
        error_message: Optional[str] = None

        channel = grpc.insecure_channel(
            grpc_addr, options=dash_sc_grpc_client_channel_options()
        )
        try:
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)

            def _req_iter():
                yield request

            for resp in stub.ModelStreamInfer(_req_iter()):
                if resp.error_message:
                    error_message = resp.error_message
                    break
                if not resp.HasField("infer_response"):
                    continue
                infer = resp.infer_response
                chunk_ids, _finish, ptn, ptcn = _parse_infer_chunk(infer)
                if chunk_ids:
                    accumulated.extend(chunk_ids)
                if ptn is not None:
                    last_prompt_token_num = ptn
                if ptcn is not None:
                    last_prompt_cached = ptcn
        finally:
            channel.close()

        if error_message:
            raise SmokeException(
                QueryStatus.VISIT_FAILED,
                f"dash_sc_grpc error: {error_message}",
            )

        decoded = tokenizer.decode(accumulated) if accumulated else ""
        actual_dict: dict[str, Any] = {"response": decoded}
        # Align with HTTP smoke when golden includes aux-style metrics from last chunk.
        aux: dict[str, Any] = {}
        if last_prompt_token_num is not None:
            aux["input_len"] = last_prompt_token_num
        if last_prompt_cached is not None:
            aux["reuse_len"] = last_prompt_cached
        if aux:
            actual_dict["aux_info"] = aux

        test_with_sleep = bool(int(os.environ.get("TEST_WITH_SLEEP", 0)))
        if test_with_sleep:
            time.sleep(3600 * 100)

        actual_result = self.format_result(actual_dict)
        self.tracer.actual_result = actual_result
        expect_result = self.format_result(self.qr_info["result"])
        self.tracer.expect_result = expect_result
        self._maybe_rewrite_expect_result(actual_result, expect_result, query_info)
        self._dump_actual_to_artifact(actual_result)
        if save_response():
            self.qr_info["result"] = actual_result.model_dump(exclude_defaults=True)
        if no_compare():
            return
        self.compare_result(expect_result, actual_result)
