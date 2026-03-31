"""Real inference path: async ``backend_visitor.enqueue`` bridged to streaming ModelStreamInfer."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import queue
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch

from rtp_llm.bailian.bailian_grpc_enqueue_loop import resolve_loop_for_enqueue
from rtp_llm.bailian.bailian_grpc_request import OtherParams, SamplingParams
from rtp_llm.bailian.bailian_grpc_response_real import (
    build_stream_response_from_generate_outputs,
)
from rtp_llm.bailian.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import GenerateInput, GenerateOutputs


def _derive_rtp_llm_request_id(request_id: str | Any) -> int:
    """Stable int64 for ``GenerateInput.request_id`` / log tags (not ``hash()`` — salt varies per process)."""
    s = request_id if isinstance(request_id, str) else str(request_id)
    digest8 = hashlib.sha256(s.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest8, "little", signed=True)


def stream_log_tag(*, request_id_numeric: int, trace_id: str) -> str:
    """Align with C++ ``GenerateStream::streamLogTag()`` for log correlation."""
    return f"request_id={request_id_numeric} trace_id={trace_id}"


# End of chunked stream from ``_iter_enqueue_sync`` queue.
_ITER_ENQUEUE_DONE = object()


def _iter_enqueue_sync(visitor, generate_input):
    """Yield enqueue stream chunks from the gRPC worker thread as they arrive.

    Runs ``visitor.enqueue`` on the loop from ``resolve_loop_for_enqueue()`` (typically
    the Uvicorn main loop). Does not buffer the full stream before returning.
    """
    out_q: queue.Queue = queue.Queue()
    pump_error: list[BaseException] = []

    async def pump():
        try:
            stream = await visitor.enqueue(generate_input)
            async for x in stream:
                out_q.put(x)
        except BaseException as e:
            pump_error.append(e)
        finally:
            out_q.put(_ITER_ENQUEUE_DONE)

    loop = resolve_loop_for_enqueue()
    fut = asyncio.run_coroutine_threadsafe(pump(), loop)
    try:
        while True:
            item = out_q.get()
            if item is _ITER_ENQUEUE_DONE:
                break
            yield item
        if pump_error:
            raise pump_error[0]
        fut.result()
    finally:
        if not fut.done():
            fut.cancel()


def iter_real_model_stream_infer(
    request,
    input_ids_list: list[int],
    sampling: SamplingParams,
    other: OtherParams,
    backend_visitor: Any,
    *,
    run_enqueue_sync: Callable[[Any, GenerateInput], Iterable[GenerateOutputs]],
) -> Iterator[predict_v2_pb2.ModelStreamInferResponse]:
    """Run enqueue on ``backend_visitor`` and yield one proto per chunk as the backend streams."""
    trace_str = str(request.id)
    rtp_llm_request_id = _derive_rtp_llm_request_id(trace_str)
    tag = stream_log_tag(request_id_numeric=rtp_llm_request_id, trace_id=trace_str)
    logging.debug(
        "[BailianGrpc] [%s] real infer start: model_name=%s input_len=%s sampling=%s",
        tag,
        request.model_name,
        len(input_ids_list),
        sampling,
    )
    try:
        generate_config = sampling.to_generate_config(other=other)
        generate_config.trace_id = trace_str
        token_ids = torch.tensor(input_ids_list, dtype=torch.int)
        generate_input = GenerateInput(
            request_id=rtp_llm_request_id,
            token_ids=token_ids,
            mm_inputs=[],
            generate_config=generate_config,
        )
        logging.debug("[BailianGrpc] [%s] generate_input: %s", tag, generate_input)
        request_shape = list(request.inputs[0].shape) if request.inputs else None
        chunk_idx = 0
        for go in run_enqueue_sync(backend_visitor, generate_input):
            chunk_idx += 1
            logging.debug("[BailianGrpc] [%s] real infer chunk %s", tag, chunk_idx)
            yield build_stream_response_from_generate_outputs(
                bailian_request_id=request.id,
                model_name=request.model_name,
                go=go,
                request_log_tag=tag,
                request_input_ids=input_ids_list,
                return_input_ids=other.return_input_ids,
                _request_shape=request_shape,
            )
        if chunk_idx:
            logging.debug(
                "[BailianGrpc] [%s] real infer done: output_chunks=%s",
                tag,
                chunk_idx,
            )
        if chunk_idx == 0:
            logging.warning(
                "[BailianGrpc] [%s] empty outputs_list",
                tag,
            )
            yield predict_v2_pb2.ModelStreamInferResponse(
                error_message="empty outputs_list from backend",
            )
    except Exception as e:
        logging.exception("[BailianGrpc] [%s] enqueue failed: %s", tag, e)
        yield predict_v2_pb2.ModelStreamInferResponse(error_message=str(e))
