"""Default-on FULL source decode fork/join with eligibility-based fallback."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer, _v41_decode_topk
from rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv import decode_compute_qkv
from rtp_llm.models_py.modules.dsv41.fp8.decode.global_overlap import (
    _GLOBAL_COMPUTE_STREAMS,
    _get_compute_stream,
    supports_paged_global,
)

_INDEXER_STREAMS = threading.local()


def _qb_pdl_enabled(mega) -> bool:
    if (
        os.environ.get("DSV41_FULL_DECODE_QB_PDL", "1") != "1"
        or getattr(mega, "merged_qb", None) is None
    ):
        return False
    import deep_gemm

    # Do not change the engine-global DeepGEMM launch policy. Its QB consumer
    # must already launch with PDL and wait before reading the producer data.
    return hasattr(deep_gemm, "get_pdl") and deep_gemm.get_pdl()


def is_supported(attn, x, positions, req_ids, starts) -> bool:
    if (
        os.environ.get("DSV41_FULL_DECODE_MULTISTREAM", "1") != "1"
        or not x.is_cuda
        or not attn.is_kv_source
        or not attn.is_index_source
        or attn.layer_id != attn.kv_source_layer_id
        or attn.tp_size != 1
        or (attn.index_n_heads, attn.index_head_dim) != (32, 128)
        or attn.index_topk != 512
    ):
        return False
    return supports_paged_global(attn, x, positions, req_ids, starts)


@dataclass
class FullDecodeWork:
    current: torch.cuda.Stream
    global_stream: torch.cuda.Stream
    auxiliary: torch.cuda.Stream
    weights: torch.Tensor | None = None
    metadata: _v41_decode_indexer.DecodeIndexerMetadata | None = None
    inputs: list[torch.Tensor] = field(default_factory=list)
    outputs: list[torch.Tensor] = field(default_factory=list)
    finished: bool = False

    def wait_indexer(self) -> None:
        self.current.wait_stream(self.auxiliary)

    def wait_global(self) -> None:
        self.current.wait_stream(self.global_stream)

    def select_candidates(
        self, logits, lengths, block_size, count, *, complete=False, build_flags=True
    ):
        self.inputs.extend((logits, lengths))
        self.auxiliary.wait_stream(self.current)
        with torch.cuda.stream(self.auxiliary):
            result = _v41_decode_topk.select_candidates(
                logits,
                lengths,
                block_size,
                count,
                **({"complete": True} if complete else {}),
                **({"build_flags": False} if not build_flags else {}),
            )
        self.outputs.extend(tensor for tensor in result if tensor is not None)
        return result

    def finish(self) -> None:
        if self.finished:
            return
        self.wait_global()
        self.current.wait_stream(self.auxiliary)
        # Side-allocated results are read by the caller and subsequent layers.
        # Caller-owned inputs are retained until their side-stream uses join.
        for tensor in self.outputs:
            tensor.record_stream(self.current)
        self.inputs.clear()
        self.finished = True


def try_full_decode(
    attn, x, metadata, positions, freqs, mega, swa_pool, *, shared_input_quant=None
):
    """Return None before forking when unsupported or not warmed for capture."""
    if (
        os.environ.get("DSV41_FULL_DECODE_MULTISTREAM", "1") != "1"
        or not attn.is_kv_source
        or not attn.is_index_source
        or attn.layer_id != attn.kv_source_layer_id
        or attn.tp_size != 1
    ):
        return None
    batch, span, _ = x.shape
    rows = batch * span
    req = metadata.req_id_per_token[:rows]
    starts = metadata.start_pos[:batch]
    if not is_supported(attn, x, positions, req, starts):
        return None
    from rtp_llm.models_py.modules.dsv41.fp8.attention import _cuda_device_index

    device = _cuda_device_index(x.device)
    current = torch.cuda.current_stream(x.device)
    with torch.cuda.device(x.device):
        capturing = torch.cuda.is_current_stream_capturing()
        global_stream, auxiliary = (
            _get_compute_stream(cache, x.device, current, capturing=capturing)
            for cache in (_GLOBAL_COMPUTE_STREAMS, _INDEXER_STREAMS)
        )
    if global_stream is None or auxiliary is None:
        return None
    warm_key = (
        device,
        tuple(x.shape),
        global_stream.cuda_stream,
        auxiliary.cuda_stream,
    )
    warmed = getattr(attn, "_full_decode_overlap_warmed", set())
    if capturing and warm_key not in warmed:
        return None
    work = FullDecodeWork(current, global_stream, auxiliary)
    work.inputs.extend((x, positions, req, starts))
    # Fork both streams from the caller before enqueueing any branch work.
    global_stream.wait_stream(current)
    auxiliary.wait_stream(current)
    try:
        with torch.cuda.stream(global_stream):
            from rtp_llm.models_py.modules.dsv41.fp8._v41_full_projection import project

            projected = project(attn, x)
            if projected is None:
                attn._produce_global_decode(x, positions, req, starts)
            else:
                values, scores, work.weights = projected
                work.outputs.extend(projected)
                # Publish the head weights before compression/cache writes.
                auxiliary.wait_stream(global_stream)
                attn._produce_global_decode(
                    x, positions, req, starts, projected=(values, scores)
                )
        keys = attn._shared_attention["global"][attn.layer_id]
        with torch.cuda.stream(auxiliary):
            if work.weights is None:
                work.weights = F.linear(x.reshape(rows, -1), attn.index_weights)
                work.outputs.append(work.weights)
            work.metadata = _v41_decode_indexer.get_decode_metadata(
                attn._shared_attention,
                keys,
                positions,
                batch_size=batch,
                query_len=span,
                compress_ratio=attn.compress_ratio,
            )
            if work.metadata is not None:
                plan = work.metadata
                work.outputs.extend(
                    (
                        plan.lengths,
                        plan.safe_table,
                        plan.paged.lengths,
                        plan.paged.tables,
                        plan.paged.request_ids,
                        plan.paged.schedule,
                    )
                )
        swa_slots = metadata.pool_write_slot_mappings.get(SWA_KV)
        qkv = decode_compute_qkv(
            attn,
            x,
            positions,
            freqs_cis=freqs,
            swa_pool=swa_pool,
            swa_slots=None if swa_slots is None else swa_slots[:rows],
            mega_attention=mega,
            shared_input_quant=shared_input_quant,
            qb_pdl=_qb_pdl_enabled(mega),
        )
        if not qkv.swa_written:
            attn._decode_write_swa_fp8(qkv.kv, batch, span, metadata)
        slot_plan = None
        if os.environ.get("DSV41_FUSED_TOPK_SLOTS", "0") == "1":
            from rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils import (
                require_pool_tokens_per_block,
            )

            region = attn._global_region()
            global_pool = attn._source_pool(region)
            raw_tpb = require_pool_tokens_per_block(attn._kv_cache, region=region)
            cp = attn._cp_ctx
            sharded = cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded
            if not sharded and raw_tpb % attn.compress_ratio == 0:
                slot_plan = _v41_decode_topk.make_slot_plan(
                    metadata.req_id_per_token[:rows],
                    attn._block_tables_by_type[region],
                    entries=global_pool.shape[1],
                    logical_entries=raw_tpb // attn.compress_ratio,
                )
        selected = attn._select_indices_decode(
            x,
            qkv.qr,
            positions,
            qkv.qr_quantized,
            qkv.index_q,
            overlap=work,
            **({"slot_plan": slot_plan} if slot_plan is not None else {}),
        )
        slots = None if slot_plan is None else slot_plan.slots
        if slots is None:
            slots = (
                attn._decode_global_slots(selected, metadata.req_id_per_token[:rows])
                .to(dtype=torch.int32)
                .contiguous()
            )
        mega_positions = attn._mega_position_chunks(metadata.position_ids, rows, rows)[
            0
        ]
        with record_function_range("dsv41.decode.mega_attention"):
            pair = mega.decode(
                qkv.q,
                swa_pool,
                metadata.swa_global_slots[:rows],
                mega_positions,
                extra_pool=attn._source_pool(attn._global_region()),
                extra_indices=slots,
            )
        out = mega.project_output(pair, attn.wo_b)
        attn._prefill_output_all_reduce(out)
    finally:
        work.finish()
    if not capturing:
        warmed.add(warm_key)
        attn._full_decode_overlap_warmed = warmed
    return out.view(batch, span, attn.dim)
