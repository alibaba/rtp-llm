"""Default-on FULL source decode fork/join with eligibility-based fallback."""

from __future__ import annotations

import os
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
    auxiliary: torch.cuda.Stream
    weights: torch.Tensor | None = None
    metadata: _v41_decode_indexer.DecodeIndexerMetadata | None = None
    inputs: list[torch.Tensor] = field(default_factory=list)
    outputs: list[torch.Tensor] = field(default_factory=list)
    pending: bool = False

    def fork(self) -> None:
        self.auxiliary.wait_stream(self.current)
        self.pending = True

    def join(self) -> None:
        if self.pending:
            self.current.wait_stream(self.auxiliary)
            self.pending = False

    def select_candidates(
        self, logits, lengths, block_size, count, *, complete=False, build_flags=True
    ):
        if not self.pending:
            self.fork()
        self.inputs.extend((logits, lengths))
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

    def handoff_candidates(self, plan_work) -> None:
        # The next indexer joins the same queue through the plan. Keep its
        # candidate inputs alive across the intervening model layers.
        plan_work.inputs += tuple(self.inputs)
        self.inputs.clear()
        self.pending = False

    def finish(self) -> None:
        self.join()
        # The caller consumes side-allocated results on its own stream.
        for tensor in self.outputs:
            tensor.record_stream(self.current)
        self.outputs.clear()
        self.inputs.clear()


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
        # The target's Engram lookups run in earlier layers. Keep a separate
        # queue if a different model layout prefetches during this attention.
        auxiliary = getattr(metadata, "_engram_lookup_stream", None)
        if (
            auxiliary is None
            or auxiliary.device != x.device
            or getattr(attn, "_engram_lookup_schedule", None) is not None
        ):
            auxiliary = _get_compute_stream(
                _GLOBAL_COMPUTE_STREAMS, x.device, current, capturing=capturing
            )
    if auxiliary is None:
        return None
    warm_key = (
        device,
        tuple(x.shape),
        current.cuda_stream,
        auxiliary.cuda_stream,
    )
    warmed = getattr(attn, "_full_decode_overlap_warmed", set())
    if capturing and warm_key not in warmed:
        return None
    work = FullDecodeWork(current, auxiliary)

    def project_qkv():
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
        return qkv

    from rtp_llm.models_py.modules.dsv41.fp8._v41_full_projection import project

    work.fork()
    try:
        # Capture the main branch first so replay keeps successive QKV/FFN
        # layers on the launch stream. Both branches start at the same fork.
        if capturing:
            qkv = project_qkv()
        with torch.cuda.stream(auxiliary):
            projected = project(attn, x, defer_head_cast=True)
            if projected is None:
                attn._produce_global_decode(x, positions, req, starts)
                work.weights = F.linear(x.reshape(rows, -1), attn.index_weights)
            else:
                values, scores, work.weights = projected
                attn._produce_global_decode(
                    x, positions, req, starts, projected=(values, scores)
                )
            keys = attn._shared_attention["global"][attn.layer_id]
            work.metadata = _v41_decode_indexer.get_decode_metadata(
                attn._shared_attention,
                keys,
                positions,
                batch_size=batch,
                query_len=span,
                compress_ratio=attn.compress_ratio,
                request_ids=attn._shared_attention.get("decode_packed_request_ids"),
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
            if projected is not None:
                # Only the indexer query needs BF16 heads. Issue the KV work first.
                work.weights = work.weights.to(torch.bfloat16)
            work.outputs.append(work.weights)
        if not capturing:
            qkv = project_qkv()
        work.join()
        slot_plan = attn._decode_slot_plan(metadata.req_id_per_token[:rows])
        selected = attn._select_indices_decode(
            x,
            qkv.qr,
            positions,
            qkv.qr_quantized,
            qkv.index_q,
            overlap=work,
            **({"slot_plan": slot_plan} if slot_plan is not None else {}),
        )
        slots = (
            attn._decode_global_slots(
                selected,
                metadata.req_id_per_token[:rows],
                slots=None if slot_plan is None else slot_plan.slots,
            )
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
