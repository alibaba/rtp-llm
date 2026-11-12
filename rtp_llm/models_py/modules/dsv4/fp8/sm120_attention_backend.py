"""SM120 sparse attention preparation, separate from shared CP orchestration."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import (
    SparseAttentionBackend,
    SparsePrefillInputs,
    SparsePrefillRun,
)

_FP8_KV_ENTRY_BYTES = 584


class Sm120AttentionBackend(SparseAttentionBackend):
    uses_deepgemm_einsum = False
    supports_shared_qkv_quant = False
    uses_tilelang_sparse = False
    supports_cp_raw_q_merge = False

    def validate_heads(self, n_heads: int) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
            validate_sm120_head_count,
        )

        validate_sm120_head_count(n_heads)

    def validate_decode_window(self, window_size: int) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
            validate_sm120_swa_topk_width,
        )

        validate_sm120_swa_topk_width(window_size, context="DeepSeek-V4 decode")

    def preserve_packed_prefix(self, n_heads: int) -> bool:
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
            SM120_HEAD_COUNTS,
        )

        return n_heads in SM120_HEAD_COUNTS

    def prepare_decode(
        self, device: torch.device, n_heads: int, softmax_scale: float
    ) -> Callable[..., torch.Tensor]:
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
            forward_decode,
            prepare_decode,
        )

        prepare_decode(device, n_heads)
        return partial(forward_decode, softmax_scale=softmax_scale)

    def initialize_workspace(
        self, workspace: torch.Tensor, compressed_width: int
    ) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
            zero_extra_sentinel_row,
        )

        # Invalid extra-table tails read row 0 after clamping. Keep it finite
        # even when request 0 owns no compressed rows; live gather overwrites it.
        zero_extra_sentinel_row(workspace, compressed_width)

    def workspace_inputs(
        self,
        *,
        workspace,
        indices,
        lengths,
        M,
        N,
        window_size,
        extra_width,
        ratio,
        device,
        prefix_cache,
        prefix_slots,
        prefix_lens,
    ) -> SparsePrefillInputs:
        from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
            split_sm120_combined_tables,
        )

        dim = workspace.shape[-1]
        extra_kv = workspace[:, :N, :].contiguous().view(-1, dim)
        swa_kv = workspace[:, N:, :].contiguous().view(-1, dim)
        swa_indices, swa_lens, extra_indices, extra_lens, extra_page_size = (
            split_sm120_combined_tables(
                indices,
                lengths,
                M=M,
                N=N,
                window_size=window_size,
                extra_width=extra_width,
                ratio=ratio,
                device=device,
            )
        )
        return SparsePrefillInputs(
            prefix_cache=prefix_cache,
            prefix_slots=prefix_slots,
            prefix_lens=prefix_lens,
            swa_kv=swa_kv,
            extra_kv=extra_kv,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
            extra_page_size=extra_page_size,
        )

    def prepare_prefill(
        self,
        *,
        n_heads,
        q,
        kv,
        indices,
        topk_length,
        head_dim,
        softmax_scale,
        attn_sink,
        prefill_workspace,
        backend_inputs: Optional[SparsePrefillInputs] = None,
    ) -> SparsePrefillRun:
        if self.uses_flash_mla_prefill(n_heads):
            return super().prepare_prefill(
                q=q,
                kv=kv,
                indices=indices,
                topk_length=topk_length,
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
            )
        return _prepare_prefill(
            q=q,
            kv=kv,
            indices=indices,
            topk_length=topk_length,
            head_dim=head_dim,
            softmax_scale=softmax_scale,
            attn_sink=attn_sink,
            prefill_workspace=prefill_workspace,
            backend_inputs=backend_inputs,
        )

    def prepare_grouped_output(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        groups: int,
        rank: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # FlashInfer consumes compact FP32 block scales, not DeepGEMM's
        # row-expanded packed layout. Convert once, before any graph capture.
        return (
            weight.view(groups, rank, width).contiguous(),
            scales.float().view(groups, rank // 128, width // 128).contiguous(),
        )

    def grouped_output(self, activation, scales, weight, weight_scales, rank):
        rows, groups, width = activation.shape
        expected = (groups, rank // 128, width // 128)
        if (
            weight_scales.dtype != torch.float32
            or tuple(weight_scales.shape) != expected
        ):
            raise ValueError(
                "SM120 grouped output projection requires compact FP32 weight "
                f"scales {expected} from prepare_grouped_output; got "
                f"{weight_scales.dtype} {tuple(weight_scales.shape)}"
            )
        from flashinfer.gemm import gemm_fp8_nt_groupwise

        def _ue8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
            scale_bytes = (
                scale.contiguous().view(torch.uint8).reshape(*scale.shape[:-1], -1)
            )
            return (scale_bytes.to(torch.int32) - 127).float().exp2()

        padded_rows = (rows + 3) & ~3
        a = torch.zeros(
            (groups, padded_rows, width),
            dtype=activation.dtype,
            device=activation.device,
        )
        a[:, :rows].copy_(activation.transpose(0, 1))
        a_scale = torch.ones(
            (groups, padded_rows, width // 128),
            dtype=torch.float32,
            device=activation.device,
        )
        a_scale[:, :rows].copy_(_ue8m0_to_fp32(scales).transpose(0, 1))
        projected = torch.stack(
            [
                gemm_fp8_nt_groupwise(
                    a[g],
                    weight[g],
                    a_scale[g],
                    weight_scales[g],
                    scale_granularity_mnk=(1, 128, 128),
                    scale_major_mode="K",
                    out_dtype=torch.bfloat16,
                )
                for g in range(groups)
            ],
            dim=0,
        )
        return projected[:, :rows].transpose(0, 1).contiguous()


SM120_ATTENTION = Sm120AttentionBackend()


def _prepare_prefill(
    *,
    q,
    kv,
    indices,
    topk_length,
    head_dim,
    softmax_scale,
    attn_sink,
    prefill_workspace,
    backend_inputs: Optional[SparsePrefillInputs] = None,
) -> SparsePrefillRun:
    inputs = backend_inputs if backend_inputs is not None else SparsePrefillInputs()
    sm120_swa_kv = inputs.swa_kv
    sm120_extra_kv = inputs.extra_kv
    sm120_swa_cache = inputs.swa_cache
    sm120_extra_cache = inputs.extra_cache
    sm120_prefix_cache = inputs.prefix_cache
    sm120_prefix_slots = inputs.prefix_slots
    sm120_prefix_lens = inputs.prefix_lens
    sm120_swa_indices = inputs.swa_indices
    sm120_swa_lens = inputs.swa_lens
    sm120_extra_indices = inputs.extra_indices
    sm120_extra_lens = inputs.extra_lens
    sm120_extra_page_size = inputs.extra_page_size
    # Paged pool views come from the caller; the caches built below are
    # packed in-method with every slot written. Only paged block 0 is
    # never guaranteed initialized. Capture each pool before the extra
    # cache may be rebound to the in-method packed one below.
    paged_swa_pool = sm120_swa_cache is not None
    paged_extra_pool = sm120_extra_cache is not None
    sm120_cache = sm120_swa_cache
    from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
        remap_padding_to_first_entry,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
        SM120_EXTRA_TOPK_WIDTHS,
        canonical_topk,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
        run as run_sm120_sparse_mla,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
        run_chunked_reference,
        uses_generic_fallback,
    )

    if sm120_cache is None:
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )

        logical_kv = (
            (sm120_swa_kv if sm120_swa_kv is not None else kv.reshape(-1, head_dim))
            .to(torch.bfloat16)
            .contiguous()
        )
        token_count = int(logical_kv.shape[0])
        sm120_cache = prefill_workspace.sm120_packed_cache(
            "swa",
            num_tokens=token_count,
            page_size=64,
            entry_bytes=_FP8_KV_ENTRY_BYTES,
        )
        quantize_and_insert_k_cache(
            logical_kv,
            sm120_cache,
            prefill_workspace.sm120_linear_slots("swa", token_count),
            prefix_cache=sm120_prefix_cache,
            prefix_slot_mapping=sm120_prefix_slots,
            prefix_gather_lens=sm120_prefix_lens,
        )
    if (
        sm120_extra_cache is None
        and sm120_extra_kv is not None
        and sm120_extra_kv.shape[0] > 0
    ):
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )

        assert sm120_extra_page_size in (2, 64)
        extra_page_size = int(sm120_extra_page_size)
        extra_kv = sm120_extra_kv.to(torch.bfloat16).contiguous()
        extra_tokens = int(extra_kv.shape[0])
        extra_role = f"extra_{extra_page_size}"
        sm120_extra_cache = prefill_workspace.sm120_packed_cache(
            extra_role,
            num_tokens=extra_tokens,
            page_size=extra_page_size,
            entry_bytes=_FP8_KV_ENTRY_BYTES,
        )
        quantize_and_insert_k_cache(
            extra_kv,
            sm120_extra_cache,
            prefill_workspace.sm120_linear_slots(extra_role, extra_tokens),
        )

    def forward(start: int, end: int) -> torch.Tensor:
        assert sm120_cache is not None
        dual_cache = sm120_extra_cache is not None
        chunk_indices = (
            sm120_swa_indices[start:end]
            if sm120_swa_indices is not None
            else indices[start:end]
        )
        if chunk_indices.dim() == 3:
            chunk_indices = chunk_indices.squeeze(1)
        supported = (128, 512, 1024) if end - start <= 64 else (128, 512, 1024, 2048)
        chunk_lens = (
            sm120_swa_lens[start:end]
            if sm120_swa_lens is not None
            else topk_length[start:end]
        )
        raw_swa_width = int(chunk_indices.shape[-1])
        raw_extra_width = (
            int(sm120_extra_indices.shape[-1])
            if dual_cache and sm120_extra_indices is not None
            else 0
        )
        generic_fallback = uses_generic_fallback(raw_swa_width, raw_extra_width)
        chunk_indices, chunk_lens = canonical_topk(
            chunk_indices,
            chunk_lens,
            (raw_swa_width,) if generic_fallback else supported,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import validate_slot_mapping

        validate_slot_mapping(
            "sm120.prefill.swa_indices",
            chunk_indices,
            block_size=int(sm120_cache.shape[1]),
            num_blocks=int(sm120_cache.shape[0]),
            negative_mode="skip_minus_one",
        )
        chunk_extra_indices = chunk_extra_lens = None
        if dual_cache:
            assert sm120_extra_indices is not None
            assert sm120_extra_lens is not None
            chunk_extra_indices = sm120_extra_indices[start:end]
            if chunk_extra_indices.dim() == 3:
                chunk_extra_indices = chunk_extra_indices.squeeze(1)
            chunk_extra_indices, chunk_extra_lens = canonical_topk(
                chunk_extra_indices,
                sm120_extra_lens[start:end],
                ((raw_extra_width,) if generic_fallback else SM120_EXTRA_TOPK_WIDTHS),
            )
            validate_slot_mapping(
                "sm120.prefill.extra_indices",
                chunk_extra_indices,
                block_size=int(sm120_extra_cache.shape[1]),
                num_blocks=int(sm120_extra_cache.shape[0]),
                negative_mode="skip_minus_one",
            )
        o_part = torch.empty_like(q[start:end])
        sparse_runner = (
            run_chunked_reference if generic_fallback else run_sm120_sparse_mla
        )
        if not generic_fallback:
            # Paged pools are caller-owned and block 0 is never
            # guaranteed initialized, yet the kernel still loads
            # clamped entries: point padding at the row's first
            # live slot (canonical_topk keeps it valid whenever
            # the row loads tiles); lengths still mask the tail.
            if paged_swa_pool:
                chunk_indices = remap_padding_to_first_entry(chunk_indices)
            else:
                chunk_indices.clamp_min_(0)
            if chunk_extra_indices is not None:
                if paged_extra_pool:
                    chunk_extra_indices = remap_padding_to_first_entry(
                        chunk_extra_indices
                    )
                else:
                    chunk_extra_indices.clamp_min_(0)
        sparse_runner(
            query=q[start:end].contiguous(),
            swa_cache=sm120_cache,
            swa_indices=chunk_indices,
            out=o_part,
            scale=softmax_scale,
            sinks=attn_sink.float(),
            swa_lens=chunk_lens,
            extra_cache=sm120_extra_cache if dual_cache else None,
            extra_indices=chunk_extra_indices,
            extra_lens=chunk_extra_lens,
            **({"is_prefill": True} if not generic_fallback else {}),
        )

        return o_part

    return SparsePrefillRun(forward, request_aligned=True)
