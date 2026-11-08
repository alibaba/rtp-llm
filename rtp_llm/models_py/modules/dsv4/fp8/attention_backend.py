"""Sparse attention provider selection shared by prefill, decode and warmup."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Callable, Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8.backend_selection import (
    _is_sm120_index,
    cuda_device_index,
)


@dataclass
class SparsePrefillRun:
    forward: Callable[[int, int], torch.Tensor]
    request_aligned: bool = False


@dataclass(frozen=True)
class SparsePrefillInputs:
    """Optional packed pools and split index tables for sparse prefill."""

    swa_kv: Optional[torch.Tensor] = None
    extra_kv: Optional[torch.Tensor] = None
    swa_cache: Optional[torch.Tensor] = None
    extra_cache: Optional[torch.Tensor] = None
    prefix_cache: Optional[torch.Tensor] = None
    prefix_slots: Optional[torch.Tensor] = None
    prefix_lens: Optional[torch.Tensor] = None
    swa_indices: Optional[torch.Tensor] = None
    swa_lens: Optional[torch.Tensor] = None
    extra_indices: Optional[torch.Tensor] = None
    extra_lens: Optional[torch.Tensor] = None
    extra_page_size: Optional[int] = None


def _prepare_wo_a_stacked(
    weight_fp8: torch.Tensor,
    scale_raw: torch.Tensor,
    G: int,
    R: int,
    K: int,
) -> tuple:
    """Stack V4 wo_a ckpt (``[G*R, K]`` fp8 + ``[G*R/128, K/128]`` e8m0fnu)
    into the ``fp8_einsum``-expected layout, computed once at init:

    - weight: ``[G, R, K]`` fp8 contiguous (free view)
    - scale : ``[G, R, K/512]`` int32 UE8M0 packed, MN-major TMA-aligned
      (stride ``(K/512 * tma_R, 1, tma_R)``)

    Cast e8m0fnu → fp32, row-repeat by 128 along R so
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor`` (which operates on
    fp32 ``[*, mn, k/128]``) sees the full [G, R, K/128] grid.  The helper
    floor-log2 bitcasts each block scale and packs 4 UE8M0 bytes per
    int32; output shape ``[G, R, K/512]`` matches
    ``deep_gemm.fp8_einsum(..., recipe=(1, 1, 128))`` expectations."""
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

    w_stk = weight_fp8.view(G, R, K).contiguous()
    scale_fp32 = scale_raw.float().view(G, R // 128, K // 128)
    idx = torch.arange(R, device=scale_raw.device) // 128
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()  # [G, R, K/128]
    s_stk = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return w_stk, s_stk


class SparseAttentionBackend:
    uses_deepgemm_einsum = True
    supports_shared_qkv_quant = True
    uses_tilelang_sparse = True
    supports_cp_raw_q_merge = True

    def validate_heads(self, n_heads: int) -> None:
        pass

    def validate_decode_window(self, window_size: int) -> None:
        pass

    def preserve_packed_prefix(self, n_heads: int) -> bool:
        return False

    def uses_flash_mla_prefill(self, n_heads: int) -> bool:
        return not self.preserve_packed_prefix(n_heads)

    def prepare_decode(
        self, device: torch.device, n_heads: int, softmax_scale: float
    ) -> Optional[Callable[..., torch.Tensor]]:
        return None

    def initialize_workspace(
        self, workspace: torch.Tensor, compressed_width: int
    ) -> None:
        pass

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
        return SparsePrefillInputs()

    def prepare_prefill(
        self,
        *,
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        softmax_scale,
        n_heads=None,
        head_dim=None,
        prefill_workspace=None,
        backend_inputs: Optional[SparsePrefillInputs] = None,
    ) -> SparsePrefillRun:
        from rtp_llm.models_py.utils.flash_mla_legacy import flash_mla_sparse_fwd

        def forward(start: int, end: int) -> torch.Tensor:
            out, _, _ = flash_mla_sparse_fwd(
                q=q[start:end],
                kv=kv,
                indices=indices[start:end],
                sm_scale=softmax_scale,
                attn_sink=attn_sink,
                topk_length=topk_length[start:end],
            )
            return out

        return SparsePrefillRun(forward)

    def prepare_grouped_output(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        groups: int,
        rank: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare immutable per-layer weight buffers for the selected kernel."""
        return _prepare_wo_a_stacked(weight, scales, groups, rank, width)

    def grouped_output(self, activation, scales, weight, weight_scales, rank):
        import deep_gemm

        rows, groups, _ = activation.shape
        out = torch.empty(
            rows, groups, rank, dtype=torch.bfloat16, device=activation.device
        )
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (activation, scales),
            (weight, weight_scales),
            out,
            recipe=(1, 1, 128),
        )
        return out


_DEFAULT = SparseAttentionBackend()


def get_sparse_attention_backend(device: torch.device) -> SparseAttentionBackend:
    """Choose a stateless provider using the resolved execution-device index."""
    index = cuda_device_index(device)
    return _DEFAULT if index is None else _backend_for_cuda(index)


@cache
def _backend_for_cuda(index: int) -> SparseAttentionBackend:
    if _is_sm120_index(index):
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        return SM120_ATTENTION
    return _DEFAULT
