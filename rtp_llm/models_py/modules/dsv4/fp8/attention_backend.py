"""Sparse attention provider selection shared by prefill, decode and warmup."""

from __future__ import annotations

import importlib
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, Optional, Sequence, TypeVar

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.fp8.backend_selection import (
    _is_sm120_index,
    cuda_device_index,
)
from rtp_llm.models_py.utils.arch import is_sm90

T = TypeVar("T")


class SparseAttentionWorkspace:
    """Arch-neutral per-model lazy holder for backend-owned sparse attention scratch."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: Any = None

    def get_or_create(self, factory: Callable[[], T]) -> T:
        owner = self._owner
        if owner is None:
            with self._lock:
                owner = self._owner
                if owner is None:
                    owner = factory()
                    self._owner = owner
        return owner


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
    """Stack V4 wo_a (``[G*R, K]`` fp8 + compact or prepacked ``int32`` scale)
    into the ``fp8_einsum``-expected layout:

    - weight: ``[G, R, K]`` fp8 contiguous
    - scale : ``[G, R, ceil(K/512)]`` int32 UE8M0 packed, MN-major TMA-aligned
      (stride ``(packed_K * tma_R, 1, tma_R)``).
    """
    if R % 128 != 0 or K % 128 != 0:
        raise ValueError(
            f"_prepare_wo_a_stacked requires R % 128 == 0 and K % 128 == 0, "
            f"got R={R}, K={K}"
        )
    if weight_fp8.device != scale_raw.device:
        raise ValueError(
            f"_prepare_wo_a_stacked weight device {weight_fp8.device} != scale device {scale_raw.device}"
        )
    try:
        layout_mod = importlib.import_module("deep_gemm.utils.layout")
    except ImportError:
        layout_mod = None
    w_stk = weight_fp8.view(G, R, K).contiguous()

    if scale_raw.dtype == torch.int32:
        packed_k = (K + 511) // 512
        expected_shape = (G * R, packed_k)
        if scale_raw.dim() != 2 or tuple(scale_raw.shape) != expected_shape:
            raise ValueError(
                f"packed wo_a scale shape mismatch: expected {expected_shape}, "
                f"got {tuple(scale_raw.shape)}"
            )
        get_tma = getattr(layout_mod, "get_tma_aligned_size", None)
        tma_r = (
            get_tma(R, scale_raw.element_size())
            if get_tma is not None
            else ((R + 3) // 4) * 4
        )
        src = scale_raw.contiguous().view(G, R, packed_k)
        buf = torch.zeros(
            (G, packed_k, tma_r), dtype=torch.int32, device=scale_raw.device
        )
        s_stk = buf.permute(0, 2, 1)[:, :R, :]
        s_stk.copy_(src)
        return w_stk, s_stk

    scale_fp32 = scale_raw.float().view(G, R // 128, K // 128)
    idx = torch.arange(R, device=scale_raw.device) // 128
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()  # [G, R, K/128]
    if weight_fp8.is_cuda and is_sm90(weight_fp8.device):
        return w_stk, scale_rep
    if layout_mod is None or not hasattr(
        layout_mod, "get_mn_major_tma_aligned_packed_ue8m0_tensor"
    ):
        # Environments without deep_gemm.utils.layout keep expanded FP32 scales
        # so grouped_output can use its portable fallback path.
        return w_stk, scale_rep
    dev_ctx = (
        torch.cuda.device(scale_raw.device) if scale_raw.is_cuda else nullcontext()
    )
    with dev_ctx:
        s_stk = layout_mod.get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return w_stk, s_stk


class SparseAttentionBackend:
    uses_deepgemm_einsum = True
    supports_shared_qkv_quant = True
    uses_tilelang_sparse = True
    supports_cp_raw_q_merge = True
    requires_packed_int32_scales = False

    def validate_heads(self, n_heads: int) -> None:
        pass

    def validate_decode_window(self, window_size: int) -> None:
        pass

    def preserve_packed_prefix(self, n_heads: int) -> bool:
        return False

    def uses_flash_mla_prefill(self, n_heads: int) -> bool:
        return not self.preserve_packed_prefix(n_heads)

    def project_indexer_prefill(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return F.linear(x, weight)

    def project_indexer_decode(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return F.linear(x, weight)

    def compressor_linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        gemm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        out_2d = gemm(x_2d, weight)
        return out_2d.reshape(*leading_shape, weight.shape[0])

    def compressor_prefill(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        chunk_lengths: Optional[Sequence[int]],
    ) -> torch.Tensor:
        del chunk_lengths
        return project(x, weight)

    def prepare_decode(
        self,
        device: torch.device,
        n_heads: int,
        softmax_scale: float,
        *,
        attn_workspace: Any = None,
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
        attn_workspace: Any = None,
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
        rows, groups, width = activation.shape
        if weight_scales.dtype == torch.int32:
            packed_k = (width + 511) // 512
            expected_ws_shape = (groups, rank, packed_k)
            if tuple(weight_scales.shape) != expected_ws_shape:
                raise ValueError(
                    f"grouped_output weight_scales shape mismatch: expected "
                    f"{expected_ws_shape}, got {tuple(weight_scales.shape)}"
                )
            ws_strides = weight_scales.stride()
            if ws_strides[1] != 1:
                raise ValueError(
                    "grouped_output expects MN-major weight_scales (stride(1) == 1), "
                    f"got strides {ws_strides}"
                )
            if ws_strides[2] < rank or (ws_strides[2] * 4) % 16 != 0:
                raise ValueError(
                    "grouped_output weight_scales inner pitch must be >= rank and "
                    f"16-byte TMA aligned, got rank={rank}, strides={ws_strides}"
                )
            if (
                ws_strides[0] != packed_k * ws_strides[2]
                or (ws_strides[0] * 4) % 16 != 0
            ):
                raise ValueError(
                    "grouped_output weight_scales cross-group pitch must equal "
                    f"packed_k * inner_pitch and be 16-byte aligned: packed_k={packed_k}, "
                    f"strides={ws_strides}"
                )
        elif (
            weight_scales.dtype == torch.float32
            and not self.requires_packed_int32_scales
        ):
            expected_ws_shape = (groups, rank, width // 128)
            if tuple(weight_scales.shape) != expected_ws_shape:
                raise ValueError(
                    f"grouped_output float32 weight_scales shape mismatch: expected "
                    f"{expected_ws_shape}, got {tuple(weight_scales.shape)}"
                )
        else:
            raise ValueError(
                f"grouped_output expects int32 or float32 weight_scales, got {weight_scales.dtype}"
            )

        try:
            deep_gemm = importlib.import_module("deep_gemm")
        except ImportError:
            deep_gemm = None
        fp8_einsum = getattr(deep_gemm, "fp8_einsum", None) if deep_gemm is not None else None
        if callable(fp8_einsum):
            dev_ctx = (
                torch.cuda.device(activation.device)
                if activation.is_cuda
                else nullcontext()
            )
            with dev_ctx:
                out = torch.empty(
                    rows, groups, rank, dtype=torch.bfloat16, device=activation.device
                )
                fp8_einsum(
                    "bhr,hdr->bhd",
                    (activation, scales),
                    (weight, weight_scales),
                    out,
                    recipe=(1, 1, 128),
                )
                return out

        # Portable fallback when deep_gemm.fp8_einsum is absent (e.g. CUDA 12.9).
        k_groups = width // 128
        act_f = activation.float().reshape(rows, groups, k_groups, 128)
        if scales.dtype == torch.int32:
            act_s = _unpack_packed_ue8m0_scales(scales, k_groups).reshape(
                rows, groups, k_groups, 1
            )
        else:
            act_s = scales.float().reshape(rows, groups, k_groups, 1)
        act_deq = (act_f * act_s).reshape(rows, groups, width)

        w_f = weight.float().reshape(groups, rank, k_groups, 128)
        if weight_scales.dtype == torch.int32:
            w_s = _unpack_packed_ue8m0_scales(weight_scales, k_groups).reshape(
                groups, rank, k_groups, 1
            )
        else:
            w_s = weight_scales.float().reshape(groups, rank, k_groups, 1)
        w_deq = (w_f * w_s).reshape(groups, rank, width)
        return torch.einsum("bhr,hdr->bhd", act_deq, w_deq).to(torch.bfloat16)


def _unpack_packed_ue8m0_scales(
    sf_packed: torch.Tensor, num_k_groups: int
) -> torch.Tensor:
    shifts = torch.arange(4, device=sf_packed.device, dtype=torch.int32) * 8
    exps = ((sf_packed.unsqueeze(-1) >> shifts) & 0xFF).to(torch.int32)
    return (exps << 23).view(torch.float32).flatten(-2)[..., :num_k_groups]


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
