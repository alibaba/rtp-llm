"""Input packing for the SM90 MegaMoE symmetric buffer.

The Hopper protocol stages FP8 activations with plain FP32 K128 scales
(DeepGEMM ``per_token_cast_to_fp8(use_ue8m0=False, gran_k=128)``), unlike the
SM100 buffer's packed UE8M0 K32 scales. The scale formula matches DeepGEMM's
``utils.math.per_token_cast_to_fp8`` so both sides quantize identically.
"""

from __future__ import annotations

from typing import Tuple

import torch

SM90_ACT_SF_GRAN_K = 128
_FP8_MAX = 448.0
_AMAX_FLOOR = 1e-4


def per_token_cast_to_fp8_fp32_scale(
    x: torch.Tensor, gran_k: int = SM90_ACT_SF_GRAN_K
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast ``[M, N]`` BF16/FP32 rows to FP8 with FP32 per-``gran_k`` scales.

    Mirrors ``deep_gemm.utils.per_token_cast_to_fp8(use_ue8m0=False)`` without
    its ``.all()`` debug assertion (which would synchronize CUDA->host and is
    illegal inside ``cudaStreamCapture``).
    """
    m, n = x.shape
    padded_n = ((n + gran_k - 1) // gran_k) * gran_k
    if padded_n != n:
        x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
        x_padded[:, :n] = x
    else:
        x_padded = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(_AMAX_FLOOR)
    scale = amax / _FP8_MAX
    x_fp8 = (
        (x_view * (1.0 / scale).unsqueeze(2))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_fp8, scale.contiguous()


class MegaMoeSM90InputPacker:
    """Stage routed MegaMoE inputs into a ``SymmBufferSM90``.

    ``buf`` exposes ``x`` ``[capacity, hidden]`` FP8, ``x_sf``
    ``[capacity, hidden/128]`` FP32, ``topk_idx`` ``[capacity, topk]`` int64 and
    ``topk_weights`` ``[capacity, topk]`` FP32.
    """

    name = "torch"

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        x_fp8, x_sf = per_token_cast_to_fp8_fp32_scale(safe_x)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())


def get_mega_sm90_input_packer() -> MegaMoeSM90InputPacker:
    return MegaMoeSM90InputPacker()
