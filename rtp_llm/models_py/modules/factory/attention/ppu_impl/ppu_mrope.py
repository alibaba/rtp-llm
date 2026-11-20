# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph-safe MRoPE for the PPU Qwen3.5 FA3 attention path.

Background
----------
The earlier diagnostic path recomputed cos/sin every forward via
``torch.arange`` + ``positions.cpu()`` + ``einsum`` and then launched a Triton
kernel. Inside a decode CUDA-graph capture that path is illegal on two counts:

1. ``positions.cpu()`` is a device->host copy ("operation not permitted when
   stream is capturing").
2. the Triton kernel JIT-compiles on first launch, which also happens during
   capture.

This op fixes both while preserving the exact prefill MRoPE numerics that are
accuracy-critical (turning MRoPE off drops the C-Eval gate hard):

* attention instances on the same device with matching RoPE parameters share
  FP32 cos/sin tables.
* forward only does a device gather ``cache[positions]`` -> no host sync;
* the Triton kernel is JIT-warmed once at construction so capture binds an
  already-compiled kernel.

Both prefill and decode share this single op (as before), so the numerical
contract is identical across phases.
"""

import logging
from functools import lru_cache
from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.ppu_impl.mrope_kernel import (
    apply_mrope_triton_inplace,
)
from rtp_llm.ops import AttentionConfigs


@lru_cache(maxsize=4)
def _get_mrope_tables(
    rotary_dim: int, base: float, max_pos: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the original FP32 CPU math for bitwise parity.
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    positions = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    return freqs.cos().to(device), freqs.sin().to(device)


class PpuMRopeOp(MhaRotaryEmbeddingOp):
    """Graph-safe MRoPE: precomputed device cos/sin cache + gather-by-positions."""

    def __init__(
        self,
        attn_config: AttentionConfigs,
        cos_sin_cache: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(attn_config, cos_sin_cache)
        rope_config = attn_config.rope_config
        rotary_dim = rope_config.dim
        base = rope_config.base
        max_pos = attn_config.max_seq_len + attn_config.gen_num_per_cycle + 1

        self._rotary_dim = rotary_dim
        self._sections = [
            rope_config.mrope_dim1,
            rope_config.mrope_dim2,
            rope_config.mrope_dim3,
        ]

        # Eager attention instances are recreated every engine step.
        device = torch.device("cuda", torch.cuda.current_device())
        self._cos_cache, self._sin_cache = _get_mrope_tables(
            rotary_dim, base, max_pos, device
        )

        self._warm_kernel(attn_config, device)

    def _warm_kernel(self, attn_config: AttentionConfigs, device: torch.device) -> None:
        """JIT the Triton MRoPE kernel eagerly so graph capture binds a compiled kernel."""
        try:
            hd = self.head_size
            n_qh = attn_config.head_num
            n_kh = attn_config.kv_head_num
            rd = self._rotary_dim
            q = torch.zeros((1, n_qh, hd), dtype=torch.bfloat16, device=device)
            k = torch.zeros((1, n_kh, hd), dtype=torch.bfloat16, device=device)
            cos = self._cos_cache[:1].to(torch.bfloat16)
            sin = self._sin_cache[:1].to(torch.bfloat16)
            cos = torch.stack((cos, cos, cos), dim=0)
            sin = torch.stack((sin, sin, sin), dim=0)
            apply_mrope_triton_inplace(
                q, k, cos, sin, self._sections, hd, rd, mrope_interleaved=True
            )
            torch.cuda.synchronize()
        except Exception as exc:  # best-effort warmup; never fatal
            logging.warning("PpuMRopeOp kernel warmup failed: %s", exc)

    def _apply_rope(
        self, query: torch.Tensor, key: torch.Tensor, rope_params: Any
    ) -> None:
        positions = rope_params.positions_d
        if positions.ndim != 1 or positions.numel() != query.shape[0]:
            raise ValueError(
                "PpuMRopeOp supports text positions with one position per token"
            )
        # Device gather only -> no host sync, capture-safe.
        # Protect gathers on other streams if a table is evicted.
        stream = torch.cuda.current_stream(query.device)
        self._cos_cache.record_stream(stream)
        self._sin_cache.record_stream(stream)
        cos = self._cos_cache[positions].to(query.dtype)
        sin = self._sin_cache[positions].to(query.dtype)
        cos = torch.stack((cos, cos, cos), dim=0)
        sin = torch.stack((sin, sin, sin), dim=0)
        apply_mrope_triton_inplace(
            query,
            key,
            cos,
            sin,
            self._sections,
            self.head_size,
            self._rotary_dim,
            mrope_interleaved=True,
        )
