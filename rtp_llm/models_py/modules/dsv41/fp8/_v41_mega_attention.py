# SPDX-License-Identifier: Apache-2.0
"""FlashMLA Q-RoPE/attention/inverse-RoPE/FP8 fusion for V4.1 on SM10x.

The installed FlashMLA extension owns both CUDA kernels. We permute projection
weights once, so their head-chunk layouts require no activation shuffle. The
ordinary weights remain intact for unsupported inputs and the disabled path.
No compilation or CUDA initialization occurs while importing this module.
Layout conventions follow vLLM's deepseek_v41/nvidia/flash_mla_mega_attn.py
and deepseek_v41/common/ops/fused_layout.py. Calls use RTP's pybind ABI.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache

import torch

from rtp_llm.models_py.modules.dsv41._fused_inv_rope_fp8_quant_triton import (
    _device_capability,
)

_HEAD_DIM = 512
_ROPE_DIM = 64
_HEADS_PER_GROUP = 8


@lru_cache(maxsize=1)
def _backend():
    # RTP uses FlashMLA's pybind package, not vLLM's torch.ops namespace. The
    # original pybind ABI returns freshly allocated output tensors; allocations
    # are captured normally by CUDA graphs. Do not pass vLLM-only out arguments.
    try:
        module = importlib.import_module("flash_mla.cuda")
    except (ImportError, OSError):
        return None
    required = (
        "fused_norm_rope_attn_rope_cast_fwd",
        "fused_norm_rope_attn_rope_cast_decode",
        "permute_q_b_proj",
        "permute_wv_proj",
    )
    return (
        module
        if all(callable(getattr(module, name, None)) for name in required)
        else None
    )


def is_supported(attn) -> bool:
    """Startup-only capability gate; unsupported configurations keep split ops.

    Native mega attention takes 64 or 128 query heads and eight heads per wo_a
    group. Smaller TP shards pad only the projected Q activation; the GEMM
    keeps its local head count. CP partitions query tokens without changing
    the sparse KV slot/length contract.
    """
    if os.environ.get("DSV41_MEGA_ATTENTION", "0") != "1":
        return False
    from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

    linear = getattr(attn, "wq_b", None)
    weight = getattr(attn, "_wo_a_stk_w", None)
    scale = getattr(attn, "_wo_a_stk_s", None)
    table = getattr(attn, "freqs_cis", None)
    sink = getattr(attn, "attn_sink", None)
    if not (
        type(linear) is V41MXFP8Linear
        and isinstance(weight, torch.Tensor)
        and isinstance(scale, torch.Tensor)
        and isinstance(table, torch.Tensor)
        and isinstance(sink, torch.Tensor)
        and getattr(attn, "skip_post_q_norm", False)
        and attn.n_heads in (8, 16, 32, 64, 128)
        and attn.head_dim == _HEAD_DIM
        and attn.rope_head_dim == _ROPE_DIM
        and attn.n_groups * _HEADS_PER_GROUP == attn.n_heads
    ):
        return False
    qw, qs = linear.weight, linear.weight_scales
    if not (
        qw.is_cuda
        and torch.version.hip is None
        and qw.dtype == weight.dtype == torch.float8_e4m3fn
        and qs.dtype == scale.dtype == torch.int32
        and qw.shape == (attn.n_heads * _HEAD_DIM, linear.K)
        and linear.K % 128 == 0
        and qs.shape == (qw.shape[0], linear.K // 128)
        and qs.stride(0) == 1
        and qw.is_contiguous()
        and weight.shape == (attn.n_groups, attn.o_lora_rank, 4096)
        and scale.shape == (attn.n_groups, attn.o_lora_rank, 32)
        and scale.stride(1) == 1
        and weight.is_contiguous()
        and table.dtype == torch.complex64
        and table.ndim == 2
        and table.shape[1] == 32
        and not table.is_conj()
        and sink.dtype == torch.float32
        and sink.shape == (attn.n_heads,)
        and sink.is_contiguous()
        and all(t.device == qw.device for t in (qs, weight, scale, table, sink))
        and all(not t.requires_grad for t in (qw, qs, weight, scale, table, sink))
    ):
        return False
    return _device_capability(qw.device)[0] == 10 and _backend() is not None


class MegaAttention(torch.nn.Module):
    """Prepared weights registered as ordinary nonpersistent model buffers."""

    def __init__(
        self,
        backend,
        query_weight,
        query_scale,
        output_weight,
        output_scale,
        cos_sin,
        sink,
        softmax_scale,
        heads,
        groups,
    ):
        super().__init__()
        self.backend = backend
        for name, value in (
            ("query_weight", query_weight),
            ("query_scale", query_scale),
            ("output_weight", output_weight),
            ("output_scale", output_scale),
            ("cos_sin", cos_sin),
            ("sink", sink),
        ):
            self.register_buffer(name, value, persistent=False)
        self.softmax_scale = softmax_scale
        self.heads = heads
        self.groups = groups

    @torch.no_grad()
    def reload_weights(self, attn):
        """Refresh derived weights in place without replacing captured buffers.

        Explicit refresh ignores the experiment env because already captured
        graphs may still reference these prepared tensors.
        """
        qw, qs, ow, oscale = _permuted_weights(attn, self.backend)
        table = attn.freqs_cis
        values = (
            (self.query_weight, qw),
            (self.query_scale, qs),
            (self.output_weight, ow),
            (self.output_scale, oscale),
            (self.cos_sin, torch.cat((table.real, table.imag), -1)),
            (
                self.sink,
                torch.nn.functional.pad(
                    attn.attn_sink,
                    (0, self.padded_heads - self.heads),
                    value=float("inf"),
                ),
            ),
        )
        for target, source in values:
            if target.shape != source.shape or target.dtype != source.dtype:
                raise RuntimeError(
                    "mega attention weight geometry changed during reload"
                )
            target.copy_(source)

    @property
    def padded_heads(self):
        return max(64, self.heads)

    @property
    def padded_groups(self):
        return self.padded_heads // _HEADS_PER_GROUP

    def pad_q(self, q):
        if q.shape[-2] == self.padded_heads:
            return q.reshape(-1, self.padded_heads, _HEAD_DIM)
        # Q is chunk-interleaved across local heads. Padding a nominal
        # [heads, 512] dimension would corrupt the layout. Pad the head axis
        # inside every 16-channel chunk, in one CUDA pad/copy operation.
        q = q.reshape(-1, 32, self.heads, 16)
        return torch.nn.functional.pad(
            q, (0, 0, 0, self.padded_heads - self.heads)
        ).view(-1, self.padded_heads, _HEAD_DIM)

    def supports_input(self, x):
        return (
            x.device == self.query_weight.device
            and x.dtype == torch.bfloat16
            and x.ndim in (2, 3)
            and x.numel() > 0
            and not x.requires_grad
        )

    def supports_prefill(self, kv, indices, lengths=None, positions=None):
        rows = indices.shape[0] if indices.ndim else 0
        return (
            kv.device == indices.device == self.query_weight.device
            and kv.dtype == torch.bfloat16
            and kv.ndim == 3
            and kv.shape[1:] == (1, _HEAD_DIM)
            and kv.is_contiguous()
            and indices.dtype == torch.int32
            and (indices.ndim == 2 or (indices.ndim == 3 and indices.shape[1] == 1))
            and indices.shape[-1] > 0
            and indices.shape[-1] % 128 == 0
            and indices.stride(-1) == 1
            and self._vector_supported(lengths, rows)
            and self._vector_supported(positions, rows)
        )

    def _vector_supported(self, value, rows):
        return value is None or (
            value.device == self.query_weight.device
            and value.dtype == torch.int32
            and value.shape == (rows,)
            and value.is_contiguous()
        )

    def supports_pool(self, value, record_bytes):
        return (
            isinstance(value, torch.Tensor)
            and value.device == self.query_weight.device
            and value.dtype == torch.uint8
            and value.ndim == 3
            and value.shape[-1] == record_bytes
            and value.stride(-1) == 1
            and value.stride(-2) == record_bytes
            and value.stride(0) >= value.shape[1] * record_bytes
            and value.stride(0) % record_bytes == 0
            and value.data_ptr() % 16 == 0
        )

    def supports_decode(self, pool, indices, extra_pool=None, extra_indices=None):
        def indices_ok(value):
            return (
                value.device == self.query_weight.device
                and value.dtype == torch.int32
                and value.ndim == 2
                and value.shape[1] > 0
                and value.shape[1] % 128 == 0
                and value.is_contiguous()
            )

        if not self.supports_pool(pool, 528) or not indices_ok(indices):
            return False
        if extra_pool is None:
            return extra_indices is None
        return (
            extra_indices is not None
            and self.supports_pool(extra_pool, 288)
            and indices_ok(extra_indices)
            and extra_indices.shape[0] == indices.shape[0]
        )

    def supports_query(self, qr: torch.Tensor) -> bool:
        return (
            qr.device == self.query_weight.device
            and qr.dtype == torch.bfloat16
            and qr.ndim in (2, 3)
            and qr.shape[-1] == self.query_weight.shape[1]
            and qr.numel() > 0
            and not qr.requires_grad
        )

    def project_q(self, linear, qr, qr_quantized=None, out=None):
        """Emit fused Q layout directly from Wq; no normalization or rotation."""
        import deep_gemm

        quantized, scales = (
            linear._quantize_input(qr) if qr_quantized is None else qr_quantized
        )
        rows = qr.numel() // self.query_weight.shape[1]
        if out is None:
            out = torch.empty(
                (rows, self.heads * _HEAD_DIM), dtype=qr.dtype, device=qr.device
            )
        deep_gemm.fp8_fp4_gemm_nt(
            (quantized.reshape(rows, -1), scales),
            (self.query_weight, self.query_scale),
            out.reshape(rows, self.heads * _HEAD_DIM),
            recipe=(1, 1, 32),
        )
        return out.view(rows, self.heads, _HEAD_DIM)

    def prefill(self, q, kv, indices, positions, topk_length=None):
        """Gathered BF16 KV with the caller's existing sparse/causal indices.

        CP and ragged request boundaries have already been resolved into those
        indices. In particular, query positions are absolute request positions,
        not a chunk-local arange. Invalid (-1) indices and per-row lengths are
        passed through to FlashMLA unchanged.
        """
        output, scales, _, _ = self.backend.fused_norm_rope_attn_rope_cast_fwd(
            self.pad_q(q),
            kv,
            indices.unsqueeze(1) if indices.ndim == 2 else indices,
            self.softmax_scale,
            _HEAD_DIM,
            self.sink,
            topk_length,
            False,
            0.0,
            positions,
            False,
            _ROPE_DIM,
            self.cos_sin,
            self.padded_groups,
            32,
            True,
            True,
            True,
        )
        return output, scales

    def decode(
        self,
        q,
        swa_pool,
        swa_indices,
        positions,
        *,
        swa_lengths=None,
        extra_pool=None,
        extra_indices=None,
        extra_lengths=None,
    ):
        """Joint softmax over native 528-byte SWA and optional 288-byte FP4.

        Pool strides, including whole-token page padding, are preserved. Slot IDs are
        physical ``page * entries_per_page + offset``; no block translation or
        host reads are introduced here. S>1 target verification is flattened in
        the same token order as positions and per-token SWA/global slot rows.
        """
        q = self.pad_q(q)
        rows = q.shape[0]
        output, scales, _ = self.backend.fused_norm_rope_attn_rope_cast_decode(
            q,
            swa_pool.unsqueeze(2),
            swa_indices.view(rows, -1),
            self.softmax_scale,
            _HEAD_DIM,
            self.sink,
            swa_lengths,
            None if extra_pool is None else extra_pool.unsqueeze(2),
            None if extra_indices is None else extra_indices.view(rows, -1),
            extra_lengths,
            False,
            0.0,
            positions,
            False,
            _ROPE_DIM,
            self.cos_sin,
            self.padded_groups,
            32,
            True,
            True,
            True,
        )
        return output, scales

    def project_output(self, pair, wo_b, out=None):
        """Consume the kernel's interleaved FP8 result without inverse RoPE."""
        import deep_gemm

        data, scales = pair
        if self.groups != self.padded_groups:
            pair = (data[:, : self.groups], scales[:, : self.groups])
        rows = data.shape[0]
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_wo_a_quant
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        if type(wo_b) is V41MXFP8Linear:
            quantized = _v41_wo_a_quant.try_grouped_quant(
                pair, (self.output_weight, self.output_scale)
            )
            if quantized is not None:
                return wo_b.forward_quantized(*quantized, out=out)
        rank = self.output_weight.shape[1]
        projected = torch.empty(
            (rows, self.groups, rank), dtype=torch.bfloat16, device=data.device
        )
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            pair,
            (self.output_weight, self.output_scale),
            projected,
            recipe=(1, 1, 32),
        )
        return wo_b(projected.view(rows, self.groups * rank), out=out)


def _permuted_weights(attn, backend):
    # The native permuter only accepts native head counts in some FlashMLA
    # versions. This equivalent startup gather also handles TP2/TP4/TP8.
    permutation = (
        torch.arange(attn.n_heads * _HEAD_DIM, device=attn.wq_b.weight.device)
        .view(attn.n_heads, 32, 16)
        .permute(1, 0, 2)
        .reshape(-1)
    )
    query_weight = attn.wq_b.weight.view(torch.uint8).index_select(0, permutation)
    query_weight = query_weight.view(attn.wq_b.weight.dtype)
    query_scale = attn.wq_b.weight_scales.T.index_select(1, permutation).contiguous().T
    output_weight, output_scale = backend.permute_wv_proj(
        attn._wo_a_stk_w, attn._wo_a_stk_s, _HEADS_PER_GROUP, _HEAD_DIM
    )
    return query_weight, query_scale, output_weight, output_scale


def prepare(attn, *, shared=None):
    """Create a complete optional execution path before any measured forward.

    Keep the original linears untouched so disabling mega attention, unsupported
    input dtypes, and other execution paths continue using their original layouts.
    RoPE tables with the same storage are converted once per model, not per layer.
    """
    if not is_supported(attn):
        return None
    backend = _backend()
    query_weight, query_scale, output_weight, output_scale = _permuted_weights(
        attn, backend
    )
    table = attn.freqs_cis
    cache = {} if shared is None else shared.setdefault("mega_rope_tables", {})
    key = (table.data_ptr(), tuple(table.shape), table.device)
    entry = cache.get(key)
    if entry is None:
        cos_sin = torch.cat((table.real, table.imag), dim=-1).contiguous()
        # Retain the source tensor to prevent a recycled pointer hitting cache.
        cache[key] = (table, cos_sin)
    else:
        _, cos_sin = entry
    return MegaAttention(
        backend,
        query_weight,
        query_scale,
        output_weight,
        output_scale,
        cos_sin,
        torch.nn.functional.pad(
            attn.attn_sink,
            (0, max(64, attn.n_heads) - attn.n_heads),
            value=float("inf"),
        ),
        float(attn.softmax_scale),
        attn.n_heads,
        attn.n_groups,
    )


def warmup(attn, *, rows, decode, swa_entries=32, global_entries=128):
    """Run the complete optional chain using exclusively private KV storage.

    The caller deduplicates shapes and owns synchronization/warmup timing.
    ``swa_entries`` must match the serving page size to prepare the Triton
    insert specialization as well as the precompiled FlashMLA module.
    """
    state = getattr(attn, "_mega_attention", None)
    if state is None or rows <= 0:
        return False
    device = state.query_weight.device
    qr = torch.zeros(rows, attn.wq_b.K, dtype=torch.bfloat16, device=device)
    positions = torch.zeros(rows, dtype=torch.int32, device=device)
    q = state.project_q(attn.wq_b, qr)
    width = attn.window_size + (attn.index_topk if attn.compress_ratio else 0)
    if not decode:
        kv = torch.zeros(128, 1, 512, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(rows, width, dtype=torch.int32, device=device)
        lengths = torch.ones(rows, dtype=torch.int32, device=device)
        pair = state.prefill(q, kv, indices, positions, lengths)
    else:
        from ._v41_qkv_rope_cache_triton import try_fused_qkv_rope_cache

        swa_pool = torch.zeros(1, swa_entries, 528, dtype=torch.uint8, device=device)
        swa_pool.view(-1)[swa_entries * 512 :].fill_(127)
        swa_indices = torch.zeros(
            rows, attn.window_size, dtype=torch.int32, device=device
        )
        # Writes are masked, and even an accidental valid write cannot touch
        # model cache: this pool is an allocation local to startup warmup.
        q_source = q.view(rows, 1, state.heads, 512)
        position_long = positions.long()
        selected_freqs = attn.freqs_cis.index_select(0, position_long).contiguous()
        # Production may retain KV as a strided view of the merged Q/KV GEMM.
        raw_kv = torch.zeros(
            rows, 1, attn.wq_b.K + 512, dtype=torch.bfloat16, device=device
        )[..., :512]
        result = None
        for slot_dtype in (torch.int32, torch.int64):
            result = try_fused_qkv_rope_cache(
                q_source,
                raw_kv,
                attn.kv_norm,
                position_long,
                attn.freqs_cis,
                swa_pool,
                torch.full((rows,), -1, dtype=slot_dtype, device=device),
                eps=attn.eps,
                freqs_cis=selected_freqs,
                apply_q_rope=False,
                pad_query_heads=state.padded_heads,
            )
        if result is not None:
            q = result.q
        extra_pool = extra_indices = None
        if attn.compress_ratio:
            extra_pool = torch.zeros(
                1, global_entries, 288, dtype=torch.uint8, device=device
            )
            extra_pool.view(-1)[global_entries * 256 :].fill_(56)
            extra_indices = torch.zeros(
                rows, attn.index_topk, dtype=torch.int32, device=device
            )
        pair = state.decode(
            q,
            swa_pool,
            swa_indices,
            positions,
            extra_pool=extra_pool,
            extra_indices=extra_indices,
        )
    state.project_output(pair, attn.wo_b)
    return True
