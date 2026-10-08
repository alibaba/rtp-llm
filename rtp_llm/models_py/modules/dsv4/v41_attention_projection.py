"""TP1 V4.1 query/index and grouped output projections for CEP/DP ranks."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import (
    V41MXFP8Linear,
    _decode_scales,
    _scale_bytes,
)
from rtp_llm.models_py.modules.dsv4.v41_projection import (
    V41KVProjection,
    _inputs,
    _rms_norm,
    _rotate,
    _tensor,
)


@dataclass(frozen=True)
class V41QueryFrame:
    query: torch.Tensor
    index_query: torch.Tensor | None
    index_weights: torch.Tensor | None
    frequencies: torch.Tensor
    valid_rows: torch.Tensor


def _grouped_bf16(weight, scales):
    """Match checkpoint convert.py, bounding temporary FP32 expansion to 128 rows."""
    if (
        weight.ndim != 2
        or min(weight.shape) <= 0
        or weight.shape[0] % 32
        or weight.shape[1] != 4096
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or weight.requires_grad
    ):
        raise ValueError("V4.1 grouped output weight requires E4M3 [8*rank,4096]")
    raw = _scale_bytes(scales)
    _tensor(
        raw,
        (weight.shape[0] // 32, 128),
        torch.uint8,
        weight.device,
        "grouped output scales",
    )
    if weight.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("V4.1 grouped output conversion must precede graph capture")
    result = torch.empty_like(weight, dtype=torch.bfloat16)
    for begin in range(0, weight.shape[0], 128):
        scale = _decode_scales(raw[begin // 32 : (begin + 128) // 32])
        expanded = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
        result[begin : begin + 128].copy_(
            (weight[begin : begin + 128].float() * expanded).bfloat16()
        )
    return result


class V41AttentionProjection:
    """Full 64-head TP1 attention, 32-head indexer and eight output groups.

    Query frames own their position/mask snapshots so output rotation cannot
    accidentally use the next scheduler step's metadata. Hidden rows already
    passed the layer attention norm. Index weights remain unscaled BF16; the
    indexer applies the reference 1/64 exactly once.
    """

    def __init__(
        self,
        sources: V41LayerSources,
        wq_a: V41MXFP8Linear,
        wq_b: V41MXFP8Linear,
        q_norm: torch.Tensor,
        wo_a: torch.Tensor,
        wo_b: V41MXFP8Linear,
        sinks: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        index_wq_b: V41MXFP8Linear | None = None,
        index_weights: torch.Tensor | None = None,
        max_rows: int = 128,
    ):
        if not all(isinstance(op, V41MXFP8Linear) for op in (wq_a, wq_b, wo_b)):
            raise ValueError("V4.1 projections require explicit group-32 linears")
        device = wq_a.weight.device
        if (
            type(max_rows) is not int
            or not 1 <= max_rows <= 128
            or wq_b.N != 64 * 512
            or wq_b.K != wq_a.N
            or wo_b.N != wq_a.K
            or wo_b.K % 8
            or any(
                op.backend != wq_a.backend or op.weight.device != device
                for op in (wq_b, wo_b)
            )
        ):
            raise ValueError("V4.1 TP1 query/output geometry or backend mismatch")
        _tensor(q_norm, (wq_a.N,), torch.bfloat16, device, "query norm")
        _tensor(wo_a, (wo_b.K, 4096), torch.bfloat16, device, "grouped output weight")
        _tensor(sinks, (64,), torch.float32, device, "attention sinks")
        if frequencies.ndim != 2 or not 1 <= frequencies.shape[0] <= 32768:
            raise ValueError("V4.1 RoPE table exceeds the bounded 32K profile")
        _tensor(frequencies, (len(frequencies), 32), torch.complex64, device, "RoPE")
        if sources.owns_index:
            if (
                not isinstance(index_wq_b, V41MXFP8Linear)
                or index_wq_b.N != 32 * 128
                or index_wq_b.K != wq_a.N
                or index_wq_b.backend != wq_a.backend
                or index_wq_b.weight.device != device
            ):
                raise ValueError("V4.1 index source requires group-32 query projection")
            _tensor(
                index_weights,
                (32, wq_a.K),
                torch.bfloat16,
                device,
                "index head weights",
            )
        elif index_wq_b is not None or index_weights is not None:
            raise ValueError(
                "V4.1 non-source must reuse selections, not project index queries"
            )
        self.sources, self.wq_a, self.wq_b, self.q_norm = sources, wq_a, wq_b, q_norm
        self.wo_a, self.wo_b, self.sinks = wo_a.view(8, wo_b.K // 8, 4096), wo_b, sinks
        self.frequencies, self.max_rows = frequencies, max_rows
        self.index_wq_b, self.index_weights = index_wq_b, index_weights

    def prepare(self, hidden, positions, valid_rows):
        x, safe = _inputs(
            hidden, positions, valid_rows, self.wq_a, self.frequencies, self.max_rows
        )
        rows = x.shape[0]
        frequencies = self.frequencies[safe]
        qr = _rms_norm(self.wq_a(x), self.q_norm)
        query = _rotate(self.wq_b(qr).view(rows, 64, 512), frequencies)
        index_query, index_weights = None, None
        if self.index_wq_b is not None:
            index_query = _rotate(self.index_wq_b(qr).view(rows, 32, 128), frequencies)
            index_weights = F.linear(x, self.index_weights)
        return V41QueryFrame(
            query, index_query, index_weights, frequencies, valid_rows.clone()
        )

    def output(self, attention, frame: V41QueryFrame):
        rows = frame.valid_rows.numel()
        if rows > self.max_rows:
            raise ValueError("V4.1 attention output exceeds its token row bound")
        device = self.wq_a.weight.device
        _tensor(frame.valid_rows, (rows,), torch.bool, device, "query frame mask")
        _tensor(
            frame.frequencies, (rows, 32), torch.complex64, device, "query frame RoPE"
        )
        if (
            attention.shape != (rows, 64, 512)
            or attention.dtype != torch.bfloat16
            or attention.device != device
            or not attention.is_contiguous()
            or attention.requires_grad
        ):
            raise ValueError("V4.1 attention output must be contiguous BF16 [R,64,512]")
        if device.type == "cpu" and not bool(
            torch.isfinite(attention[frame.valid_rows]).all()
        ):
            raise ValueError("V4.1 live attention outputs must be finite")
        clean = torch.where(frame.valid_rows[:, None, None], attention, 0)
        unrotated = _rotate(clean, frame.frequencies, inverse=True).view(rows, 8, 4096)
        grouped = (
            torch.einsum("rgd,god->rgo", unrotated, self.wo_a)
            .reshape(rows, self.wo_b.K)
            .contiguous()
        )
        return self.wo_b(grouped)


def build_v41_projections(
    sources,
    weights,
    frequencies,
    *,
    backend="flashinfer",
    max_rows=128,
    kv_max_rows=None,
):
    """Bind loader W names without consuming/mutating its dictionary on failure.

    The caller releases loader references only after complete layer admission.
    Conversion adds BF16 grouped-output and (for ratio 2) FP32 compressor weights.
    This entry supports replicated TP1 weights used by CEP4PP2 and DP8EP8.
    """
    from rtp_llm.utils.model_weight import W

    def dense(weight, scale):
        return V41MXFP8Linear(weights[weight], weights[scale], backend=backend)

    kv = V41KVProjection(
        sources,
        dense(W.v4_attn_wkv_w, W.v4_attn_wkv_s),
        weights[W.v4_attn_kv_norm],
        frequencies,
        compressor_weight=weights.get(W.v4_compressor_wkv),
        compressor_norm=weights.get(W.v4_compressor_norm),
        gate_weight=weights.get(W.v4_compressor_wgate),
        index_weight=weights.get(W.v41_indexer_wk),
        index_norm=weights.get(W.v41_indexer_k_norm),
        max_rows=max_rows if kv_max_rows is None else kv_max_rows,
    )
    index_q = (
        dense(W.v4_indexer_wq_b_w, W.v4_indexer_wq_b_s) if sources.owns_index else None
    )
    attention = V41AttentionProjection(
        sources,
        dense(W.v4_attn_wq_a_w, W.v4_attn_wq_a_s),
        dense(W.v4_attn_wq_b_w, W.v4_attn_wq_b_s),
        weights[W.v4_attn_q_norm],
        _grouped_bf16(weights[W.v4_attn_wo_a_w], weights[W.v4_attn_wo_a_s]),
        dense(W.v4_attn_wo_b_w, W.v4_attn_wo_b_s),
        weights[W.v4_attn_sink],
        frequencies,
        index_wq_b=index_q,
        index_weights=weights.get(W.v4_indexer_weights_proj_w),
        max_rows=max_rows,
    )
    return attention, kv
