"""V4.1 grouped FP8xFP4 expert arithmetic with an explicit SM120 provider.

Callers own routing and transport. Masked rows occupy an expert-local prefix;
contiguous rows occupy aligned expert segments. Returned BF16 rows already
include route weights and must be combined in FP32 without reweighting.
"""

from contextlib import nullcontext

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv4.v41_expert import (
    V41_EXPERT_RECIPE,
    _check_limit,
    _check_rows,
    pack_group32_scales,
    quantize_swiglu,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import (
    GROUP_SIZE,
    _decode_scales,
    _require_sm120,
    _scale_bytes,
    quantize_group32,
)

GROUPED_ALIGNMENT = 128


def _fp4_bytes(weight):
    if weight.dtype in (torch.uint8, torch.int8) or weight.dtype == getattr(
        torch, "float4_e2m1fn_x2", None
    ):
        return weight.view(torch.uint8)
    raise ValueError("V4.1 FP4 weights require packed E2M1 bytes")


def _fp8_fp4_eager(values, scales, weight, weight_scales):
    """CPU grouped-equation oracle, bounded to 32x128 output tiles."""
    rows, dim = values.shape
    width = weight.shape[0]
    a_scale, b_scale = _decode_scales(scales), _decode_scales(weight_scales)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    result = torch.empty((rows, width), dtype=torch.bfloat16)
    for m in range(0, rows, 32):
        for n in range(0, width, 128):
            accum = torch.zeros((min(32, rows - m), min(128, width - n)))
            for group in range(dim // GROUP_SIZE):
                raw = weight[n : n + 128, group * 16 : (group + 1) * 16]
                codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
                decoded = levels[(codes & 7).long()]
                decoded = torch.where((codes & 8) != 0, -decoded, decoded)
                a = values[m : m + 32, group * 32 : (group + 1) * 32].float()
                accum.add_(
                    (a @ decoded.T)
                    * a_scale[m : m + 32, group, None]
                    * b_scale[None, n : n + 128, group]
                )
            result[m : m + 32, n : n + 128] = accum.bfloat16()
    return result


class V41RoutedExperts(nn.Module):
    """Gate-first FP4 W13/W2 stacks on their immutable final device.

    ``backend='eager'`` is CPU diagnostics only. ``backend='deepgemm'`` requires
    an explicitly installed SM120-capable DeepGEMM provider. Symbol presence is
    not hardware qualification; provider failures propagate without fallback.
    No shared global scratch or host counts reads occur in the masked path.
    Warm each layout/capacity/expected-M outside capture before graph use.
    """

    recipe = V41_EXPERT_RECIPE

    def __init__(self, expert_weights, *, swiglu_limit=10.0, backend="deepgemm"):
        super().__init__()
        _check_limit(swiglu_limit)
        w13, w2 = expert_weights["w13_w"], expert_weights["w2_w"]
        s13, s2 = expert_weights["w13_s"], expert_weights["w2_s"]
        if (
            w13.ndim != 3
            or w2.ndim != 3
            or min(w13.shape) <= 0
            or min(w2.shape) <= 0
            or w13.shape[0] != w2.shape[0]
            or w13.shape[1] != w2.shape[2] * 4
            or w13.shape[2] * 2 != w2.shape[1]
            or w2.shape[1] % GROUP_SIZE
            or w2.shape[2] * 2 % GROUP_SIZE
        ):
            raise ValueError("V4.1 routed gate/up and down geometry disagree")
        self.experts, self.dim, packed_inter = w2.shape
        self.inter_dim = packed_inter * 2
        self.device = w13.device
        if any(
            tensor.device != self.device or not tensor.is_contiguous()
            for tensor in (w13, w2, s13, s2)
        ):
            raise ValueError("V4.1 routed weights require contiguous colocated tensors")
        self.w13, self.w2 = _fp4_bytes(w13), _fp4_bytes(w2)
        self.s13, self.s2 = _scale_bytes(s13), _scale_bytes(s2)
        if self.s13.shape != (
            self.experts,
            2 * self.inter_dim,
            self.dim // 32,
        ) or self.s2.shape != (self.experts, self.dim, self.inter_dim // 32):
            raise ValueError(
                "V4.1 routed scales require per-output-channel group-32 UE8M0"
            )
        if backend not in ("eager", "deepgemm"):
            raise ValueError("Unknown V4.1 routed backend")
        self.backend, self.swiglu_limit = backend, swiglu_limit
        self._warmed = set()
        self._packed_s13 = self._packed_s2 = None
        if backend == "eager":
            if self.device.type != "cpu":
                raise ValueError("V4.1 eager routed experts are CPU diagnostics only")
            return
        _require_sm120(self.device)
        if (
            min(self.dim, self.inter_dim) < 128
            or self.dim % 128
            or self.inter_dim % 128
        ):
            raise ValueError("SM120 grouped FP4 requires D/I multiples of 128")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "V4.1 routed weight preparation must precede capture"
                )
            from deep_gemm import (
                m_grouped_fp8_fp4_gemm_nt_contiguous,
                m_grouped_fp8_fp4_gemm_nt_masked,
            )

            self._contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
            self._masked = m_grouped_fp8_fp4_gemm_nt_masked
            self._packed_s13 = pack_group32_scales(self.s13, column_major=True)
            self._packed_s2 = pack_group32_scales(self.s2, column_major=True)

    def _validate_input(self, x, row_shape, route_weights, out, metadata):
        if (
            x.shape != (*row_shape, self.dim)
            or x.device != self.device
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or (torch.is_grad_enabled() and x.requires_grad)
        ):
            raise ValueError(
                "V4.1 routed input must be contiguous inference BF16 on the weight device"
            )
        if out is not None:
            if (
                out.shape != x.shape
                or out.dtype != x.dtype
                or out.device != x.device
                or not out.is_contiguous()
                or out.requires_grad
            ):
                raise ValueError(
                    "V4.1 routed output must match contiguous BF16 input metadata"
                )
            protected = (
                x,
                route_weights,
                metadata,
                self.w13,
                self.w2,
                self.s13,
                self.s2,
                self._packed_s13,
                self._packed_s2,
            )
            if out.numel() and any(
                t is not None
                and t.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
                for t in protected
            ):
                raise ValueError(
                    "V4.1 routed output must not share input or weight storage"
                )

    def _run(self, x, valid, route_weights, key, gemm, *, out):
        active = _check_rows(valid.shape, self.device, valid, route_weights)
        context = (
            nullcontext() if self.backend == "eager" else torch.cuda.device(self.device)
        )
        with context:
            capturing = (
                self.backend != "eager" and torch.cuda.is_current_stream_capturing()
            )
            if not x.numel():
                return torch.empty_like(x) if out is None else out
            if capturing and key not in self._warmed:
                raise RuntimeError(
                    "V4.1 routed layout/capacity must be warmed before capture"
                )
            safe_x = torch.where(active[..., None], x, 0)
            quant_backend = "eager" if self.backend == "eager" else "triton"
            values, scales = quantize_group32(safe_x, backend=quant_backend)
            scales = scales.reshape(*valid.shape, self.dim // 32)
            scales = torch.where(active[..., None], scales, 0)
            gate_up = gemm(values, scales, self.w13, self.s13, self._packed_s13)
            gate, up = gate_up.chunk(2, dim=-1)
            values, scales = quantize_swiglu(
                gate,
                up,
                valid,
                route_weights=route_weights,
                clamp_limit=self.swiglu_limit,
                backend=quant_backend,
            )
            scales = scales.reshape(*valid.shape, self.inter_dim // 32)
            result = gemm(values, scales, self.w2, self.s2, self._packed_s2)
            result = torch.where(active[..., None], result, 0)
            if not capturing:
                self._warmed.add(key)
            return result if out is None else out.copy_(result)

    def forward_masked(self, x, counts, route_weights, *, expected_m=None, out=None):
        """Execute [E,M,D] with active prefixes ``counts[E]`` and route weights.

        CUDA counts must be in [0,M]; a stream-ordered device assertion checks
        them before provider calls. Zero-weight rows inside prefixes are allowed.
        ``expected_m`` is an explicit scheduling hint, not an occupancy limit.
        """
        if x.ndim != 3 or x.shape[0] != self.experts:
            raise ValueError("V4.1 masked experts require [E,M,D]")
        capacity = x.shape[1]
        self._validate_input(x, (self.experts, capacity), route_weights, out, counts)
        if (
            counts.shape != (self.experts,)
            or counts.dtype != torch.int32
            or counts.device != self.device
            or not counts.is_contiguous()
        ):
            raise ValueError("V4.1 masked counts require contiguous device int32 [E]")
        expected_m = capacity if expected_m is None else expected_m
        if (
            type(expected_m) is not int
            or not min(1, capacity) <= expected_m <= capacity
        ):
            raise ValueError("V4.1 masked expected-M must be within row capacity")
        if self.backend != "eager" and capacity % GROUPED_ALIGNMENT:
            raise ValueError("V4.1 masked capacity must be 128-aligned")
        legal_counts = ((counts >= 0) & (counts <= capacity)).all()
        if self.backend == "eager":
            if not bool(legal_counts):
                raise ValueError("V4.1 masked counts exceed row capacity")
        else:
            torch._assert_async(legal_counts, "V4.1 masked counts exceed row capacity")
        valid = torch.arange(capacity, device=self.device)[None, :] < counts[:, None]

        def gemm(values, scales, weight, raw_scales, packed_scales):
            result = torch.zeros(
                (*valid.shape, weight.shape[1]),
                device=self.device,
                dtype=torch.bfloat16,
            )
            if self.backend == "eager":
                for expert in range(self.experts):
                    rows = int(counts[expert])
                    result[expert, :rows] = _fp8_fp4_eager(
                        values[expert, :rows],
                        scales[expert, :rows],
                        weight[expert],
                        raw_scales[expert],
                    )
            else:
                self._masked(
                    (values, pack_group32_scales(scales, column_major=True)),
                    (weight.view(torch.int8), packed_scales),
                    result,
                    counts,
                    expected_m,
                    recipe_a=(1, 32),
                    recipe_b=(1, 32),
                    disable_ue8m0_cast=False,
                )
            return result

        return self._run(
            x, valid, route_weights, ("masked", capacity, expected_m), gemm, out=out
        )

    def forward_contiguous(
        self, x, rows_per_expert, valid_rows, route_weights, *, out=None
    ):
        """Execute aligned expert segments, including explicit padding masks.

        Segment sizes are host admission metadata, not read from device counts.
        Every segment must be 128-aligned. Zero-sized experts are valid.
        """
        if (
            type(rows_per_expert) is not tuple
            or len(rows_per_expert) != self.experts
            or any(
                type(n) is not int or n < 0 or n % GROUPED_ALIGNMENT
                for n in rows_per_expert
            )
        ):
            raise ValueError(
                "V4.1 contiguous segments require a tuple of 128-aligned expert row counts"
            )
        rows = sum(rows_per_expert)
        self._validate_input(x, (rows,), route_weights, out, valid_rows)
        _check_rows((rows,), self.device, valid_rows, route_weights)

        def gemm(values, scales, weight, raw_scales, packed_scales):
            result = torch.empty(
                (rows, weight.shape[1]), device=self.device, dtype=torch.bfloat16
            )
            if self.backend == "eager":
                start = 0
                for expert, count in enumerate(rows_per_expert):
                    result[start : start + count] = _fp8_fp4_eager(
                        values[start : start + count],
                        scales[start : start + count],
                        weight[expert],
                        raw_scales[expert],
                    )
                    start += count
            else:
                layout = torch.cat(
                    [
                        torch.full((n,), e, dtype=torch.int32, device=self.device)
                        for e, n in enumerate(rows_per_expert)
                    ]
                )
                self._contiguous(
                    (values, pack_group32_scales(scales, column_major=True)),
                    (weight.view(torch.int8), packed_scales),
                    result,
                    layout,
                    recipe_a=(1, 32),
                    recipe_b=(1, 32),
                    disable_ue8m0_cast=False,
                )
            return result

        return self._run(
            x, valid_rows, route_weights, ("contiguous", rows_per_expert), gemm, out=out
        )
