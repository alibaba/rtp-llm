"""Reuse the committed target-feature quantization across DSpark WKV layers.

Only the V4.1 group32 contract is admitted. The original per-layer dispatch
remains the fallback for other linears. Packed scales are column-major with a
TMA-aligned leading dimension; concatenating their rows directly is incorrect.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch

from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear
from rtp_llm.utils.model_weight import W


def sleep_enabled():
    # This baseline does not ship weight_memory_saver; retain the guard for
    # deployments that enable checkpoint sleep/reload.
    try:
        from rtp_llm.model_loader.weight_memory_saver import is_enabled
    except ModuleNotFoundError as error:
        if error.name != "rtp_llm.model_loader.weight_memory_saver":
            raise
        return False
    return is_enabled()


def _supported(linears: Sequence[torch.nn.Module]) -> bool:
    if len(linears) != 3 or any(type(w) is not V41MXFP8Linear for w in linears):
        return False
    first = linears[0]
    return all(
        w.K == first.K
        and w.N == first.N
        and w.K > 0
        and w.N > 0
        and w.K % 128 == 0
        and w.N % 128 == 0
        and w.weight.is_cuda
        and w.weight.device == first.weight.device
        and w.weight.dtype == torch.float8_e4m3fn
        and w.weight.shape == (w.N, w.K)
        and w.weight.is_contiguous()
        and w.weight_scales.device == w.weight.device
        and w.weight_scales.dtype == torch.int32
        and w.weight_scales.shape == (w.N, w.K // 128)
        and w.weight_scales.stride() == (1, w.N)
        for w in linears
    )


class DSparkCommitWKV(torch.nn.Module):
    """Model-owned WKV storage shared by stacked and per-layer projections.

    Checkpoint entries and original linears point into the resident allocation.
    Reload refreshes packed scales and any proposal QKV copies in place before
    graph replay. Replacing linears/geometry requires model and graph rebuild.
    """

    def __init__(
        self,
        linears: Sequence[torch.nn.Module],
        mode: str = "auto",
        *,
        layer_weights=None,
        merged_linears=None,
    ):
        super().__init__()
        if mode not in ("auto", "baseline", "shared", "stacked"):
            raise ValueError(
                "DSV4_DSPARK_COMMIT_WKV must be auto, baseline, shared, or stacked"
            )
        self.mode = mode
        self.linears = tuple(linears)
        self.merged_linears = tuple(merged_linears or (None,) * len(linears))
        self.stacked = None
        first = linears[0]
        measured = (
            first.N == 512
            and first.K == 5120
            and torch.cuda.get_device_capability(first.weight.device) == (10, 3)
        )
        if (mode == "stacked" or (mode == "auto" and measured)) and not sleep_enabled():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("DSpark WKV weights must be prepared before capture")
            stacked = V41MXFP8Linear.__new__(V41MXFP8Linear)
            torch.nn.Module.__init__(stacked)
            stacked.N, stacked.K = sum(w.N for w in linears), first.K
            stacked.register_buffer(
                "weight",
                torch.cat([w.weight.view(torch.uint8) for w in linears]).view(
                    first.weight.dtype
                ),
                persistent=False,
            )
            # Transposed concatenation preserves the logical packed bytes and
            # gives each scale column the NEW total-N leading dimension.
            stacked.register_buffer(
                "weight_scales",
                torch.cat([w.weight_scales.t() for w in linears], dim=1)
                .contiguous()
                .t(),
                persistent=False,
            )
            self.stacked = stacked
            for linear, view in zip(linears, stacked.weight.split(first.N, dim=0)):
                linear.weight = view
        self.sources = tuple((w.weight, w.weight_scales) for w in linears)
        self.addresses = tuple((a.data_ptr(), b.data_ptr()) for a, b in self.sources)
        if layer_weights is not None:
            # Initial values already came from these dictionaries. Redirect the
            # loader to the shared views without copying the weights again.
            for linear, weights in zip(self.linears, layer_weights):
                weights[W.v4_attn_wkv_w] = linear.weight

    @torch.no_grad()
    def reload(self, layer_weights) -> None:
        """Refresh existing graph buffers after checkpoint updates, before replay."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSpark WKV reload must run outside graph capture")
        if not self.matches(self.linears) or len(layer_weights) != len(self.linears):
            raise RuntimeError("DSpark WKV ownership changed; rebuild model and graphs")
        for linear, weights in zip(self.linears, layer_weights):
            source = weights[W.v4_attn_wkv_w]
            scales = weights[W.v4_attn_wkv_s]
            if (
                source.shape != linear.weight.shape
                or source.dtype != linear.weight.dtype
                or source.device != linear.weight.device
                or scales.shape != (linear.N // 32, linear.K // 32)
                or scales.device != linear.weight.device
            ):
                raise RuntimeError(
                    "DSpark WKV checkpoint geometry changed; rebuild model and graphs"
                )
        for index, (linear, weights, merged) in enumerate(
            zip(self.linears, layer_weights, self.merged_linears)
        ):
            linear.weight.copy_(weights[W.v4_attn_wkv_w])
            weights[W.v4_attn_wkv_w] = linear.weight
            linear.weight_scales.copy_(linear.pack_scales(weights[W.v4_attn_wkv_s]))
            if self.stacked is not None:
                begin = index * linear.N
                self.stacked.weight_scales[begin : begin + linear.N].copy_(
                    linear.weight_scales
                )
            # Decode proposal retains its fused QKV allocation. Its Q
            # region is unchanged; update the derived WKV region only.
            if merged is not None:
                merged.weight[-linear.N :].copy_(linear.weight)
                merged.weight_scales[-linear.N :].copy_(linear.weight_scales)

    def matches(self, linears: Sequence[torch.nn.Module]) -> bool:
        return (
            _supported(linears)
            and len(linears) == len(self.linears)
            and all(
                w is old
                and w.weight is source[0]
                and w.weight_scales is source[1]
                and (w.weight.data_ptr(), w.weight_scales.data_ptr()) == address
                for w, old, source, address in zip(
                    linears, self.linears, self.sources, self.addresses
                )
            )
        )

    def project(self, x: torch.Tensor) -> Optional[tuple[torch.Tensor, ...]]:
        if (
            self.mode == "baseline"
            or x.ndim != 2
            or x.dtype != torch.bfloat16
            or x.device != self.linears[0].weight.device
            or x.shape[1] != self.linears[0].K
            or x.shape[0] == 0
            or (self.mode == "auto" and (self.stacked is None or x.shape[0] > 3072))
        ):
            return None
        quantized = self.linears[0]._quantize_input(x)
        if self.stacked is not None:
            output = self.stacked.forward_quantized(*quantized)
            # fused_rmsnorm_rope consumes the row stride directly.
            return tuple(output.split(self.linears[0].N, dim=-1))
        return tuple(w.forward_quantized(*quantized) for w in self.linears)


def prepare_dspark_commit_wkv(
    linears: Sequence[torch.nn.Module],
    mode: Optional[str] = None,
    *,
    layer_weights=None,
    merged_linears=None,
) -> Optional[DSparkCommitWKV]:
    if not _supported(linears):
        return None
    if mode is None:
        mode = os.environ.get("DSV4_DSPARK_COMMIT_WKV", "auto").strip().lower()
    if layer_weights is not None and len(layer_weights) != len(linears):
        raise ValueError("DSpark WKV requires one checkpoint dictionary per layer")
    if merged_linears is not None and len(merged_linears) != len(linears):
        raise ValueError("DSpark WKV requires one proposal projection per layer")
    return DSparkCommitWKV(
        linears, mode, layer_weights=layer_weights, merged_linears=merged_linears
    )
