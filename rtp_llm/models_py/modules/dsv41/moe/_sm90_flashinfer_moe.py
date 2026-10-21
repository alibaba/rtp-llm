"""SM90 local experts using the *opensource* FlashInfer CUTLASS ABI.

This is the SGLang W4A8 execution order, not a bitwise replacement for RTP's
Humming implementation: routing weights multiply GEMM2 outputs, rather than
SwiGLU outputs before activation quantization. FlashInfer returns BF16; asking
for FP32 below only widens that result. Qualification must retain both the
independent mathematical reference and the original RTP numerical comparison.

Reference: SGLang's mxfp4_flashinfer_cutlass_moe.py and flashinfer_cutlass.py,
as installed alongside FlashInfer 0.7.1. No upstream kernels are copied here.
Only flashinfer_opensource is imported; the internal flashinfer is untouched.
One instance/workspace belongs to one serialized CUDA stream execution context.
"""

from contextlib import nullcontext
from typing import Dict, Optional

import torch
from torch import nn

from ._sm90_flashinfer_glue import finish_flashinfer_output, prepare_flashinfer_inputs

# Forty layers execute serially on one stream; do not retain forty copies of
# potentially large CUTLASS scratch. Never resize a buffer referenced by a graph.
_STREAM_WORKSPACES = {}
# Eager prefill sees route-dependent row counts at every layer. Keep only the
# largest scratch per stream, separate from storage pinned by captured graphs.
_EAGER_STREAM_WORKSPACES = {}


def _eager_workspace(device, nbytes):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("eager workspace cannot be acquired during graph capture")
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    workspace = _EAGER_STREAM_WORKSPACES.get(key)
    if workspace is not None and workspace.numel() >= nbytes:
        return workspace
    # No layer owns this tensor. Drop the old slot AND local reference before
    # allocating its replacement, avoiding an old+new scratch peak. All uses
    # are serialized on this same stream, so allocator reuse is stream ordered.
    _EAGER_STREAM_WORKSPACES.pop(key, None)
    del workspace
    workspace = torch.empty(nbytes, device=device, dtype=torch.uint8)
    _EAGER_STREAM_WORKSPACES[key] = workspace
    return workspace


def validate_lossless_mxfp4_scales(scales: torch.Tensor) -> None:
    """Reject upstream's lossy exponent-range clamp before weight preparation.

    This is the existing RTP native-A8 weight contract, independent of any
    Humming package. Byte zero has special E8M0 semantics not implemented by
    the fused FP4-to-FP8 normalization; expert-local exponent span must be at
    most 11 steps (this is not an 11-bit scale encoding).
    """
    if scales.dtype != torch.float8_e8m0fnu or scales.ndim != 3 or not scales.numel():
        raise ValueError("expected nonempty [E,N,K/32] native E8M0 scales")
    raw = scales.contiguous().view(torch.uint8).flatten(1).to(torch.int16)
    if bool(((raw < 1) | (raw > 252)).any()):
        raise ValueError("W4A8 requires E8M0 bytes in [1,252]")
    if bool(((raw.amax(1) - raw.amin(1)) > 11).any()):
        raise ValueError(
            "W4A8 requires per-expert exponent span <=11; refusing lossy clipping"
        )


def _prepare_experts_in_chunks(prep, weight, scale):
    # Upstream expands FP4 codes and scale offsets to int64. Preparing all
    # 96 EP4 experts together needs two 16.88 GiB temporaries on H20.
    # Scale reduction and interleave are independent for each expert.
    if weight.shape[0] <= 8:
        return prep(weight, scale)
    outputs = None
    for start in range(0, weight.shape[0], 8):
        end = min(start + 8, weight.shape[0])
        chunk = prep(weight[start:end], scale[start:end])
        if outputs is None:
            outputs = tuple(
                value.new_empty((weight.shape[0], *value.shape[1:])) for value in chunk
            )
        for output, value in zip(outputs, chunk):
            output[start:end].copy_(value)
        del chunk
    return outputs


class Sm90FlashInferLocalExperts(nn.Module):
    """Local expert IDs are [0, E); -1/nonlocal IDs are masked, not routed.

    Call setup_weights and prepare for each (rows, topk) before graph capture.
    Routing values and activations remain device-dynamic on every replay.
    This class does not dispatch, communicate, or apply shared experts.
    """

    route_weight_position = "gemm2_finalize_after_activation_quantization"
    activation_quantization = "flashinfer_sm90_humming_dynamic_row_fp8"
    native_output_dtype = torch.bfloat16

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.n_routed_experts // cfg.ep_size
        self.hidden_size = cfg.dim
        self.intermediate_size = cfg.moe_inter_dim
        if cfg.n_routed_experts % cfg.ep_size:
            raise ValueError("expert count must be divisible by EP size")
        if self.num_experts < 1 or any(
            d <= 0 or d % 128 for d in (self.hidden_size, self.intermediate_size)
        ):
            raise ValueError("SM90 CUTLASS requires positive 128-aligned dimensions")
        self._workspaces = {}
        self._ready = False

    def setup_weights(self, layer_weights: Dict) -> None:
        from flashinfer_opensource.fused_moe import (
            ActivationType,
            cutlass_fused_moe,
            cutlass_fused_moe_workspace_size,
            preprocess_moe_weights_for_sm90_mixed_gemm_humming,
        )

        from rtp_llm.utils.model_weight import W

        if self._ready:
            raise RuntimeError("setup_weights may only be called once")
        keys = (
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
        )
        w1, s1, w3, s3, w2, s2 = (layer_weights[k] for k in keys)
        e, k, i = self.num_experts, self.hidden_size, self.intermediate_size
        device = w1.device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("opensource FlashInfer W4A8 adapter requires SM90")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("weight preparation must happen outside graph capture")
        for tensor, shape, scale in (
            (w1, (e, i, k // 2), False),
            (w3, (e, i, k // 2), False),
            (w2, (e, k, i // 2), False),
            (s1, (e, i, k // 32), True),
            (s3, (e, i, k // 32), True),
            (s2, (e, k, i // 32), True),
        ):
            if tensor.device != device or tuple(tensor.shape) != shape:
                raise ValueError(f"invalid MXFP4 tensor geometry; expected {shape}")
            expected = (torch.float8_e8m0fnu,) if scale else (torch.int8, torch.uint8)
            if tensor.dtype not in expected:
                raise TypeError(f"expected native {expected}, got {tensor.dtype}")

        # FlashInfer consumes contiguous [up; gate], unlike RTP's [gate; up].
        # The real loader produces int8; this is a byte reinterpretation, not
        # signed numeric conversion of the two packed FP4 nibbles.
        w31 = torch.cat(
            (w3.view(torch.uint8), w1.view(torch.uint8)), dim=1
        ).contiguous()
        s31 = torch.cat((s3.view(torch.uint8), s1.view(torch.uint8)), dim=1)
        validate_lossless_mxfp4_scales(s31.view(torch.float8_e8m0fnu))
        validate_lossless_mxfp4_scales(s2)
        prep = preprocess_moe_weights_for_sm90_mixed_gemm_humming
        # Model construction runs under torch.device("meta"). Upstream lazily
        # caches a CPU rewrite LUT using a factory without an explicit device;
        # allowing that factory to inherit meta permanently poisons its cache.
        # Tensor-derived allocations and explicit CUDA devices remain on CUDA.
        with torch.device("cpu"):
            fw1, fs1, residual1 = _prepare_experts_in_chunks(
                prep, w31, s31.contiguous()
            )
            fw2, fs2, residual2 = _prepare_experts_in_chunks(
                prep,
                w2.view(torch.uint8).contiguous(),
                s2.view(torch.uint8).contiguous(),
            )
        for name, value in (
            ("fc1_weight", fw1),
            ("fc2_weight", fw2),
            ("fc1_scale", fs1.view(torch.int32)),
            ("fc2_scale", fs2.view(torch.int32)),
            ("fc1_residual", (residual1 * 64.0).contiguous()),
            ("fc2_residual", (residual2 * 64.0).contiguous()),
            (
                "fc2_activation_scale",
                torch.ones((), device=device, dtype=torch.float32),
            ),
            (
                "swiglu_limit",
                torch.full(
                    (e,),
                    float(self.cfg.swiglu_limit),
                    device=device,
                    dtype=torch.float32,
                ),
            ),
        ):
            self.register_buffer(name, value)
        self._op = cutlass_fused_moe
        self._workspace_size = cutlass_fused_moe_workspace_size
        self._activation_type = ActivationType.Swiglu
        self._ready = True
        # Consume the original dictionary only after all validation/preparation.
        for key in keys:
            layer_weights.pop(key)

    def _workspace_nbytes(self, num_rows: int, topk: int) -> int:
        return self._workspace_size(
            num_rows,
            self.hidden_size,
            self.intermediate_size,
            self.num_experts,
            topk,
            x_dtype=torch.bfloat16,
            weight_dtype=self.fc1_weight.dtype,
            output_dtype=torch.bfloat16,
            activation_type=self._activation_type,
            use_w4_group_scaling=True,
            use_wfp4afp8_humming=True,
            use_fused_finalize=True,
            device=self.fc1_weight.device,
        )

    def prepare(self, num_rows: int, topk: int, *, tune: bool = False) -> None:
        """Allocate scratch and warm up outside capture; optionally autotune.

        A separate workspace for each exact shape avoids reallocating storage
        already referenced by a captured graph. No host route values are cached.
        """
        if not self._ready:
            raise RuntimeError("setup_weights must precede prepare")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare/tuning cannot run during graph capture")
        if num_rows < 0 or not 1 <= topk <= 8:
            raise ValueError("expected nonnegative rows and topk in [1, 8]")
        if num_rows == 0:
            return
        key = (num_rows, topk)
        if key in self._workspaces:
            return
        device = self.fc1_weight.device
        nbytes = self._workspace_nbytes(num_rows, topk)
        stream = torch.cuda.current_stream(device).cuda_stream
        workspace_key = (device.index, stream, nbytes)
        if workspace_key not in _STREAM_WORKSPACES:
            _STREAM_WORKSPACES[workspace_key] = torch.empty(
                nbytes, device=device, dtype=torch.uint8
            )
        self._workspaces[key] = _STREAM_WORKSPACES[workspace_key]
        x = torch.zeros(
            (num_rows, self.hidden_size), device=device, dtype=torch.bfloat16
        )
        indices = torch.arange(num_rows * topk, device=device).view(num_rows, topk)
        indices = indices.remainder(self.num_experts)
        weights = torch.full(
            (num_rows, topk), 1.0 / topk, device=device, dtype=torch.float32
        )
        context = nullcontext()
        if tune:
            from flashinfer_opensource import autotune

            context = autotune(tune_mode=True)
        try:
            with context:
                self.forward_local(x, weights, indices)
        except Exception:
            del self._workspaces[key]
            raise

    def forward_local(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        output_dtype: torch.dtype = torch.float32,
        *,
        workspace: Optional[torch.Tensor] = None,
        recv_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._ready:
            raise RuntimeError("setup_weights must precede forward_local")
        if output_dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("output dtype must be BF16 or FP32")
        if x.ndim != 2 or x.shape[1] != self.hidden_size:
            raise ValueError("expected [rows, hidden_size] activations")
        if (
            indices.ndim != 2
            or weights.shape != indices.shape
            or indices.shape[0] != x.shape[0]
        ):
            raise ValueError("weights and indices must have shape [rows, topk]")
        if not 1 <= indices.shape[1] <= 8:
            raise ValueError("topk must be in [1, 8]")
        for tensor, dtype in (
            (x, torch.bfloat16),
            (weights, torch.float32),
            (indices, torch.int64),
        ):
            if tensor.dtype != dtype or tensor.device != self.fc1_weight.device:
                raise TypeError("invalid input dtype/device")
            if not tensor.is_contiguous():
                raise ValueError("local expert inputs must be contiguous")
        if x.shape[0] == 0:
            return torch.empty(x.shape, device=x.device, dtype=output_dtype)
        if workspace is None:
            key = (x.shape[0], indices.shape[1])
            if key not in self._workspaces:
                raise RuntimeError(f"prepare{key} is required before forward/capture")
            workspace = self._workspaces[key]
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("explicit eager workspace is not a graph workspace")
            if (
                workspace.device != x.device
                or workspace.dtype != torch.uint8
                or workspace.ndim != 1
                or not workspace.is_contiguous()
            ):
                raise ValueError(
                    "expected contiguous uint8 eager workspace on input device"
                )
            # The upstream op checks capacity against the actual shape before
            # launching. A larger buffer is valid; it does not change tactics.
        # Framework side-stream warmup and capture may differ from the setup
        # stream. Their owner must establish stream dependencies and serialize
        # uses of this scratch, including replay: changing streams alone is not
        # evidence of concurrent use and must not reject legal graph warmup.

        # Nonlocal expert sentinel E is recognized by the upstream routing
        # kernels. Zero invalid weights explicitly: 0 * NaN is not masking.
        safe_x, safe_weights, safe_indices, live_rows = prepare_flashinfer_inputs(
            x, weights, indices, self.num_experts, recv_count=recv_count
        )
        out = torch.empty_like(x)
        self._op(
            input=safe_x,
            token_selected_experts=safe_indices,
            token_final_scales=safe_weights,
            fc1_expert_weights=self.fc1_weight,
            fc2_expert_weights=self.fc2_weight,
            output_dtype=torch.bfloat16,
            quant_scales=[
                self.fc1_scale,
                self.fc1_residual,
                self.fc2_activation_scale,
                self.fc2_scale,
                self.fc2_residual,
            ],
            swiglu_limit=self.swiglu_limit,
            activation_type=self._activation_type,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            use_w4_group_scaling=True,
            use_wfp4afp8_humming=True,
            use_fused_finalize=True,
            output=out,
            tune_max_num_tokens=1 << (x.shape[0] - 1).bit_length(),
            workspace_buffer=workspace,
        )
        # Even a no-work expert launch must not expose uninitialized outputs.
        return finish_flashinfer_output(out, live_rows, output_dtype)

    def _forward_into_buf(self, x, weights, indices, local_start, local_end):
        """Existing DeepEP local-compute ABI, without changing dispatch/combine."""
        if local_end - local_start != self.num_experts:
            raise ValueError("DeepEP local expert range does not match loaded weights")
        if not self._ready:
            raise RuntimeError("setup_weights must precede forward")
        local_indices = torch.where(
            (indices >= local_start) & (indices < local_end),
            indices - local_start,
            -1,
        )
        if x.shape[0] == 0 or torch.cuda.is_current_stream_capturing():
            return self.forward_local(x, weights, local_indices)
        workspace = _eager_workspace(
            x.device, self._workspace_nbytes(x.shape[0], indices.shape[1])
        )
        try:
            # This actual eager invocation performs any lazy JIT initialization.
            # Do not warm up synthetic inputs or retain route-dependent shapes.
            return self.forward_local(x, weights, local_indices, workspace=workspace)
        finally:
            del workspace
