"""SM90 native block32 DeepGEMM adapter with one resident weight layout."""

import json
import os
import platform
from functools import lru_cache

import torch


def select_sm90_dense_backend(n, k):
    """Choose once per layer; P/D may supply different process-local maps."""
    native_default = (
        "deepgemm" if platform.machine().lower() in ("x86_64", "amd64") else "humming"
    )
    default = os.environ.get("DSV41_SM90_DENSE_BACKEND", native_default).strip().lower()
    raw = os.environ.get("DSV41_SM90_DENSE_BACKEND_MAP", "{}").strip()
    return _backend_policy(default, raw).get(f"{n}x{k}", default)


@lru_cache(maxsize=16)
def _backend_policy(default, raw):
    """Validate a process policy once, including overrides not used by a layer."""
    if default not in ("humming", "deepgemm"):
        raise ValueError("DSV41_SM90_DENSE_BACKEND must be humming or deepgemm")
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("DSV41_SM90_DENSE_BACKEND_MAP must be a JSON object") from exc
    if not isinstance(overrides, dict) or any(
        not isinstance(shape, str)
        or len(shape.split("x")) != 2
        or not all(part.isdigit() and int(part) > 0 for part in shape.split("x"))
        or backend not in ("humming", "deepgemm")
        for shape, backend in overrides.items()
    ):
        raise ValueError("Expected a map of positive NxK shapes to humming/deepgemm")
    return overrides


class Sm90DeepGEMMMXFP8Linear:
    """Native E4M3/UE8M0 data, FP8 WGMMA float accumulator and BF16 output.

    The original FP8 weight remains the executable weight. No Humming repack,
    BF16 weight cache, or activation/weight requantization is retained.
    """

    backend = "deepgemm"
    retains_checkpoint_weight = True

    def __init__(self, weight, scales):
        if (
            not weight.is_cuda
            or torch.cuda.get_device_capability(weight.device) != (9, 0)
            or weight.ndim != 2
            or weight.dtype != torch.float8_e4m3fn
        ):
            raise ValueError("Expected a CUDA SM90 E4M3 [N,K] weight")
        self.N, self.K = weight.shape
        if self.N <= 0 or self.N % 8 or self.K <= 0 or self.K % 32:
            raise ValueError("SM90 block32 requires N%8==0 and K%32==0")
        if (
            scales.dtype != torch.float8_e8m0fnu
            or scales.device != weight.device
            or tuple(scales.shape) != ((self.N + 31) // 32, self.K // 32)
        ):
            raise ValueError("Expected same-device UE8M0 block32 weight scales")
        import deep_gemm

        self._forward = deep_gemm.fp8_gemm_nt
        self.weight = weight.contiguous()
        self.retains_checkpoint_weight = self.weight is weight
        # Exactly represent every UE8M0 value, including byte zero and NaN.
        self.weight_scale = scales.float().contiguous()

    def forward_quantized(self, x_q, x_s, out=None):
        if (
            x_q.dtype != torch.float8_e4m3fn
            or x_q.shape[-1] != self.K
            or x_q.device != self.weight.device
            or x_s.device != x_q.device
        ):
            raise ValueError("Expected same-device prequantized E4M3 activations")
        rows = x_q.numel() // self.K
        if x_s.dtype != torch.int32 or tuple(x_s.shape) != (
            rows,
            (self.K // 32 + 3) // 4,
        ):
            raise ValueError("Expected packed group32 UE8M0 int32 scales")
        from ._sm90_quant_scale_fused import unpack_ue8m0_to_fp32

        scales = unpack_ue8m0_to_fp32(x_s)[:, : self.K // 32]
        return self._forward_with_scales(x_q, scales, out)

    def forward_bf16(self, x, out=None):
        from ._sm90_quant_scale_fused import quantize_g32

        if (
            x.dtype != torch.bfloat16
            or x.device != self.weight.device
            or x.shape[-1] != self.K
            or x.numel() // self.K > 16
        ):
            raise ValueError("Direct G32 scales require same-device BF16 with M<=16")
        q, scales = quantize_g32(x.reshape(-1, self.K).contiguous())
        return self._forward_with_scales(q.view(x.shape), scales, out)

    def _forward_with_scales(self, x_q, scales, out=None):
        rows = x_q.numel() // self.K
        shape = (*x_q.shape[:-1], self.N)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=x_q.device)
        elif (
            tuple(out.shape) != shape
            or out.dtype != torch.bfloat16
            or out.device != x_q.device
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous BF16 with output shape/device")
        if rows:
            # The package converts row-major F32 scales to its MN-major TMA
            # layout inside this call, so graph replay sees changed scales.
            self._forward(
                (x_q.reshape(rows, self.K).contiguous(), scales),
                (self.weight, self.weight_scale),
                out.view(rows, self.N),
                recipe=(1, 32, 32),
                compiled_dims="",
            )
        return out
