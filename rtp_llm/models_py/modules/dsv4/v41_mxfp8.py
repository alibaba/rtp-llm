"""V4.1 group-32 dense projections and the explicit SM120 FlashInfer ABI.

Quantization follows checkpoint inference/kernel.py: amax floor 1e-4, FP32
reciprocal multiplication, bitwise ceil-to-power-of-two and E4M3 RTNE. The eager
backend is a CPU equation oracle. Construct weights on their final device, as
with the existing DSV4 factory linears; module buffer traversal does not own them.
"""

import torch
from torch import nn

GROUP_SIZE = 32
AMAX_FLOOR = 1e-4


def _scale_bytes(scales):
    if scales.dtype == torch.uint8:
        return scales
    if scales.dtype == getattr(torch, "float8_e8m0fnu", None):
        return scales.view(torch.uint8)
    raise ValueError("V4.1 scales must be raw UE8M0 bytes or float8_e8m0fnu")


def _decode_scales(scales):
    encoded = _scale_bytes(scales)
    result = torch.exp2(encoded.float() - 127)
    return torch.where(encoded == 255, float("nan"), result)


def quantize_group32(x: torch.Tensor, *, backend="triton"):
    """Return E4M3 values in the input shape and linear uint8 scales [M,K/32].

    Inputs must be finite, floating, contiguous and have a group-aligned K.
    The GPU kernel provides debug device assertions for nonfinite inputs.
    """
    if (
        x.ndim < 2
        or x.shape[-1] <= 0
        or x.shape[-1] % GROUP_SIZE
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError(
            "V4.1 quantization requires contiguous floating [...,K] with K divisible by 32"
        )
    if torch.is_grad_enabled() and x.requires_grad:
        raise ValueError("V4.1 quantization is inference-only")
    if backend == "eager":
        if x.device.type != "cpu":
            raise ValueError("V4.1 eager quantization is a CPU diagnostic path")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("V4.1 quantization requires finite inputs")
        grouped = x.float().reshape(-1, x.shape[-1] // GROUP_SIZE, GROUP_SIZE)
        scaled = grouped.abs().amax(-1).clamp_min(AMAX_FLOOR) * (1.0 / 448.0)
        bits = scaled.view(torch.int32)
        encoded = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        scales = (encoded << 23).view(torch.float32)
        values = (grouped / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return values.reshape(x.shape), encoded.to(torch.uint8)
    if backend != "triton" or not x.is_cuda:
        raise ValueError(
            "V4.1 quantization requires an explicit eager or CUDA Triton backend"
        )
    from rtp_llm.models_py.modules.dsv4._v41_mxfp8_triton import quantize_rows

    return quantize_rows(x)


def _require_sm120(device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("V4.1 FlashInfer MXFP8 requires an SM120 CUDA device")


class V41MXFP8Linear(nn.Module):
    """Native E4M3 weights with 32x32 UE8M0 blocks and BF16 output.

    SM120 consumes 128x4-swizzled uint8 scale bytes, not DeepGEMM packed int32.
    Weight scales expand along N once; the E4M3 weight allocation is preserved.
    Weights and scales must remain immutable on their final device. Warm each
    input row count, dtype and pointer alignment outside capture to initialize
    JIT and workspace.
    Provider failures propagate without a different numerical backend.
    """

    def __init__(self, weight, scales, *, backend="flashinfer"):
        super().__init__()
        if (
            weight.ndim != 2
            or min(weight.shape) <= 0
            or weight.shape[1] % GROUP_SIZE
            or weight.dtype != torch.float8_e4m3fn
            or not weight.is_contiguous()
            or not scales.is_contiguous()
            or scales.device != weight.device
        ):
            raise ValueError(
                "V4.1 dense weights require contiguous E4M3 [N,K] and colocated scales"
            )
        self.N, self.K = weight.shape
        raw_scales = _scale_bytes(scales)
        if raw_scales.shape != ((self.N + 31) // GROUP_SIZE, self.K // GROUP_SIZE):
            raise ValueError(
                "V4.1 weight scales require the checkpoint 32x32 block layout"
            )
        if backend not in ("flashinfer", "eager"):
            raise ValueError("Unknown V4.1 dense backend")
        self.weight, self.scales = weight, raw_scales
        self.backend = backend
        self._warmed_rows = set()
        self._warmed_inputs = set()
        self._packed_scales = None
        if backend == "eager":
            if weight.device.type != "cpu":
                raise ValueError("V4.1 eager GEMM is a CPU diagnostic path")
            return
        _require_sm120(weight.device)
        if min(self.N, self.K) < 128 or self.N % GROUP_SIZE:
            raise ValueError("SM120 MXFP8 GEMM requires N,K >= 128 and multiples of 32")
        with torch.cuda.device(weight.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V4.1 weight preparation must precede graph capture")
            from flashinfer import block_scale_interleave
            from flashinfer.gemm import mm_mxfp8

            self._interleave, self._gemm = block_scale_interleave, mm_mxfp8
            expanded = raw_scales.repeat_interleave(GROUP_SIZE, dim=0)
            self._packed_scales = self._interleave(expanded)
            self._check_packed(self._packed_scales, self.N)

    def _check_packed(self, scales, rows):
        expected = ((rows + 127) // 128 * 128) * ((self.K // 32 + 3) // 4 * 4)
        if (
            scales.shape != (expected,)
            or scales.dtype != torch.uint8
            or scales.device != self.weight.device
            or not scales.is_contiguous()
        ):
            raise ValueError(
                "FlashInfer returned an invalid 128x4-swizzled scale buffer"
            )

    def _output(self, x, out, other_inputs=()):
        if (
            x.ndim < 2
            or x.shape[-1] != self.K
            or not x.is_contiguous()
            or x.device != self.weight.device
        ):
            raise ValueError(
                "V4.1 dense input shape/layout/device disagrees with its weight"
            )
        shape = (*x.shape[:-1], self.N)
        if out is not None:
            if (
                out.shape != shape
                or out.dtype != torch.bfloat16
                or out.device != x.device
                or not out.is_contiguous()
                or out.requires_grad
            ):
                raise ValueError(
                    "V4.1 output must be contiguous BF16 with matching shape/device"
                )
            if out.numel():
                inputs = (
                    x,
                    self.weight,
                    self.scales,
                    self._packed_scales,
                    *other_inputs,
                )
                if any(
                    t is not None
                    and t.untyped_storage().data_ptr()
                    == out.untyped_storage().data_ptr()
                    for t in inputs
                ):
                    raise ValueError(
                        "V4.1 output must not share input or weight storage"
                    )
        return (
            out
            if out is not None
            else torch.empty(shape, device=x.device, dtype=torch.bfloat16)
        )

    def forward_quantized(self, values, scales, *, out=None):
        if values.dtype != torch.float8_e4m3fn:
            raise ValueError("V4.1 prequantized input must be E4M3")
        output = self._output(values, out, (scales,))
        rows = values.numel() // self.K
        raw_scales = _scale_bytes(scales)
        if (
            raw_scales.shape != (rows, self.K // GROUP_SIZE)
            or raw_scales.device != values.device
            or not raw_scales.is_contiguous()
        ):
            raise ValueError("V4.1 activations require linear UE8M0 [M,K/32] scales")
        if torch.is_grad_enabled() and any(
            t.requires_grad for t in (values, self.weight)
        ):
            raise ValueError("V4.1 MXFP8 linear is inference-only")
        if not rows:
            return output
        flat, result = values.reshape(rows, self.K), output.view(rows, self.N)
        if self.backend == "eager":
            # Bound diagnostic scratch to a 32x128 output tile and one K group.
            a_scale, b_scale = _decode_scales(raw_scales), _decode_scales(self.scales)
            for m in range(0, rows, 32):
                for n in range(0, self.N, 128):
                    end = min(n + 128, self.N)
                    accum = torch.zeros(
                        (min(32, rows - m), end - n), dtype=torch.float32
                    )
                    for group in range(self.K // GROUP_SIZE):
                        begin = group * GROUP_SIZE
                        a = flat[m : m + 32, begin : begin + 32].float()
                        b = self.weight[n:end, begin : begin + 32].float()
                        scale_b = b_scale[
                            n // 32 : (end + 31) // 32, group
                        ].repeat_interleave(32)[: end - n]
                        accum.add_(
                            (a @ b.T)
                            * a_scale[m : m + 32, group, None]
                            * scale_b[None, :]
                        )
                    result[m : m + 32, n:end].copy_(accum.to(torch.bfloat16))
            return output
        with torch.cuda.device(values.device):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and rows not in self._warmed_rows:
                raise RuntimeError(
                    "V4.1 GEMM shape must be warmed before graph capture"
                )
            a_scales = self._interleave(raw_scales)
            self._check_packed(a_scales, rows)
            self._gemm(
                flat,
                self.weight.T,
                a_scales,
                self._packed_scales,
                out=result,
                out_dtype=torch.bfloat16,
                use_8x4_sf_layout=False,
                backend="cutlass",
            )
            if not capturing:
                self._warmed_rows.add(rows)
        return output

    def forward(self, x, *, out=None):
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("V4.1 dense input must be BF16, FP16 or FP32")
        if torch.is_grad_enabled() and x.requires_grad:
            raise ValueError("V4.1 MXFP8 linear is inference-only")
        output = self._output(x, out)
        backend = "eager" if self.backend == "eager" else "triton"
        if self.backend == "eager":
            return self.forward_quantized(
                *quantize_group32(x, backend=backend), out=output
            )
        rows = x.numel() // self.K
        with torch.cuda.device(x.device):
            capturing = torch.cuda.is_current_stream_capturing()
            key = (rows, x.dtype, x.data_ptr() % 16)
            if capturing and rows and key not in self._warmed_inputs:
                raise RuntimeError(
                    "V4.1 input shape, dtype and alignment must be warmed before graph capture"
                )
            result = self.forward_quantized(
                *quantize_group32(x, backend=backend), out=output
            )
            if not capturing:
                self._warmed_inputs.add(key)
            return result
