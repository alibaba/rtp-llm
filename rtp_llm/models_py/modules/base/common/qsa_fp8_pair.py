"""Three-compute-kernel SM90 QSA prefill projection using a library GEMM.

Preparation copies frozen weights/scales once. Runtime uses two quantization
kernels and one rowwise-scaled GEMM; the row scales repeat one *global* scale,
so this does not change activation quantization to per-token quantization.
Profiler checks include all device operations, including library copies and
memsets, rather than counting only compute kernels.

The visible output has 3072 gate + 640 Q/K columns, with no extra weight
columns. The native entry uses the pinned PyTorch 2.11 scaled-mm dispatcher
with accurate accumulation. Physical M has 16-row alignment; only real input
rows participate in the global amax. Returned column slices keep the physical
output row stride and are views. The CUDA13 / Torch2.11 / SM90 lane remains a
validation boundary and must be revalidated when dependencies change.
"""

import torch

from .fp8_linear import Fp8Linear

_INPUT_WIDTH = 2560
_GATE_WIDTH = 3072
_QK_WIDTH = 640
_PACKED_WIDTH = _GATE_WIDTH + _QK_WIDTH


def _native_projection_available():
    """Query the compiled ABI capability during preparation, never forward."""
    from rtp_llm.ops import compute_ops

    available = getattr(compute_ops, "qsa_prefill_fp8_available", None)
    return (
        available is not None
        and hasattr(compute_ops, "QsaFp8ProjectionGuard")
        and bool(available())
    )


def quantize_fp8_prefill_input(x):
    """Return padded FP8 input and repeated global scale in two CUDA kernels.

    The native entry validates aligned SM90 BF16 input and safe element counts.
    It allocates independent quantized output, row scales and partial scratch
    per call. There is no persistent state shared between requests or CUDA streams.
    """
    from rtp_llm.ops.compute_ops import qsa_prefill_fp8_quant

    return qsa_prefill_fp8_quant(x)


def _create_fp8_source_guard(sources):
    """Capture source metadata once; CPU integration tests mock this boundary."""
    from rtp_llm.ops.compute_ops import QsaFp8ProjectionGuard

    return QsaFp8ProjectionGuard(sources)


def _project_fp8_prefill_native(x, weight_t, column_scales, guard, sources):
    """One native entry validates sources, quantizes and executes the GEMM."""
    return guard.project(x, weight_t, column_scales, sources)


def _has_hooks_or_override(projection):
    return (
        "forward" in projection.__dict__
        or projection._forward_pre_hooks
        or projection._forward_hooks
    )


def _owner_flags_supported(first, second):
    return (
        type(first) is Fp8Linear
        and type(second) is Fp8Linear
        and first.use_scaled_mm
        and second.use_scaled_mm
        and first.bias is None
        and second.bias is None
        and not _has_hooks_or_override(first)
        and not _has_hooks_or_override(second)
        and not torch.nn.modules.module._global_forward_pre_hooks
        and not torch.nn.modules.module._global_forward_hooks
        and first.role in ("PREFILL", "PDFUSION")
        and second.role in ("PREFILL", "PDFUSION")
    )


def _owners_supported(first, second):
    # Full tensor metadata checks run at preparation. The native guard retains
    # the same checks in the compute entry, avoiding Python metadata dispatch.
    return (
        _owner_flags_supported(first, second)
        and tuple(first.weight.shape) == (_GATE_WIDTH, _INPUT_WIDTH)
        and tuple(second.weight.shape) == (_QK_WIDTH, _INPUT_WIDTH)
        and first.weight.dtype == second.weight.dtype == torch.float8_e4m3fn
        and first.weight.device == second.weight.device
        and first.weight.is_cuda
        and first.weight.is_contiguous()
        and second.weight.is_contiguous()
        and not first.weight.requires_grad
        and not second.weight.requires_grad
        and all(
            isinstance(p.scale, torch.Tensor)
            and p.scale.dtype == torch.float32
            and p.scale.device == p.weight.device
            and p.scale.numel() == 1
            and p.scale.is_contiguous()
            and not p.scale.requires_grad
            for p in (first, second)
        )
    )


class PreparedFp8Pair:
    """Prepared immutable SM90 pair. Construct through ``prepare_fp8_pair``.

    ``project`` only handles eager prefill-shaped input; the model owner must
    gate phase, graph mode and semantic overrides. Unsupported or invalidated
    inputs return None *before* launching. Once execution starts, errors are
    propagated rather than disguising a fallback as the three-kernel path.
    """

    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.device = first.weight.device
        self._source_tensors = (
            first.weight,
            first.scale,
            second.weight,
            second.scale,
        )
        self._source_guard = _create_fp8_source_guard(self._source_tensors)
        self.weight = torch.cat((first.weight, second.weight), dim=0)
        self.weight_t = self.weight.t()
        self.weight_scales = torch.cat(
            (
                first.scale.reshape(1).expand(_GATE_WIDTH),
                second.scale.reshape(1).expand(_QK_WIDTH),
            )
        ).reshape(1, _PACKED_WIDTH)

    def _sources_unchanged(self):
        # These checks concern Python dispatch and object identity. Every tensor
        # metadata/version check still runs, inside the single native call.
        return (
            _owner_flags_supported(self.first, self.second)
            and self.first.weight is self._source_tensors[0]
            and self.first.scale is self._source_tensors[1]
            and self.second.weight is self._source_tensors[2]
            and self.second.scale is self._source_tensors[3]
        )

    def project(self, x, *, first, second):
        if first is not self.first or second is not self.second:
            return None
        if not isinstance(x, torch.Tensor) or not self._sources_unchanged():
            return None
        # Keep the existing capture boundary; input tensor metadata is checked
        # inside the same native guard call as the source snapshots.
        if torch.cuda.is_current_stream_capturing():
            return None
        output = _project_fp8_prefill_native(
            x,
            self.weight_t,
            self.weight_scales,
            self._source_guard,
            self._source_tensors,
        )
        if output is None:
            return None
        rows = x.shape[0]
        return (
            output[:rows, :_GATE_WIDTH],
            output[:rows, _GATE_WIDTH : _GATE_WIDTH + _QK_WIDTH],
        )


def prepare_fp8_pair(first, second):
    """Pack a supported pair once; unsupported platforms retain the old path."""
    if torch.version.hip is not None or not _owners_supported(first, second):
        return None
    if not _native_projection_available():
        return None
    if torch.cuda.get_device_capability(first.weight.device) != (9, 0):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    return PreparedFp8Pair(first, second)
