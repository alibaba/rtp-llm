"""Bind-time packing of the existing PTPC Q-a and KV-a projections.

Only the output-channel layout changes: resident E4M3 bytes and FP32 channel
scales are concatenated once, then the existing PTPC linear launches one GEMM.
The caller controls Prefill opt-in; this module never reads feature flags.

Model weights are immutable after binding. Ordinary tensor writes, rebinding
and metadata changes invalidate this helper before activation quantization.
Inference tensors have no version counter, so their immutable-weight contract
also applies to writes through aliases and to captured Graph replays.
"""

import torch

from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear


def _physical_pair_supported(first, second):
    return first.weight.is_cuda and second.weight.is_cuda


def _linear_supported(op):
    if type(op) is not PPUPTPCLinear:
        return False
    weight, scale = getattr(op, "weight", None), getattr(op, "scale", None)
    return (
        isinstance(weight, torch.Tensor)
        and isinstance(scale, torch.Tensor)
        and weight.layout == scale.layout == torch.strided
        and weight.dtype == torch.float8_e4m3fn
        and weight.ndim == 2
        and min(weight.shape) > 0
        and scale.dtype == torch.float32
        and tuple(scale.shape) == (weight.shape[0], 1)
        and weight.device == scale.device
        and weight.is_contiguous()
        and scale.is_contiguous()
        and not weight.is_conj()
        and not weight.is_neg()
        and not scale.is_conj()
        and not scale.is_neg()
        and not weight.requires_grad
        and not scale.requires_grad
        and getattr(op, "output_dtype", None) == torch.bfloat16
        and getattr(op, "quantization", None) == "fp8_per_channel_per_token"
        and (
            not weight.is_cuda
            or callable(getattr(op, "_gemm", None))
        )
    )


def _tensor_key(tensor):
    # Inference tensors deliberately have no counter. Reading it is a
    # metadata-only RuntimeError, not a reason to scan/synchronize the weight.
    version = None if torch.is_inference(tensor) else tensor._version
    return (
        id(tensor), tensor.untyped_storage()._cdata, tensor.data_ptr(),
        tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
        tensor.dtype, tensor.device, tensor.layout, tensor.requires_grad,
        tensor.is_conj(), tensor.is_neg(), version,
    )


def _linear_key(op):
    return (
        _tensor_key(op.weight), _tensor_key(op.scale),
        op.output_dtype, op.quantization, op._gemm,
    )


class PreparedQKV:
    """A frozen packed PTPC linear retaining both original weight owners."""

    def __init__(self, first, second, packed):
        self._sources = (first, second)
        self._source_keys = tuple(_linear_key(op) for op in self._sources)
        self._linear = packed
        self._packed_key = _linear_key(packed)
        self._q_width = first.weight.shape[0]

    @classmethod
    def try_prepare(cls, wq_a, wkv):
        """Pack compatible resident weights once, or decline using metadata.

        No quantization, dequantization or scale conversion is introduced.
        Initialization/allocation failures propagate instead of being hidden
        behind a second provider.
        """
        if not (_linear_supported(wq_a) and _linear_supported(wkv)):
            return None
        if (
            wq_a.weight.device != wkv.weight.device
            or wq_a.weight.shape[1] != wkv.weight.shape[1]
            or wq_a._gemm is not wkv._gemm
            or not _physical_pair_supported(wq_a, wkv)
        ):
            return None
        # Some CPU Torch builds do not implement cat for float8. Concatenating
        # its byte view preserves every already-quantized E4M3 bit pattern.
        weight = torch.cat(
            (wq_a.weight.view(torch.uint8), wkv.weight.view(torch.uint8)), dim=0
        ).view(torch.float8_e4m3fn)
        scale = torch.cat((wq_a.scale, wkv.scale), dim=0)
        packed = PPUPTPCLinear(weight, scale)
        packed._gemm = wq_a._gemm
        return cls(wq_a, wkv, packed)

    def _current(self):
        return (
            all(_linear_supported(op) for op in self._sources)
            and tuple(_linear_key(op) for op in self._sources) == self._source_keys
            and _linear_supported(self._linear)
            and _linear_key(self._linear) == self._packed_key
        )

    def _input_supported(self, value):
        return (
            isinstance(value, torch.Tensor)
            and value.layout == torch.strided
            and value.ndim == 2
            and value.dtype == torch.bfloat16
            and value.shape[-1] == self._linear.weight.shape[1]
            and value.device == self._linear.weight.device
            and value.is_contiguous()
            and not value.is_conj()
            and not value.is_neg()
            and not value.requires_grad
            and not torch.is_grad_enabled()
        )

    @staticmethod
    def _pair_supported(pair, value):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return False
        payload, scale = pair
        return (
            isinstance(payload, torch.Tensor)
            and isinstance(scale, torch.Tensor)
            and payload.layout == scale.layout == torch.strided
            and payload.dtype == torch.float8_e4m3fn
            and scale.dtype == torch.float32
            and tuple(payload.shape) == tuple(value.shape)
            and tuple(scale.shape) == (*value.shape[:-1], 1)
            and payload.device == scale.device == value.device
            and payload.is_contiguous()
            and scale.is_contiguous()
            and not payload.is_conj()
            and not payload.is_neg()
            and not scale.is_conj()
            and not scale.is_neg()
            and not payload.requires_grad
            and not scale.requires_grad
        )

    def project(self, x, *, quantized_input=None):
        """Return two raw BF16 views, or None before any activation/device work.

        A supplied pair belongs to this invocation. No activation or output
        cache survives a call; native quantizer/GEMM failures propagate.
        """
        if not self._current() or not self._input_supported(x):
            return None
        if quantized_input is not None and not self._pair_supported(quantized_input, x):
            return None
        pair = (
            self._linear.quantize_input(x)
            if quantized_input is None else quantized_input
        )
        projected = self._linear.forward_quantized(*pair)
        # Both views retain the fresh GEMM output. Norm accepts this explicit
        # row stride, so no forward-time concatenation or raw-output copy occurs.
        return projected[:, :self._q_width], projected[:, self._q_width:]
