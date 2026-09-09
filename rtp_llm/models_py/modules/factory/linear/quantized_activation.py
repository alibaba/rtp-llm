"""Explicit group128 activation data shared by producers and FP8 consumers.

The wire storage is contiguous; the GEMM scale is a view of that storage.
This is deliberately not a Tensor subclass: CUDA/cache consumers must request
the retained BF16 value explicitly rather than accidentally dropping scales.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class QuantizedActivation:
    values: torch.Tensor
    scale_wire: torch.Tensor
    bf16: Optional[torch.Tensor] = None

    def __post_init__(self):
        m, k = self.values.shape
        if self.values.dtype != torch.float8_e4m3fn or not self.values.is_contiguous():
            raise ValueError("activation must be contiguous E4M3 [M,K]")
        if k % 128 or self.scale_wire.shape != ((k + 511) // 512, (m + 3) // 4 * 4):
            raise ValueError("invalid group128 packed UE8M0 activation scale shape")
        if (
            self.scale_wire.dtype != torch.int32
            or not self.scale_wire.is_contiguous()
            or self.scale_wire.device != self.values.device
        ):
            raise ValueError(
                "activation scales must be contiguous int32 on the input device"
            )
        if self.bf16 is not None and (
            self.bf16.shape != self.values.shape
            or self.bf16.dtype != torch.bfloat16
            or self.bf16.device != self.values.device
        ):
            raise ValueError("retained activation must be matching BF16 data")

    @property
    def scales(self):
        return self.scale_wire.T[: self.shape[0]]

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return 2

    @property
    def device(self):
        return self.values.device

    @property
    def is_cuda(self):
        return self.values.is_cuda

    @property
    def dtype(self):
        return self.values.dtype

    def reshape(self, *shape):
        # Only the logical flattening already performed by the producer is
        # allowed. Reshaping groups would silently change scale ownership.
        if tuple(shape) != tuple(self.shape):
            raise ValueError("cannot reshape group128 activation boundaries")
        return self

    def contiguous(self):
        return self

    def dim(self):
        return 2

    def is_contiguous(self):
        return True

    def narrow_rows(self, start: int, length: int):
        if start < 0 or length < 0 or start + length > self.shape[0]:
            raise ValueError("activation row slice outside logical rows")
        # RS may split at non-TMA-aligned rows. Repack only the small scale
        # buffer; never requantize the values or reconstruct BF16 activations.
        wire = torch.full(
            (self.scale_wire.shape[0], (length + 3) // 4 * 4),
            0x7F7F7F7F,
            dtype=torch.int32,
            device=self.device,
        )
        wire[:, :length].copy_(self.scale_wire[:, start : start + length])
        retained = None if self.bf16 is None else self.bf16.narrow(0, start, length)
        return QuantizedActivation(self.values.narrow(0, start, length), wire, retained)

    def pad_rows(self, m: int):
        if m < self.shape[0]:
            raise ValueError("cannot pad to fewer activation rows")
        if m == self.shape[0]:
            return self
        values = torch.zeros((m, self.shape[1]), dtype=self.dtype, device=self.device)
        values[: self.shape[0]].copy_(self.values)
        wire = torch.full(
            (self.scale_wire.shape[0], (m + 3) // 4 * 4),
            0x7F7F7F7F,
            dtype=torch.int32,
            device=self.device,
        )
        wire[:, : self.shape[0]].copy_(self.scale_wire[:, : self.shape[0]])
        return QuantizedActivation(values, wire)


def retained_bf16(value):
    if isinstance(value, QuantizedActivation):
        if value.bf16 is None:
            raise ValueError("this consumer requires a retained BF16 activation")
        return value.bf16
    return value


def allocate_quantized(m, k, device, *, retain_bf16=False):
    return QuantizedActivation(
        torch.empty((m, k), dtype=torch.float8_e4m3fn, device=device),
        torch.empty(
            ((k + 511) // 512, (m + 3) // 4 * 4), dtype=torch.int32, device=device
        ),
        (
            torch.empty((m, k), dtype=torch.bfloat16, device=device)
            if retain_bf16
            else None
        ),
    )
