"""Module-owned AWQ/GPTQ checkpoint conversion, independent of DeviceImpl.

This is a loading/conversion building block, NOT an inference registration.
CUDA/ROCm mixed-GEMM layouts differ, and an executable grouped-GEMM backend
must explicitly consume the result before NewLoader can enable inference.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


def _unpack_last(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if tensor.dtype != torch.int32 or tensor.ndim != 2:
        raise TypeError("AWQ/GPTQ packed tensors must be two-dimensional int32")
    # Preserve the exporter's byte-oriented unpacking: expanding every packed
    # word to int64 lanes would add an avoidable 8-byte/value peak allocation.
    packed_bytes = tensor.contiguous().view(torch.uint8)
    if bits == 8:
        return packed_bytes.to(torch.int16)
    values = torch.empty(
        packed_bytes.shape[0],
        packed_bytes.shape[1] * 2,
        dtype=torch.int16,
        device=tensor.device,
    )
    values[:, ::2] = packed_bytes & 15
    values[:, 1::2] = packed_bytes >> 4
    return values


def _reverse_awq(tensor: torch.Tensor) -> torch.Tensor:
    # The legacy exporter reverses this eight-element interleave for both
    # four- and eight-bit AWQ. Preserve its byte-level contract.
    if tensor.shape[-1] % 8:
        raise ValueError("AWQ output width must be divisible by eight")
    return tensor.reshape(-1, 2, 4).transpose(1, 2).reshape(tensor.shape)


@dataclass(frozen=True)
class GroupwiseCheckpointLayout:
    method: str
    bits: int
    group_size: int
    input_size: int
    output_size: int
    gptq_v2: bool = False

    def __post_init__(self):
        if not isinstance(self.gptq_v2, bool) or (self.gptq_v2 and self.method != "gptq"):
            raise ValueError("GPTQ v2 zero points require a GPTQ checkpoint")
        if self.method not in ("awq", "gptq") or self.bits not in (4, 8):
            raise ValueError("Expected AWQ/GPTQ with four- or eight-bit weights")
        if self.group_size <= 0 or self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("Grouped checkpoint dimensions must be positive")
        if self.output_size % self.pack_factor:
            raise ValueError("Output width must contain complete zero-point words")
        if self.method == "gptq" and self.input_size % self.pack_factor:
            raise ValueError("GPTQ input width must contain complete packed words")
        if self.method == "awq" and self.output_size % 8:
            raise ValueError("AWQ output width must preserve eight-element interleaves")

    @property
    def pack_factor(self):
        return 32 // self.bits

    @property
    def shapes(self):
        groups = (self.input_size + self.group_size - 1) // self.group_size
        return {
            "qweight": (
                (self.input_size, self.output_size // self.pack_factor)
                if self.method == "awq"
                else (self.input_size // self.pack_factor, self.output_size)
            ),
            "qzeros": (groups, self.output_size // self.pack_factor),
            "scales": (groups, self.output_size),
        }

    def local_layout(self, partition_dim, tp_size, tp_rank):
        if partition_dim not in (0, 1) or tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError("Invalid logical partition dimension or TP rank")
        # Logical checkpoint orientation is [K, N], not Linear's [N, K].
        dimensions = [self.input_size, self.output_size]
        if dimensions[partition_dim] % tp_size:
            raise ValueError("Logical weight dimension is not divisible by TP size")
        dimensions[partition_dim] //= tp_size
        if partition_dim == 0 and tp_size > 1 and dimensions[0] % self.group_size:
            raise ValueError("TP partitions must contain complete quantization groups")
        return GroupwiseCheckpointLayout(
            self.method, self.bits, self.group_size, *dimensions, self.gptq_v2
        )

    def read_spec(self, name, shape, partition_dim=1, tp_size=1, tp_rank=0):
        parameter = name.rsplit(".", 1)[-1]
        if parameter not in self.shapes or tuple(shape) != self.shapes[parameter]:
            raise ValueError(f"Invalid grouped checkpoint shape for {name}: {shape}")
        self.local_layout(partition_dim, tp_size, tp_rank)
        if tp_size == 1:
            return ReadSpec.full(name, tuple(shape))
        return make_sharded_read_spec(
            name, tuple(shape), partition_dim, tp_size, tp_rank
        )


class PackedGroupwiseWeights(nn.Module):
    """Own packed parameters for one logical projection or rank-local fusion.

    Use one instance per projection while loading Q/K/V with different GQA
    ranks. ``fuse`` then concatenates only already-local projection parameters;
    no full-precision matrix or second W.* dictionary is materialized.
    """

    def __init__(self, layout, dtype=torch.float16, *, device=None):
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Group scales must use a floating-point compute dtype")
        self.layout = layout
        self._loaded = set()
        for name, shape in layout.shapes.items():
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.empty(
                        shape,
                        dtype=dtype if name == "scales" else torch.int32,
                        device=device,
                    ),
                    requires_grad=False,
                ),
            )

    @classmethod
    def from_tensors(
        cls, qweight, qzeros, scales, *, method, bits, group_size, gptq_v2=False
    ):
        """Adopt already-sharded checkpoint storage without allocating a copy.

        The offline loader has already performed model-specific padding/fusion.
        This boundary deliberately does not repeat those transformations.
        """
        if method not in ("awq", "gptq") or bits not in (4, 8):
            raise ValueError("Expected AWQ/GPTQ with four- or eight-bit weights")
        if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
            raise ValueError("Grouped projection tensors must be two-dimensional")
        if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
            raise TypeError("Grouped projection weights and zeros must be int32")
        if scales.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Grouped scales must be floating point")
        if len({qweight.device, qzeros.device, scales.device}) != 1:
            raise ValueError("Grouped projection tensors must be on the same device")
        k = qweight.shape[0] * (32 // bits if method == "gptq" else 1)
        n = qweight.shape[1] * (32 // bits if method == "awq" else 1)
        layout = GroupwiseCheckpointLayout(method, bits, group_size, k, n, gptq_v2)
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.layout = layout
        module._loaded = set(layout.shapes)
        for name, tensor in (
            ("qweight", qweight),
            ("qzeros", qzeros),
            ("scales", scales),
        ):
            if tuple(tensor.shape) != layout.shapes[name]:
                raise ValueError(
                    f"Invalid grouped export shape for {name}: {tensor.shape}"
                )
            module.register_parameter(name, nn.Parameter(tensor, requires_grad=False))
        return module

    def load_checkpoint(
        self, weights, source_layout=None, *, partition_dim=1, tp_size=1, tp_rank=0
    ):
        source_layout = source_layout or self.layout
        if source_layout.local_layout(partition_dim, tp_size, tp_rank) != self.layout:
            raise ValueError("Source checkpoint partition does not match this module")
        for name, value in weights.items():
            parameter = name.rsplit(".", 1)[-1]
            if parameter not in self.layout.shapes:
                raise ValueError(f"Unknown grouped checkpoint tensor {name}")
            if parameter in self._loaded:
                raise RuntimeError(f"Duplicate grouped checkpoint tensor {name}")
            tensor, spec = unwrap_checkpoint_tensor(value)
            expected = source_layout.read_spec(
                name, source_layout.shapes[parameter], partition_dim, tp_size, tp_rank
            )
            if spec is not None:
                if spec != expected:
                    raise ValueError("ReadSpec does not match this grouped module/rank")
            else:
                if tuple(tensor.shape) != source_layout.shapes[parameter]:
                    raise ValueError(f"Invalid source shape for {name}: {tensor.shape}")
                if expected.slices is not None:
                    tensor = tensor[expected.slices]
            target = getattr(self, parameter)
            if tensor.dtype != target.dtype or tuple(tensor.shape) != tuple(
                target.shape
            ):
                raise ValueError(f"Grouped checkpoint dtype/shape mismatch for {name}")
            with torch.no_grad():
                target.copy_(tensor)
            self._loaded.add(parameter)

    def _check_complete(self):
        missing = set(self.layout.shapes) - self._loaded
        if missing:
            raise RuntimeError(
                f"Missing grouped checkpoint parameters: {sorted(missing)}"
            )

    @classmethod
    def fuse(cls, projections):
        if not projections:
            raise ValueError("Cannot fuse an empty projection list")
        first = projections[0]
        for projection in projections:
            projection._check_complete()
            if (
                projection.layout.method,
                projection.layout.gptq_v2,
                projection.layout.bits,
                projection.layout.group_size,
                projection.layout.input_size,
                projection.scales.dtype,
                projection.scales.device,
            ) != (
                first.layout.method,
                first.layout.gptq_v2,
                first.layout.bits,
                first.layout.group_size,
                first.layout.input_size,
                first.scales.dtype,
                first.scales.device,
            ):
                raise ValueError("Fused grouped projections have incompatible layouts")
        layout = GroupwiseCheckpointLayout(
            first.layout.method,
            first.layout.bits,
            first.layout.group_size,
            first.layout.input_size,
            sum(p.layout.output_size for p in projections),
            first.layout.gptq_v2,
        )
        result = cls(layout, first.scales.dtype, device=first.scales.device)
        result.load_checkpoint(
            {
                name: torch.cat([getattr(p, name) for p in projections], 1)
                for name in layout.shapes
            }
        )
        return result

    def _checkpoint_codes(self):
        self._check_complete()
        bits = self.layout.bits
        if self.layout.method == "awq":
            values = _reverse_awq(_unpack_last(self.qweight, bits))
            zeros = _reverse_awq(_unpack_last(self.qzeros, bits))
        else:
            values = _unpack_last(self.qweight.T, bits).T.contiguous()
            zeros = _unpack_last(self.qzeros, bits)
            if not self.layout.gptq_v2:
                zeros = zeros + 1
        return values, zeros

    def decode_control_weight(self, dtype):
        """Import a small [K,N] control projection for an existing floating consumer."""
        values, zeros = self._checkpoint_codes()
        rows = (
            torch.arange(values.shape[0], device=values.device) // self.layout.group_size
        )
        weight = ((values - zeros[rows]).float() * self.scales[rows].float()).to(dtype)
        if not torch.isfinite(weight).all():
            raise ValueError("Non-finite decoded grouped control weight")
        return weight

    def convert(self, weight_layout: Callable):
        """Convert to the existing mixed-GEMM export ABI, with explicit backend.

        ``weight_layout(signed_packed_weight, bits)`` owns platform swizzling.
        There is deliberately no automatic backend or full-precision fallback.
        """
        values, zeros = self._checkpoint_codes()
        bits = self.layout.bits
        shift = 1 << (bits - 1)
        signed = (values - shift).to(torch.int8)
        if bits == 4:
            unsigned = signed.to(torch.uint8) & 15
            signed = (unsigned[:, ::2] | (unsigned[:, 1::2] << 4)).view(torch.int8)
        offsets = ((-zeros + shift) * self.scales).half()
        return (
            weight_layout(signed.contiguous(), bits),
            offsets.contiguous(),
            self.scales,
        )
