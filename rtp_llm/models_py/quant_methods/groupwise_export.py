"""AWQ/GPTQ offline runtime-layout export. No inference backend registration.

The CUDA and ROCm permutations below preserve their existing DeviceImpl export
ABI. They operate on explicit module parameters and require no device singleton.
"""

import torch
from torch import nn

from rtp_llm.models_py.quant_methods.groupwise_checkpoint import PackedGroupwiseWeights


def cuda_groupwise_layout(tensor, bits, arch):
    if bits not in (4, 8) or tensor.dtype != torch.int8 or tensor.ndim != 2:
        raise ValueError("Expected signed packed two-dimensional 4/8-bit weights")
    tensor = tensor.unsqueeze(0)
    rows, columns = tensor.shape[1:]
    rows_per_mma, elements_per_register = 128 // bits, 32 // bits
    if rows % rows_per_mma or columns % 8:
        raise ValueError("CUDA mixed-GEMM weights violate tile alignment")
    if arch >= 75:
        if rows % 64:
            raise ValueError("CUDA mixed-GEMM input width must align to 64")
        permutation = [
            8 * ((i % elements_per_register) // 2)
            + i % 2
            + 2 * (i // elements_per_register)
            for i in range(rows_per_mma)
        ]
        row_indices = [
            i // rows_per_mma * rows_per_mma + permutation[i % rows_per_mma]
            for i in range(rows)
        ]
        tensor = tensor[:, row_indices, :]
        original_shape = tensor.shape
        if bits == 4:
            unsigned = tensor.view(torch.uint8)
            high = (unsigned >> 4).permute(0, 2, 1).unsqueeze(2)
            low = (unsigned & 15).permute(0, 2, 1).unsqueeze(2)
            unpacked = torch.cat([low, high], 2).reshape(1, -1, rows)
            tensor = (
                (unpacked[:, :, ::2] + unpacked[:, :, 1::2] * 16)
                .view(torch.int8)
                .reshape(original_shape)
            )
        else:
            tensor = tensor.permute(0, 2, 1).reshape(original_shape)
        interleave = 16 // bits
        tensor = tensor.reshape(
            1, -1, interleave, rows // 64, 64 * 4 // elements_per_register
        )
        tensor = tensor.permute(0, 1, 3, 2, 4).reshape(original_shape)
    if bits == 8:
        tensor = (tensor.int() + 128).char()
        tensor = tensor.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(tensor.shape)
    else:
        unsigned = tensor.view(torch.uint8)
        unpacked = torch.stack([unsigned & 15, unsigned >> 4], -1).reshape(1, rows, -1)
        unpacked = unpacked.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(
            unpacked.shape
        )
        unpacked += -16 * (unpacked > 7).byte() + 8
        tensor = (unpacked[:, :, ::2] + unpacked[:, :, 1::2] * 16).view(torch.int8)
    return tensor.squeeze(0).contiguous()


def rocm_groupwise_layout(tensor, bits):
    if bits not in (4, 8) or tensor.dtype != torch.int8 or tensor.ndim != 2:
        raise ValueError("Expected signed packed two-dimensional 4/8-bit weights")
    if bits == 4:
        unsigned = tensor.view(torch.uint8)
        values = torch.stack([unsigned & 15, unsigned >> 4], -1).reshape(
            tensor.shape[0], -1
        )
        if values.shape[0] % 2:
            raise ValueError("ROCm grouped input width must be even")
        column_packed = (values[::2] * 16 + values[1::2]) ^ 0x88
        tensor = column_packed.view(torch.int8).T.contiguous().view(column_packed.shape)
    if tensor.numel() % 4:
        raise ValueError("ROCm mixed-GEMM storage must align to four bytes")
    # Preserve existing ROCm byte-interleave, including its eight-bit behavior.
    unsigned = tensor.reshape(-1).to(torch.uint8)
    nibbles = torch.stack([unsigned >> 4, unsigned & 15], 1).reshape(-1, 8)
    reordered = nibbles[:, [2, 0, 6, 4, 3, 1, 7, 5]].reshape(-1)
    packed = ((reordered[::2] << 4) | reordered[1::2]).to(torch.int8)
    return packed.reshape(list(reversed(tensor.shape))).T.contiguous()


class GroupwiseExportModule(nn.Module):
    """Own dense or stacked-expert parameters through the offline export stage.

    Stacked legacy tensors have [expert, N, K] orientation after checkpoint
    fusion. The module adopts transposed views, never repeats padding, sharding,
    or fused projection construction already performed by the source reader.
    """

    def __init__(
        self,
        qweight,
        qzeros,
        scales,
        *,
        method,
        bits,
        group_size,
        gptq_v2=False,
        stacked=False,
        activation_scale=None,
    ):
        super().__init__()
        self.stacked = stacked
        if stacked:
            if qweight.ndim != 3 or qzeros.ndim != 3 or scales.ndim != 3:
                raise ValueError(
                    "Stacked grouped export requires three-dimensional tensors"
                )
            if (
                not qweight.shape[0]
                or len({qweight.shape[0], qzeros.shape[0], scales.shape[0]}) != 1
            ):
                raise ValueError(
                    "Stacked grouped tensors have inconsistent expert counts"
                )
            inputs = zip(qweight.unbind(), qzeros.unbind(), scales.unbind())
            inputs = ((weight.T, zero.T, scale.T) for weight, zero, scale in inputs)
        else:
            inputs = [(qweight, qzeros, scales)]
        self.projections = nn.ModuleList(
            [
                PackedGroupwiseWeights.from_tensors(
                    weight, zero, scale, method=method, bits=bits,
                    group_size=group_size, gptq_v2=gptq_v2,
                )
                for weight, zero, scale in inputs
            ]
        )
        self.register_parameter(
            "activation_scale",
            (
                None
                if activation_scale is None
                else nn.Parameter(activation_scale, requires_grad=False)
            ),
        )

    def export(self, *, platform, arch=0, device="cpu"):
        if platform in ("Cuda", "Ppu"):
            layout = lambda tensor, bits: cuda_groupwise_layout(tensor, bits, int(arch))
        elif platform == "ROCm":
            layout = rocm_groupwise_layout
        else:
            raise ValueError(f"No grouped offline export layout for {platform}")
        converted = [projection.convert(layout) for projection in self.projections]
        result = tuple(
            torch.stack(items) if self.stacked else items[0]
            for items in zip(*converted)
        )
        return tuple(tensor.contiguous().to(device) for tensor in result)
