"""AllGather of independently quantized NVFP4 source shards."""

from typing import NamedTuple

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import Fp4Activation
from rtp_llm.models_py.utils.module_profiler import profile_scope


def _align(size, alignment):
    return (size + alignment - 1) // alignment * alignment


def nvfp4_packet_capacity(rows, width, *, rowwise=False):
    """Capacity for FlashInfer's E2M1 values and CUTLASS 128x4 scale layout."""
    scale_bytes = _align(4 * rows, 16) if rowwise else 16
    return (
        _align(rows * (width // 2), 16)
        + _align(rows, 128) * _align(width // 16, 4)
        + scale_bytes
    )


class PackedNvfp4Activation(NamedTuple):
    wire: torch.Tensor
    shape: torch.Size
    rowwise: bool = False

    @property
    def device(self):
        return self.wire.device

    @classmethod
    def empty(cls, rows, width, device, *, rowwise=False):
        wire = torch.empty(
            nvfp4_packet_capacity(rows, width, rowwise=rowwise),
            dtype=torch.uint8,
            device=device,
        )
        return cls(wire, torch.Size((rows, width)), rowwise)

    def unpack(self, source):
        rows, width = self.shape
        values_end = rows * (width // 2)
        block_offset = _align(values_end, 16)
        scale_offset = block_offset + _align(rows, 128) * _align(width // 16, 4)
        scale_count = rows if self.rowwise else 1
        return Fp4Activation(
            source[:values_end].view(rows, width // 2),
            source[block_offset:scale_offset],
            source[scale_offset : scale_offset + 4 * scale_count].view(torch.float32),
        )


def decode_packet_capacity(rows, width, head_gate_columns=0, *, rowwise=False):
    size = nvfp4_packet_capacity(rows, width, rowwise=rowwise)
    if head_gate_columns:
        size += rows * width + _align(rows, 4) * (width // 128)
        size += rows * head_gate_columns * 4
    return size


class DsaDecodeActivation(NamedTuple):
    """One packet holding FP4/FP8 activations, their scales and FP32 head gates."""

    wire: torch.Tensor
    local_hidden: torch.Tensor
    head_gate_columns: int
    rowwise: bool = False

    @classmethod
    def empty(cls, local_hidden, head_gate_columns, *, rowwise=False):
        rows, width = local_hidden.shape
        size = decode_packet_capacity(rows, width, head_gate_columns, rowwise=rowwise)
        return cls(
            local_hidden.new_empty(size, dtype=torch.uint8),
            local_hidden,
            head_gate_columns,
            rowwise,
        )

    def views(self):
        from rtp_llm.models_py.distributed.collective_torch import Fp8Activation

        rows, width = self.local_hidden.shape
        fp4_end = nvfp4_packet_capacity(rows, width, rowwise=self.rowwise)
        fp8_end = fp4_end + rows * width
        scales_end = fp8_end + _align(rows, 4) * (width // 128)
        return (
            PackedNvfp4Activation(
                self.wire[:fp4_end], self.local_hidden.shape, self.rowwise
            ),
            Fp8Activation(
                self.wire[fp4_end:fp8_end].view(torch.float8_e4m3fn).view(rows, width),
                self.wire[fp8_end:scales_end].view(torch.int32).view(width // 512, -1),
            ),
            self.wire[scales_end:]
            .view(torch.float32)
            .view(rows, self.head_gate_columns),
        )


class DecodeNvfp4AllGather:
    def __init__(self, group, direct):
        self.group = group
        self.direct = direct

    def __call__(self, packet):
        from rtp_llm.models_py.triton_kernels.common.nvfp4 import unpack_decode_nvfp4

        wire = packet.wire
        tp_size = self.group.size()
        with profile_scope("RTP::attention.decode.nvfp4_packet_ag"):
            if self.direct is not None and self.direct.can_gather_packed(wire.numel()):
                gathered = self.direct.all_gather_packed(wire)
            else:
                gathered = wire.new_empty(tp_size * wire.numel())
                dist.all_gather_into_tensor(gathered, wire, group=self.group)
        with profile_scope("RTP::attention.decode.nvfp4_unpack"):
            return unpack_decode_nvfp4(gathered, packet, tp_size)


def _pack_nvfp4(local):
    values, block_scales, scale = local
    values_bytes = values.reshape(-1).view(torch.uint8)
    block_bytes = block_scales.reshape(-1).view(torch.uint8)
    # Each field and the next rank's packet start on a 16-byte boundary.
    values_end = values_bytes.numel()
    block_offset = _align(values_end, 16)
    block_end = block_offset + block_bytes.numel()
    scale_offset = _align(block_end, 16)
    padding = values_bytes.new_zeros(16)
    packet = torch.cat(
        (
            values_bytes,
            padding[: block_offset - values_end],
            block_bytes,
            padding[: scale_offset - block_end],
            scale.view(torch.uint8),
            padding[:12],
        )
    )

    def unpack(source):
        return Fp4Activation(
            source[:values_end].view(values.dtype).view(values.shape),
            source[block_offset:block_end]
            .view(block_scales.dtype)
            .view(block_scales.shape),
            source[scale_offset : scale_offset + 4].view(torch.float32),
        )

    return packet, unpack


def all_gather_nvfp4_gemm(hidden, projection, group, *, overlap, direct=None):
    if overlap and direct is not None:
        raise ValueError("NVFP4 Direct AllGather runs on one stream")
    world_size = group.size()
    rows = hidden.shape[0]
    if isinstance(hidden, PackedNvfp4Activation):
        packet, unpack = hidden.wire, hidden.unpack
        local = unpack(packet)
    else:
        with profile_scope("RTP::attention.kda.fp4_quantize_local"):
            local = projection.quantize_dynamic(hidden)
        with profile_scope("RTP::attention.kda.fp4_pack"):
            packet, unpack = _pack_nvfp4(local)
    output = packet.new_empty(
        (world_size * rows, projection.output_size), dtype=torch.bfloat16
    )
    out_views = output.split(rows)
    with profile_scope("RTP::attention.kda.fp4_prepare_gemm"):
        gemms = projection.prepare_quantized_gemms(
            local,
            out_views[group.rank()],
            num_streams=2 if overlap else 1,
        )

    def consume(source, rank, lane):
        with profile_scope("RTP::attention.kda.fp4_shard_gemm"):
            projection.forward_quantized(
                unpack(source),
                out=out_views[rank],
                gemm=gemms[lane],
            )

    if overlap:
        from torch.distributed import _symmetric_memory as symm

        gathered = packet.new_empty(world_size * packet.numel())
        compute = torch.cuda.current_stream(packet.device)
        side = symm._get_backend_stream()

        def consume_on_current_stream(shard, rank):
            # As in FP8, each stream owns both the copy and its consumer.
            # CUTLASS needs a different workspace on each of those streams.
            lane = int(torch.cuda.current_stream(packet.device) != compute)
            consume(shard[0], rank, lane)

        try:
            with profile_scope("RTP::attention.kda.fp4_packet_ag_overlap"):
                symm._pipelined_multi_all_gather_and_consume(
                    [packet],
                    consume_on_current_stream,
                    [gathered],
                    group.group_name,
                    ag_out_needed=False,
                )
        except Exception:
            # The helper normally joins its side stream before returning.
            # Keep the buffers and CUTLASS workspace live on a failed launch too.
            compute.wait_stream(side)
            raise
    else:
        if direct is not None and direct.can_gather_packed(packet.numel()):
            with profile_scope("RTP::attention.kda.fp4_packet_ag_direct"):
                gathered = direct.all_gather_packed(packet)
        else:
            gathered = packet.new_empty(world_size * packet.numel())
            with profile_scope("RTP::attention.kda.fp4_packet_ag_nccl"):
                dist.all_gather_into_tensor(gathered, packet, group=group)
        for rank, shard in enumerate(gathered.split(packet.numel())):
            consume(shard, rank, 0)
    return output
