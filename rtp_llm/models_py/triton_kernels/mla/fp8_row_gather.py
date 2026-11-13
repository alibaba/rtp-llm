"""Single-launch gather of directed FP8 rows and their packed scales."""

import torch
import triton
import triton.language as tl

_VALUE_BYTES_PER_TILE = 16 * 1024
_SCALE_TILE = 512


@triton.jit(do_not_specialize=["value_tasks", "scale_tasks", "stride", "rows"])
def _gather(
    value_descriptors,
    scale_descriptors,
    out_values,
    out_scales,
    value_tasks,
    scale_tasks,
    stride,
    rows,
    VALUE_WORDS: tl.constexpr,
    SCALE_TILE: tl.constexpr,
    GROUPS: tl.constexpr,
    PADDING_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < value_tasks:
        base = pid * 3
        address = tl.multiple_of(tl.load(value_descriptors + base), 16)
        source = address.to(tl.pointer_type(tl.int32))
        destination = tl.multiple_of(tl.load(value_descriptors + base + 1), 4)
        length = tl.multiple_of(tl.load(value_descriptors + base + 2).to(tl.int32), 4)
        offsets = tl.arange(0, VALUE_WORDS)
        bits = tl.load(
            source + offsets, mask=offsets < length, other=0, cache_modifier=".cg"
        )
        tl.store(out_values + destination + offsets, bits, mask=offsets < length)
    elif pid < value_tasks + scale_tasks:
        scale_base = (pid - value_tasks) * 4
        scale_source = tl.load(scale_descriptors + scale_base).to(
            tl.pointer_type(tl.int32)
        )
        scale_destination = tl.load(scale_descriptors + scale_base + 1)
        scale_length = tl.load(scale_descriptors + scale_base + 2).to(tl.int32)
        scale_index = tl.load(scale_descriptors + scale_base + 3).to(
            tl.int32
        ) + tl.arange(0, SCALE_TILE)
        scale_group = scale_index // scale_length
        scale_row = scale_index % scale_length
        packed = tl.load(
            scale_source + scale_group * stride + scale_row,
            mask=scale_group < GROUPS,
            other=0,
            cache_modifier=".cg",
        )
        tl.store(
            out_scales + scale_group * stride + scale_destination + scale_row,
            packed,
            mask=scale_group < GROUPS,
        )
    else:
        padding = tl.maximum(stride - rows, 1)
        padding_index = tl.arange(0, PADDING_TILE)
        padding_group = padding_index // padding
        padding_row = rows + padding_index % padding
        tl.store(
            out_scales + padding_group * stride + padding_row,
            0x7F7F7F7F,
            mask=padding_group < GROUPS,
        )


class Fp8RowGather:
    def __init__(self, plan, peers, width, device):
        # Descriptors are owned by the prepared plan and only used on the
        # stream that uploads them. They point at model-lifetime peer buffers.
        self.rows = plan.local_rows
        self.stride = (self.rows + 3) // 4 * 4
        self.groups = width // 512
        value_tasks, scale_tasks = [], []
        for segment in plan.to_natural:
            values, scales = peers[segment.peer]
            source = values.data_ptr() + segment.source_offset * width
            destination = segment.destination_offset * width
            length = segment.rows * width
            if source % 16:
                raise ValueError("FP8 source tiles must be 16-byte aligned")
            for offset in range(0, length, _VALUE_BYTES_PER_TILE):
                count = min(_VALUE_BYTES_PER_TILE, length - offset)
                value_tasks.append(
                    (source + offset, (destination + offset) // 4, count // 4)
                )
            source = scales.data_ptr() + 4 * segment.source_offset
            for offset in range(0, segment.rows * self.groups, _SCALE_TILE):
                scale_tasks.append(
                    (source, segment.destination_offset, segment.rows, offset)
                )
        self.value_tasks, self.scale_tasks = len(value_tasks), len(scale_tasks)
        self.values = torch.tensor(value_tasks, dtype=torch.int64, device="cpu").to(
            device, non_blocking=True
        )
        self.scales = torch.tensor(scale_tasks, dtype=torch.int64, device="cpu").to(
            device, non_blocking=True
        )

    def __call__(self, values, scales):
        grid = (self.value_tasks + self.scale_tasks + int(self.stride != self.rows),)
        _gather[grid](
            self.values,
            self.scales,
            values.view(torch.int32),
            scales,
            self.value_tasks,
            self.scale_tasks,
            self.stride,
            self.rows,
            VALUE_WORDS=_VALUE_BYTES_PER_TILE // 4,
            SCALE_TILE=_SCALE_TILE,
            GROUPS=self.groups,
            PADDING_TILE=triton.next_power_of_2(self.groups * 3),
            num_warps=4,
        )
