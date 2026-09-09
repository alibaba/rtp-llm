"""Separate collective/GEMM reference implementations for tests only."""

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_into,
    get_process_group,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)


def all_gather_gemm_reference(local_input, projections, *, logical_m, group=Group.TP):
    pg = get_process_group(group)
    size = int(pg.size())
    if not isinstance(local_input, QuantizedActivation) and not isinstance(
        projections[0], torch.Tensor
    ):
        values, scales = projections[0].quantize_input(local_input)
        m, k = values.shape
        aligned = (m + 3) // 4 * 4
        local_input = QuantizedActivation(
            values, scales.as_strided(((k + 511) // 512, aligned), (aligned, 1))
        )
    m, k = local_input.shape
    if isinstance(local_input, QuantizedActivation):
        if size == 1 or m == 0:
            return [
                p.forward_quantized(local_input.values, local_input.scales)[:logical_m]
                for p in projections
            ]
        values = torch.empty(
            (size * m, k), dtype=local_input.dtype, device=local_input.device
        )
        wire = torch.empty(
            (size * local_input.scale_wire.shape[0], local_input.scale_wire.shape[1]),
            dtype=torch.int32,
            device=local_input.device,
        )
        all_gather_into(
            local_input.values.view(torch.uint8), values.view(torch.uint8), group
        )
        all_gather_into(local_input.scale_wire, wire, group)
        outputs = [
            torch.empty(
                (size * m, p.N), dtype=torch.bfloat16, device=local_input.device
            )
            for p in projections
        ]
        for rank, (v, s) in enumerate(zip(values.chunk(size), wire.chunk(size))):
            for p, output in zip(projections, outputs):
                p.forward_quantized(v, s.T[:m], out=output.narrow(0, rank * m, m))
    else:
        gathered = (
            local_input
            if size == 1 or m == 0
            else all_gather_into(
                local_input, local_input.new_empty((size * m, k)), group
            )
        )
        outputs = [torch.matmul(gathered, weight) for weight in projections]
    return [out[:logical_m] for out in outputs]


def gemm_reduce_scatter_reference(
    x, weight, group, *, pad_rows=True, ordered_sum=False
):
    size = int(group.size())
    m, k = x.shape
    physical = (m + size - 1) // size * size if pad_rows else m
    if physical % size:
        raise ValueError("reference RS needs equal destination shards")
    rows = physical // size
    if isinstance(x, QuantizedActivation):
        x = x.pad_rows(physical)
    elif physical != m:
        padded = x.new_zeros((physical, k))
        padded[:m].copy_(x)
        x = padded
    n = weight.shape[1] if isinstance(weight, torch.Tensor) else weight.N
    if physical == 0:
        return torch.empty(
            (0, n),
            dtype=x.dtype if isinstance(weight, torch.Tensor) else torch.bfloat16,
            device=x.device,
        )
    if isinstance(weight, torch.Tensor):
        partial = torch.matmul(x, weight)
    else:
        partial = torch.cat(
            [
                weight(
                    x.narrow_rows(i * rows, rows)
                    if isinstance(x, QuantizedActivation)
                    else x.narrow(0, i * rows, rows)
                )
                for i in range(size)
            ]
        )
    if size == 1:
        return partial
    output = torch.empty((rows, n), dtype=partial.dtype, device=partial.device)
    if ordered_sum:
        # Fused RS accumulates BF16 source slots in rank order in FP32.
        sources = [torch.empty_like(partial) for _ in range(size)]
        dist.all_gather(sources, partial, group=group)
        rank = dist.get_rank(group)
        acc = torch.zeros((rows, n), dtype=torch.float32, device=partial.device)
        for source in sources:
            acc.add_(source[rank * rows : (rank + 1) * rows].float())
        output.copy_(acc)
    else:
        dist.reduce_scatter_tensor(output, partial.contiguous(), group=group)
    return output
