from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["count"],
    do_not_specialize_on_alignment=[
        "draft_ids",
        "positions",
        "do_sample",
        "cap",
        "absolute_positions",
        "expanded_mapping",
        "local_positions",
        "cumulative",
        "rates",
    ],
)
def _prepare_rejection_metadata(
    draft_ids,
    positions,
    do_sample,
    cap,
    packed,
    absolute_positions,
    mapping,
    expanded_mapping,
    local_positions,
    cumulative,
    temperature,
    rates,
    count,
    rows,
    draft_stride,
    draft_column_stride,
    position_stride,
    sample_stride,
    cap_stride,
    BLOCK: tl.constexpr,
    graph_input_addresses=None,
    seed_output=None,
):
    if graph_input_addresses is not None:
        draft_ids = tl.load(graph_input_addresses + 1).to(tl.uint64).to(draft_ids.dtype)
        positions = tl.load(graph_input_addresses + 3).to(tl.uint64).to(positions.dtype)
        do_sample = tl.load(graph_input_addresses + 4).to(tl.uint64).to(do_sample.dtype)
        if cap is not None:
            cap = tl.load(graph_input_addresses + 5).to(tl.uint64).to(cap.dtype)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if graph_input_addresses is not None:
        seed_source = (
            tl.load(graph_input_addresses + 2).to(tl.uint64).to(seed_output.dtype)
        )
        seeds = tl.load(seed_source + index, mask=index < count, other=0)
        tl.store(seed_output + index, seeds, mask=index < count)
    request = index // rows
    column = index % rows
    active = index < count * rows
    proposal = tl.load(
        draft_ids + request * draft_stride + (column - 1) * draft_column_stride,
        mask=active & (column > 0),
        other=-1,
    ).to(tl.int64)
    if cap is not None:
        limit = tl.load(cap + request * cap_stride, mask=active, other=0)
        limit = tl.minimum(tl.maximum(limit, 0), rows - 1)
        proposal = tl.where(column > limit, -1, proposal)
    position = tl.load(positions + request * position_stride, mask=active, other=0)
    tl.store(packed + index, proposal, active)
    tl.store(absolute_positions + index, position + column, active)
    tl.store(expanded_mapping + index, request, active)
    tl.store(local_positions + index, column, active)
    tl.store(mapping + index, index, index < count)
    tl.store(cumulative + index, index * rows, index <= count)
    sample = tl.load(do_sample + index * sample_stride, mask=index < count, other=0)
    tl.store(temperature + index, sample, index < count)
    tl.store(rates + index, 1.0, index < rows - 1)


@dataclass(frozen=True, slots=True)
class RejectionMetadata:
    packed_draft: torch.Tensor
    absolute_positions: torch.Tensor
    mapping: torch.Tensor
    expanded_mapping: torch.Tensor
    local_positions: torch.Tensor
    cumulative: torch.Tensor
    temperature: torch.Tensor
    rates: torch.Tensor

    @classmethod
    def prepare(
        cls,
        draft_ids: torch.Tensor,
        positions: torch.Tensor,
        do_sample: torch.Tensor,
        acceptance_cap: torch.Tensor | None,
        graph_input_addresses: torch.Tensor | None = None,
        seed_output: torch.Tensor | None = None,
    ) -> "RejectionMetadata":
        count, steps = draft_ids.shape
        if graph_input_addresses is not None:
            assert graph_input_addresses.shape == (8,)
            assert graph_input_addresses.dtype in (torch.int64, torch.uint64)
            assert graph_input_addresses.device == draft_ids.device
            assert graph_input_addresses.is_contiguous()
            assert seed_output is not None and seed_output.shape == (count,)
            assert seed_output.dtype == torch.int64
            assert (
                seed_output.device == draft_ids.device and seed_output.is_contiguous()
            )
        rows = steps + 1
        total = count * rows
        integers = torch.empty(2 * total, device=draft_ids.device, dtype=torch.int64)
        indices = torch.empty(
            2 * total + 2 * count + 1, device=draft_ids.device, dtype=torch.int32
        )
        floats = torch.empty(
            count + steps, device=draft_ids.device, dtype=torch.float32
        )
        metadata = cls(
            integers[:total],
            integers[total:],
            indices[:count],
            indices[count : count + total],
            indices[count + total : count + 2 * total],
            indices[count + 2 * total :],
            floats[:count],
            floats[count:],
        )
        _prepare_rejection_metadata[(triton.cdiv(max(total, count + 1), 256),)](
            draft_ids,
            positions,
            do_sample,
            acceptance_cap,
            metadata.packed_draft,
            metadata.absolute_positions,
            metadata.mapping,
            metadata.expanded_mapping,
            metadata.local_positions,
            metadata.cumulative,
            metadata.temperature,
            metadata.rates,
            count,
            rows,
            draft_ids.stride(0),
            draft_ids.stride(1),
            positions.stride(0),
            do_sample.stride(0),
            acceptance_cap.stride(0) if acceptance_cap is not None else 0,
            BLOCK=256,
            graph_input_addresses=graph_input_addresses,
            seed_output=seed_output,
        )
        return metadata


@triton.jit(
    do_not_specialize=["count"],
    do_not_specialize_on_alignment=["output_ids", "output_lengths"],
)
def _finalize_rejection_output(
    sampled,
    lengths,
    output_ids,
    output_lengths,
    count,
    rows,
    sampled_stride,
    output_stride,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = index // rows
    column = index % rows
    active = index < count * rows
    length = tl.load(lengths + request, mask=active, other=0)
    token = tl.load(
        sampled + request * sampled_stride + column,
        mask=active & (column < length),
        other=-1,
    )
    tl.store(output_ids + request * output_stride + column, token, active)
    tl.store(output_lengths + request, length, active & (column == 0))


def finalize_rejection_output(
    sampled: torch.Tensor,
    lengths: torch.Tensor,
    output_ids: torch.Tensor,
    output_lengths: torch.Tensor,
) -> None:
    count, rows = output_ids.shape
    _finalize_rejection_output[(triton.cdiv(count * rows, 256),)](
        sampled,
        lengths,
        output_ids,
        output_lengths,
        count,
        rows,
        sampled.stride(0),
        output_ids.stride(0),
        BLOCK=256,
    )
