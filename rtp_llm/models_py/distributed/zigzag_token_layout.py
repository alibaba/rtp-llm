"""Layout conversion for dedicated Prefill MLA context parallelism.

The surrounding model remains in contiguous token sequence parallelism (SP).
Only MLA uses per-request zig-zag context parallelism (CP).  Conversion uses a
single variable-size AllToAll in each direction and tiny O(batch * world_size)
fragment descriptors; it never builds or transfers a per-token index tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.sequence_parallel import TokenShardLayout


@dataclass(frozen=True)
class _Fragment:
    sp_rank: int
    cp_rank: int
    sp_offset: int
    cp_offset: int
    canonical_offset: int
    length: int


@triton.jit
def _copy_fragments_kernel(
    input_ptr,
    output_ptr,
    descriptors_ptr,
    row_width,
    max_fragment_rows,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    fragment = tl.program_id(0)
    row_offsets = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.program_id(2) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    desc = descriptors_ptr + fragment * 3
    input_start = tl.load(desc).to(tl.int64)
    output_start = tl.load(desc + 1).to(tl.int64)
    fragment_rows = tl.load(desc + 2).to(tl.int64)
    row_mask = (row_offsets < fragment_rows) & (row_offsets < max_fragment_rows)
    column_mask = columns < row_width
    offsets = row_offsets[:, None].to(tl.int64) * row_width + columns[None, :].to(
        tl.int64
    )
    values = tl.load(
        input_ptr + input_start * row_width + offsets,
        mask=row_mask[:, None] & column_mask[None, :],
    )
    tl.store(
        output_ptr + output_start * row_width + offsets,
        values,
        mask=row_mask[:, None] & column_mask[None, :],
    )


def _descriptor_tensor(
    descriptors: Iterable[tuple[int, int, int]], device: torch.device
) -> tuple[torch.Tensor, int]:
    values = list(descriptors)
    if not values:
        return torch.empty((0, 3), dtype=torch.int64, device=device), 0
    return (
        torch.tensor(values, dtype=torch.int64, device=device),
        max(value[2] for value in values),
    )


def _copy_fragments(
    source: torch.Tensor,
    destination: torch.Tensor,
    descriptors: torch.Tensor,
    max_fragment_rows: int,
) -> None:
    if descriptors.numel() == 0:
        return
    if not source.is_cuda or not destination.is_cuda:
        raise ValueError("MLA CP fragment copies require CUDA tensors")
    if not source.is_contiguous() or not destination.is_contiguous():
        raise ValueError("MLA CP fragment copies require contiguous tensors")
    if source.shape[1:] != destination.shape[1:]:
        raise ValueError(
            "MLA CP fragment copy row mismatch: "
            f"source={tuple(source.shape)} destination={tuple(destination.shape)}"
        )
    row_width = source.numel() // source.shape[0]
    grid = (
        descriptors.shape[0],
        triton.cdiv(max_fragment_rows, 8),
        triton.cdiv(row_width, 256),
    )
    _copy_fragments_kernel[grid](
        source,
        destination,
        descriptors,
        row_width,
        max_fragment_rows,
        BLOCK_ROWS=8,
        BLOCK_COLS=256,
        num_warps=8,
    )


class ZigzagTokenLayout:
    """One invocation's reusable SP <-> per-request zig-zag CP plan."""

    def __init__(
        self,
        q_lens: Sequence[int],
        sp_layout: TokenShardLayout,
        world_size: int,
        rank: int,
        device: torch.device,
    ) -> None:
        self.q_lens = tuple(int(value) for value in q_lens)
        self.sp_layout = sp_layout
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.device = device
        if not self.q_lens or any(value <= 0 for value in self.q_lens):
            raise ValueError(f"MLA CP requires positive Q lengths, got {q_lens}")
        if sum(self.q_lens) != sp_layout.logical_tokens:
            raise ValueError(
                "MLA CP Q lengths disagree with SP layout: "
                f"sum={sum(self.q_lens)} logical={sp_layout.logical_tokens}"
            )
        if self.world_size <= 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"invalid MLA CP topology size={self.world_size} rank={self.rank}"
            )

        self.chunk_lens = tuple(
            2 * ((q_len + 2 * self.world_size - 1) // (2 * self.world_size))
            for q_len in self.q_lens
        )
        self.local_cp_tokens = sum(self.chunk_lens)
        self.fragments = self._build_fragments()
        self._build_device_plans()

    def context_parallel_info(self):
        """GLM5.2 ZigZagProcessor metadata, built once per invocation on CPU."""
        from rtp_llm.ops.compute_ops import PyContextParallelParams

        info = PyContextParallelParams()
        chunks = torch.tensor(self.chunk_lens, dtype=torch.int32)
        info.prefill_cp_chunk_lengths = chunks
        info.prefill_actual_input_lengths_cpu = torch.tensor(
            self.q_lens, dtype=torch.int32
        )
        info.prefill_cp_padding_lengths = (
            chunks * self.world_size - info.prefill_actual_input_lengths_cpu
        )
        total = self.local_cp_tokens * self.world_size
        restore = torch.empty(total, dtype=torch.int32)
        mask = torch.zeros(total, dtype=torch.int32)
        seq_offset = local_offset = 0
        for q_len, chunk in zip(self.q_lens, self.chunk_lens, strict=True):
            pair = chunk // 2
            mask[seq_offset : seq_offset + q_len] = 1
            for rank in range(self.world_size):
                base = rank * self.local_cp_tokens + local_offset
                front = seq_offset + rank * pair
                back = seq_offset + (2 * self.world_size - 1 - rank) * pair
                restore[front : front + pair] = torch.arange(
                    base, base + pair, dtype=torch.int32
                )
                restore[back : back + pair] = torch.arange(
                    base + pair, base + chunk, dtype=torch.int32
                )
            seq_offset += chunk * self.world_size
            local_offset += chunk
        info.prefill_qkv_restore_indice = restore
        info.prefill_qkv_padding_mask = mask
        info.prefill_shuffle_indices = torch.empty(0, dtype=torch.int32)
        return info

    def _build_fragments(self) -> tuple[_Fragment, ...]:
        fragments: list[_Fragment] = []
        request_start = 0
        cp_request_offsets = [0] * self.world_size
        sp_tokens = self.sp_layout.local_tokens
        for q_len, chunk_len in zip(self.q_lens, self.chunk_lens, strict=True):
            pair = chunk_len // 2
            request_end = request_start + q_len
            for cp_rank in range(self.world_size):
                cp_base = cp_request_offsets[cp_rank]
                ranges = (
                    (request_start + cp_rank * pair, cp_base),
                    (
                        request_start + (2 * self.world_size - 1 - cp_rank) * pair,
                        cp_base + pair,
                    ),
                )
                for canonical_start, cp_start in ranges:
                    canonical_end = min(canonical_start + pair, request_end)
                    cursor = canonical_start
                    while cursor < canonical_end:
                        sp_rank = cursor // sp_tokens
                        sp_rank_start = sp_rank * sp_tokens
                        fragment_end = min(canonical_end, sp_rank_start + sp_tokens)
                        fragments.append(
                            _Fragment(
                                sp_rank=sp_rank,
                                cp_rank=cp_rank,
                                sp_offset=cursor - sp_rank_start,
                                cp_offset=cp_start + cursor - canonical_start,
                                canonical_offset=cursor,
                                length=fragment_end - cursor,
                            )
                        )
                        cursor = fragment_end
                cp_request_offsets[cp_rank] += chunk_len
            request_start = request_end
        return tuple(fragments)

    def _build_device_plans(self) -> None:
        send_counts = [0] * self.world_size
        forward_pack: list[tuple[int, int, int]] = []
        send_offset = 0
        for cp_rank in range(self.world_size):
            for fragment in self.fragments:
                if fragment.sp_rank == self.rank and fragment.cp_rank == cp_rank:
                    forward_pack.append(
                        (fragment.sp_offset, send_offset, fragment.length)
                    )
                    send_counts[cp_rank] += fragment.length
                    send_offset += fragment.length

        recv_counts = [0] * self.world_size
        forward_unpack: list[tuple[int, int, int]] = []
        recv_offset = 0
        for sp_rank in range(self.world_size):
            for fragment in self.fragments:
                if fragment.cp_rank == self.rank and fragment.sp_rank == sp_rank:
                    forward_unpack.append(
                        (recv_offset, fragment.cp_offset, fragment.length)
                    )
                    recv_counts[sp_rank] += fragment.length
                    recv_offset += fragment.length

        reverse_pack: list[tuple[int, int, int]] = []
        reverse_send_offset = 0
        for sp_rank in range(self.world_size):
            for fragment in self.fragments:
                if fragment.cp_rank == self.rank and fragment.sp_rank == sp_rank:
                    reverse_pack.append(
                        (fragment.cp_offset, reverse_send_offset, fragment.length)
                    )
                    reverse_send_offset += fragment.length

        reverse_unpack: list[tuple[int, int, int]] = []
        reverse_recv_offset = 0
        reverse_recv_counts = [0] * self.world_size
        for cp_rank in range(self.world_size):
            for fragment in self.fragments:
                if fragment.sp_rank == self.rank and fragment.cp_rank == cp_rank:
                    reverse_unpack.append(
                        (reverse_recv_offset, fragment.sp_offset, fragment.length)
                    )
                    reverse_recv_counts[cp_rank] += fragment.length
                    reverse_recv_offset += fragment.length

        self.send_counts = send_counts
        self.recv_counts = recv_counts
        self.reverse_send_counts = recv_counts
        self.reverse_recv_counts = reverse_recv_counts
        if sum(send_counts) != self.sp_layout.local_valid_tokens:
            raise AssertionError(
                "MLA CP forward plan did not consume every valid SP row: "
                f"planned={sum(send_counts)} valid={self.sp_layout.local_valid_tokens}"
            )
        if sum(recv_counts) > self.local_cp_tokens:
            raise AssertionError("MLA CP receive plan exceeds local padded rows")
        if reverse_recv_counts != send_counts:
            raise AssertionError("MLA CP reverse AllToAll counts disagree")

        self.forward_pack, self.forward_pack_max = _descriptor_tensor(
            forward_pack, self.device
        )
        self.forward_unpack, self.forward_unpack_max = _descriptor_tensor(
            forward_unpack, self.device
        )
        self.reverse_pack, self.reverse_pack_max = _descriptor_tensor(
            reverse_pack, self.device
        )
        self.reverse_unpack, self.reverse_unpack_max = _descriptor_tensor(
            reverse_unpack, self.device
        )

    def _all_to_all(
        self,
        tensor: torch.Tensor,
        *,
        pack: torch.Tensor,
        pack_max: int,
        unpack: torch.Tensor,
        unpack_max: int,
        send_counts: list[int],
        recv_counts: list[int],
        output_rows: int,
    ) -> torch.Tensor:
        # Keep the CPU-only layout planner independent from the compiled RTP
        # extension. Production reaches this import only for the CUDA transfer.
        from rtp_llm.models_py.distributed.collective_torch import (
            Group,
            get_process_group,
        )

        send = tensor.new_empty((sum(send_counts), *tensor.shape[1:]))
        _copy_fragments(tensor, send, pack, pack_max)
        recv = tensor.new_empty((sum(recv_counts), *tensor.shape[1:]))
        torch.distributed.all_to_all_single(
            recv,
            send,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=get_process_group(Group.TP),
        )
        output = tensor.new_zeros((output_rows, *tensor.shape[1:]))
        _copy_fragments(recv, output, unpack, unpack_max)
        return output

    def sp_to_cp(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != self.sp_layout.local_tokens:
            raise ValueError(
                "MLA CP expected the invocation's SP shard: "
                f"rows={tensor.shape[0]} expected={self.sp_layout.local_tokens}"
            )
        return self._all_to_all(
            tensor.contiguous(),
            pack=self.forward_pack,
            pack_max=self.forward_pack_max,
            unpack=self.forward_unpack,
            unpack_max=self.forward_unpack_max,
            send_counts=self.send_counts,
            recv_counts=self.recv_counts,
            output_rows=self.local_cp_tokens,
        )

    def cp_to_sp(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != self.local_cp_tokens:
            raise ValueError(
                "MLA CP expected the invocation's CP shard: "
                f"rows={tensor.shape[0]} expected={self.local_cp_tokens}"
            )
        return self._all_to_all(
            tensor.contiguous(),
            pack=self.reverse_pack,
            pack_max=self.reverse_pack_max,
            unpack=self.reverse_unpack,
            unpack_max=self.reverse_unpack_max,
            send_counts=self.reverse_send_counts,
            recv_counts=self.reverse_recv_counts,
            output_rows=self.sp_layout.local_tokens,
        )


__all__ = ["ZigzagTokenLayout"]
