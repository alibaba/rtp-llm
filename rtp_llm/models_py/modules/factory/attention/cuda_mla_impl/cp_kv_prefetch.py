"""Prefetch immutable round-robin CP history from a future staged MLA layer."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.pynccl_cp import all_gather_kv_overlap
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _starts(lengths):
    return [0, *torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()[:-1]]


@dataclass(frozen=True)
class PrefixPlan:
    history_actual: list[int]
    history_padded: list[int]
    history_restore: torch.Tensor
    history_dest: torch.Tensor
    new_local_rows: torch.Tensor
    new_restore: torch.Tensor
    new_dest: torch.Tensor
    new_padded: int


def build_prefix_plan(prefixes, lengths, world_size, rank, owner_tokens):
    """Build a CPU plan for unpadded old and new rows in global order."""
    if (
        not prefixes
        or len(prefixes) != len(lengths)
        or not 0 <= rank < world_size
        or owner_tokens <= 0
        or any(x < 0 for x in [*prefixes, *lengths])
    ):
        raise ValueError("invalid CP history layout")

    cycle = world_size * owner_tokens
    padded = [((prefix + cycle - 1) // cycle) * owner_tokens for prefix in prefixes]
    per_rank_history = sum(padded)
    actual = [
        prefix // cycle * owner_tokens
        + max(0, min(owner_tokens, prefix % cycle - rank * owner_tokens))
        for prefix in prefixes
    ]
    history_restore, history_dest, new_dest, row_owners = [], [], [], []
    global_start = 0
    for prefix, count, history_start in zip(prefixes, lengths, _starts(padded)):
        blocks = torch.arange((prefix + owner_tokens - 1) // owner_tokens)
        offsets = torch.arange(min(prefix, owner_tokens))
        starts = (blocks % world_size) * per_rank_history + (
            blocks // world_size
        ) * owner_tokens
        history_restore.append(
            (starts[:, None] + offsets + history_start).flatten()[:prefix]
        )
        history_dest.append(torch.arange(global_start, global_start + prefix))
        new_dest.append(
            torch.arange(global_start + prefix, global_start + prefix + count)
        )
        row_owners.append(
            torch.arange(prefix, prefix + count) // owner_tokens % world_size
        )
        global_start += prefix + count

    owners = torch.cat(row_owners)
    new_rows = [
        torch.where(owners == owner_rank)[0] for owner_rank in range(world_size)
    ]
    new_padded = max(map(len, new_rows))
    new_restore = torch.empty(sum(lengths), dtype=torch.int64)
    for owner_rank, rows in enumerate(new_rows):
        new_restore[rows] = owner_rank * new_padded + torch.arange(len(rows))
    return PrefixPlan(
        actual,
        padded,
        torch.cat(history_restore),
        torch.cat(history_dest),
        new_rows[rank],
        new_restore,
        torch.cat(new_dest),
        new_padded,
    )


@triton.jit
def _restore_rows_kernel(
    Source,
    SourceRows,
    Target,
    TargetRows,
    count,
    width: tl.constexpr,
    rows_per_tile: tl.constexpr,
    cols_per_tile: tl.constexpr,
):
    rows = tl.program_id(0) * rows_per_tile + tl.arange(0, rows_per_tile)
    cols = tl.arange(0, cols_per_tile)
    source_rows = tl.load(SourceRows + rows, rows < count, other=0).to(tl.int64)
    target_rows = tl.load(TargetRows + rows, rows < count, other=0).to(tl.int64)
    mask = (rows[:, None] < count) & (cols[None, :] < width)
    values = tl.load(
        Source + source_rows[:, None] * width + cols[None, :], mask, other=0
    )
    tl.store(Target + target_rows[:, None] * width + cols[None, :], values, mask)


def _restore_rows(source, source_rows, target, target_rows):
    count = source_rows.numel()
    if count:
        width = target.shape[1]
        _restore_rows_kernel[(triton.cdiv(count, 4),)](
            source,
            source_rows,
            target,
            target_rows,
            count,
            width,
            4,
            triton.next_power_of_2(width),
        )


class CrossLayerKvPrefetch:
    """Runs prefix and fresh-row gathers in a fixed order on one CP side stream."""

    def __init__(self, op, staged, stream, group):
        self.op, self.staged, self.stream, self.group = op, staged, stream, group
        self.table = op.block_table
        self.device = self.table.device
        self.pending = None
        prefix = op.attn_inputs.prefix_lengths.reshape(-1).tolist()
        lengths = op.cp_info.prefill_actual_input_lengths_cpu.reshape(-1).tolist()
        self.plan = plan = build_prefix_plan(
            prefix,
            lengths,
            op.prefill_cp_size,
            op.prefill_cp_rank,
            op.kv_owner_tokens_per_block,
        )

        def upload(values, dtype=torch.int64):
            return torch.as_tensor(values, dtype=dtype).to(
                self.device, non_blocking=True
            )

        self.history_lengths = upload(plan.history_actual, torch.int32)
        self.history_starts = upload(_starts(plan.history_actual), torch.int32)
        self.history_actual = sum(plan.history_actual)
        self.history_padded = sum(plan.history_padded)
        self.history_scatter = upload(
            torch.cat(
                [
                    torch.arange(start, start + size)
                    for start, size in zip(
                        _starts(plan.history_padded), plan.history_actual
                    )
                ]
            )
        )
        self.history_restore = upload(plan.history_restore)
        self.history_dest = upload(plan.history_dest)
        self.new_local_rows = upload(plan.new_local_rows)
        self.new_restore = upload(plan.new_restore)
        self.new_dest = upload(plan.new_dest)
        self.new_length = upload([len(plan.new_local_rows)], torch.int32)
        self.zero = upload([0], torch.int32)
        self.total_rows = sum(prefix) + sum(lengths)
        self.width = op.kv_lora_rank + op.qk_rope_head_dim

    def _buffer(self, rows, zero=False):
        create = torch.zeros if zero else torch.empty
        return create((rows, self.width), dtype=torch.bfloat16, device=self.device)

    @staticmethod
    def _pack(source, target, table, lengths, starts, count):
        if count:
            rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                source.view(torch.uint8),
                target,
                table,
                lengths,
                starts,
                table.shape[0],
                count,
            )

    def prefetch(self, layer):
        if layer >= len(self.staged.backing):
            return
        source, table, ready = self.staged.borrow_next_layer(layer, self.table)
        if ready is not None:
            self.stream.wait_event(ready)
        # Actual owned lengths avoid tail padding whose pages a rank doesn't own.
        local = self._buffer(self.history_padded, zero=True)
        actual = self._buffer(self.history_actual)
        self._pack(
            source,
            actual,
            table,
            self.history_lengths,
            self.history_starts,
            self.history_actual,
        )
        local.index_copy_(0, self.history_scatter, actual)
        del actual
        gathered = all_gather_kv_overlap(local, self.group)
        del local
        # Restore straight into the future consumer buffer. No compact
        # prefix copy has to stay live next to the full KV during attention.
        result = self._buffer(self.total_rows)
        _restore_rows(gathered, self.history_restore, result, self.history_dest)
        done = torch.cuda.Event()
        done.record(self.stream)
        self.pending = layer, result, done

    def finish_layer(self, layer, kv_cache, slots):
        """Join prefetched history and append just the new, re-read FP8 rows."""
        if self.pending is None or self.pending[0] != layer:
            raise RuntimeError(f"CP history for layer {layer} was not prefetched")
        _, result, _ = self.pending
        self.pending = None
        if self.plan.new_padded:
            local = self._buffer(self.plan.new_padded, zero=True)
            selected = (
                slots.index_select(0, self.new_local_rows)
                .to(torch.int32)
                .reshape(1, -1)
            )
            # Each row in the flat cache is a one-token page. Read the exact
            # packed bytes written by MLA, then use the existing FP8 upconverter.
            source = kv_cache.kv_cache_base.flatten(0, 1).unsqueeze(1)
            self._pack(
                source,
                local,
                selected,
                self.new_length,
                self.zero,
                len(self.plan.new_local_rows),
            )
            received = all_gather_kv_overlap(local, self.group)
            _restore_rows(received, self.new_restore, result, self.new_dest)
        return result
