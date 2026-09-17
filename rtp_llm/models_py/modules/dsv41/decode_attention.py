"""Decode attention over fixed graph metadata and engine-owned compact pages.

Wrap the target's forty ``V41Attention`` modules with ``V41DecodeAttention``.
Refresh the shared context before replay and call ``begin_forward`` at the
start of the captured target forward. The existing target blocks, Engram, MoE
and aux selection remain caller-owned. No alternate GraphRunner is created.
"""

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.attention import AttentionOwnerCache, V41Attention
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    GLOBAL_OWNERS,
    INDEX_QUERY_OWNERS,
    PAIR_OWNERS,
    SWA_WINDOW,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.compact_reader import (
    GlobalBinding,
    SwaBinding,
)
from rtp_llm.models_py.modules.dsv41.compact_writer import write_compact
from rtp_llm.models_py.modules.dsv41.decode_compressor import (
    V41DecodeOwnerCompressor,
    _inverse_frequencies,
    _tensor,
    is_supported,
)
from rtp_llm.models_py.modules.dsv41.indexer import QUERY_TILE, select_index_positions
from rtp_llm.models_py.modules.dsv41.math import grouped_wo_a, rms_norm


def _same_pages(current, updated):
    if (
        current.region != updated.region
        or current.entries_per_page != updated.entries_per_page
        or current.data.data_ptr() != updated.data.data_ptr()
        or current.data.shape != updated.data.shape
        or current.data.stride() != updated.data.stride()
    ):
        raise ValueError("decode graph pool backing or layout changed")


def _rotate(values, frequencies, inverse=False):
    if inverse:
        frequencies = frequencies.conj()
    tail = torch.view_as_complex(values[..., -64:].float().unflatten(-1, (-1, 2)))
    if values.ndim == 3:
        frequencies = frequencies[:, None, :]
    result = values.clone()
    result[..., -64:] = torch.view_as_real(tail * frequencies).flatten(-2)
    return result


class V41DecodeAttentionContext:
    """One fixed B x Q bucket; page IDs and bounds are refreshed in place.

    Bindings must describe complete local pages after PD/CPRR restoration, not
    CP byte slices. Pool addresses and table capacities are immutable. ``prepare``
    requires complete target SWA and accepted pair state at ``start_positions``;
    a global-only data hit cannot be used as an execution-state certificate.
    """

    def __init__(self, layout, swa, owners, *, batch_size, query_width, max_tokens):
        if (
            type(batch_size) is not int
            or batch_size <= 0
            or type(query_width) is not int
            or query_width not in (1, 6)
            or type(max_tokens) is not int
            or not 0 < max_tokens <= 1048576
            or set(swa) != set(range(40))
            or set(owners) != set(GLOBAL_OWNERS)
        ):
            raise ValueError("decode context needs fixed geometry and all target owners")
        device = swa[0].pages.data.device
        if not is_supported(torch.empty(0, dtype=torch.bfloat16, device=device)):
            raise RuntimeError("decode attention requires a Blackwell CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("create fixed decode context before graph capture")
        self.layout, self.batch_size, self.query_width, self.max_tokens = (
            layout,
            batch_size,
            query_width,
            max_tokens,
        )
        self.device = device
        self.rows = batch_size * query_width
        self.swa, self.owners, self.layers = {}, {}, {}
        self.replay_floors = {}
        for layer, binding in swa.items():
            if (
                binding.validate(device) != batch_size
                or binding.pages.entries_per_page < SWA_WINDOW + query_width - 1
                or binding.pages.entries_per_page != layout.swa_entries
            ):
                raise ValueError("target SWA must retain the entire verify window")
            self.swa[layer] = SwaBinding(
                binding.pages,
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                torch.zeros(batch_size, dtype=torch.int32, device=device),
            )
            self.replay_floors[layer] = torch.zeros(
                batch_size, dtype=torch.int32, device=device
            )
        for layer, binding in owners.items():
            ratio = layer_sources(layer).ratio
            binding.global_kv.validate(batch_size, device)
            binding.index_pages.validate(device)
            table = binding.global_kv.page_table
            if (
                binding.global_kv.compress_ratio != ratio
                or binding.global_kv.pages.entries_per_page
                != layout.token_block_size // ratio
                or binding.index_pages.entries_per_page
                != layout.token_block_size // ratio
                or table.shape[1] * layout.token_block_size < max_tokens
            ):
                raise ValueError("owner pages do not cover the fixed decode capacity")
            _tensor(table, table.shape, torch.int32, device, "global page table")
            _tensor(
                binding.index_table, table.shape, torch.int32, device, "index page table"
            )
            self.owners[layer] = AttentionOwnerCache(
                GlobalBinding(binding.global_kv.pages, torch.zeros_like(table), ratio),
                binding.index_pages,
                torch.zeros_like(binding.index_table),
            )
        self.start_positions = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self.valid_rows = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.offsets = torch.arange(query_width, dtype=torch.int64, device=device)
        self.request_ids = torch.arange(
            batch_size, dtype=torch.int32, device=device
        ).repeat_interleave(query_width)
        self.positions = torch.zeros(self.rows, dtype=torch.int64, device=device)
        self.reader_positions = torch.full(
            (self.rows,), -1, dtype=torch.int32, device=device
        )
        self.active = torch.zeros(self.rows, dtype=torch.bool, device=device)
        self.completed = torch.zeros(40, dtype=torch.bool, device=device)
        self.topk = {
            layer: torch.full((self.rows, 512), -1, dtype=torch.int32, device=device)
            for layer in INDEX_QUERY_OWNERS
        }
        self.candidates = torch.full(
            (self.rows, 2048), -1, dtype=torch.int32, device=device
        )
        self.index_status = {
            layer: torch.zeros(self.rows, dtype=torch.int32, device=device)
            for layer in INDEX_QUERY_OWNERS
        }
        self.inverse_global = _inverse_frequencies(device)
        dims = torch.arange(0, 64, 2, dtype=torch.float32, device=device)
        self.inverse_swa = 1.0 / (10000.0 ** (dims / 64))
        self.global_frequencies = torch.ones(
            (self.rows, 32), dtype=torch.complex64, device=device
        )
        self.swa_frequencies = torch.ones_like(self.global_frequencies)

    @torch.inference_mode()
    def prepare(
        self,
        start_positions,
        valid_rows,
        *,
        swa,
        owners,
        pair_states,
        replay_floors=None,
    ):
        """Run after copy completion, before replay; retain pool/table addresses."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare decode metadata before graph replay")
        _tensor(
            start_positions, (self.batch_size,), torch.int64, self.device, "starts"
        )
        _tensor(valid_rows, (self.batch_size,), torch.int32, self.device, "row counts")
        if (
            set(swa) != set(self.swa)
            or set(owners) != set(self.owners)
            or set(pair_states) != set(PAIR_OWNERS)
            or (replay_floors is not None and set(replay_floors) != set(self.swa))
        ):
            raise ValueError("decode recovery must cover all target regions and pairs")
        self.start_positions.copy_(start_positions)
        self.valid_rows.copy_(valid_rows)
        for layer, destination in self.swa.items():
            source = swa[layer]
            if source.validate(self.device) != self.batch_size:
                raise ValueError("restored SWA has a different request count")
            _same_pages(destination.pages, source.pages)
            floor = self.replay_floors[layer]
            if replay_floors is None:
                floor.zero_()
            else:
                _tensor(
                    replay_floors[layer], floor.shape, floor.dtype, self.device, "floor"
                )
                floor.copy_(replay_floors[layer])
            destination.page_ids.copy_(source.page_ids)
            destination.valid_starts.copy_(source.valid_starts)
            destination.valid_ends.copy_(source.valid_ends)
        for layer, destination in self.owners.items():
            source = owners[layer]
            _same_pages(destination.global_kv.pages, source.global_kv.pages)
            _same_pages(destination.index_pages, source.index_pages)
            for updated, target in (
                (source.global_kv.page_table, destination.global_kv.page_table),
                (source.index_table, destination.index_table),
            ):
                _tensor(updated, target.shape, target.dtype, self.device, "page table")
                target.copy_(updated)
            if layer in self.layers:
                self.layers[layer].compressor.prepare(
                    start_positions,
                    valid_rows,
                    pair_state=pair_states.get(layer),
                )

    @torch.inference_mode()
    def begin_forward(self):
        """Include these metadata kernels in the captured target forward."""
        active = self.offsets[None, :] < self.valid_rows[:, None]
        positions = self.start_positions[:, None] + self.offsets[None, :]
        self.active.copy_(active.flatten())
        self.positions.copy_(torch.where(active, positions, 0).flatten())
        self.reader_positions.copy_(torch.where(active, positions, -1).flatten())
        self.completed.zero_()
        for inverse, frequencies in (
            (self.inverse_global, self.global_frequencies),
            (self.inverse_swa, self.swa_frequencies),
        ):
            phases = self.positions.float()[:, None] * inverse[None, :]
            frequencies.copy_(torch.polar(torch.ones_like(phases), phases))

    def owner_slots(self, layer, table, entries):
        logical = (
            self.positions.view(self.batch_size, self.query_width)
            // layer_sources(layer).ratio
        )
        ids = table.gather(1, (logical // entries).clamp(0, table.shape[1] - 1))
        return (ids.to(torch.int64) * entries + logical % entries).contiguous()

    def check(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("check decode completion only after graph replay")
        statuses = [(~self.completed).to(torch.int32)]
        for layer in self.layers.values():
            statuses.extend((layer.writer_status.flatten(), layer.reader_status.flatten()))
            if layer.compressor is not None:
                statuses.extend(
                    (
                        layer.compressor.global_status.flatten(),
                        layer.compressor.index_status.flatten(),
                    )
                )
        statuses.extend(status.flatten() for status in self.index_status.values())
        # Successful replay needs one host fence; retain ordered diagnostics on error.
        if not bool(torch.cat(statuses).ne(0).any()):
            return
        if not bool(self.completed.all()):
            raise RuntimeError("decode target did not complete all forty attention layers")
        for layer in self.layers.values():
            if bool((layer.writer_status != 0).any()) or bool(
                (layer.reader_status != 0).any()
            ):
                raise RuntimeError("decode attention rejected a compact writer or reader")
            if layer.compressor is not None and (
                bool((layer.compressor.global_status != 0).any())
                or bool((layer.compressor.index_status != 0).any())
            ):
                raise RuntimeError("decode owner did not publish both compact regions")
        if any(bool((status != 0).any()) for status in self.index_status.values()):
            raise RuntimeError("decode indexer rejected query or page metadata")


class V41DecodeAttention(nn.Module):
    """Use loaded target projections with graph-safe producer/scorer/reader calls."""

    def __init__(self, attention, context):
        super().__init__()
        if not isinstance(attention, V41Attention):
            raise TypeError("decode wrapper needs an existing V4.1 attention module")
        if attention.layer in context.layers:
            raise ValueError("decode context already has this attention layer")
        self.attention = attention
        self.layer, self.source = attention.layer, attention.source
        if attention.q_norm.device != context.device:
            raise ValueError("decode attention and graph resources use different devices")
        if (
            attention.compressor is not None
            and attention.compressor.layout != context.layout
        ):
            raise ValueError("decode owner and target graph use different cache layouts")
        self.context = context
        from rtp_llm.models_py.modules.dsv41.flashmla import flashmla_compact_attention

        self.reader = flashmla_compact_attention
        self.compressor = (
            V41DecodeOwnerCompressor.from_owner(
                attention.compressor,
                attention.index_wk,
                attention.index_norm,
                batch_size=context.batch_size,
                query_width=context.query_width,
            )
            if self.source.writes_global
            else None
        )
        self.register_buffer(
            "reader_output",
            torch.empty(
                (context.rows, 64, 512), dtype=torch.bfloat16, device=context.device
            ),
        )
        self.register_buffer(
            "reader_lse",
            torch.empty((context.rows, 64), dtype=torch.float32, device=context.device),
        )
        self.register_buffer(
            "reader_status",
            torch.zeros((context.rows, 64), dtype=torch.int32, device=context.device),
        )
        self.register_buffer(
            "writer_status",
            torch.zeros(context.rows, dtype=torch.int32, device=context.device),
        )
        context.layers[self.layer] = self

    def _score(self, hidden, qr):
        context, attention = self.context, self.attention
        state = context.owners[self.source.index_k_owner]
        query = _rotate(
            attention.index_wq_b(qr).reshape(-1, 32, 128),
            context.global_frequencies,
        )
        weights = (
            F.linear(hidden, attention.index_weights) * (128**-0.5 * 32**-0.5)
        ).contiguous()
        visible = torch.where(
            context.active,
            (context.positions + 1) // self.source.ratio,
            0,
        ).to(torch.int32)
        for first in range(0, context.rows, QUERY_TILE):
            last = min(first + QUERY_TILE, context.rows)
            result = select_index_positions(
                query[first:last].contiguous(),
                weights[first:last],
                state.index_pages,
                state.index_table,
                context.request_ids[first:last],
                visible[first:last],
                layer=self.layer,
                max_visible_length=max(1, context.max_tokens // self.source.ratio),
                candidate_blocks=(
                    context.candidates[first:last] if self.layer > 20 else None
                ),
            )
            context.topk[self.layer][first:last].copy_(result.topk)
            context.index_status[self.layer][first:last].copy_(result.status)
            if self.layer == 20:
                context.candidates[first:last].copy_(result.candidate_blocks)

    @torch.inference_mode()
    def forward(self, hidden, context):
        if context is not self.context:
            raise ValueError("decode wrapper received another graph's context")
        _tensor(
            hidden,
            (context.rows, 5120),
            torch.bfloat16,
            context.device,
            "attention input",
        )
        hidden = hidden.masked_fill(~context.active[:, None], 0)
        attention = self.attention
        frequencies = (
            context.global_frequencies if self.source.ratio else context.swa_frequencies
        )
        qr = rms_norm(attention.wq_a(hidden), attention.q_norm)
        query = _rotate(attention.wq_b(qr).reshape(-1, 64, 512), frequencies)
        kv = _rotate(rms_norm(attention.wkv(hidden), attention.kv_norm), frequencies)
        if self.compressor is not None:
            state = context.owners[self.layer]
            self.compressor(hidden.view(context.batch_size, context.query_width, 5120))
            self.compressor.store(
                state.global_kv.pages,
                state.index_pages,
                context.owner_slots(
                    self.layer,
                    state.global_kv.page_table,
                    state.global_kv.pages.entries_per_page,
                ),
                context.owner_slots(
                    self.layer, state.index_table, state.index_pages.entries_per_page
                ),
            )
        if self.source.scores_queries:
            self._score(hidden, qr)
        swa = context.swa[self.layer]
        entries = swa.pages.entries_per_page
        slots = (
            swa.page_ids.to(torch.int64)[:, None] * entries
            + context.positions.view(context.batch_size, context.query_width) % entries
        ).flatten()
        slots = torch.where(context.active, slots, -1)
        write_compact(kv.contiguous(), swa.pages, slots, status=self.writer_status)
        end = context.start_positions + context.valid_rows
        live = context.valid_rows > 0
        swa.valid_starts.copy_(
            torch.where(
                live,
                torch.maximum(swa.valid_starts, (end - entries).clamp_min(0)),
                swa.valid_starts,
            )
        )
        swa.valid_ends.copy_(torch.where(live, end, swa.valid_ends))
        floors = context.replay_floors[self.layer].repeat_interleave(context.query_width)
        self.reader(
            query.contiguous(),
            context.request_ids,
            context.reader_positions,
            floors,
            swa,
            attention.sinks,
            global_kv=(
                context.owners[self.source.global_owner].global_kv
                if self.source.ratio
                else None
            ),
            global_indices=(
                context.topk[self.source.topk_owner] if self.source.ratio else None
            ),
            output=self.reader_output,
            lse=self.reader_lse,
            status=self.reader_status,
        )
        output = _rotate(self.reader_output, frequencies, inverse=True)
        output = attention.wo_b(
            grouped_wo_a(output.reshape(-1, 8, 4096), attention.wo_a).flatten(1)
        )
        context.completed[self.layer].fill_(True)
        return output.masked_fill(~context.active[:, None], 0)
