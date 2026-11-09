"""Instance-owned Graph for PPU Decode and fixed-width MTP metadata."""

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
    DSv4DecodeFmhaImplFP8,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import primary_attention_inputs
from rtp_llm.models_py.modules.dsv4.ppu.metadata.ppu_metadata_update import (
    update_decode_metadata_in_place_ppu,
)


def _table_identity(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class _MtpCommitLayout:
    """Fixed-size inverse maps between packed framework rows and Graph rows."""

    def __init__(self, lengths, batch, width):
        self.lengths = lengths
        self.rows = torch.arange(batch * width, dtype=torch.long, device=lengths.device)
        self.columns = self.rows[:width]
        self.padded_to_compact = torch.empty_like(self.rows)
        self._reverse = torch.empty(
            batch * width * 2, dtype=torch.long, device=lengths.device
        )
        self.compact_to_padded = self._reverse[: batch * width]
        self.valid_rows = torch.empty_like(self.rows, dtype=torch.bool)
        self.update()

    def update(self):
        offsets = self.lengths.cumsum(dim=0) - self.lengths
        valid = self.columns[None, :] < self.lengths[:, None]
        compact = (offsets[:, None] + self.columns[None, :]).reshape(-1)
        self.valid_rows.copy_(valid.reshape(-1))
        self.padded_to_compact.copy_(torch.where(self.valid_rows, compact, 0))
        # Give every padding row a distinct sink. Repeated scatter indices
        # could race with a live row or with each other on the device.
        destinations = torch.where(
            self.valid_rows, compact, self.rows + self.rows.numel()
        )
        self.compact_to_padded.zero_()
        self._reverse.scatter_(0, destinations, self.rows)

    def mask_writes(self, write_slots):
        write_slots.masked_fill_(~self.valid_rows, -1)


class PpuDecodeMetadataGraph(DSv4DecodeFmhaImplFP8):
    """Replay existing metadata arithmetic from framework-owned block tables.

    CudaGraphRunner retains one set of input buffers per captured batch. Keep
    strong references and reject any change to their addresses or geometry.
    Positions are copied into an owned device buffer before each replay, so
    no host tensor address is captured. Metadata output storage remains owned
    by the base class and shared with the model's captured attention kernels.
    """

    def __init__(
        self,
        config,
        device,
        attn_inputs,
        *,
        fused_state_slots=False,
        shared_rope_tables=(),
        mask_inactive=False,
    ):
        device = torch.device(device)
        if (
            device.type != "cuda"
            or config.q_len not in (1, 2, 3, 4, 6)
            or not 0 < config.max_batch_size <= 128
            or not config.paged_pool_specs
        ):
            raise ValueError(
                "PPU metadata Graph requires paged PPU Decode q_len=1/2/3/4/6"
            )
        super().__init__(config, device, attn_inputs)
        self.metadata.active_token_mask = None
        self._metadata_graph = None
        self._commit_layout = None
        self._source_tables = None
        self._source_identity = None
        self._capture_stream = None
        self._positions = torch.empty_like(self.metadata.start_pos)
        if mask_inactive:
            from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

            if SWA_KV not in config.paged_pool_specs:
                raise ValueError("Inactive MoE masking requires a paged SWA pool")
            # Keep capture warmup's original full route workload. Replay
            # updates this owned allocation after the framework copies tables.
            self.metadata.active_token_mask = torch.ones(
                config.max_batch_size * config.q_len,
                dtype=torch.bool,
                device=device,
            )
        self._state_slot_updater = None
        if fused_state_slots:
            from rtp_llm.models_py.kernels.ppu.attention.ppu_decode_state_slots import (
                update_compressor_state_slots,
            )

            self._state_slot_updater = update_compressor_state_slots
        self._rope_sources = {}
        for table in shared_rope_tables:
            if (
                not isinstance(table, torch.Tensor)
                or table.device != self._positions.device
                or table.dtype != torch.complex64
                or table.ndim != 2
                or table.shape[0] < config.max_seq_len
                or table.shape[1] == 0
            ):
                raise ValueError("Shared RoPE requires full complex64 device tables")
            self._rope_sources[id(table)] = table
        self.metadata.rope_freqs_by_source = {
            key: torch.empty(
                (config.max_batch_size * config.q_len, table.shape[1]),
                dtype=table.dtype,
                device=table.device,
            )
            for key, table in self._rope_sources.items()
        }
        self._rope_identity = {
            key: (
                _table_identity(table),
                _table_identity(self.metadata.rope_freqs_by_source[key]),
            )
            for key, table in self._rope_sources.items()
        }
        # Initialize rows for the model's warmup/capture forward without
        # changing the base constructor's full-width MLA scheduling metadata.
        with torch.inference_mode():
            self._update_rope()

    def enable_mtp_commit_layout(self, lengths):
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

        batch, width = self.config.max_batch_size, self.config.q_len
        if (
            width not in (3, 4)
            or set(self.config.paged_pool_specs) != {SWA_KV}
            or lengths.shape != (batch,)
            or lengths.dtype != torch.int32
            or lengths.device != self._positions.device
            or not lengths.is_contiguous()
        ):
            raise ValueError(
                "MTP commit requires a fixed-width SWA-only Graph and captured lengths"
            )
        self._commit_layout = _MtpCommitLayout(lengths, batch, width)
        self.metadata.mtp_commit_layout = self._commit_layout
        self._commit_layout.mask_writes(self.metadata.pool_write_slot_mappings[SWA_KV])

    def _update_rope(self):
        for key, table in self._rope_sources.items():
            torch.index_select(
                table,
                0,
                self.metadata.position_ids_long,
                out=self.metadata.rope_freqs_by_source[key],
            )

    def _validate_inputs(self, attn_inputs):
        rows = self.metadata.rope_freqs_by_source
        if set(rows) != set(self._rope_sources) or any(
            (_table_identity(table), _table_identity(rows[key]))
            != self._rope_identity[key]
            for key, table in self._rope_sources.items()
        ):
            raise ValueError("Shared RoPE table or output storage changed")
        primary = primary_attention_inputs(attn_inputs)
        if primary is None:
            raise ValueError("PPU metadata Graph requires attention inputs")
        verify = getattr(primary, "is_target_verify", False)
        prefill = getattr(primary, "is_prefill", False)
        if self.config.q_len == 1 and (verify or prefill):
            raise ValueError("Single-token metadata Graph requires Decode inputs")
        # Match the reference updater: C++ refreshes prefix_lengths for
        # verify/draft catch-up, while sequence_lengths may be a stale sentinel.
        positions = (
            primary.prefix_lengths
            if verify or (prefill and self.config.q_len > 1)
            else primary.sequence_lengths
        )
        if verify or (prefill and self.config.q_len > 1):
            device_positions = getattr(primary, "prefix_lengths_device", None)
            if device_positions is not None and device_positions.numel():
                if not device_positions.is_cuda:
                    raise ValueError("prefix_lengths_device must be on device")
                positions = device_positions
        else:
            plus_one = getattr(primary, "sequence_lengths_plus_1_device", None)
            if plus_one is not None and plus_one.numel():
                if not plus_one.is_cuda:
                    raise ValueError("sequence_lengths_plus_1_device must be on device")
                positions = plus_one - 1
        batch = self.config.max_batch_size
        if (
            positions.shape != (batch,)
            or positions.dtype != torch.int32
            or not positions.is_contiguous()
            or (positions.is_cuda and positions.device != self._positions.device)
        ):
            raise ValueError("Decode positions must match the captured int32 batch")
        if self._commit_layout is not None:
            if (
                not prefill
                or verify
                or _table_identity(primary.input_lengths_device)
                != _table_identity(self._commit_layout.lengths)
            ):
                raise ValueError(
                    "MTP commit replay changed captured lengths storage or Graph role"
                )
        tables = self._extract_paged_block_tables(attn_inputs)
        if tables is None or set(tables) != set(self._paged_entries_per_block):
            raise ValueError("PPU metadata Graph requires every configured cache tag")
        for table in tables.values():
            if (
                table.ndim != 2
                or table.shape[0] < batch
                or table.shape[1] == 0
                or table.dtype != torch.int32
                or table.device != self._positions.device
            ):
                raise ValueError(
                    "Cache tables must be int32 device matrices for the batch"
                )
        identity = {tag: _table_identity(tensor) for tag, tensor in tables.items()}
        if self._source_identity is not None and identity != self._source_identity:
            raise ValueError(
                "Framework cache table storage changed after metadata capture"
            )
        return positions, tables, identity

    def _update(self):
        layout = self._commit_layout
        if layout is not None:
            layout.update()
        update_decode_metadata_in_place_ppu(
            self.metadata,
            self._positions,
            forbid_realloc=True,
            paged_block_tables=self._source_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
            paged_pool_tokens_per_block=self._paged_tokens_per_block,
            compressor_state_slot_updater=self._state_slot_updater,
            commit_valid_rows=None if layout is None else layout.valid_rows,
        )
        self._update_rope()
        if layout is not None:
            from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

            layout.mask_writes(self.metadata.pool_write_slot_mappings[SWA_KV])
        if self.metadata.active_token_mask is not None:
            from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

            batch, width = self.config.max_batch_size, self.config.q_len
            # Zero is reserved for graph padding and engine fake streams.
            # Scan the entire row: a real SWA block need not be in column 0.
            active = (self._source_tables[SWA_KV][:batch] > 0).any(dim=1)
            self.metadata.active_token_mask.view(batch, width).copy_(
                active[:, None].expand(batch, width)
            )
            if layout is not None:
                self.metadata.active_token_mask.logical_and_(layout.valid_rows)

    def prepare_cuda_graph(self, attn_inputs):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Metadata preparation must precede model Graph replay")
        positions, tables, identity = self._validate_inputs(attn_inputs)
        with torch.inference_mode(), torch.cuda.device(self._positions.device):
            self._positions.copy_(positions, non_blocking=True)
            if self._metadata_graph is None:
                # Bind after model capture, when C++ supplies all tagged tables.
                # Base construction keeps the full-width FlashMLA warmup lengths.
                self._source_tables, self._source_identity = tables, identity
                current = torch.cuda.current_stream(self._positions.device)
                stream = torch.cuda.Stream(device=self._positions.device)
                stream.wait_stream(current)
                try:
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self._update()
                    graph = torch.cuda.CUDAGraph()
                    # RPC threads synchronously copy independent MTP handoff
                    # tensors while this engine thread captures owned metadata.
                    with torch.cuda.graph(
                        graph, stream=stream, capture_error_mode="thread_local"
                    ):
                        self._update()
                finally:
                    current.wait_stream(stream)
                self._capture_stream, self._metadata_graph = stream, graph
            self._metadata_graph.replay()
