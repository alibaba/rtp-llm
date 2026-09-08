"""Instance-owned Graph for single-token PPU Decode metadata preparation."""

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    update_decode_metadata_in_place_fp8,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
    DSv4DecodeFmhaImplFP8,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import primary_attention_inputs


def _table_identity(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


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
        shared_rope_tables=()
    ):
        device = torch.device(device)
        if (
            device.type != "cuda"
            or torch.cuda.get_device_name(device) != "ZW-M890P"
            or config.q_len != 1
            or not 0 < config.max_batch_size <= 128
            or not config.paged_pool_specs
        ):
            raise ValueError("PPU metadata Graph requires paged M890P Decode q_len=1")
        super().__init__(config, device, attn_inputs)
        self._metadata_graph = None
        self._source_tables = None
        self._source_identity = None
        self._capture_stream = None
        self._positions = torch.empty_like(self.metadata.start_pos)
        self._state_slot_updater = None
        if fused_state_slots:
            from rtp_llm.platforms.ppu.kernels.ppu_decode_state_slots import (
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
                (config.max_batch_size, table.shape[1]),
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
        if (
            primary is None
            or getattr(primary, "is_target_verify", False)
            or getattr(primary, "is_prefill", False)
        ):
            raise ValueError("PPU metadata Graph accepts single-token Decode only")
        positions = primary.sequence_lengths
        batch = self.config.max_batch_size
        if (
            positions.shape != (batch,)
            or positions.dtype != torch.int32
            or not positions.is_contiguous()
            or (positions.is_cuda and positions.device != self._positions.device)
        ):
            raise ValueError("Decode positions must match the captured int32 batch")
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
        update_decode_metadata_in_place_fp8(
            self.metadata,
            self._positions,
            forbid_realloc=True,
            paged_block_tables=self._source_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
            paged_pool_tokens_per_block=self._paged_tokens_per_block,
            compressor_state_slot_updater=self._state_slot_updater,
        )
        self._update_rope()

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
                    with torch.cuda.graph(graph, stream=stream):
                        self._update()
                finally:
                    current.wait_stream(stream)
                self._capture_stream, self._metadata_graph = stream, graph
            self._metadata_graph.replay()
