"""PPU metadata adapter for the unchanged native Graph preparation protocol."""

import os

import torch

from rtp_llm.config.v41_runtime import env_bool
from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_tagged_attention_inputs
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import PPUDecodeGraphImpl
from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import (
    GRAPH_PACKED_ATTENTION_ENV,
    GRAPH_VISIBLE_INDEXER_ENV,
    GraphPackedAttention,
)
from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import DEPENDENCY_ENV


class SharedPPUDecodeGraphImpl(PPUDecodeGraphImpl):
    """Single-token Decode buckets; the public runner owns bucket/replay state."""

    def __init__(self, model, inputs):
        args, views = model._v4_args, model.shared_weight_views
        attn = as_tagged_attention_inputs(inputs.attention_inputs)
        if not model._is_decode_role or bool(attn.is_prefill):
            raise ValueError("shared Graph requires ordinary DP8 Decode; Prefill stays eager")
        if model._is_speculative or bool(getattr(attn, "is_target_verify", False)):
            raise ValueError("shared speculative Graph modes belong to N9")
        if (args.dp_size, args.ep_size, args.tp_size) != (8, 8, 1):
            raise ValueError("shared Graph requires DP8/EP8 with Attention TP1")
        batch = int(attn.input_lengths.numel())
        if not 0 < batch <= model.shared_graph_runtime.max_batch_size or inputs.input_ids.numel() != batch:
            raise ValueError("shared Graph requires one token per captured request within capacity")
        if inputs.multimodal_inputs.multimodal_features:
            raise ValueError("N4 shared Graph requires text-only Decode")
        self.runtime = model.shared_graph_runtime
        self.rows = batch
        self.runtime.validate()
        if self.runtime.grouped is not None:
            from rtp_llm.models_py.modules.dsv41.ppu.graph_metadata import (
                device_rows,
                require_device_tables,
            )

            require_device_tables(inputs.attention_inputs, model.kv_cache, self.runtime.device)
            device_rows(attn, "sequence_lengths_plus_1_device", self.runtime.device, batch)
        config = dict(views.config, max_position_embeddings=int(args.max_seq_len))
        super().__init__(config, model.kv_cache, inputs.attention_inputs,
                         device=self.runtime.device, batch_size=batch)
        megamoe_graph = self.runtime.grouped is None
        packed_default = megamoe_graph and bool(os.environ.get(DEPENDENCY_ENV))
        self.packed_attention = (
            GraphPackedAttention(self.cache, views.geometry)
            if env_bool(os.environ, GRAPH_PACKED_ATTENTION_ENV, default=packed_default)
            else None
        )
        self.visible_indexer = env_bool(
            os.environ, GRAPH_VISIBLE_INDEXER_ENV, default=megamoe_graph
        )
        if self.runtime.grouped is not None:
            # Native runner warmup owns only block zero. The first real replay
            # fills the count on the prepare stream.
            self.cuda_graph_valid_request_count_device.zero_()

    def prepare_cuda_graph(self, attention_inputs, *, valid_request_count=None):
        if self.runtime.grouped is not None:
            from rtp_llm.models_py.modules.dsv41.ppu.graph_metadata import refresh_device_metadata

            refresh_device_metadata(self, attention_inputs)
            # The runner's prepare fill already updated the aliased scalar.
            destination = self.cuda_graph_valid_request_count_device
            if isinstance(valid_request_count, torch.Tensor):
                if (valid_request_count.shape != destination.shape or
                    valid_request_count.dtype != destination.dtype or
                    valid_request_count.device != destination.device):
                    raise ValueError("expected the public device int32 live-count scalar")
                if valid_request_count.data_ptr() != destination.data_ptr():
                    destination.copy_(valid_request_count)
            elif type(valid_request_count) is int and 0 <= valid_request_count <= self.batch_size:
                destination.fill_(valid_request_count)
            else:
                raise ValueError("invalid grouped Graph live request count")
            if not self.cuda_graph_preparation_in_graph:
                self._update_metadata()
            return
        super().prepare_cuda_graph(attention_inputs, valid_request_count=valid_request_count)

    def prepare_in_graph(self, attention_inputs):
        if self.runtime.grouped is None:
            return super().prepare_in_graph(attention_inputs)
        self._update_metadata()
        if torch.cuda.is_current_stream_capturing():
            self._metadata_in_graph = True
