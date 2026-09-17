import os
from typing import Any

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.glm53_prefill_parallel import MlaCPParallelismView, mla_cp_enabled
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_trim,
    all_reduce,
)
from rtp_llm.models_py.distributed.sequence_parallel import shard_tokens
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    Embedding,
    LinearFactory,
    MultimodalEmbeddingInjector,
    RMSNorm,
    RMSResNorm,
)
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    write_typed_aux_cache_regions,
)
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    prepare_mtp_multimodal_inputs,
)
from rtp_llm.models_py.modules.dsv4.forward_metadata import scoped_forward_metadata
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.hybrid.glm5_cmp import should_enable_glm5_cmp
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from torch import nn

_MTP_INDEXER_ROLE_NORMAL = 0
_MTP_INDEXER_ROLE_SEED = 1
_MTP_INDEXER_ROLE_REUSE = 2


def _mtp_indexer_share_enabled() -> bool:
    # Hard, process-wide opt-in. Model metadata must not silently enable an
    # experimental execution path.
    return os.getenv("RTP_LLM_ENABLE_MTP_INDEXER_SHARE") == "1"


def _mtp_indexer_share_active(
    model_config: ModelConfig,
    parallelism_config: ParallelismConfig,
    layer_num: int,
    topk: int,
) -> bool:
    cp_config = getattr(parallelism_config, "prefill_cp_config", None)
    context_parallel_enabled = bool(cp_config is not None and cp_config.is_enabled())
    return bool(
        _mtp_indexer_share_enabled()
        and getattr(model_config, "index_share_for_mtp_iteration", False)
        and not context_parallel_enabled
        and layer_num == 1
        and topk > 0
    )


class GenericMoeMTPModel(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.moe_config = moe_config
        self.max_generate_batch_size = max_generate_batch_size
        self.device_resource_config = device_resource_config
        cp_config = getattr(parallelism_config, "prefill_cp_config", None)
        self.context_parallel_enabled = bool(
            cp_config is not None and cp_config.is_enabled()
        )
        self.prefill_mla_cp = mla_cp_enabled(
            model_config.model_type,
            parallelism_config.role_type,
            is_glm53_mtp=getattr(model_config, "is_glm53_mtp", False),
        )
        self.mla_parallelism = (
            MlaCPParallelismView(parallelism_config)
            if self.prefill_mla_cp
            else parallelism_config
        )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.pre_fc_norm_embedding = RMSNorm(
            weights.global_weights[W.multi_tokens_predict_enorm],
            eps=model_config.layernorm_eps,
        )
        self.pre_fc_norm_hidden = RMSNorm(
            weights.global_weights[W.multi_tokens_predict_hnorm],
            eps=model_config.layernorm_eps,
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.multi_tokens_predict_eh_proj
        )

        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.global_weights[W.multi_tokens_predict_final_ln_gamma],
            eps=model_config.layernorm_eps,
        )
        topk = int(model_config.attn_config.indexer_topk)
        self._mtp_indexer_share_enabled = _mtp_indexer_share_active(
            model_config, parallelism_config, self.layer_num, topk
        )
        self._mtp_indexer_role = _MTP_INDEXER_ROLE_NORMAL
        if getattr(model_config, "is_glm53_mtp", False):
            topk = int(model_config.attn_config.sparse_attention_topk)
        buffer_device = weights.global_weights[W.multi_tokens_predict_enorm].device
        buffer_shape = (
            (max_generate_batch_size, topk)
            if self._mtp_indexer_share_enabled
            else (0, 0)
        )
        self._mtp_shared_topk_indices = torch.zeros(
            buffer_shape,
            dtype=torch.int32,
            device=buffer_device,
        )

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        if not super().initialize(init_resource):
            return False
        from rtp_llm.models_py.modules.hybrid.indexer import (
            bind_indexer_block_table_group_ids,
        )

        bind_indexer_block_table_group_ids(self.layers, self.kv_cache)
        return True

    def clone_for_cuda_graph(self) -> "GenericMoeMTPModel":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)

        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.weight = self.weight
        clone.fmha_config = self.fmha_config
        clone.py_hw_kernel_config = self.py_hw_kernel_config
        clone.micro_batch_size = self.micro_batch_size
        clone.layer_num = self.layer_num
        clone.vocab_size = self.vocab_size
        clone.kv_cache = None
        clone.device_type = self.device_type
        clone.params_dict = {}
        clone.moe_config = self.moe_config
        clone.max_generate_batch_size = self.max_generate_batch_size
        clone.device_resource_config = self.device_resource_config
        clone.context_parallel_enabled = self.context_parallel_enabled
        clone.prefill_mla_cp = self.prefill_mla_cp
        clone.mla_parallelism = self.mla_parallelism

        clone.embed_tokens = self.embed_tokens
        clone.multimodal_embedding_injector = self.multimodal_embedding_injector
        clone.pre_fc_norm_embedding = self.pre_fc_norm_embedding
        clone.pre_fc_norm_hidden = self.pre_fc_norm_hidden
        clone.fc = self.fc
        clone.layers = nn.ModuleList(
            [
                (
                    layer.clone_for_cuda_graph(draft_prefill=True)
                    if hasattr(layer, "clone_for_cuda_graph")
                    else layer
                )
                for layer in self.layers
            ]
        )
        clone.norm = self.norm
        clone._mtp_indexer_share_enabled = self._mtp_indexer_share_enabled
        clone._mtp_indexer_role = _MTP_INDEXER_ROLE_NORMAL
        clone._mtp_shared_topk_indices = self._mtp_shared_topk_indices

        return clone

    def set_mtp_indexer_role(self, role: int) -> None:
        if role not in (
            _MTP_INDEXER_ROLE_NORMAL,
            _MTP_INDEXER_ROLE_SEED,
            _MTP_INDEXER_ROLE_REUSE,
        ):
            raise ValueError(f"invalid MTP indexer role: {role}")
        self._mtp_indexer_role = (
            role if self._mtp_indexer_share_enabled else _MTP_INDEXER_ROLE_NORMAL
        )

    def load_mtp_indexer_topk(self, topk_indices: torch.Tensor) -> None:
        if not self._mtp_indexer_share_enabled:
            return
        batch_size = topk_indices.size(0)
        expected_topk = self._mtp_shared_topk_indices.size(1)
        if (
            topk_indices.dtype != torch.int32
            or topk_indices.device != self._mtp_shared_topk_indices.device
            or topk_indices.dim() != 2
            or topk_indices.size(1) != expected_topk
            or batch_size > self._mtp_shared_topk_indices.size(0)
        ):
            raise RuntimeError(
                f"invalid request indexer seed: shape={topk_indices.shape}, "
                f"dtype={topk_indices.dtype}"
            )
        self._mtp_shared_topk_indices[:batch_size].copy_(topk_indices)

    def snapshot_mtp_indexer_topk(self, batch_size: int) -> torch.Tensor:
        if not self._mtp_indexer_share_enabled:
            return self._mtp_shared_topk_indices
        if batch_size < 0 or batch_size > self._mtp_shared_topk_indices.size(0):
            raise RuntimeError(f"invalid MTP indexer snapshot batch size: {batch_size}")
        return self._mtp_shared_topk_indices[:batch_size]

    def _get_mtp_reuse_topk_indices(
        self, hidden_states: torch.Tensor, fmha_impl: Any
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        topk = self._mtp_shared_topk_indices.size(1)
        if batch_size > self._mtp_shared_topk_indices.size(0):
            raise RuntimeError(
                "MTP indexer share batch exceeds fixed buffer: "
                f"batch={batch_size}, capacity={self._mtp_shared_topk_indices.size(0)}"
            )
        topk_indices = self._mtp_shared_topk_indices[:batch_size, :topk]
        if getattr(self.config, "is_glm53_mtp", False):
            # Official index_share_for_mtp_iteration freezes the expanded raw
            # selection, including the first step's incomplete KPool tail.
            return topk_indices
        positions = getattr(fmha_impl.fmha_params, "positions_d", None)
        if (
            not torch.is_tensor(positions)
            or positions.numel() != batch_size
            or positions.device != topk_indices.device
        ):
            raise RuntimeError("MTP indexer share requires device positions_d per row")

        # The shared selection lives in compressed KPool space. A four-token
        # group becomes addressable only when the current raw position closes
        # that group; incomplete tokens are supplied separately by SparseMLA's
        # causal tail. Ratio one keeps the original per-token behavior.
        group_size = int(getattr(self.config.attn_config, "indexer_compress_ratio", 1))
        if group_size <= 0:
            raise RuntimeError(
                f"invalid indexer_compress_ratio for MTP reuse: {group_size}"
            )
        group_complete = torch.remainder(positions + 1, group_size) == 0
        pooled_position_column = torch.div(
            positions, group_size, rounding_mode="floor"
        ).reshape(-1, 1)
        with_current_position = torch.cat(
            [pooled_position_column, topk_indices[:, :-1]], dim=1
        )
        position_present = (topk_indices == pooled_position_column).any(
            dim=1, keepdim=True
        )
        should_insert = group_complete.reshape(-1, 1) & ~position_present
        topk_indices.copy_(
            torch.where(should_insert, with_current_position, topk_indices)
        )
        return topk_indices

    def _store_mtp_topk_indices(
        self, topk_indices: torch.Tensor, seed_rows: torch.Tensor, positions=None
    ) -> None:
        batch_size = seed_rows.numel()
        topk = int(self.config.attn_config.indexer_topk)
        if topk_indices is None and getattr(self.config, "is_glm53_mtp", False):
            if positions is None:
                raise RuntimeError("GLM53 MTP seed requires raw token positions")
            selected = self._expand_mtp_seed(
                None, positions.index_select(0, seed_rows.long()) + 1
            )
            self._mtp_shared_topk_indices[:batch_size].copy_(selected)
            return
        valid = (
            torch.is_tensor(topk_indices)
            and topk_indices.dtype == torch.int32
            and topk_indices.device == seed_rows.device
            and topk_indices.dim() == 2
            and topk_indices.size(1) == topk
            and seed_rows.dtype == torch.int32
            and seed_rows.dim() == 1
            and batch_size <= self._mtp_shared_topk_indices.size(0)
        )
        if not valid:
            raise RuntimeError(
                "invalid MTP indexer share output: "
                f"shape={getattr(topk_indices, 'shape', None)}, "
                f"dtype={getattr(topk_indices, 'dtype', None)}, "
                f"batch={batch_size}, topk={topk}"
            )
        selected = topk_indices.index_select(0, seed_rows.to(torch.int64))
        if getattr(self.config, "is_glm53_mtp", False):
            if positions is None:
                raise RuntimeError("GLM53 MTP seed requires raw token positions")
            selected = self._expand_mtp_seed(
                selected, positions.index_select(0, seed_rows.long()) + 1
            )
        self._mtp_shared_topk_indices[:batch_size].copy_(selected)

    def _expand_mtp_seed(self, pooled, raw_lengths):
        from rtp_llm.models_py.modules.indexer_grouping import (
            append_incomplete_tail_indices,
            expand_indexer_group_indices,
            fused_expand_indexer_groups_with_tail,
        )

        group_size = int(self.config.attn_config.indexer_compress_ratio)
        if pooled is None:
            # Dense short-context fast path attends to every causal token.
            pooled = torch.arange(
                int(self.config.attn_config.indexer_topk),
                dtype=torch.int32,
                device=raw_lengths.device,
            ).expand(raw_lengths.numel(), -1)
        expanded = fused_expand_indexer_groups_with_tail(
            pooled, raw_lengths, group_size
        )
        if expanded is None:
            expanded = append_incomplete_tail_indices(
                expand_indexer_group_indices(
                    pooled, group_size, raw_sequence_lengths=raw_lengths
                ),
                raw_lengths,
                group_size,
            )
        return expanded

    def _store_prefill_cp_seed(self, topk_indices, fmha, attn_inputs):
        layout = fmha.glm53_cp_layout
        batch_size = len(layout.q_lens)
        if batch_size > self._mtp_shared_topk_indices.size(0):
            raise RuntimeError("MTP CP seed batch exceeds fixed buffer")
        device = self._mtp_shared_topk_indices.device
        raw_lengths = attn_inputs.prefix_lengths.to(
            device=device, dtype=torch.int32
        ) + torch.tensor(layout.q_lens, device=device, dtype=torch.int32)
        selected = None
        if topk_indices is not None:
            # Indexer returns valid local queries, excluding CP padding. Restore
            # their CP offsets, then communicate only each request's final row.
            local = torch.full(
                (layout.local_cp_tokens, topk_indices.shape[-1]),
                -1,
                dtype=torch.int32,
                device=device,
            )
            local.index_copy_(0, fmha.cp_params.total_local_ids.long(), topk_indices)
            selected = torch.zeros(
                (batch_size, local.shape[1]), dtype=torch.int32, device=device
            )
            end = 0
            for request, length in enumerate(layout.q_lens):
                end += length
                for fragment in layout.fragments:
                    if (
                        fragment.cp_rank == layout.rank
                        and fragment.canonical_offset
                        <= end - 1
                        < fragment.canonical_offset + fragment.length
                    ):
                        row = fragment.cp_offset + end - 1 - fragment.canonical_offset
                        selected[request].copy_(local[row])
                        break
            selected = all_reduce(selected, Group.TP)
        self._mtp_shared_topk_indices[:batch_size].copy_(
            self._expand_mtp_seed(selected, raw_lengths)
        )

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        if not self.prefill_mla_cp:
            return super().prepare_fmha_impl(inputs, is_cuda_graph)
        from rtp_llm.models_py.distributed.glm53_mla_cp import prepare_mla_cp_fmha

        return prepare_mla_cp_fmha(self, inputs, is_cuda_graph)

    @staticmethod
    def _mask_prefill_position_zero_embeddings(inputs_embeds, attention_inputs):
        # CP positions describe rank-local zigzag rows. The embedding still has
        # canonical rows here; mask each zero-prefix request's first row before
        # sharding, including ragged/multimodal batches and cache-hit suffixes.
        starts = attention_inputs.cu_seqlens[:-1].to(
            device=inputs_embeds.device, dtype=torch.int64
        )
        prefix = attention_inputs.prefix_lengths.to(device=inputs_embeds.device)
        if starts.numel() != prefix.numel():
            raise ValueError("MTP CP prefix lengths disagree with request boundaries")
        mask = torch.zeros(
            inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device
        )
        mask.scatter_(0, starts, prefix == 0)
        return torch.where(mask.unsqueeze(-1), 0, inputs_embeds)

    @scoped_forward_metadata
    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        typed_aux_cache_store = create_write_cache_store_impl(
            inputs.attention_inputs, self.kv_cache
        )
        multimodal_features = inputs.multimodal_inputs.multimodal_features
        if multimodal_features:
            input_ids, shifted_features, shifted_locs = prepare_mtp_multimodal_inputs(
                input_ids,
                multimodal_features,
                inputs.multimodal_inputs.mm_features_locs,
                inputs.attention_inputs.cu_seqlens,
                getattr(inputs.attention_inputs, "cu_seqlens_host", None),
                already_shifted=self.context_parallel_enabled,
            )
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.multimodal_embedding_injector(
                inputs_embeds, shifted_features, shifted_locs
            )
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        last_hidden_states = inputs.input_hiddens
        cp_layout = fmha_impl.glm53_cp_layout if self.prefill_mla_cp else None
        if cp_layout is not None:
            if inputs_embeds.shape != last_hidden_states.shape:
                raise ValueError("MTP CP embedding and input hidden rows must align")
            inputs_embeds = self._mask_prefill_position_zero_embeddings(
                inputs_embeds, inputs.attention_inputs
            )
            inputs_embeds = shard_tokens(inputs_embeds, cp_layout.sp_layout)
            last_hidden_states = shard_tokens(last_hidden_states, cp_layout.sp_layout)
        else:
            inputs_embeds = self._mask_position_zero_embeddings(
                inputs_embeds, fmha_impl
            )

        e_norm = self.pre_fc_norm_embedding(inputs_embeds)
        h_norm = self.pre_fc_norm_hidden(last_hidden_states)
        cat_hidden_states = torch.cat([e_norm, h_norm], -1)
        hidden_states = self.fc(cat_hidden_states)

        # These front-end activations are not consumed by the decoder layer.
        # Without an explicit del, Python keeps their independent storages live
        # until the whole MTP forward returns, overlapping every Indexer/SparseMLA
        # layer. CUDA allocator stream tracking makes this release asynchronous
        # and safe; do not synchronize or call empty_cache on the hot path.
        del inputs_embeds, e_norm, h_norm, cat_hidden_states

        residual = torch.zeros_like(hidden_states)
        reuse_topk_indices = (
            self._mtp_indexer_share_enabled
            and self._mtp_indexer_role == _MTP_INDEXER_ROLE_REUSE
            and not inputs.attention_inputs.is_prefill
        )
        prev_topk_indices = (
            self._get_mtp_reuse_topk_indices(hidden_states, fmha_impl)
            if reuse_topk_indices
            else None
        )
        enable_cmp = should_enable_glm5_cmp(
            self.layers,
            self.layer_num,
            hidden_states,
            fmha_impl,
            self.kv_cache,
            force_reuse_topk_indices=reuse_topk_indices,
        )
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            if cp_layout is not None:
                select_block_map_for_layer(fmha_impl.attn_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                global_kv_cache=self.kv_cache,
                prev_topk_indices=prev_topk_indices,
                enable_cmp=enable_cmp,
                force_reuse_topk_indices=reuse_topk_indices,
                mla_cp_layout=cp_layout,
            )
            hidden_states = output.hidden_states
            residual = output.residual
            prev_topk_indices = output.topk_indices
            write_typed_aux_cache_regions(typed_aux_cache_store, self.kv_cache, i)

        if self._mtp_indexer_share_enabled and (
            self._mtp_indexer_role == _MTP_INDEXER_ROLE_SEED
            or (
                inputs.attention_inputs.is_prefill
                and not getattr(inputs.attention_inputs, "is_draft_extend", False)
                and not inputs.attention_inputs.is_target_verify
            )
        ):
            if cp_layout is not None:
                self._store_prefill_cp_seed(
                    prev_topk_indices, fmha_impl, inputs.attention_inputs
                )
            else:
                self._store_mtp_topk_indices(
                    prev_topk_indices,
                    inputs.attention_inputs.mtp_indexer_seed_rows,
                    fmha_impl.fmha_params.positions_d,
                )

        hidden_states, _ = self.norm(hidden_states, residual)
        if cp_layout is not None:
            hidden_states = all_gather_trim(
                hidden_states, cp_layout.sp_layout.logical_tokens, Group.TP
            )
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)

    def _mask_position_zero_embeddings(
        self, inputs_embeds: torch.Tensor, fmha_impl: Any
    ) -> torch.Tensor:
        fmha_params = getattr(fmha_impl, "fmha_params", None)
        positions = getattr(fmha_params, "positions_d", None)
        if (
            positions is None
            or not torch.is_tensor(positions)
            or positions.numel() == 0
        ):
            return inputs_embeds
        positions = positions.reshape(-1)
        if positions.size(0) != inputs_embeds.size(0):
            return inputs_embeds
        if positions.device != inputs_embeds.device:
            positions = positions.to(device=inputs_embeds.device)
        return torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
