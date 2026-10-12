import os
from collections.abc import Mapping
from typing import Any

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.glm5 import Glm5ModelBase
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import LinearFactory, RMSNorm, RMSResNorm
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
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


class GenericMoeMTPModel(Glm5ModelBase):
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
            moe_config,
            max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
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

        self._initialize_decoder_layers()
        self.norm = RMSResNorm(
            weights.global_weights[W.multi_tokens_predict_final_ln_gamma],
            eps=model_config.layernorm_eps,
        )
        topk = int(model_config.attn_config.indexer_topk)
        self._mtp_indexer_share_enabled = _mtp_indexer_share_active(
            model_config, parallelism_config, self.layer_num, topk
        )
        self._mtp_indexer_role = _MTP_INDEXER_ROLE_NORMAL
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

    def clone_for_cuda_graph(
        self, *, share_mega_buf: bool = False
    ) -> "GenericMoeMTPModel":
        clone = object.__new__(type(self))
        GptModelBase.__init__(
            clone,
            self.config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        # Clones using the same KV arena must share its token-to-slot mapping.
        # initialize() rebuilds the working sets if the arena changes.
        clone.device_type = self.device_type
        clone.moe_config = self.moe_config
        clone.max_generate_batch_size = self.max_generate_batch_size
        clone.device_resource_config = self.device_resource_config

        clone.embed_tokens = self.embed_tokens
        clone.pre_fc_norm_embedding = self.pre_fc_norm_embedding
        clone.pre_fc_norm_hidden = self.pre_fc_norm_hidden
        clone.fc = self.fc
        # Only the seed-decode clone shares communication state. The default
        # draft-prefill clone keeps its existing, separate communication arena.
        cloned_layers = []
        for layer in self.layers:
            if share_mega_buf:
                cloned_layers.append(
                    layer.clone_for_cuda_graph(draft_prefill=True, share_mega_buf=True)
                )
            elif hasattr(layer, "clone_for_cuda_graph"):
                cloned_layers.append(layer.clone_for_cuda_graph(draft_prefill=True))
            else:
                cloned_layers.append(layer)
        clone.layers = nn.ModuleList(cloned_layers)
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
        topk = int(self.config.attn_config.indexer_topk)
        if batch_size > self._mtp_shared_topk_indices.size(0):
            raise RuntimeError(
                "MTP indexer share batch exceeds fixed buffer: "
                f"batch={batch_size}, capacity={self._mtp_shared_topk_indices.size(0)}"
            )
        topk_indices = self._mtp_shared_topk_indices[:batch_size, :topk]
        positions = getattr(fmha_impl.fmha_params, "positions_d", None)
        if (
            not torch.is_tensor(positions)
            or positions.numel() != batch_size
            or positions.device != topk_indices.device
        ):
            raise RuntimeError("MTP indexer share requires device positions_d per row")

        # Each reuse iteration appends one causal KV position. Prepend it and
        # shift the previous selection right: while the anchor is short this
        # only evicts a trailing -1; once top-k is full it evicts the tail. This
        # keeps positions generated earlier in the same proposal cycle at the
        # front instead of making them invisible to later draft tokens. Avoid
        # inserting duplicates when the current position is already selected.
        position_column = positions.reshape(-1, 1)
        with_current_position = torch.cat(
            [position_column, topk_indices[:, :-1]], dim=1
        )
        position_present = (topk_indices == position_column).any(dim=1, keepdim=True)
        topk_indices.copy_(
            torch.where(position_present, topk_indices, with_current_position)
        )
        return topk_indices

    def _store_mtp_topk_indices(
        self, topk_indices: torch.Tensor, seed_rows: torch.Tensor
    ) -> None:
        batch_size = seed_rows.numel()
        topk = int(self.config.attn_config.indexer_topk)
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
        self._mtp_shared_topk_indices[:batch_size].copy_(selected)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        inputs_embeds = self.embed_tokens(input_ids)
        default_fmha = (
            fmha_impl["default"] if isinstance(fmha_impl, Mapping) else fmha_impl
        )
        inputs_embeds = self._mask_position_zero_embeddings(inputs_embeds, default_fmha)
        last_hidden_states = inputs.input_hiddens

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

        reuse_topk_indices = (
            self._mtp_indexer_share_enabled
            and self._mtp_indexer_role == _MTP_INDEXER_ROLE_REUSE
            and not get_primary_attention_inputs(inputs).is_prefill
        )
        prev_topk_indices = (
            self._get_mtp_reuse_topk_indices(hidden_states, default_fmha)
            if reuse_topk_indices
            else None
        )
        result = self._forward_decoder_layers(
            hidden_states,
            fmha_impl,
            prev_topk_indices,
            force_reuse_topk_indices=reuse_topk_indices,
        )

        if (
            self._mtp_indexer_share_enabled
            and self._mtp_indexer_role == _MTP_INDEXER_ROLE_SEED
        ):
            self._store_mtp_topk_indices(
                result.topk_indices,
                get_primary_attention_inputs(inputs).mtp_indexer_seed_rows,
            )

        hidden_states, _ = self.norm(result.hidden_states, result.residual)
        self.capture_cp_mtp_hidden(hidden_states)
        return PyModelOutputs(hidden_states)

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
