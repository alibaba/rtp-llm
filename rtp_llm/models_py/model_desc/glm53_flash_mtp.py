"""GLM-5.3-Flash draft layer on the main engine's grouped cache API."""

from typing import Any, Optional

import torch
from rtp_llm.models.glm53_prefill_parallel import MlaCPParallelismView, mla_cp_enabled
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather_trim
from rtp_llm.models_py.distributed.glm53_mla_cp import prepare_glm53_fmha
from rtp_llm.models_py.distributed.sequence_parallel import shard_tokens
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.model_desc.glm53_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, LinearFactory, RMSNorm, RMSResNorm
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
    prepare_mtp_multimodal_inputs,
)
from rtp_llm.models_py.modules.dsv4.forward_metadata import scoped_forward_metadata
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W
from torch import nn


class Glm53FlashMtpModel(GptModelBase):
    def __init__(
        self,
        config,
        parallelism,
        weights,
        moe_config,
        max_generate_batch_size,
        fmha_config=None,
        py_hw_kernel_config=None,
    ):
        super().__init__(
            config,
            parallelism,
            weights,
            max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
        )
        self.prefill_mla_cp = mla_cp_enabled(
            config.model_type, parallelism.role_type, is_glm53_mtp=True
        )
        self.mla_parallelism = (
            MlaCPParallelismView(parallelism) if self.prefill_mla_cp else parallelism
        )
        self.embed_tokens = Embedding(
            config, parallelism, weights.get_global_weight(W.embedding)
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.enorm = RMSNorm(
            weights.get_global_weight(W.multi_tokens_predict_enorm),
            eps=config.layernorm_eps,
        )
        self.hnorm = RMSNorm(
            weights.get_global_weight(W.multi_tokens_predict_hnorm),
            eps=config.layernorm_eps,
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.multi_tokens_predict_eh_proj
        )
        self.layers = nn.ModuleList(
            GenericMoeDecoderLayer(
                config,
                parallelism,
                weights.weights[i],
                weights.global_weights,
                i,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=bool(
                    py_hw_kernel_config and py_hw_kernel_config.enable_cuda_graph
                ),
                hw_kernel_config=py_hw_kernel_config,
            )
            for i in range(self.layer_num)
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.multi_tokens_predict_final_ln_gamma),
            eps=config.layernorm_eps,
        )

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ):
        return prepare_glm53_fmha(self, inputs, is_cuda_graph)

    @scoped_forward_metadata
    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        group_inputs = get_attention_inputs_value(inputs)
        full_inputs = select_attention_inputs_for_tag(group_inputs, "full")
        fmha_impl = fmha_impl or self.prepare_fmha_impl(inputs)
        fmha_impl.glm53_group_inputs = group_inputs
        token_ids = inputs.input_ids
        features = inputs.multimodal_inputs.multimodal_features
        if features:
            token_ids, features, locs = prepare_mtp_multimodal_inputs(
                token_ids,
                features,
                inputs.multimodal_inputs.mm_features_locs,
                full_inputs.cu_seqlens_device,
                full_inputs.cu_seqlens,
                already_shifted=self.parallelism_config.prefill_cp_config.is_enabled(),
            )
            embeds = self.multimodal_embedding_injector(
                self.embed_tokens(token_ids), features, locs
            )
        else:
            embeds = self.embed_tokens(token_ids)

        layout = getattr(fmha_impl, "glm53_cp_layout", None)
        hiddens = inputs.input_hiddens
        if embeds.shape != hiddens.shape:
            raise ValueError("GLM53 MTP embedding and target hidden rows must align")
        if layout is not None:
            starts = full_inputs.cu_seqlens[:-1].to(embeds.device, dtype=torch.int64)
            prefix = full_inputs.prefix_lengths.to(embeds.device)
            first = torch.zeros(embeds.shape[0], device=embeds.device, dtype=torch.bool)
            first.scatter_(0, starts, prefix == 0)
            embeds = torch.where(first.unsqueeze(-1), 0, embeds)
            embeds = shard_tokens(embeds, layout.sp_layout)
            hiddens = shard_tokens(hiddens, layout.sp_layout)
        else:
            pos = getattr(fmha_impl.fmha_params, "positions_d", None)
            if pos is not None and pos.numel() == embeds.shape[0]:
                embeds = torch.where(
                    pos.reshape(-1, 1).to(embeds.device) == 0, 0, embeds
                )

        hidden = self.fc(torch.cat((self.enorm(embeds), self.hnorm(hiddens)), -1))
        residual = torch.zeros_like(hidden)
        for i, layer in enumerate(self.layers):
            cache = self.kv_cache.get_layer_cache(i, "full") if self.kv_cache else None
            out = layer(
                hidden,
                residual,
                fmha_impl,
                kv_cache=cache,
                global_kv_cache=self.kv_cache,
                mla_cp_layout=layout,
            )
            hidden, residual = out.hidden_states, out.residual
            if (
                self.kv_cache
                and full_inputs.is_prefill
                and not full_inputs.is_target_verify
            ):
                for tag in ("indexer_kv", "indexer_state"):
                    group = select_attention_inputs_for_tag(group_inputs, tag)
                    if group.cache_store_writer and group.cache_store_inputs:
                        group.cache_store_writer.write(
                            group.cache_store_inputs,
                            self.kv_cache.get_layer_cache(i, tag),
                        )
        hidden, _ = self.norm(hidden, residual)
        if layout is not None:
            hidden = all_gather_trim(hidden, layout.sp_layout.logical_tokens, Group.TP)
        return PyModelOutputs(hidden)
