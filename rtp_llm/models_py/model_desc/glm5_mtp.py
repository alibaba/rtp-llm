"""GLM NextN draft using the target's grouped sparse MLA/cache adapters."""

from typing import Any

import torch

from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_group
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import LinearFactory, RMSNorm
from rtp_llm.utils.model_weight import W


class Glm5MtpModel(GenericMoeModel):
    final_norm_weight_name = W.multi_tokens_predict_final_ln_gamma

    def __init__(self, model_config, parallelism_config, weights, *args, **kwargs):
        super().__init__(model_config, parallelism_config, weights, *args, **kwargs)
        self.pre_fc_norm_embedding = RMSNorm(
            weights.get_global_weight(W.multi_tokens_predict_enorm),
            eps=model_config.layernorm_eps,
        )
        self.pre_fc_norm_hidden = RMSNorm(
            weights.get_global_weight(W.multi_tokens_predict_hnorm),
            eps=model_config.layernorm_eps,
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.global_weights, W.multi_tokens_predict_eh_proj
        )

    def initialize(self, init_resource):
        # Draft decode and draft extend construct separate graph wrappers for
        # this same Python model. Keep the runtime/writer addresses captured by
        # the first wrapper valid when the second wrapper initializes.
        if getattr(self, "offload_plans", {}):
            if init_resource.kv_cache is None:
                raise ValueError(
                    "GLM MTP graph wrappers require their existing KV storage"
                )
            for (layer, tag), plan in self.offload_plans.items():
                storage = init_resource.kv_cache.get_offload_storage(layer, tag)
                if storage is None or (
                    [plane.data_ptr() for plane in storage.resident_planes]
                    != [plane.data_ptr() for plane in plan.storage.resident_planes]
                    or storage.generations.data_ptr()
                    != plan.storage.generations.data_ptr()
                ):
                    raise ValueError(
                        "GLM MTP graph wrappers must share their KV storage"
                    )
            return GptModelBase.initialize(self, init_resource)
        return super().initialize(init_resource)

    def embedding(self, inputs, fmha_impl: Any = None):
        previous = inputs.input_hiddens
        embedding = self.embed_tokens(inputs.input_ids)
        if previous is None or previous.shape != embedding.shape:
            raise ValueError("GLM MTP requires one previous hidden state per token")
        positions = select_fmha_impl_for_group(
            fmha_impl, "default"
        ).rope_params.positions_d
        # The first prompt token has no predecessor. Match GLM's trained NextN
        # contract, including draft extend after a partially accepted round.
        embedding = torch.where(positions.reshape(-1, 1) == 0, 0, embedding)
        return self.fc(
            torch.cat(
                (
                    self.pre_fc_norm_embedding(embedding),
                    self.pre_fc_norm_hidden(previous),
                ),
                dim=-1,
            )
        )
