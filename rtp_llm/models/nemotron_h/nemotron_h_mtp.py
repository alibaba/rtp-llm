from typing import Any, List

from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.nemotron_h.nemotron_h import NemotronH
from rtp_llm.models.nemotron_h.nemotron_h_weight import (
    NemotronHWeight,
    make_mamba_reorder,
)
from rtp_llm.ops import (
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


class NemotronHMTPWeight(NemotronHWeight):
    """NemotronH MTP weights.

    The MTP block is two sub-layers under ``mtp.layers.{i}.``: an attention
    layer (0) and a moe layer (1) with the final layernorm.  Embedding and
    lm_head are shared with the main backbone checkpoint namespace; the
    enorm/hnorm/eh_proj fusion head lives on the attention sub-layer.
    """

    def __init__(self, *args: List[Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.prefix = "backbone."

    def _layer_prefix(self) -> str:
        return "mtp.layers.{i}."

    def _get_weight_info(self):
        weights: List = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("backbone.embeddings.weight", identity)],
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [CkptWeightInfo("mtp.layers.0.enorm.weight", identity)],
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [CkptWeightInfo("mtp.layers.0.hnorm.weight", identity)],
            ),
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [CkptWeightInfo("mtp.layers.0.eh_proj.weight", identity)],
                transpose,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("mtp.layers.1.final_layernorm.weight", identity)],
            ),
        ]
        # Reuse the parent per-layer creators: block types resolve to
        # [attention, moe] from the MTP config's hybrid types and moe index.
        reorder_in_proj, reorder_conv = make_mamba_reorder(self.model_config)
        all_layer_weights: List[List] = []
        for idx in range(self._num_layers):
            layer_weight: List = []
            layer_weight.extend(self._create_layer_norm_weight(idx))
            if self._block_type(idx) == "mamba":
                layer_weight.extend(
                    self._create_mamba_weight(idx, reorder_in_proj, reorder_conv)
                )
            elif self._block_type(idx) == "attention":
                layer_weight.extend(self._create_mqa_weight(idx))
            else:
                layer_weight.extend(self._create_moe_weight(idx))
            all_layer_weights.append(layer_weight)
        return ModelWeightInfo(layer_weights=all_layer_weights, weights=weights)


class NemotronHMTPMixin:
    _mtp_moe_layer_index = (1,)

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        # mtp_layers_block_type = [attention, moe]: the draft block is
        # full-attention + routed experts, attention-free on the moe layer.
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.NONE,
            HybridAttentionType.NONE,
        ]
        config.moe_layer_index = list(cls._mtp_moe_layer_index)
        config.num_layers = 2
        config.is_mtp = True
        return config

    @classmethod
    def _post_build_model_config(cls, model_config) -> None:
        full_desc = KVCacheSpecDesc()
        full_desc.tag = "full"
        full_desc.cache_type = KVCacheSpecType.MHA
        model_config.kv_cache_spec_descs = [[full_desc], []]

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.nemotron_h_mtp import NemotronHMTPModel

        self.py_model = NemotronHMTPModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            moe_config=self.moe_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


class NemotronHMTP(NemotronHMTPMixin, NemotronH):
    @staticmethod
    def get_weight_cls():
        return NemotronHMTPWeight


register_model("nemotron_h_mtp", NemotronHMTP, ["NemotronHForCausalLMNextN"])
