"""DeepSeek V4.1 text configuration and stage-local weight descriptors.

Execution is gated until the delayed-mHC, Engram and V4.1 cache paths land.
"""

import torch

from rtp_llm.config.dsv41_config import V41Config, dsv41_pp_partition
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.pp_layout import register_pp_partitioner
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v4 import DeepSeekV4, DeepSeekV4Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


class DeepSeekV41Weight(DeepSeekV4Weight):
    def _process_meta(self, meta_dict, weight_keys):
        super()._process_meta(meta_dict, weight_keys)
        # Validate before the common loader expands per-expert checkpoint keys.
        experts = self.model_config.expert_num
        if self.ep_size <= 0 or experts % self.ep_size:
            raise ValueError("V4.1 expert count must be divisible by ep_size")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("V4.1 ep_rank is outside its EP group")

    def _get_hf_layer_weight_info(self, layer_id):
        weights = super()._get_hf_layer_weight_info(layer_id)
        config = self.model_config.deepseek_v41_config
        if layer_id in config["kv_source_layer_ids"]:
            tensors = [
                (W.v4_compressor_wkv, "attn.compressor.wkv.weight"),
                (W.v4_compressor_norm, "attn.compressor.norm.weight"),
                (W.v41_indexer_wk, "attn.indexer.wk.weight"),
                (W.v41_indexer_k_norm, "attn.indexer.k_norm.weight"),
            ]
            if self._compress_ratio(layer_id) == 2:
                tensors.append((W.v4_compressor_wgate, "attn.compressor.wgate.weight"))
            for name, suffix in tensors:
                weights.append(
                    AtomicWeight(
                        name,
                        [CkptWeightInfo(self._key(suffix), identity)],
                        identity,
                        data_type=torch.bfloat16,
                    )
                )
        if layer_id in config["index_source_layer_ids"]:
            weights += self._build_indexer(layer_id)
        # Engram owns its shared host tables and small projection tensors.
        # They must never pass through the ordinary GPU checkpoint iterator.
        return weights

    def _get_weight_info(self):
        info = super()._get_weight_info()
        obsolete_head = {W.v4_hc_head_base, W.v4_hc_head_fn, W.v4_hc_head_scale}
        info.weights = [w for w in info.weights if w.name not in obsolete_head]
        return info


class DeepSeekV41(DeepSeekV4):
    @classmethod
    def _create_config(cls, ckpt_path):
        metadata = V41Config.from_path(ckpt_path)
        text = metadata.runtime_text_config(ckpt_path)
        config = ModelConfig()
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"
        DeepSeekV4._from_hf(config, ckpt_path, config_json=text)
        config.is_deepseek_v41 = True
        config.deepseek_v41_config = text
        config.mm_model_config.is_multimodal = False
        config.max_seq_len = text["max_position_embeddings"]
        config.config_dtype = metadata.dtype
        config.moe_inter_size = text["moe_intermediate_size"]
        # YaRN modifies RoPE; the V4.1 attention softmax uses head_dim**-0.5.
        config.attn_config.softmax_extra_scale = 1.0
        config.special_tokens.bos_token_id = metadata.bos_token_id
        config.special_tokens.eos_token_id = metadata.eos_token_id
        return config

    @classmethod
    def _post_build_model_config(cls, model_config):
        raise NotImplementedError(
            "DeepSeek V4.1 config and weight descriptors are available; serving "
            "requires cache codecs/replica transport, delayed-mHC, Engram and SM120 operators"
        )

    def support_pp(self) -> bool:
        return False

    def _create_python_model(self):
        raise NotImplementedError("DeepSeek V4.1 execution is not integrated yet")

    @staticmethod
    def get_weight_cls():
        return DeepSeekV41Weight


register_pp_partitioner("deepseek_v41", dsv41_pp_partition)
register_model("deepseek_v41", DeepSeekV41, ["DeepseekV41ForCausalLM"])
