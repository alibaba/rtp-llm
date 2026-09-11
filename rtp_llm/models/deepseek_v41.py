"""V4.1 configuration and checkpoint descriptors for the existing model factory.

The runtime guard is intentional until the typed cache, shared Engram mapping
and V4.1 attention driver are connected. V4's ratio dispatch is incompatible.
"""

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.config.dsv41_weights import build_v41_manifest, validate_inventory
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models_py.modules.dsv41.math import dequantize_block32
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


def _wo_a_bf16(tensors):
    return dequantize_block32(tensors[0], tensors[1])


class DeepSeekV41Weight(ModelDeployWeightInfo):
    def _process_meta(self, meta_dict, weight_keys):
        validate_inventory(
            build_v41_manifest(self.model_config.dsv41_config), weight_keys
        )

    @property
    def host_weight_specs(self):
        return {
            name: spec
            for name, spec in build_v41_manifest(self.model_config.dsv41_config).items()
            if spec.placement == "host_shared"
        }

    def get_weight_info(self) -> ModelWeightInfo:
        # V4.1 carries its own raw FP8/FP4 scales. Generic V4 block128
        # conversions and merged-FFN transforms must not rewrite this graph.
        return self._get_weight_info()

    def _get_weight_info(self) -> ModelWeightInfo:
        config = self.model_config.dsv41_config
        config.validate_parallelism(tp_size=self.tp_size, ep_size=self.ep_size)
        if self.tp_size != 1:
            raise NotImplementedError(
                "V4.1 descriptors currently require attention TP1"
            )
        specs = build_v41_manifest(config)
        layers = [[] for _ in range(config.text["num_hidden_layers"])]
        globals_ = []
        aliases = {
            "embed.weight": W.embedding,
            "head.weight": W.lm_head,
            "norm.weight": W.final_ln_gamma,
        }
        dtypes = {
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "I8": torch.int8,
            "F8_E4M3": torch.float8_e4m3fn,
            "F8_E8M0": torch.float8_e8m0fnu,
        }
        for name, spec in specs.items():
            if name.startswith("mtp.") or spec.placement == "host_shared":
                continue
            if spec.conversion == "dequantize_bf16" and name.endswith(".scale"):
                continue
            parts = name.split(".")
            if ".ffn.experts." in name:
                expert = int(parts[4])
                local = config.text["n_routed_experts"] // self.ep_size
                if not self.ep_rank * local <= expert < (self.ep_rank + 1) * local:
                    continue
            source_weights = [CkptWeightInfo(name, identity)]
            process = identity
            dtype = dtypes[spec.dtype]
            if spec.conversion == "dequantize_bf16":
                source_weights.append(
                    CkptWeightInfo(name.removesuffix(".weight") + ".scale", identity)
                )
                process, dtype = _wo_a_bf16, torch.bfloat16
            elif spec.conversion in ("fp32_logits", "fp32_norm"):
                dtype = torch.float32
            is_layer = name.startswith("layers.")
            key = "v41." + (".".join(parts[2:]) if is_layer else name)
            descriptor = AtomicWeight(
                aliases.get(name, key), source_weights, process, data_type=dtype
            )
            if is_layer:
                layers[int(parts[1])].append(descriptor)
            else:
                globals_.append(descriptor)
        return ModelWeightInfo(weights=globals_, layer_weights=layers)


class DeepSeekV41(DeepSeekV2):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        parsed = V41Config.from_path(ckpt_path)
        t = parsed.text
        config = ModelConfig()
        config.dsv41_config = parsed
        config.model_type = "deepseek_v41"
        config.num_layers = t["num_hidden_layers"]
        config.hidden_size = t["hidden_size"]
        config.vocab_size = t["vocab_size"]
        config.max_seq_len = t["max_position_embeddings"]
        config.layernorm_eps = t["rms_norm_eps"]
        config.norm_type = "rmsnorm"
        config.activation_type = "SiGLU"
        config.has_post_decoder_layernorm = True
        config.tie_word_embeddings = False
        config.enable_fp32_lm_head = True
        config.config_dtype = parsed.dtype
        config.special_tokens.bos_token_id = parsed.bos_token_id
        config.special_tokens.eos_token_id = parsed.eos_token_id
        config.special_tokens.pad_token_id = parsed.pad_token_id
        a = config.attn_config
        a.dsv41_cache_layout_version = 1
        a.head_num = t["num_attention_heads"]
        a.kv_head_num = t["num_key_value_heads"]
        a.size_per_head = t["head_dim"]
        a.rope_head_dim = t["qk_rope_head_dim"]
        a.nope_head_dim = t["head_dim"] - t["qk_rope_head_dim"]
        a.v_head_dim = t["head_dim"]
        a.q_lora_rank = t["q_lora_rank"]
        a.kv_lora_rank = 0
        a.use_mla = False
        a.o_groups = t["o_groups"]
        a.o_lora_rank = t["o_lora_rank"]
        a.sliding_window = t["sliding_window"]
        a.layer_compress_ratios = t["compress_ratios"][: config.num_layers]
        a.compress_rope_theta = t["compress_rope_theta"]
        a.rope_config.base = t["rope_theta"]
        a.rope_config.dim = t["qk_rope_head_dim"]
        a.rope_config.offset = t["head_dim"] - t["qk_rope_head_dim"]
        a.rope_config.style = 0
        scaling = t["rope_scaling"]
        a.rope_config.scale = scaling["factor"]
        a.rope_config.factor1 = float(scaling["beta_slow"])
        a.rope_config.factor2 = float(scaling["beta_fast"])
        a.rope_config.max_pos = scaling["original_max_position_embeddings"]
        a.is_sparse = True
        a.indexer_head_dim = t["index_head_dim"]
        a.indexer_head_num = t["index_n_heads"]
        a.indexer_topk = t["index_topk"]
        config.hc_mult = t["hc_mult"]
        config.hc_sinkhorn_iters = t["hc_sinkhorn_iters"]
        config.hc_eps = t["hc_eps"]
        config.swiglu_limit = t["swiglu_limit"]
        config.num_hash_layers = 0
        config.scoring_func = 2
        config.expert_num = t["n_routed_experts"]
        config.moe_k = t["num_experts_per_tok"]
        config.moe_style = 2
        config.moe_n_group = 0
        config.moe_topk_group = 0
        config.has_moe_norm = t["norm_topk_prob"]
        config.moe_layer_index = list(range(config.num_layers))
        config.routed_scaling_factor = t["routed_scaling_factor"]
        config.moe_inter_size = t["moe_intermediate_size"]
        config.inter_size = t["n_shared_experts"] * t["moe_intermediate_size"]
        config.dspark_noise_token_id = t["dspark_noise_token_id"]
        config.dspark_target_layer_ids = list(t["dspark_target_layer_ids"])
        config.dspark_markov_rank = t["dspark_markov_rank"]
        return config

    @classmethod
    def from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "V4.1 runtime requires its typed cache/attention and host-shared Engram integration; V4 execution is not compatible"
        )

    def _create_python_model(self):
        raise NotImplementedError("V4.1 runtime integration is incomplete")

    @staticmethod
    def get_weight_cls():
        return DeepSeekV41Weight


register_model("deepseek_v41", DeepSeekV41, ["DeepseekV41ForCausalLM"])
