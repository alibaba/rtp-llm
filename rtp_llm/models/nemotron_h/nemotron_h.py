import json
import os
from typing import Any, Dict, List, Tuple

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import (
    ModelConfig,
    ssm_state_dtype_str_to_data_type,
)
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.nemotron_h.nemotron_h_weight import NemotronHWeight
from rtp_llm.ops import (
    DataType,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
)


class NemotronHBase(BaseModel):
    """NemotronH: hybrid Mamba-2 / GQA attention / MoE architecture.

    Layer kinds come from the checkpoint's ``layers_block_type`` list:
    ``mamba`` (SSM + conv state cache), ``attention`` (GQA KV cache) and
    ``moe`` (attention-free; no cache group).
    """

    mamba_layer_type = "mamba"
    attention_layer_type = "attention"
    moe_layer_type = "moe"

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        cls._parse_basic_config(config_json, config)
        cls._parse_rope_config(config_json, config)
        cls._parse_normalization_config(config_json, config)
        cls._parse_moe_config(config_json, config)
        cls._parse_hybrid_attention_config(config_json, config)
        cls._parse_linear_attention_config(config_json, config)

        return config

    @classmethod
    def _parse_basic_config(cls, config_json: dict, config: ModelConfig):
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
        config.attn_config.size_per_head = config_json["head_dim"]
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.max_seq_len = config_json["max_position_embeddings"]
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = config_json.get("rope_theta", 10000)
        config.partial_rotary_factor = config_json.get("partial_rotary_factor", 1.0)
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )

    @classmethod
    def _parse_normalization_config(cls, config_json: dict, config: ModelConfig):
        eps = config_json.get("layer_norm_epsilon", config_json.get("norm_eps", 1e-5))
        config.layernorm_eps = eps
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.qk_norm = False

    @classmethod
    def _parse_moe_config(cls, config_json: Dict[str, Any], config: ModelConfig):
        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["n_routed_experts"]
        config.moe_inter_size = config_json["moe_intermediate_size"]
        config.inter_size = config_json.get("moe_shared_expert_intermediate_size", 0)
        # One independently-sized shared expert whose output is added on top
        # of the routed experts (moe_shared_expert_overlap=True semantics).
        config.n_shared_experts = int(config.inter_size > 0)
        config.has_moe_norm = config_json.get("norm_topk_prob", True)
        config.moe_style = 2 if config.n_shared_experts > 0 else 1
        # Ungated up/down experts with a squared-ReLU activation.
        config.moe_w1_layout = "up_only"
        # DeepSeek-style sigmoid router with bias-corrected ranking.
        config.moe_n_group = config_json.get("n_group", 1)
        config.moe_topk_group = config_json.get("topk_group", 1)
        config.routed_scaling_factor = config_json.get("routed_scaling_factor", 1.0)

        block_types = cls._layer_block_types(config_json, config)
        moe_layer_index = [
            i for i, t in enumerate(block_types) if t == cls.moe_layer_type
        ]
        config.moe_layer_index = moe_layer_index

    @classmethod
    def _layer_block_types(cls, config_json: dict, config: ModelConfig) -> List[str]:
        block_types = config_json["layers_block_type"]
        if len(block_types) != config.num_layers:
            raise ValueError(
                f"layers_block_type length {len(block_types)} != num_hidden_layers "
                f"{config.num_layers}"
            )
        return block_types

    @classmethod
    def _parse_hybrid_attention_config(cls, config_json: dict, config: ModelConfig):
        block_types = cls._layer_block_types(config_json, config)
        config.hybrid_attention_config.enable_hybrid_attention = True
        hybrid_layer_types: List[HybridAttentionType] = []
        for block_type in block_types:
            if block_type == cls.mamba_layer_type:
                hybrid_layer_types.append(HybridAttentionType.LINEAR)
            else:
                # attention and attention-free moe blocks both use the
                # non-LINEAR (full attention / no linear state) code path.
                hybrid_layer_types.append(HybridAttentionType.NONE)
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types

    @classmethod
    def _parse_linear_attention_config(cls, config_json: dict, config: ModelConfig):
        # Mamba-2 state geometry mapped onto the framework's GDN-shaped
        # linear attention config:
        #   key heads   = SSM B/C groups (n_groups), key head dim = d_state
        #   value heads = SSM heads (mamba_num_heads), value head dim = head dim
        # so the SSM state is (value heads, value dim, key dim) ==
        # (nheads, headdim, dstate) and the conv channels are
        # [B; C; x] == [q; k; v].
        config.linear_attention_config.linear_conv_kernel_dim = config_json[
            "conv_kernel"
        ]
        config.linear_attention_config.linear_key_head_dim = config_json[
            "ssm_state_size"
        ]
        config.linear_attention_config.linear_num_key_heads = config_json["n_groups"]
        config.linear_attention_config.linear_num_value_heads = config_json[
            "mamba_num_heads"
        ]
        config.linear_attention_config.linear_value_head_dim = config_json[
            "mamba_head_dim"
        ]
        ssm_dtype = config_json.get("mamba_ssm_cache_dtype", "float32")
        if ssm_dtype is not None:
            config.linear_attention_config.ssm_state_dtype = (
                ssm_state_dtype_str_to_data_type(ssm_dtype)
            )

    @classmethod
    def _apply_kv_cache_config(
        cls, model_config: ModelConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        remote_cache_enabled = (
            kv_cache_config.reuse_cache and kv_cache_config.enable_remote_cache
        )
        if remote_cache_enabled:
            raise ValueError(
                "NemotronH does not support remote cache (legacy connector "
                "assumes a single contiguous KV allocation)"
            )

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        linear_desc = KVCacheSpecDesc()
        linear_desc.tag = "linear"
        linear_desc.cache_type = KVCacheSpecType.LINEAR

        full_desc = KVCacheSpecDesc()
        full_desc.tag = "full"
        full_desc.cache_type = KVCacheSpecType.MHA

        block_types = cls._layer_block_types_from_config(model_config)
        spec_descs = []
        for block_type in block_types:
            if block_type == cls.mamba_layer_type:
                spec_descs.append([linear_desc])
            elif block_type == cls.attention_layer_type:
                spec_descs.append([full_desc])
            else:
                # Attention-free moe blocks carry no cache descriptor.
                spec_descs.append([])
        model_config.kv_cache_spec_descs = spec_descs

    @classmethod
    def _layer_block_types_from_config(cls, model_config: ModelConfig) -> List[str]:
        """Recover per-layer block types from the model config.

        The hybrid attention types only distinguish mamba (LINEAR) from
        non-mamba (NONE); the moe/attention split is recovered from
        moe_layer_index: non-mamba blocks that are not moe layers are
        attention blocks.
        """
        hybrid_types = model_config.hybrid_attention_config.hybrid_attention_types
        moe_layers = set(model_config.moe_layer_index)
        block_types: List[str] = []
        for i, hybrid_type in enumerate(hybrid_types):
            if hybrid_type == HybridAttentionType.LINEAR:
                block_types.append(cls.mamba_layer_type)
            elif i in moe_layers:
                block_types.append(cls.moe_layer_type)
            else:
                block_types.append(cls.attention_layer_type)
        return block_types

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        from rtp_llm.models_py.utils.arch import get_device_type, is_cuda

        if not is_cuda():
            raise RuntimeError(
                "NemotronH has no python-model implementation for "
                f"{get_device_type().name}"
            )
        from rtp_llm.models_py.model_desc.nemotron_h import NemotronHModel

        self.py_model = NemotronHModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


class NemotronH(NemotronHBase):
    @staticmethod
    def get_weight_cls():
        return NemotronHWeight


register_model("nemotron_h", NemotronH, ["NemotronHForCausalLM"])
