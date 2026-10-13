import json
import logging
import os
from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.hybrid_kv_cache import build_hybrid_kv_cache_spec_descs
from rtp_llm.models.lfm2.lfm2_weight import Lfm2Weight
from rtp_llm.ops import DataType, HybridAttentionType, KVCacheSpecType


class Lfm2Vl(BaseModel):
    """LiquidAI LFM2-VL (d1 series backbone).

    Text backbone: 30 layers mixing 22 gated short-conv layers with 8 GQA
    full-attention layers (RoPE theta 1e6, per-head q/k RMSNorm, SwiGLU MLP).
    Vision tower/projector are wired through the multimodal mixin path.

    The short-conv layer keeps a (kernel - 1) x hidden gated-input state per
    sequence.  It reuses the LINEAR hybrid cache group (same mechanism as
    kimi_linear): ``linear_attention_config`` below is a layout contract, not
    a KDA model -- it is chosen so that the LINEAR block layout stores exactly
    conv_state = (3 - 1) x 2048 elements with a minimal unused ssm head:

      qkv = 1*1*2 + 1*2046        = 2048  (conv input width = hidden)
      conv_elems = (3-1) * 2048   = 4096  (the real conv state)
      ssm_elems = 2046 * 1 * 1    = 2046  (dead space, ~0.3% of a block)

    The Python model never reads or writes the ssm part.
    """

    # layout contract for the LINEAR cache group backing conv states
    LAYOUT_KEY_HEAD_DIM = 1
    LAYOUT_VALUE_HEAD_DIM = 1
    LAYOUT_KEY_HEADS = 1
    LAYOUT_VALUE_HEADS = 2046
    CONV_KERNEL = 3

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        text_config = config_json.get("text_config", config_json)
        cls._parse_basic_config(text_config, config, config_json)
        cls._parse_rope_config(text_config, config)
        cls._parse_hybrid_attention_config(text_config, config)
        cls._parse_conv_state_layout_config(config)
        cls._populate_mm_config(ckpt_path, config_json, config)

        eos_token_id = config_json.get(
            "eos_token_id", text_config.get("eos_token_id", 0)
        )
        config.special_tokens.eos_token_id = eos_token_id
        # ChatML turn end doubles as the generation stop word
        config.special_tokens.stop_words_id_list = [[eos_token_id]]
        return config

    @staticmethod
    def _populate_mm_config(ckpt_path: str, config_json: dict, config: ModelConfig):
        image_token_id = config_json.get("image_token_id", 124907)

        config.mm_model_config.is_multimodal = True
        # single-token placeholder: the chat template emits one <image>
        # (124907) that the multimodal processor expands to N feature tokens
        config.mm_model_config.mm_sep_tokens = [[image_token_id]]
        # plain sequential RoPE positions for expanded image tokens
        config.mm_model_config.mm_position_ids_style = 0

        sanitized = {k: v for k, v in config_json.items() if not k.startswith("_")}
        config.mm_related_params.config["vl_config"] = sanitized
        vision_config = config_json.get("vision_config", {})
        config.mm_related_params.config["vision_config"] = {
            k: v for k, v in vision_config.items() if not k.startswith("_")
        }
        config.mm_related_params.config["ckpt_path"] = ckpt_path
        config.mm_related_params.special_token_ids["image_token_index"] = image_token_id
        config.mm_related_params.special_tokens.update({"default_mm_token": "<image>"})

        processor_config_path = os.path.join(ckpt_path, "processor_config.json")
        if os.path.exists(processor_config_path):
            with open(processor_config_path) as reader:
                processor_config = json.loads(reader.read())
            config.mm_related_params.config["image_processor_config"] = (
                processor_config.get("image_processor", {})
            )

    @staticmethod
    def _parse_basic_config(
        config_json: dict, config: ModelConfig, vl_config: dict = None
    ):
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config_json["num_attention_heads"]
        )
        config.attn_config.size_per_head = config_json.get(
            "head_dim",
            config_json["hidden_size"] // config_json["num_attention_heads"],
        )
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.inter_size = config_json["intermediate_size"]
        config.max_seq_len = config_json.get("max_position_embeddings", 8192)
        config.layernorm_eps = config_json.get("norm_eps", 1e-05)
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        # per-head q/k RMSNorm (Lfm2Attention.q_layernorm / k_layernorm)
        config.qk_norm = True
        config.activation_type = "SiGLU"
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        # text_config carries dtype on real d1 checkpoints; fall back to the
        # top-level VL config for hand-written test fixtures
        vl_dtype = (vl_config or {}).get("dtype")
        config.config_dtype = config_json.get(
            "dtype", config_json.get("torch_dtype", vl_dtype)
        )

    @staticmethod
    def _parse_rope_config(config_json: dict, config: ModelConfig):
        rope_parameters = config_json.get("rope_parameters", {})
        config.attn_config.rope_config.base = int(
            rope_parameters.get("rope_theta", config_json.get("rope_theta", 10000))
        )
        # half-split rotary over the full head dim, llama-style
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.dim = config.attn_config.size_per_head

    @staticmethod
    def _parse_hybrid_attention_config(config_json: dict, config: ModelConfig):
        layer_types = config_json.get("layer_types", [])
        if len(layer_types) != config.num_layers:
            raise ValueError(
                f"layer_types length {len(layer_types)} != num_hidden_layers "
                f"{config.num_layers}"
            )

        config.hybrid_attention_config.enable_hybrid_attention = True
        hybrid_layer_types: List[HybridAttentionType] = []
        for layer_type in layer_types:
            if layer_type == "full_attention":
                hybrid_layer_types.append(HybridAttentionType.NONE)
            elif layer_type == "conv":
                hybrid_layer_types.append(HybridAttentionType.LINEAR)
            else:
                raise ValueError(f"unsupported lfm2 layer type: {layer_type}")
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types
        config.attn_config.use_mla = False

    @staticmethod
    def _parse_conv_state_layout_config(config: ModelConfig):
        layout = config.linear_attention_config
        layout.linear_key_head_dim = Lfm2Vl.LAYOUT_KEY_HEAD_DIM
        layout.linear_value_head_dim = Lfm2Vl.LAYOUT_VALUE_HEAD_DIM
        layout.linear_num_key_heads = Lfm2Vl.LAYOUT_KEY_HEADS
        layout.linear_num_value_heads = Lfm2Vl.LAYOUT_VALUE_HEADS
        layout.linear_conv_kernel_dim = Lfm2Vl.CONV_KERNEL
        # unused ssm head of the LINEAR layout; fp32 keeps the C++ spec check
        # happy and matches the kimi_linear convention. Overwritten by
        # resolve_ssm_state_dtype only when the user passes an explicit flag.
        layout.ssm_state_dtype = DataType.TYPE_FP32

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        if model_config.kv_cache_spec_descs:
            return

        full_cache_type = KVCacheSpecType.MHA
        model_config.kv_cache_spec_descs = build_hybrid_kv_cache_spec_descs(
            model_config.hybrid_attention_config.hybrid_attention_types,
            full_cache_type,
        )

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.lfm2 import Lfm2VlModel

        self.py_model = Lfm2VlModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return Lfm2Weight


register_model("lfm2_vl", Lfm2Vl, ["Lfm2VlForConditionalGeneration"])
