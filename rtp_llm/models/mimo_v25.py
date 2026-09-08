"""MiMo-V2.5: a 48-layer MoE model with per-layer hybrid attention.

Nine layers (0, 5, 11, 17, 23, 29, 35, 41, 47) use global attention with 4 KV
heads; the other 39 use a 128-token sliding window with 8 KV heads and a
per-head learnable attention sink.  QK head dim is 192 while V is 128, RoPE
covers only the leading 64 dims of each head, and the two layer kinds use
different RoPE bases.
"""

import json
import math
import os
from typing import Any, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.mimo_v25_weight import MiMoV25Weight
from rtp_llm.ops import (
    CacheGroupType,
    CacheReusePolicyDesc,
    CacheTailPolicyDesc,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
)

# Cache group tags. Each becomes one independent KV cache pool, and the model
# description keys its per-layer-kind FMHA implementations off the same names --
# it carries its own copy (models_py/model_desc/mimo_v25.py) because model_desc
# modules sit below this package in the dependency order. Keep both in sync.
GA_KV_TAG = "ga_kv"
SWA_KV_TAG = "swa_kv"


def build_mimo_v25_kv_cache_spec_descs(
    hybrid_attention_types: List[HybridAttentionType],
    ga_kv_head_num: int,
    swa_kv_head_num: int,
    v_size_per_head: int,
    window_size: int,
    tokens_per_block: int,
) -> List[List[KVCacheSpecDesc]]:
    """Describe the two KV cache pools, one per attention kind.

    The pools cannot be merged: their KV head counts differ, so their block
    strides do, and the shared-pool layout in ``BlockPoolConfigHelper`` carries a
    single stride for every group.
    """
    assert window_size > 0, f"MiMo V2.5 requires a sliding window, got {window_size}"
    assert tokens_per_block > 0, f"invalid tokens_per_block={tokens_per_block}"

    ga_desc = KVCacheSpecDesc()
    ga_desc.tag = GA_KV_TAG
    ga_desc.cache_type = KVCacheSpecType.MHA
    ga_desc.kv_head_num_override = ga_kv_head_num
    ga_desc.v_size_per_head_override = v_size_per_head

    swa_desc = KVCacheSpecDesc()
    swa_desc.tag = SWA_KV_TAG
    swa_desc.cache_type = KVCacheSpecType.MHA
    swa_desc.kv_head_num_override = swa_kv_head_num
    swa_desc.v_size_per_head_override = v_size_per_head
    # SWAKVCacheGroup keeps a full-length block table addressed by absolute page,
    # but only materializes the last active_tail_blocks entries and leaves the
    # rest NULL. Per-request occupancy is therefore bounded by that count rather
    # than by sequence length.
    #
    # ceil(window / page) + 1 blocks: a window ending at an arbitrary position
    # spans ceil(window / page) pages, plus one because the window boundary
    # generally falls mid-page. At exactly that count every NULL page lies wholly
    # outside the window, so window_left masking hides the reserved block the
    # plan kernel substitutes for it. One block fewer and the oldest kept tokens
    # would fall in a NULL page.
    swa_desc.group_type = CacheGroupType.SWA
    # MiMo SWA participates in prefix reuse. The allocator uses the explicit
    # window below to require a complete contiguous SWA suffix; active_tail_blocks
    # remains an allocation/retention policy and is intentionally kept separate.
    reuse = CacheReusePolicyDesc()
    reuse.enable_prefix_reuse = True
    swa_desc.reuse = reuse
    tail = CacheTailPolicyDesc()
    tail.active_tail_blocks = math.ceil(window_size / tokens_per_block) + 1
    tail.prefix_reuse_window_tokens = window_size
    swa_desc.tail = tail

    return [
        [swa_desc if t == HybridAttentionType.SLIDING_WINDOW else ga_desc]
        for t in hybrid_attention_types
    ]


class MiMoV25(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        with open(os.path.join(ckpt_path, "config.json")) as f:
            cj = json.load(f)

        config = ModelConfig()
        config.ckpt_path = ckpt_path

        cls._parse_basic_config(cj, config)
        cls._parse_stop_words(ckpt_path, config)
        cls._parse_rope_config(cj, config)
        cls._parse_normalization_config(cj, config)
        cls._parse_hybrid_attention_config(cj, config)
        cls._parse_swa_config(cj, config)
        cls._parse_moe_config(cj, config)
        return config

    @classmethod
    def _parse_basic_config(cls, cj: Dict[str, Any], config: ModelConfig) -> None:
        config.hidden_size = cj["hidden_size"]
        config.num_layers = cj["num_hidden_layers"]
        config.vocab_size = cj["vocab_size"]
        config.max_seq_len = cj["max_position_embeddings"]
        config.config_dtype = cj.get("dtype", None)
        config.tie_word_embeddings = cj.get("tie_word_embeddings", False)
        config.special_tokens.eos_token_id = cj["eos_token_id"]

        config.attn_config.head_num = cj["num_attention_heads"]
        config.attn_config.size_per_head = cj["head_dim"]  # QK head_dim = 192
        config.attn_config.v_size_per_head = cj["v_head_dim"]  # V  head_dim = 128

        # kv_head_num is the KV head count of the global-attention layers. The
        # sliding-window count lives in swa_attention_config.swa_kv_head_num, which the
        # cache spec descs and the per-layer forward overrides both read explicitly.
        config.attn_config.kv_head_num = cj["num_key_value_heads"]  # 4 (GA)

        # Both layer kinds share the head dims set above: the fused-qkv row boundaries,
        # the o_proj input dim and both KV specs' strides are all derived from them. The
        # config schema allows the swa_* variants to differ
        # (configuration_mimo_v2.py:243-248); if they ever do, those dims must become
        # per-layer, otherwise shapes go silently wrong and only surface during numerical
        # comparison. Fail fast here instead.
        assert cj.get("swa_head_dim", cj["head_dim"]) == cj["head_dim"]
        assert cj.get("swa_v_head_dim", cj["v_head_dim"]) == cj["v_head_dim"]
        assert (
            cj.get("swa_num_attention_heads", cj["num_attention_heads"])
            == cj["num_attention_heads"]
        )
        # The weight mapping splits a single fused-qkv tensor
        assert cj.get("attention_projection_layout") == "fused_qkv"

        # V is scaled by this factor before the KV cache is written, in every layer. The
        # weight loader folds it into o_proj instead (mimo_v25_weight.py), which is
        # equivalent because attention is linear in V. None means the ckpt does not scale
        # V at all: configuration_mimo_v2.py defaults attention_value_scale to None and
        # the reference implementation multiplies only when it is set. Read from the ckpt
        # rather than assumed, so a ckpt with a different factor is followed instead of
        # silently rescaled.
        value_scale = cj.get("attention_value_scale")
        assert value_scale is None or (
            isinstance(value_scale, (int, float))
            and not isinstance(value_scale, bool)
            and value_scale > 0
        ), (
            "attention_value_scale must be a positive number or absent, "
            f"got {value_scale!r}"
        )
        config.attention_value_scale = value_scale

    @classmethod
    def _parse_stop_words(cls, ckpt_path: str, config: ModelConfig) -> None:
        # The eos in generation_config.json is usually a list (three entries in this
        # ckpt) and holds more than the single value in config.json. The framework's
        # eos_token_id is scalar, so the rest go through stop_words (cf. qwen_v2.py).
        # Everything is read from the ckpt; no token id is hardcoded.
        path = os.path.join(ckpt_path, "generation_config.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            eos = json.load(f).get("eos_token_id")
        if eos is None:
            return
        if isinstance(eos, int):
            eos = [eos]
        config.special_tokens.stop_words_id_list = [
            [t] for t in eos if t != config.special_tokens.eos_token_id
        ]

    @classmethod
    def _parse_rope_config(cls, cj: Dict[str, Any], config: ModelConfig) -> None:
        # Partial rope: rope applies only to the first int(192*0.334)=64 dims of each head
        config.attn_config.rope_config.style = 1  # RopeStyle::Base (NEOX)
        config.attn_config.rope_config.base = int(cj["rope_theta"])  # GA: 1e7
        config.partial_rotary_factor = cj["partial_rotary_factor"]  # 0.334
        config.attn_config.rope_config.dim = int(
            config.attn_config.size_per_head * config.partial_rotary_factor
        )  # -> 64
        assert config.attn_config.rope_config.dim % 2 == 0

    @classmethod
    def _parse_normalization_config(
        cls, cj: Dict[str, Any], config: ModelConfig
    ) -> None:
        config.layernorm_eps = cj["layernorm_epsilon"]  # 1e-5
        config.norm_type = "rmsnorm"
        # The MLP is a gate/up/down triple (modeling_mimo_v2.py:131), i.e. gated silu
        assert cj["hidden_act"] == "silu", cj["hidden_act"]
        config.activation_type = "SiGLU"
        config.qk_norm = False
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True

    @classmethod
    def _parse_hybrid_attention_config(
        cls, cj: Dict[str, Any], config: ModelConfig
    ) -> None:
        config.hybrid_attention_config.enable_hybrid_attention = True
        pattern = cj["hybrid_layer_pattern"]  # 0=GA(full), 1=SWA
        assert len(pattern) == config.num_layers
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.NONE if v == 0 else HybridAttentionType.SLIDING_WINDOW
            for v in pattern
        ]

    @classmethod
    def _parse_swa_config(cls, cj: Dict[str, Any], config: ModelConfig) -> None:
        swa = config.hybrid_attention_config.swa_attention_config
        swa.window_size = int(cj["sliding_window"])  # 128
        swa.swa_kv_head_num = cj["swa_num_key_value_heads"]  # 8
        swa.swa_rope_theta = float(cj["swa_rope_theta"])  # 1e4
        swa.ga_kv_head_num = cj["num_key_value_heads"]  # 4
        swa.add_sink_bias = bool(cj["add_swa_attention_sink_bias"])  # True
        # A single model-wide value that marks "this model uses a sliding window".
        # Whether a given layer is actually windowed is decided by the per-layer
        # override in the model description before it reaches the kernel.
        config.attn_config.sliding_window = swa.window_size
        assert not cj.get("add_full_attention_sink_bias", False), (
            "GA layers must not have a sink bias; if the ckpt changes, the "
            "per-layer-kind routing in the model description has to be updated"
        )
        # Another model-wide marker, read by the fmha implementations that cannot do
        # asymmetric K/V so they exclude themselves in support(). Whether a layer
        # actually receives a sink is decided per-layer-kind in the model description.
        config.attn_config.add_sink_bias = swa.add_sink_bias

    @classmethod
    def _parse_moe_config(cls, cj: Dict[str, Any], config: ModelConfig) -> None:
        config.moe_k = cj["num_experts_per_tok"]  # 8
        config.expert_num = cj["n_routed_experts"]  # 256
        config.moe_inter_size = cj["moe_intermediate_size"]  # 2048
        config.inter_size = cj["intermediate_size"]  # 16384, used by the dense layer 0
        config.has_moe_norm = cj.get("norm_topk_prob", True)  # True
        # moe_style 1 = routed experts only, 2 = shared + routed. This ckpt has
        # n_shared_experts=null; a variant with shared experts would need an extra path
        # in the FFN weight mapping, so fail fast here.
        assert cj.get("n_shared_experts") is None, cj.get("n_shared_experts")
        config.moe_style = 1
        # Map the gating score function from the ckpt string (cf. deepseek_v2.py)
        config.scoring_func = {"softmax": 0, "sigmoid": 1}[cj["scoring_func"]]
        # noaux_tc = sigmoid + e_score_correction_bias + grouped topk, which requires
        # loading gate.e_score_correction_bias. A different topk_method would mean
        # rewriting the gating in the model description.
        assert cj["topk_method"] == "noaux_tc", cj["topk_method"]
        config.moe_n_group = cj.get("n_group") or 1  # 1
        config.moe_topk_group = cj.get("topk_group") or 1  # 1
        config.routed_scaling_factor = cj.get("routed_scaling_factor") or 1.0  # -> 1.0

        # moe_layer_freq[i] decides whether layer i is MoE or dense: layer 0 is dense,
        # layers 1..47 are MoE
        freq = cj["moe_layer_freq"]
        assert len(freq) == config.num_layers
        config.moe_layer_index = [i for i, f in enumerate(freq) if f]

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        """Declare the two-pool hybrid cache topology.

        Runs after ``build_model_config``, so ``attn_config.tokens_per_block`` already
        reflects the CLI and can be used to size the sliding-window tail.
        """
        if model_config.kv_cache_spec_descs:
            return

        hybrid_config = model_config.hybrid_attention_config
        swa_config = hybrid_config.swa_attention_config
        attn_config = model_config.attn_config

        # Without this C++ routes to HybridConfigCreator, whose shared pool carries one
        # block stride for every group and whose FULL-MHA-group count is asserted to be
        # exactly one. Both hold only for models with a single attention shape.
        hybrid_config.enable_independent_kv_cache_pools = True

        model_config.kv_cache_spec_descs = build_mimo_v25_kv_cache_spec_descs(
            hybrid_attention_types=list(hybrid_config.hybrid_attention_types),
            ga_kv_head_num=int(swa_config.ga_kv_head_num),
            swa_kv_head_num=int(swa_config.swa_kv_head_num),
            v_size_per_head=int(attn_config.v_size_per_head),
            window_size=int(swa_config.window_size),
            tokens_per_block=int(attn_config.tokens_per_block),
        )

    @staticmethod
    def get_weight_cls():
        return MiMoV25Weight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.mimo_v25 import MiMoV25Model

        self.py_model = MiMoV25Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


register_model("mimo_v25", MiMoV25, ["MiMoV2ForCausalLM"])
