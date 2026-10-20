"""openai/whisper-large-v3 registration: config, weights, python model wiring.

Whisper is an encoder-decoder ASR model:
- the audio encoder runs in the multimodal process engine (WhisperMixin);
  its hidden states reach the decoder as request-level side state because
  `mm_model_config.mm_skip_token_expansion = True` keeps them out of the
  token stream.
- the 32-layer autoregressive decoder (self-attn + cross-attn + GELU MLP,
  learned positional embeddings, tied lm_head) is the engine "LLM": the
  scheduler drives it with the standard prefill/decode loop and the KV cache
  covers the decoder self-attention.

Checkpoint layout (1259 F16 tensors, rev 06f233fe):
  model.encoder.*           -> WhisperMixin tower (mixin loader, by ckpt name)
  model.decoder.embed_tokens.weight [51866,1280] -> W.embedding + W.lm_head (tied)
  model.decoder.embed_positions.weight [448,1280] -> W.positional_embedding
  model.decoder.layer_norm.*  -> W.final_ln_*
  model.decoder.layers.N.{self_attn,encoder_attn,fc1,fc2,*_layer_norm}.* -> layer weights
"""

import functools
import json
import os
from typing import Any, List

import torch

from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.ops import (
    CacheCapacityPolicyDesc,
    CacheGroupType,
    CacheReusePolicyDesc,
    KVCacheSpecDesc,
    KVCacheSpecType,
)
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
    WeightModule,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    transpose,
)
from rtp_llm.utils.util import get_config_from_path


def merge_qkv_b_whisper(ts: List[torch.Tensor]):
    """Whisper k_proj carries no bias: fuse [q_bias, v_bias] with zero k bias."""
    if len(ts) == 3:
        q, k, v = ts
    else:
        q, v = ts
        k = torch.zeros_like(q)
    return torch.concat([q, k, v], dim=0).contiguous()


class WhisperWeight(ModelDeployWeightInfo):
    def __init__(self, prefix=None, **kwargs: Any):
        self.prefix = prefix or ""
        self.model_prefix: str = "model."
        self.strip_model_prefix = False
        super().__init__(**kwargs)

    @property
    def support_lora(self):
        return False

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        # single supported layout: HF whisper checkpoints under model.decoder.*
        if self.model_prefix + "decoder.layers.0.self_attn.q_proj.weight" not in meta_dicts[0]:
            if "decoder.layers.0.self_attn.q_proj.weight" in meta_dicts[0]:
                self.model_prefix = ""
            else:
                raise Exception(
                    "whisper checkpoint must expose model.decoder.layers.0 weights, "
                    f"got keys like {list(meta_dicts[0].keys())[:4]}"
                )
        self.transformer_prefix = self.model_prefix + "decoder."

    def _get_weight_info(self):
        return self._get_whisper_weight_info()

    def _get_whisper_weight_info(self) -> ModelWeightInfo:
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        ffn_config = FfnConfig(
            is_gated_activation=False,
            align_size=self._align_size,
            is_moe=False,
        )
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.model_prefix + "decoder.embed_tokens.weight", identity)],
                identity,
            ),
            # lm_head is tied to the token embedding in whisper checkpoints
            # (no separate proj_out tensor exists); materialize a second copy.
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.model_prefix + "decoder.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.positional_embedding,
                [CkptWeightInfo(self.model_prefix + "decoder.embed_positions.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.model_prefix + "decoder.layer_norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [CkptWeightInfo(self.model_prefix + "decoder.layer_norm.bias", identity)],
                identity,
            ),
        ]
        layer_weights = []
        for i in range(self._num_layers):
            layer_weights.append(self._get_layer_weight_info(i, attn_config, ffn_config))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _get_layer_weight_info(self, layer_id: int, attn_config: AttnConfig, ffn_config: FfnConfig):
        p = self.transformer_prefix + f"layers.{layer_id}."
        layer_weights = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(p + "self_attn_layer_norm.weight", identity)],
                identity,
                config=attn_config,
            ),
            AtomicWeight(
                W.pre_ln_beta,
                [CkptWeightInfo(p + "self_attn_layer_norm.bias", identity)],
                identity,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(p + "self_attn.q_proj.weight", identity),
                    CkptWeightInfo(p + "self_attn.k_proj.weight", identity),
                    CkptWeightInfo(p + "self_attn.v_proj.weight", identity),
                ],
                merge_qkv_hf,
                config=attn_config,
            ),
            # whisper self-attn: q/v/out have bias, k has none (zero-filled)
            AttnAtomicWeight(
                W.attn_qkv_b,
                [
                    CkptWeightInfo(p + "self_attn.q_proj.bias", identity),
                    CkptWeightInfo(p + "self_attn.v_proj.bias", identity),
                ],
                merge_qkv_b_whisper,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(p + "self_attn.out_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_b,
                [CkptWeightInfo(p + "self_attn.out_proj.bias", identity)],
                identity,
                config=attn_config,
            ),
            AtomicWeight(
                W.cross_attn_pre_ln_gamma,
                [CkptWeightInfo(p + "encoder_attn_layer_norm.weight", identity)],
                identity,
                config=attn_config,
            ),
            AtomicWeight(
                W.cross_attn_pre_ln_beta,
                [CkptWeightInfo(p + "encoder_attn_layer_norm.bias", identity)],
                identity,
                config=attn_config,
            ),
            # cross-attn projections stay separate: q reads decoder hidden,
            # k/v read the encoder output, so a fused qkv slot cannot express it.
            AttnAtomicWeight(
                W.cross_attn_q_w,
                [CkptWeightInfo(p + "encoder_attn.q_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.cross_attn_q_b,
                [CkptWeightInfo(p + "encoder_attn.q_proj.bias", identity)],
                identity,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.cross_attn_k_w,
                [CkptWeightInfo(p + "encoder_attn.k_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.cross_attn_v_w,
                [CkptWeightInfo(p + "encoder_attn.v_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.cross_attn_o_w,
                [CkptWeightInfo(p + "encoder_attn.out_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.cross_attn_o_b,
                [CkptWeightInfo(p + "encoder_attn.out_proj.bias", identity)],
                identity,
                config=attn_config,
            ),
            AtomicWeight(
                W.post_ffn_ln_gamma,
                [CkptWeightInfo(p + "final_layer_norm.weight", identity)],
                identity,
                config=attn_config,
            ),
            AtomicWeight(
                W.post_ffn_ln_beta,
                [CkptWeightInfo(p + "final_layer_norm.bias", identity)],
                identity,
                config=attn_config,
            ),
            FfnAtomicWeight(
                W.ffn_w3,
                [CkptWeightInfo(p + "fc1.weight", identity)],
                transpose,
                config=ffn_config,
            ),
            FfnAtomicWeight(
                W.ffn_b3,
                [CkptWeightInfo(p + "fc1.bias", identity)],
                identity,
                config=ffn_config,
            ),
            FfnAtomicWeight(
                W.ffn_w2,
                [CkptWeightInfo(p + "fc2.weight", identity)],
                transpose,
                config=ffn_config,
            ),
            FfnAtomicWeight(
                W.ffn_b2,
                [CkptWeightInfo(p + "fc2.bias", identity)],
                identity,
                config=ffn_config,
            ),
        ]
        return layer_weights


class Whisper(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise Exception(f"failed to find config json from path: {ckpt_path}")
        with open(config_path) as reader:
            config_json = json.load(reader)
        cls._from_hf(config, config_json)
        return config

    @staticmethod
    def _from_hf(config: ModelConfig, config_json: dict):
        head_num = config_json.get("decoder_attention_heads", 20)
        config.attn_config.head_num = head_num
        config.attn_config.kv_head_num = head_num
        config.attn_config.size_per_head = config_json.get("d_model", 1280) // head_num
        config.attn_config.rope_config.dim = 0
        config.attn_config.rope_config.style = 0
        config.attn_config.is_causal = True
        config.attn_config.need_rope_kv_cache = False
        config.hidden_size = config_json.get("d_model", 1280)
        config.inter_size = config_json.get("decoder_ffn_dim", 5120)
        config.num_layers = config_json.get("decoder_layers", 32)
        config.vocab_size = config_json.get("vocab_size", 51866)
        config.max_seq_len = config_json.get("max_target_positions", 448)
        config.activation_type = "gelu"
        config.norm_type = "layernorm"
        config.layernorm_type = "pre_layernorm"
        config.layernorm_eps = 1e-5
        config.has_positional_encoding = True
        config.tie_word_embeddings = False  # lm_head materialized from tied ckpt tensor
        config.mm_model_config.is_multimodal = True
        # whisper consumes the encoder output through per-layer cross attention
        # (request-level side state), never through token-sequence expansion.
        config.mm_model_config.mm_skip_token_expansion = True
        config.mm_related_params.config["ckpt_path"] = config.ckpt_path

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        if model_config.kv_cache_spec_descs:
            return
        # Decoder self-attention KV: the standard reusable token-stream cache.
        default_desc = KVCacheSpecDesc()
        default_desc.tag = "default"
        default_desc.cache_type = KVCacheSpecType.MHA
        # Cross-attention KV: engine pool holding the projected encoder K/V
        # (audio frames), written once at prefill and read-only during decode.
        # Never prefix-shared and never reserved: blocks stay request-private
        # and are sized by the audio frame count, not the token stream.
        cross_desc = KVCacheSpecDesc()
        cross_desc.tag = "cross"
        cross_desc.cache_type = KVCacheSpecType.MHA
        cross_desc.group_type = CacheGroupType.FULL
        cross_desc.kv_len_from_mm_features = True
        cross_reuse = CacheReusePolicyDesc()
        cross_reuse.enable_prefix_reuse = False
        cross_desc.reuse = cross_reuse
        cross_capacity = CacheCapacityPolicyDesc()
        cross_capacity.reservable = False
        cross_desc.capacity = cross_capacity
        # Assign once: pybind deep-copies the list at assignment time.
        model_config.kv_cache_spec_descs = [
            [default_desc, cross_desc] for _ in range(model_config.num_layers)
        ]

    @staticmethod
    def get_weight_cls():
        return WhisperWeight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.whisper import WhisperModel

        self.py_model = WhisperModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    def support_cuda_graph(self) -> bool:
        # Phase A: model-side cross-KV slots are keyed per request and updated
        # from Python at prefill, which is not CUDA-graph replay safe. Graph
        # decode arrives with the engine cross-KV cache group (Phase B).
        return False


register_model("whisper", Whisper, ["WhisperForConditionalGeneration"])
