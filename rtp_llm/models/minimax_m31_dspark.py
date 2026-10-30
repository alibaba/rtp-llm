"""MiniMax-M3.1 five-layer DSpARK draft on main's speculative model contract."""

import json
from pathlib import Path

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, CustomAtomicWeight
from rtp_llm.models.minimax_m31 import MiniMaxM31
from rtp_llm.models.minimax_m31_kv_cache import MAIN_KV_TAG
from rtp_llm.models.minimax_m31_weights import MiniMaxM31Weight
from rtp_llm.ops import DataType, KVCacheSpecDesc, KVCacheSpecType, SpeculativeType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    transpose,
)

DSPARK_HIDDEN_NORM = "minimax_m31_dspark.hidden_norm"
DSPARK_FINAL_NORM = "minimax_m31_dspark.final_norm"
PREFIX = "language_model.model.dspark."


class MiniMaxM31DSparkWeight(MiniMaxM31Weight):
    """Draft-only dense MXFP8 projection and raw Gemma norm descriptors."""

    def _process_meta(self, meta_dict, weight_keys):
        if self._num_layers != 5 or not any(k.startswith(PREFIX) for k in weight_keys):
            raise ValueError("expected the five-layer MiniMax-M3.1 DSpARK checkpoint")

    def _get_hf_layer_weight_info(self, layer_id):
        root = PREFIX + "layers.{i}.decoder_layer."
        attn = root + "self_attn."
        cfg = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        result = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(root + "input_layernorm.weight", identity)],
                identity,
                disable_quantization=True,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(attn + name + "_proj.weight", identity)
                    for name in ("q", "k", "v")
                ],
                merge_qkv_hf,
                config=cfg,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(attn + "o_proj.weight", identity)],
                transpose,
                config=cfg,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(root + "post_attention_layernorm.weight", identity)],
                identity,
                disable_quantization=True,
            ),
        ]
        for name, key in (("q_norm", W.q_ln_gamma), ("k_norm", W.k_ln_gamma)):
            result.append(
                AttnAtomicWeight(
                    key,
                    [CkptWeightInfo(attn + name + ".weight", identity)],
                    identity,
                    config=cfg,
                    disable_quantization=True,
                )
            )
        result.append(self._dense_mlp(root + "mlp."))
        return result

    def _get_weight_info(self):
        # main's weight-alias loader supplies the target-owned embedding/head.
        globals = [
            AtomicWeight(W.embedding, [], identity),
            AtomicWeight(W.lm_head, [], identity),
        ]
        for name, suffix, process in (
            (W.dspark_fc_w, "fc.weight", transpose),
            (W.dspark_markov_w1, "markov_head.markov_w1.weight", identity),
            (W.dspark_markov_w2, "markov_head.markov_w2.weight", identity),
            (DSPARK_HIDDEN_NORM, "hidden_norm.weight", identity),
            (DSPARK_FINAL_NORM, "final_norm.weight", identity),
        ):
            globals.append(
                CustomAtomicWeight(
                    name,
                    [CkptWeightInfo(PREFIX + suffix, identity)],
                    process,
                    disable_quantization=True,
                )
            )
        return ModelWeightInfo(
            weights=globals,
            layer_weights=[
                self._get_hf_layer_weight_info(i) for i in range(self._num_layers)
            ],
        )


class MiniMaxM31DSpark(MiniMaxM31):
    @classmethod
    def _apply_kv_cache_config(cls, model_config, kv_cache_config) -> None:
        # DSpARK keeps BF16 GQA data, but shares the target's 128-token page table.
        attn = model_config.attn_config
        if attn.tokens_per_block == 64 and attn.kernel_tokens_per_block == 64:
            attn.tokens_per_block = 128
            attn.kernel_tokens_per_block = 128
        elif attn.tokens_per_block != 128 or attn.kernel_tokens_per_block != 128:
            raise ValueError("MiniMax-M3.1 DSpARK requires 128-token KV pages")
        desc = KVCacheSpecDesc()
        desc.tag = MAIN_KV_TAG
        desc.cache_type = KVCacheSpecType.MHA
        desc.dtype = DataType.TYPE_BF16
        model_config.kv_cache_spec_descs = [
            [desc] for _ in range(model_config.num_layers)
        ]

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        with (Path(ckpt_path) / "config.json").open() as reader:
            raw = json.load(reader)
        if raw.get("architectures") != ["DSparkMiniMaxDraftModel"]:
            raise ValueError("MiniMax-M3.1 DSpARK requires DSparkMiniMaxDraftModel")
        text = raw["text_config"]
        if text.get("layer_types") != ["sliding_attention"] * 5:
            raise ValueError(
                "MiniMax-M3.1 DSpARK requires five sliding-attention layers"
            )
        if text.get("moe_layer_freq") != [0] * 5 or not text.get("use_gemma_norm"):
            raise ValueError("MiniMax-M3.1 DSpARK requires dense Gemma layers")
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.model_type = "minimax_m31_dspark"
        config.hidden_size = int(text["hidden_size"])
        config.num_layers = 5
        config.vocab_size = int(text["vocab_size"])
        config.layernorm_eps = float(text.get("rms_norm_eps", 1e-6))
        config.max_seq_len = max(
            config.max_seq_len, int(text.get("max_position_embeddings", 8192))
        )
        config.attn_config.head_num = int(text["num_attention_heads"])
        config.attn_config.kv_head_num = int(text["num_key_value_heads"])
        config.attn_config.size_per_head = int(text["head_dim"])
        config.attn_config.is_causal = False
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = int(text.get("rope_theta", 10000))
        config.attn_config.rope_config.dim = int(
            text.get("rotary_dim", text["head_dim"])
        )
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        config.enable_fp32_lm_head = False
        config.inter_size = int(text["dense_intermediate_size"])
        config.moe_layer_index = []
        config.expert_num = 0
        config.moe_k = 0
        config.moe_style = 0
        config.swiglu_alpha = float(text.get("swiglu_alpha", 0.0))
        config.swiglu_limit = float(text.get("swiglu_limit", 0.0))
        config.dspark_noise_token_id = int(text["dspark_noise_token_id"])
        config.dspark_target_layer_ids = list(map(int, text["dspark_target_layer_ids"]))
        config.dspark_markov_rank = int(text["dspark_markov_rank"])
        config.attn_config.sliding_window = int(text["sliding_window"])
        config.is_mtp = True
        return config

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        if not isinstance(target_model, MiniMaxM31) or isinstance(
            target_model, MiniMaxM31DSpark
        ):
            raise TypeError("MiniMax-M3.1 DSpARK requires a MiniMax-M3.1 target")
        for name in ("hidden_size", "vocab_size", "data_type", "enable_fp32_lm_head"):
            if getattr(target_model.model_config, name) != getattr(
                draft_model_config, name
            ):
                raise ValueError(f"MiniMax-M3.1 DSpARK target/draft {name} mismatch")
        return (W.embedding, W.lm_head)

    @classmethod
    def configure_speculative_model(cls, sp_config, target_config, draft_config):
        if sp_config.type != SpeculativeType.DSPARK:
            raise ValueError("MiniMax-M3.1 DSpARK requires SP_TYPE=dspark")
        if max(draft_config.dspark_target_layer_ids) >= target_config.num_layers:
            raise ValueError("MiniMax-M3.1 target hidden layer does not exist")

    @staticmethod
    def get_weight_cls():
        return MiniMaxM31DSparkWeight

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        if torch.version.hip is not None:
            raise RuntimeError("MiniMax-M3.1 DSpARK requires CUDA")
        with (Path(self.model_config.ckpt_path) / "config.json").open() as reader:
            draft_text = json.load(reader)["text_config"]
        if (
            int(self.model_config.gen_num_per_cycle)
            != int(draft_text["dspark_block_size"])
            or not self.model_config.dspark_sample_from_anchor
        ):
            raise ValueError(
                "MiniMax-M3.1 DSpARK requires matching proposal width and anchor sampling"
            )
        from rtp_llm.models_py.model_desc.minimax_m31_dspark import (
            MiniMaxM31DSparkMath,
            MiniMaxM31DSparkModel,
        )

        self.py_model = MiniMaxM31DSparkModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            math_contract=MiniMaxM31DSparkMath(
                hidden_norm_gemma=True,
                final_norm_gemma=True,
                causal_query=False,
                window_left=int(self.model_config.attn_config.sliding_window) - 1,
            ),
        )
        return self.py_model


register_model("minimax_m31_dspark", MiniMaxM31DSpark, ["DSparkMiniMaxDraftModel"])
