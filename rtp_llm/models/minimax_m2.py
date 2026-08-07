"""MiniMax-M2 model support.

Architecture (per HF `MiniMaxM2ForCausalLM`):
  - 62 decoder layers, ALL full softmax attention.
  - GQA: 48 attn heads / 8 KV heads, head_dim=128.
  - Partial RoPE: rotary_dim=64 (only first half of head_dim is rotated).
  - QK-norm with `qk_norm_type="per_layer"` (single RMSNorm over the full
    H*D dimension before per-head reshape).
  - MoE in EVERY layer: 256 experts top-8, sigmoid scoring + per-layer
    `e_score_correction_bias` for routing (DeepSeek-V3 style), no shared
    expert (`shared_intermediate_size=0`), routing weights renormalized to
    sum=1 (no `routed_scaling_factor`).
  - Mixtral-style MoE tensor naming:
        model.layers.{i}.block_sparse_moe.gate.weight
        model.layers.{i}.block_sparse_moe.e_score_correction_bias
        model.layers.{i}.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight
    where w1=gate_proj, w3=up_proj, w2=down_proj (silu(w1) * w3 -> w2).
  - FP8 e4m3fn block-wise quantization (128x128). All q/k/v/o + experts'
    w1/w2/w3 are FP8 with `*.weight_scale_inv`; gate / e_score_correction_bias
    / lm_head / norms / embedding stay bf16 (matches `modules_to_not_convert`).
    FP8 is auto-detected by `rtp_llm/config/quant_config.py`; nothing extra
    needed here.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    stack_moe_w1,
    transpose,
)


class MinimaxM2Weight(QWenV2Weight):
    """Weight loader for MiniMax-M2."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # M2 attention has no q/k/v/o biases.
        self.bias = False

    def _get_hf_ffn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
        )

        up_first_then_gate: List[CkptWeightInfo] = [
            CkptWeightInfo(
                "model.layers.{i}.block_sparse_moe.experts.{expert_id}.w3.weight",
                identity,
            ),
            CkptWeightInfo(
                "model.layers.{i}.block_sparse_moe.experts.{expert_id}.w1.weight",
                identity,
            ),
        ]

        return [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.block_sparse_moe.gate.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        up_first_then_gate,
                        stack_moe_w1,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.block_sparse_moe.experts.{expert_id}.w2.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            ),
            AtomicWeight(
                W.e_score_correction_b,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.block_sparse_moe.e_score_correction_bias",
                        identity,
                    )
                ],
                identity,
                data_type=torch.float32,
            ),
        ]


class MinimaxM2(QWenV2):
    """MiniMax-M2 model registration."""

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        cls._patch_minimax_m2(ckpt_path, config)
        return config

    @classmethod
    def _patch_minimax_m2(cls, ckpt_path: str, config: ModelConfig) -> None:
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as f:
            cj: Dict[str, Any] = json.load(f)

        # ---- MoE ----
        config.expert_num = cj["num_local_experts"]
        config.moe_k = cj["num_experts_per_tok"]
        config.moe_layer_index = list(range(config.num_layers))
        # No shared expert: shared_intermediate_size==0.
        config.inter_size = 0
        config.moe_inter_size = cj["intermediate_size"]
        config.moe_style = 1  # MoE-only, no shared
        config.has_moe_norm = True
        config.scoring_func = 1
        config.routed_scaling_factor = 1.0
        config.moe_n_group = 1
        config.moe_topk_group = 1

        # ---- QK-norm ----
        config.qk_norm = bool(cj.get("use_qk_norm", False))
        qk_norm_type = cj.get("qk_norm_type", "per_layer")
        if config.qk_norm and qk_norm_type not in ("per_head", "per_layer"):
            raise NotImplementedError(
                f"[MiniMax-M2] Unsupported qk_norm_type={qk_norm_type!r}; "
                "only 'per_head' and 'per_layer' are wired."
            )
        config.qk_norm_type = qk_norm_type

        # ---- Partial RoPE ----
        rotary_dim = cj.get("rotary_dim")
        if rotary_dim is not None:
            config.attn_config.rope_config.dim = int(rotary_dim)
        else:
            partial_factor = cj.get("partial_rotary_factor", 1.0)
            config.attn_config.rope_config.dim = int(
                config.attn_config.size_per_head * partial_factor
            )

        # ---- Special tokens ----
        gen_cfg: Dict[str, Any] = {}
        gen_cfg_path = os.path.join(ckpt_path, "generation_config.json")
        if os.path.exists(gen_cfg_path):
            with open(gen_cfg_path) as gf:
                gen_cfg = json.load(gf)

        tok_cfg: Dict[str, Any] = {}
        tok_cfg_path = os.path.join(ckpt_path, "tokenizer_config.json")
        if os.path.exists(tok_cfg_path):
            with open(tok_cfg_path) as tf:
                tok_cfg = json.load(tf)

        def _lookup_added_token_id(content: str) -> Optional[int]:
            for k, v in tok_cfg.get("added_tokens_decoder", {}).items():
                if isinstance(v, dict) and v.get("content") == content:
                    return int(k)
            return None

        bos: Optional[int] = None
        bos_str = tok_cfg.get("bos_token")
        if isinstance(bos_str, dict):
            bos_str = bos_str.get("content")
        if bos_str:
            bos = _lookup_added_token_id(bos_str)
        if bos is None:
            bos = gen_cfg.get("bos_token_id", cj.get("bos_token_id"))
        if bos is not None:
            config.special_tokens.bos_token_id = bos

        eos_raw = gen_cfg.get("eos_token_id", cj.get("eos_token_id"))
        extra_eos: List[int] = []
        if isinstance(eos_raw, list):
            primary_eos = eos_raw[0] if eos_raw else None
            extra_eos = list(eos_raw[1:])
        else:
            primary_eos = eos_raw
        if primary_eos is not None:
            config.special_tokens.eos_token_id = primary_eos

        # ---- Stop words ----
        swl: List[List[int]] = []
        eot_id = _lookup_added_token_id("[e~[")
        if eot_id is not None:
            swl.append([eot_id])
        swl.extend([[int(e)] for e in extra_eos])
        config.special_tokens.stop_words_id_list = swl

        # ---- Long context ----
        max_pos = cj.get("max_position_embeddings")
        if max_pos is not None:
            config.max_seq_len = max_pos

        logging.info(
            "[MiniMax-M2] num_layers=%d hidden=%d heads=%d/%d head_dim=%d "
            "rotary_dim=%d rope_theta=%d experts=%d top-%d vocab=%d "
            "max_seq=%d qk_norm=%s(type=%s) tie_word_emb=%s",
            config.num_layers,
            config.hidden_size,
            config.attn_config.head_num,
            config.attn_config.kv_head_num,
            config.attn_config.size_per_head,
            config.attn_config.rope_config.dim,
            config.attn_config.rope_config.base,
            config.expert_num,
            config.moe_k,
            config.vocab_size,
            config.max_seq_len,
            config.qk_norm,
            qk_norm_type,
            config.tie_word_embeddings,
        )

    @staticmethod
    def get_weight_cls():
        return MinimaxM2Weight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel

        self.py_model = GenericMoeModel(
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


register_model("minimax_m2", MinimaxM2, ["MiniMaxM2ForCausalLM"])
