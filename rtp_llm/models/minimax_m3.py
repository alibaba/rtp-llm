"""MiniMax M3 (text-only LLM backbone) — adaptation for rtp-llm.

This file ports the language-model half of MiniMaxM3SparseForConditionalGeneration
(see sglang `srt/models/minimax_m3.py`). The vision tower is intentionally
ignored for now; we register only the LLM so we can compare logits against
sglang's text path before plumbing through the ViT.

Notable M3-specific points wired here:

* Checkpoint paths are prefixed with ``language_model.`` (the multi-modal
  container). QWenV2Weight already auto-detects this prefix in _process_meta;
  we keep that mechanism and slot it into Glm4Moe-style FFN logic.
* Layers 0..first_k_dense_replace-1 are dense MLP (``mlp.gate_proj``/``up_proj``/
  ``down_proj``), the rest are MoE under ``block_sparse_moe.`` with experts
  named ``experts.{j}.{w1,w2,w3}`` (HF "w1/w2/w3" convention).
* MoE has ``n_shared_experts=1`` mapped to ``block_sparse_moe.shared_experts``
  and a routing-bias tensor at ``block_sparse_moe.e_score_correction_bias``.
  Routing is sigmoid + renormalize (top-4) with routed_scaling_factor=2.0.
* Attention is GQA with QK RMSNorm per-head (Gemma-style +1 weight) and
  partial RoPE (rotary_dim=64 of head_dim=128).
* Sparse attention (MiniMax MSA) layers carry extra index_{q,k}_proj weights
  and index_{q,k}_norm. With ``disable_index_value=1`` (M3 sets this for all
  sparse layers) the index branch only writes idx_K to a dedicated cache and
  does NOT contribute additively to the attention output. These sparse layers
  are routed to ``MSAAttention`` (rtp_llm/models_py/modules/hybrid/
  msa_attention.py), which runs the ported MSA Triton kernels under
  rtp_llm/models_py/triton_kernels/sparse_msa/. The index-branch weights are
  loaded only when ``M3_LOAD_MSA_INDEX`` is set (see _get_hf_layer_weight_info).
"""

import functools
import json
import logging
import os
from typing import Any, Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    merge_qkv_lora_A,
    merge_qkv_lora_B,
    sp_0,
    sp_head_lora,
    sp_id,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
)


def _mxfp8_dequant_to_bf16(ts: List[torch.Tensor]) -> torch.Tensor:
    """Dequantize an on-disk MXFP8 (1x32 microscaling) linear weight to BF16.

    The MiniMax-M3 MSA index projections (``index_q_proj`` / ``index_k_proj``)
    ship as MXFP8: an e4m3 ``weight`` [N, K] plus a UE8M0 ``weight_scale_inv``
    [N, K/32] (one uint8 power-of-two exponent per 1x32 micro-block). The index
    branch only drives top-k *block selection* (it does not contribute to the
    attention value when ``disable_index_value=True``), so BF16 precision is
    plenty and lets us run the index GEMMs as plain ``F.linear`` — no MXFP8
    linear plumbing required.

    ``ts[0]``: e4m3 weight [N, K] (may already be cast to a wider dtype by the
               generic loader — we re-cast to fp32 defensively).
    ``ts[1]``: UE8M0 scale [N, K/32] (uint8 exponent bytes, bias 127).
    """
    w = ts[0].to(torch.float32)
    s = ts[1].to(torch.float32)
    scale = torch.exp2(s - 127.0)  # per (1,32) block power-of-two
    n, k = w.shape
    groups = scale.shape[1]
    group_size = k // groups
    scale_full = scale.repeat_interleave(group_size, dim=1)
    if scale_full.shape[1] != k:
        scale_full = scale_full[:, :k]
    return (w * scale_full).to(torch.bfloat16)


def add_unit_offset(ts: List[torch.Tensor]) -> torch.Tensor:
    """Bake the Gemma-style RMSNorm ``+1`` offset into the gamma at load time.

    MiniMax-M3 uses Gemma RMSNorm (``use_gemma_norm=True``): the effective
    per-channel scale is ``(1 + weight)`` and the checkpoint stores ``weight``
    centered at 0 (e.g. layer-0 input_layernorm has mean ~-0.94). rtp-llm's
    runtime RMSNorm ops apply plain ``x_normed * gamma``, so we add 1.0 here to
    recover the correct scale. Applied to every M3 norm gamma (pre/post layer
    norm, q/k norm, final norm, MSA index q/k norm).
    """
    return (ts[0] + 1.0).contiguous()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _should_load_raw_mxfp8_idx() -> bool:
    return _env_flag("M3_MSA_RAW_IDX_MXFP8")


class MiniMaxM3Weight(ModelDeployWeightInfo):
    """Weight loader for MiniMax-M3 text backbone.

    Mirrors Glm4MoeWeight (Dense+MoE hybrid with shared experts + routing bias)
    but:

    * Adds the ``language_model.`` checkpoint prefix.
    * Uses ``block_sparse_moe`` instead of ``mlp`` for MoE-layer FFN paths.
    * Reads ``e_score_correction_bias`` at the MoE module root (not inside
      ``mlp.gate`` like GLM4).
    * For sparse-attention layers (per ``sparse_attention_freq``), also loads
      the MSA index-branch weights into ``W.msa_idx_{q,k}_{w,norm}`` (consumed
      by ``MSAAttention``). With ``disable_index_value=True`` (M3 default for
      all sparse layers) there is NO ``index_v_proj`` / ``index_o_proj``, so
      those are intentionally not declared. Loading is gated behind
      ``M3_LOAD_MSA_INDEX``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._load_raw_mxfp8_idx = _should_load_raw_mxfp8_idx()
        super().__init__(*args, **kwargs)
        # Multi-modal checkpoints prefix all LLM tensors with `language_model.`
        self.prefix = "language_model."
        # M3 has no bias on q/k/v/o projections.
        self.bias = False
        # M3 always carries the routing bias on MoE layers.
        self.has_e_score_correction_bias = True
        # Layers where sparse attention is active (index branch present on
        # disk). Populated by the model's _create_config via the public
        # _sparse_layer_set helper, defaults to "same as MoE layers" to match
        # the actual M3 checkpoint where sparse_attention_freq == moe_layer_freq.
        # If a checkpoint diverges on that, _process_meta auto-detects.
        self._sparse_layer_set = None

    def _process_meta(self, meta_dict, weight_keys):
        # Auto-detect missing prefix (e.g. a stripped checkpoint).
        if not self._contains(weight_keys, "language_model."):
            self.prefix = ""
        # Confirm e_score_correction_bias presence; default True for M3 but
        # keep the runtime check so a future variant without it still loads.
        self.has_e_score_correction_bias = self._contains(
            weight_keys, ".block_sparse_moe.e_score_correction_bias"
        )
        # Discover sparse layers by checking which layer ids carry an
        # index_q_proj weight in the actual checkpoint. This avoids depending
        # on the config-side sparse_attention_freq list (and catches the case
        # where the checkpoint is dense-only despite a sparse config).
        sparse = set()
        for k in weight_keys:
            # Skip MTP (multi-token-prediction) layers. They live under
            # ``model.mtp.layers.{j}.transformer_layer.`` and carry their own
            # ``index_q_proj``; splitting on ".layers." would misparse that
            # ``layers.{j}`` as a *main-model* layer id (e.g. mtp layer 0 ->
            # main layer 0), wrongly marking a dense layer as sparse and making
            # the loader request a non-existent main-model index tensor.
            if ".mtp." in k:
                continue
            if ".self_attn.index_q_proj.weight" in k and ".weight_scale" not in k:
                # Path: language_model.model.layers.{i}.self_attn.index_q_proj.weight
                try:
                    layer_id = int(k.split(".layers.")[1].split(".")[0])
                    sparse.add(layer_id)
                except (IndexError, ValueError):
                    pass
        self._sparse_layer_set = sparse

    # ------------------------------------------------------------------
    # Per-layer attention weights
    # ------------------------------------------------------------------
    def _get_hf_layer_weight_info(self, layer_id: int):
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )

        layer_weights: List[WeightModule] = [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "model.layers.{i}.input_layernorm.weight",
                        identity,
                    )
                ],
                add_unit_offset,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self.prefix + "model.layers.{i}.self_attn.q_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        self.prefix + "model.layers.{i}.self_attn.k_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        self.prefix + "model.layers.{i}.self_attn.v_proj.weight",
                        identity,
                    ),
                ],
                functools.partial(merge_qkv_hf),
                config=attn_config,
                lora_a_process_func=functools.partial(
                    merge_qkv_lora_A,
                    allow_empty=False,
                    hidden_size=self._hidden_size,
                    head_num=self._head_num,
                    head_num_kv=self._head_num_kv,
                    size_per_head=self._size_per_head,
                ),
                lora_b_process_func=functools.partial(
                    merge_qkv_lora_B,
                    allow_empty=False,
                    hidden_size=self._hidden_size,
                    head_num=self._head_num,
                    head_num_kv=self._head_num_kv,
                    size_per_head=self._size_per_head,
                ),
                lora_a_split_func=sp_id,
                lora_b_split_func=sp_head_lora,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        self.prefix + "model.layers.{i}.self_attn.o_proj.weight",
                        identity,
                    )
                ],
                transpose,
                config=attn_config,
                lora_a_process_func=transpose,
                lora_b_process_func=transpose,
                lora_a_split_func=sp_0,
                lora_b_split_func=sp_id,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix
                        + "model.layers.{i}.post_attention_layernorm.weight",
                        identity,
                    )
                ],
                add_unit_offset,
                config=attn_config,
            ),
        ]

        # M3 always carries per-head QK RMSNorm weights.
        if self._use_qk_norm:
            layer_weights.extend(
                [
                    AttnAtomicWeight(
                        W.q_ln_gamma,
                        [
                            CkptWeightInfo(
                                self.prefix + "model.layers.{i}.self_attn.q_norm.weight"
                            )
                        ],
                        add_unit_offset,
                        config=attn_config,
                    ),
                    AttnAtomicWeight(
                        W.k_ln_gamma,
                        [
                            CkptWeightInfo(
                                self.prefix + "model.layers.{i}.self_attn.k_norm.weight"
                            )
                        ],
                        add_unit_offset,
                        config=attn_config,
                    ),
                ]
            )

        # MSA index-branch weights for sparse layers. M3 disables the index
        # value branch on every sparse layer, so the checkpoint only carries
        # q/k projections + their RMSNorm gammas — no v_proj, no o_proj.
        # These feed MSAAttention (the Triton sparse path) at runtime.
        # The sparse layer set is auto-discovered in _process_meta from
        # checkpoint contents; if discovery hasn't run yet (e.g. unit test
        # path) we declare NONE rather than guessing.
        sparse_set = self._sparse_layer_set
        if sparse_set is None:
            # MSA index weights only exist on the sparse-attention layers of the
            # real checkpoint. If auto-detection (_process_meta) has not run yet,
            # declare NONE rather than guessing from moe_layer_index_: dense
            # layers (0..first_k_dense-1) carry no index_* tensors, so guessing
            # wrong makes the loader request a missing tensor and crash with
            # "ts is empty".
            sparse_set = set()
        # Gate the index branch behind M3_LOAD_MSA_INDEX. When off, only the
        # main GQA chain is loaded and every layer runs dense FlashInfer
        # attention (numerically equivalent to MSA for kv_len <= topk*block);
        # when on, sparse layers load the index weights and route to MSAAttention.
        load_msa_index = _env_flag("M3_LOAD_MSA_INDEX", "false")
        if load_msa_index and layer_id in sparse_set:
            idx_prefix = self.prefix + "model.layers.{i}.self_attn."
            # Always keep BF16-dequantized idx weights for the original/F.linear
            # paths. When M3_MSA_RAW_IDX_MXFP8 is enabled, load an additional
            # raw MXFP8 copy for the fused decode projection. This lets PDFUSION
            # performance tests exercise fused decode while preserving fallback.
            idx_projection_weights = [
                AtomicWeight(
                    W.msa_idx_q_w,
                    [
                        CkptWeightInfo(idx_prefix + "index_q_proj.weight", identity),
                        CkptWeightInfo(
                            idx_prefix + "index_q_proj.weight_scale_inv", identity
                        ),
                    ],
                    _mxfp8_dequant_to_bf16,
                    data_type=torch.bfloat16,
                ),
                AtomicWeight(
                    W.msa_idx_k_w,
                    [
                        CkptWeightInfo(idx_prefix + "index_k_proj.weight", identity),
                        CkptWeightInfo(
                            idx_prefix + "index_k_proj.weight_scale_inv", identity
                        ),
                    ],
                    _mxfp8_dequant_to_bf16,
                    data_type=torch.bfloat16,
                ),
            ]
            if self._load_raw_mxfp8_idx:
                idx_projection_weights.extend(
                    [
                        AtomicWeight(
                            W.msa_idx_q_raw_w,
                            [
                                CkptWeightInfo(
                                    idx_prefix + "index_q_proj.weight", identity
                                )
                            ],
                            identity,
                            data_type=torch.float8_e4m3fn,
                        ),
                        AtomicWeight(
                            W.msa_idx_q_raw_s,
                            [
                                CkptWeightInfo(
                                    idx_prefix + "index_q_proj.weight_scale_inv",
                                    identity,
                                )
                            ],
                            identity,
                            data_type=torch.uint8,
                        ),
                        AtomicWeight(
                            W.msa_idx_k_raw_w,
                            [
                                CkptWeightInfo(
                                    idx_prefix + "index_k_proj.weight", identity
                                )
                            ],
                            identity,
                            data_type=torch.float8_e4m3fn,
                        ),
                        AtomicWeight(
                            W.msa_idx_k_raw_s,
                            [
                                CkptWeightInfo(
                                    idx_prefix + "index_k_proj.weight_scale_inv",
                                    identity,
                                )
                            ],
                            identity,
                            data_type=torch.uint8,
                        ),
                    ]
                )
            layer_weights.extend(
                [
                    *idx_projection_weights,
                    AtomicWeight(
                        W.msa_idx_q_norm,
                        [CkptWeightInfo(idx_prefix + "index_q_norm.weight", identity)],
                        add_unit_offset,
                        data_type=torch.bfloat16,
                    ),
                    AtomicWeight(
                        W.msa_idx_k_norm,
                        [CkptWeightInfo(idx_prefix + "index_k_norm.weight", identity)],
                        add_unit_offset,
                        data_type=torch.bfloat16,
                    ),
                ]
            )

        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    # ------------------------------------------------------------------
    # Per-layer FFN weights — dense MLP vs MoE switched on layer_id
    # ------------------------------------------------------------------
    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        align_size = self._align_size

        ffn_config = FfnConfig(
            align_size=align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )

        if layer_id in self.moe_layer_index_:
            moe_config = MoeConfig(
                align_size=align_size,
                expert_num=self.expert_num_,
            )
            moe_root = self.prefix + "model.layers.{i}.block_sparse_moe."

            layer_weights: List[WeightModule] = [
                # Shared expert (M3 n_shared_experts=1). Loaded into the
                # FfnWeight slot exactly like GLM4 / Qwen2-MoE so it composes
                # with the rest of the moe_style=2 inference path.
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [
                                CkptWeightInfo(
                                    moe_root + "shared_experts.gate_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [
                                CkptWeightInfo(
                                    moe_root + "shared_experts.down_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=1,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [
                                CkptWeightInfo(
                                    moe_root + "shared_experts.up_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
                # Routed experts. M3 uses HuggingFace w1/w2/w3 names directly
                # for routed experts — convention matches Qwen2-MoE.
                MoeWeight(
                    sub_weights=[
                        MoeAtomicWeight(
                            W.moe_gate,
                            [CkptWeightInfo(moe_root + "gate.weight", identity)],
                            transpose,
                            config=moe_config,
                        ),
                        MoeAtomicWeight(
                            W.moe_w2,
                            [
                                CkptWeightInfo(
                                    moe_root + "experts.{expert_id}.w2.weight",
                                    identity,
                                )
                            ],
                            stack_,
                            config=moe_config,
                        ),
                        # Note: stack_moe_w1 stacks [up, gate] in that order;
                        # FusedMoE expects gate_up packed = [gate; up]; this
                        # helper handles the interleave to match the kernel
                        # — see Qwen3-MoE / GLM4-MoE which use the same call.
                        MoeAtomicWeight(
                            W.moe_w1,
                            [
                                CkptWeightInfo(
                                    moe_root + "experts.{expert_id}.w3.weight",
                                    identity,
                                )
                            ]
                            + [
                                CkptWeightInfo(
                                    moe_root + "experts.{expert_id}.w1.weight",
                                    identity,
                                )
                            ],
                            stack_moe_w1,
                            config=moe_config,
                        ),
                    ],
                    config=moe_config,
                ),
            ]
            if self.has_e_score_correction_bias:
                layer_weights.append(
                    AtomicWeight(
                        W.e_score_correction_b,
                        [
                            CkptWeightInfo(
                                moe_root + "e_score_correction_bias",
                                identity,
                            )
                        ],
                        identity,
                        data_type=torch.float32,
                    )
                )
            return layer_weights
        else:
            # Dense MLP (M3 layers 0..first_k_dense_replace-1)
            mlp_root = self.prefix + "model.layers.{i}.mlp."
            return [
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [CkptWeightInfo(mlp_root + "gate_proj.weight", identity)],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [CkptWeightInfo(mlp_root + "down_proj.weight", identity)],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=1,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [CkptWeightInfo(mlp_root + "up_proj.weight", identity)],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
            ]

    # ------------------------------------------------------------------
    # Global weights
    # ------------------------------------------------------------------
    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.prefix + "model.norm.weight", identity)],
                add_unit_offset,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
                identity,
            ),
        ]
        for layer in range(self._num_layers):
            layer_weights.append(self._get_hf_layer_weight_info(layer))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class MiniMaxM3(DeepSeekV2):
    """MiniMax-M3 LLM backbone.

    Reuses DeepSeekV2's base infrastructure (which is the closest hybrid
    MoE+QK-norm cousin already wired for the GenericMoeModel python path).
    Dense layers run FlashInfer GQA; sparse layers run the Triton MSA path
    (MSAAttention) when ``M3_LOAD_MSA_INDEX`` is set. Dense and MSA are
    numerically equivalent whenever the total KV length is <= topk *
    block_size (16 * 128 = 2048 tokens for the default M3 config).
    """

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.attn_config.head_num = 0
        config.attn_config.kv_head_num = 0
        config.attn_config.size_per_head = 0
        config.num_layers = 0
        config.inter_size = 0
        config.vocab_size = 0
        config.max_seq_len = 8192
        config.attn_config.rope_config.style = 1
        # M3 uses "SiGLU" as the activation_type enum, but the actual math is
        # SwiGLU-OAI (clamp + sigmoid_alpha + up_plus_one). DenseMLP and the
        # MoE deepgemm executor pick up the OAI variant when both
        # ``config.swiglu_alpha > 0`` AND ``config.swiglu_limit > 0``;
        # otherwise they fall back to plain SiLU-and-mul.
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        # MiniMax-M3 checkpoints store lm_head in BF16. Keeping the framework
        # default would upcast it to FP32 at load time and dispatch a slow FP32
        # SIMT full-vocab GEMM in decode; keep the checkpoint half precision so
        # PyWrappedModel uses the BF16 tensor-core lm_head path.
        config.enable_fp32_lm_head = False

        cls._from_hf(config, ckpt_path)
        assert (
            config.attn_config.head_num > 0
            and config.attn_config.kv_head_num > 0
            and config.attn_config.size_per_head > 0
            and config.num_layers > 0
            and config.inter_size > 0
        ), (
            f"invalid m3 config: head_num={config.attn_config.head_num} "
            f"kv_head_num={config.attn_config.kv_head_num} "
            f"size_per_head={config.attn_config.size_per_head} "
            f"num_layers={config.num_layers} inter_size={config.inter_size}"
        )
        return config

    @classmethod
    def _from_hf(cls, config: "ModelConfig", ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        # MiniMax M3 nests the LLM-relevant fields under "text_config".
        text_cfg = config_json.get("text_config", config_json)
        MiniMaxM3._from_text_config(config, text_cfg)
        logging.info(
            "minimax-m3 config: layers=%d hidden=%d heads=%d/%d head_dim=%d "
            "experts=%dx top%d shared=%d rotary_dim=%d qk_norm=%s "
            "scoring=%s routed_scale=%.2f",
            config.num_layers,
            config.hidden_size,
            config.attn_config.head_num,
            config.attn_config.kv_head_num,
            config.attn_config.size_per_head,
            config.expert_num,
            config.moe_k,
            text_cfg.get("n_shared_experts", 0),
            config.attn_config.rope_config.dim,
            config.qk_norm,
            "sigmoid" if config.scoring_func == 1 else "softmax",
            config.routed_scaling_factor,
        )
        return config

    @staticmethod
    def _from_text_config(config: "ModelConfig", text_cfg: Dict[str, Any]):
        # ----- Backbone shape -----
        config.hidden_size = int(text_cfg["hidden_size"])
        config.num_layers = int(text_cfg["num_hidden_layers"])
        config.attn_config.head_num = int(text_cfg["num_attention_heads"])
        config.attn_config.kv_head_num = int(
            text_cfg.get("num_key_value_heads", config.attn_config.head_num)
        )
        config.attn_config.size_per_head = (
            int(text_cfg["head_dim"])
            if "head_dim" in text_cfg
            else config.hidden_size // config.attn_config.head_num
        )
        config.vocab_size = int(text_cfg["vocab_size"])
        config.layernorm_eps = float(text_cfg.get("rms_norm_eps", 1e-6))
        config.tie_word_embeddings = bool(text_cfg.get("tie_word_embeddings", False))
        config.max_seq_len = max(
            config.max_seq_len, int(text_cfg.get("max_position_embeddings", 8192))
        )

        # ----- RoPE (partial) -----
        config.attn_config.rope_config.base = int(
            text_cfg.get("rope_theta", config.attn_config.rope_config.base)
        )
        if "rotary_dim" in text_cfg:
            config.attn_config.rope_config.dim = int(text_cfg["rotary_dim"])
        else:
            partial = float(text_cfg.get("partial_rotary_factor", 1.0))
            config.attn_config.rope_config.dim = int(
                config.attn_config.size_per_head * partial
            )

        # ----- QK norm -----
        config.qk_norm = bool(text_cfg.get("use_qk_norm", True))

        # ----- MoE / dense split -----
        # M3 expresses the dense->MoE switch via moe_layer_freq list-of-ints
        # (0=dense, !=0=MoE), same convention as GLM4's first_k_dense_replace
        # but per-layer.
        moe_layer_freq = text_cfg.get("moe_layer_freq")
        num_layers = config.num_layers
        if moe_layer_freq is None:
            config.moe_layer_index = list(range(num_layers))
        else:
            assert (
                len(moe_layer_freq) == num_layers
            ), f"moe_layer_freq length {len(moe_layer_freq)} != num_layers {num_layers}"
            config.moe_layer_index = [i for i, v in enumerate(moe_layer_freq) if v != 0]

        config.expert_num = int(text_cfg.get("num_local_experts", 0))
        config.moe_k = int(text_cfg.get("num_experts_per_tok", 0))
        config.moe_inter_size = int(text_cfg.get("intermediate_size", 0))
        n_shared = int(text_cfg.get("n_shared_experts", 0))
        # Shared expert width = n_shared_experts * shared_intermediate_size.
        # M3 carries this in `shared_intermediate_size`; fall back to
        # intermediate_size if absent so a checkpoint without the field still
        # loads.
        shared_inter = int(
            text_cfg.get("shared_intermediate_size", text_cfg["intermediate_size"])
        )
        shared_total = n_shared * shared_inter if n_shared > 0 else 0
        # Dense MLP layers use ``dense_intermediate_size`` (M3 = 12288, much
        # larger than the per-MoE-layer shared expert width of 3072). rtp-llm
        # has a single ``config.inter_size`` slot used both for memory
        # budgeting and (in some legacy paths) for FFN buffer sizing, so we
        # pick the larger of {shared_total, dense_inter_size} to avoid
        # under-allocation on dense layers. Runtime correctness comes from
        # the loaded tensor's actual shape, not from inter_size.
        dense_inter = int(text_cfg.get("dense_intermediate_size", shared_total))
        config.inter_size = max(shared_total, dense_inter)
        # moe_style=2 = "shared + routed experts" (the GLM4/Qwen2-MoE path).
        config.moe_style = 2 if n_shared > 0 else 1

        # Sigmoid scoring + renormalize + routing-bias (e_score_correction).
        scoring = text_cfg.get("scoring_func", "softmax")
        config.scoring_func = 1 if scoring == "sigmoid" else 0
        # In sglang TopK uses renormalize=True for M3 — match that.
        config.has_moe_norm = True
        config.routed_scaling_factor = float(text_cfg.get("routed_scaling_factor", 1.0))
        # M3 does NOT use grouped TopK; leave n_group/topk_group at the
        # default 1 so SelectTopk + GroupTopK (correction_bias path) take the
        # ungrouped branch.
        config.moe_n_group = 1
        config.moe_topk_group = 1

        # SwiGLU-OAI parameters (GPT-OSS / MiniMax-M3 variant). M3 ships
        # alpha=1.702 / limit=7.0. Both must be > 0 for DenseMLP + the MoE
        # deepgemm executor to route through the OAI kernels; otherwise they
        # silently fall back to plain SiLU-and-mul.
        config.swiglu_alpha = float(text_cfg.get("swiglu_alpha", 0.0))
        config.swiglu_limit = float(text_cfg.get("swiglu_limit", 0.0))

        # ----- Sparse attention (MiniMax MSA) -----
        # Parse the checkpoint ``sparse_attention_config`` into a python-only
        # dict used by MSAAttention to wire the Triton MSA kernels for sparse
        # layers. When absent or disabled, leave None so every layer runs dense
        # GQA (which is numerically equivalent to MSA for kv_len <= topk*block).
        MiniMaxM3._parse_sparse_config(config, text_cfg)

    @staticmethod
    def _parse_sparse_config(config: "ModelConfig", text_cfg: Dict[str, Any]):
        sparse_cfg = text_cfg.get("sparse_attention_config")
        if not sparse_cfg or not sparse_cfg.get("use_sparse_attention", False):
            config.msa_sparse_config = None
            return
        num_layers = config.num_layers
        freq = sparse_cfg.get("sparse_attention_freq", [])
        sparse_layer_ids = [i for i, f in enumerate(freq) if f != 0 and i < num_layers]
        disable_value_flags = sparse_cfg.get("sparse_disable_index_value", [])
        disable_value_layer_ids = {
            i for i, f in enumerate(disable_value_flags) if f != 0
        }
        block_size = int(sparse_cfg["sparse_block_size"])
        if "sparse_init_block" in sparse_cfg:
            init_blocks = int(sparse_cfg["sparse_init_block"])
        else:
            init_tokens = int(sparse_cfg.get("sparse_init_tokens", 0))
            init_blocks = (init_tokens + block_size - 1) // block_size
        if "sparse_local_block" in sparse_cfg:
            local_blocks = int(sparse_cfg["sparse_local_block"])
        else:
            local_tokens = int(sparse_cfg.get("sparse_local_tokens", 0))
            local_blocks = (local_tokens + block_size - 1) // block_size + 1
        idx_head_dim = int(sparse_cfg["sparse_index_dim"])
        config.msa_sparse_config = {
            "sparse_layer_ids": sparse_layer_ids,
            "disable_value_layer_ids": sorted(disable_value_layer_ids),
            "idx_head_dim": idx_head_dim,
            "num_idx_heads": int(sparse_cfg["sparse_num_index_heads"]),
            "topk_blocks": int(sparse_cfg["sparse_topk_blocks"]),
            "block_size": block_size,
            "init_blocks": init_blocks,
            "local_blocks": local_blocks,
            "score_type": str(sparse_cfg.get("sparse_score_type", "max")),
        }
        # PD-compatible idx_K: when M3_IDX_PAGED is set, tell the C++ cache
        # config to enlarge the MHA scale region of the main paged pool so the
        # BF16 idx_K can be stored there (addressed by the same block table and
        # transferred with the main K/V under PD separation). We deliberately do
        # NOT set ``attn_config.is_sparse`` here: that flag flips the pool to the
        # MLA layout (BlockPoolConfigHelper: is_mla = use_mla || is_sparse) which
        # would break M3's MHA main-K/V paging. ``indexer_head_dim`` alone drives
        # the scale-region sizing in SingleConfigCreator without touching is_mla.
        if os.environ.get("M3_IDX_PAGED", "0") == "1":
            config.attn_config.indexer_head_dim = idx_head_dim
        logging.info("minimax-m3 MSA sparse config: %s", config.msa_sparse_config)

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m3 import MiniMaxM3Model

        self.py_model = MiniMaxM3Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls() -> type[MiniMaxM3Weight]:
        return MiniMaxM3Weight


# ----------------------------------------------------------------------
# Registration
#   - "minimax_m3"             : forced name
#   - HF arch via auto-mapping : MiniMaxM3SparseForConditionalGeneration
#                                (the VL container; we load only the LLM)
# ----------------------------------------------------------------------
# arch routing moved to MiniMaxM3_VL in minimax_m3_vl.py — text-only ckpts require explicit --model_type=minimax_m3
register_model(
    "minimax_m3",
    MiniMaxM3,
    [],
    [],
)
