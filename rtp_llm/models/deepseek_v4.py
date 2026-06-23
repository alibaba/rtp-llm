"""DeepSeek-V4 model entry point (skeleton).

This file establishes the registration and HF config parsing for DeepseekV4.
For M0 the forward path is intentionally not wired up — `_create_python_model`
raises so the engine fails fast if you try to actually run inference. Subsequent
milestones (M1 mHC, M2 HCA, M3 CSA, M4 heterogeneous KV cache) flesh out the
real model body in `rtp_llm/models_py/model_desc/deepseek_v4_model.py`.

Key V4 architecture differences vs V3 — see
`/home/wangyin.yx/workspace/work_memory/plans/develop_ds_v4.md` for the full
breakdown:

  - MQA single-KV-head with head_dim=512 (no kv_lora_rank, no v_head_dim split)
  - Per-layer attention type from `compress_ratios`: 0=SWA-only, 4=CSA, 128=HCA
  - mHC residual: residual stream lives in [n_hc=4, hidden_size]
  - Grouped output projection: n_h heads -> g groups -> o_lora_rank -> hidden
  - Lightning indexer with FP4 QK / BF16 score for CSA (top-k selection)
  - Sliding-window bypass + per-head learnable attention sink in softmax denom
  - sqrt(softplus) MoE scoring (scoring_func=2), no n_group/topk_group
  - First `num_hash_layers` MoE layers route via deterministic token-id hash
  - SwiGLU clamping (linear in [-swiglu_limit,limit], gate <= limit)
  - Dual RoPE bases: rope_theta=10000 main, compress_rope_theta=160000 for
    compressed K branch
"""

import functools
import json
import logging
import os
from typing import Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import (
    CacheType,
    CompressedKVCacheSpec,
    DataType,
    FixedStateCacheSpec,
    KVCacheSpec,
    KVCacheSpecDesc,
    KvCacheDataType,
)
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import (
    DeepSeekV2,
    DeepSeekV2Weight,
    DeepSeekV3Mtp,
    DeepSeekV3MtpWeight,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    concat_0,
    identity,
    stack_,
    stack_moe_w1,
    yarn_get_mscale,
    zeros,
)

SCORING_FUNC_SOFTMAX = 0
SCORING_FUNC_SIGMOID = 1
SCORING_FUNC_SQRT_SOFTPLUS = 2  # DeepSeek-V4

def _make_kv_cache_spec(
    tag: str,
    kind: str,
    entry_elems: int,
    dtype: DataType,
    compression_ratio: int = 1,
) -> KVCacheSpec:
    if kind == "compressed_kv":
        spec = CompressedKVCacheSpec()
        spec.entry_elems = int(entry_elems)
        spec.compression_ratio = int(compression_ratio)
        spec.store_dtype = dtype
    else:
        spec = FixedStateCacheSpec()
        spec.state_dim = int(entry_elems)
        spec.store_dtype = dtype
    spec.tag = tag
    spec.dtype = dtype
    return spec


def _build_dsv4_kv_cache_specs(config: ModelConfig) -> Dict[int, List[KVCacheSpec]]:
    ratios = list(config.attn_config.layer_compress_ratios)

    head_dim = int(config.attn_config.size_per_head)
    indexer_head_dim = int(config.attn_config.indexer_head_dim)
    # These defaults match BASE/BF16 cache. DeepSeekV4._post_build_model_config
    # refreshes entry sizes after kv_cache_config/quantization finalizes kv_cache_dtype.
    kv_entry_elems = head_dim * 2
    indexer_entry_elems = indexer_head_dim * 2

    csa_kv_spec = _make_kv_cache_spec(
        "csa_kv", "compressed_kv", kv_entry_elems, DataType.TYPE_UINT8, 4
    )
    hca_kv_spec = _make_kv_cache_spec(
        "hca_kv", "compressed_kv", kv_entry_elems, DataType.TYPE_UINT8, 128
    )
    indexer_kv_spec = _make_kv_cache_spec(
        "indexer_kv", "compressed_kv", indexer_entry_elems, DataType.TYPE_UINT8, 4
    )
    indexer_state_spec = _make_kv_cache_spec(
        "indexer_state", "fixed_state", 4 * indexer_head_dim, DataType.TYPE_FP32
    )
    csa_state_spec = _make_kv_cache_spec(
        "csa_state", "fixed_state", 4 * head_dim, DataType.TYPE_FP32
    )
    hca_state_spec = _make_kv_cache_spec(
        "hca_state", "fixed_state", 2 * head_dim, DataType.TYPE_FP32
    )
    swa_kv_spec = _make_kv_cache_spec(
        "swa_kv", "sliding_window_kv", kv_entry_elems, DataType.TYPE_UINT8
    )

    layer_specs: Dict[int, List[KVCacheSpec]] = {}
    for layer_id in range(config.num_layers):
        ratio = int(ratios[layer_id]) if layer_id < len(ratios) else 0
        if ratio == 4:
            layer_specs[layer_id] = [
                csa_kv_spec,
                indexer_kv_spec,
                indexer_state_spec,
                csa_state_spec,
                swa_kv_spec,
            ]
        elif ratio == 128:
            layer_specs[layer_id] = [hca_kv_spec, hca_state_spec, swa_kv_spec]
        else:
            layer_specs[layer_id] = [swa_kv_spec]
    return layer_specs


def _kv_cache_spec_to_desc(spec: KVCacheSpec) -> KVCacheSpecDesc:
    desc = KVCacheSpecDesc()
    desc.tag = spec.tag
    desc.seq_size_per_block = int(spec.seq_size_per_block)
    desc.dtype = spec.dtype
    if hasattr(spec, "compression_ratio"):
        desc.cache_type = CacheType.COMPRESSED_KV
        desc.entry_elems = int(spec.entry_elems)
        desc.compression_ratio = int(spec.compression_ratio)
        desc.store_dtype = spec.store_dtype
        desc.block_size_bytes_alignment = int(spec.block_size_bytes_alignment)
    else:
        desc.cache_type = CacheType.FIXED_STATE
        desc.entry_elems = int(spec.state_dim)
        desc.entries_per_block = int(spec.entries_per_block)
        desc.store_dtype = spec.store_dtype
        desc.block_size_bytes_override = int(spec.block_size_bytes_override)
        desc.block_size_bytes_alignment = int(spec.block_size_bytes_alignment)
        desc.block_size_alignment_min_entries = int(
            spec.block_size_alignment_min_entries
        )
        desc.is_state_cache = spec.tag != "swa_kv"
        desc.skip_prefix_reuse = bool(spec.skip_prefix_reuse)
    return desc


def _build_dsv4_kv_cache_spec_descs(
    layer_specs: Dict[int, List[KVCacheSpec]]
) -> Dict[int, List[KVCacheSpecDesc]]:
    return {
        layer_id: [_kv_cache_spec_to_desc(spec) for spec in specs]
        for layer_id, specs in layer_specs.items()
    }


def _refresh_dsv4_kv_cache_specs(config: ModelConfig) -> None:
    expected_tags = [
        "csa_kv",
        "hca_kv",
        "indexer_kv",
        "indexer_state",
        "csa_state",
        "hca_state",
        "swa_kv",
    ]
    expected_tag_set = set(expected_tags)

    specs_by_tag = {}
    for layer_id, layer_specs in config.kv_cache_specs.items():
        seen_layer_tags = set()
        for spec in layer_specs:
            if not spec.tag:
                raise ValueError("DeepSeek-V4 kv_cache spec tag must not be empty")
            if spec.tag not in expected_tag_set:
                raise ValueError(f"Unknown DeepSeek-V4 kv_cache spec tag: {spec.tag}")
            if spec.tag in seen_layer_tags:
                raise ValueError(
                    f"DeepSeek-V4 kv_cache spec tag duplicated in layer {layer_id}: {spec.tag}"
                )
            seen_layer_tags.add(spec.tag)
            specs_by_tag.setdefault(spec.tag, spec)

    fp8_kv = config.attn_config.kv_cache_dtype == KvCacheDataType.FP8
    head_dim = int(config.attn_config.size_per_head)
    indexer_head_dim = int(config.attn_config.indexer_head_dim)
    kv_entry_elems = 584 if fp8_kv else head_dim * 2
    indexer_entry_elems = 132 if fp8_kv else indexer_head_dim * 2

    for tag in expected_tags:
        spec = specs_by_tag.get(tag)
        if spec is None:
            continue
        if tag in ("csa_kv", "hca_kv", "swa_kv"):
            if hasattr(spec, "entry_elems"):
                spec.entry_elems = kv_entry_elems
            else:
                spec.state_dim = kv_entry_elems
            spec.store_dtype = DataType.TYPE_UINT8
            spec.dtype = DataType.TYPE_UINT8
        elif tag == "indexer_kv":
            spec.entry_elems = indexer_entry_elems
            spec.store_dtype = DataType.TYPE_UINT8
            spec.dtype = DataType.TYPE_UINT8
        elif tag == "indexer_state":
            spec.state_dim = 4 * indexer_head_dim
            spec.store_dtype = DataType.TYPE_FP32
            spec.dtype = DataType.TYPE_FP32
        elif tag == "csa_state":
            spec.state_dim = 4 * head_dim
            spec.store_dtype = DataType.TYPE_FP32
            spec.dtype = DataType.TYPE_FP32
        elif tag == "hca_state":
            spec.state_dim = 2 * head_dim
            spec.store_dtype = DataType.TYPE_FP32
            spec.dtype = DataType.TYPE_FP32

    config.kv_cache_spec_descs = _build_dsv4_kv_cache_spec_descs(
        config.kv_cache_specs
    )



class DeepSeekV4Weight(DeepSeekV2Weight):
    """DeepSeek-V4 weight info.

    Declares the per-layer + global ``WeightModule`` graph the framework's
    fastsafetensors loader needs to populate ``ModelWeights`` from a V4-Flash
    safetensors checkpoint.  ``DeepSeekV4Model._initialize_impl`` hands the
    populated ``ModelWeights`` directly to ``V4Transformer``; each dsv4
    sub-module reads its tensors from ``mw.global_weights[W.*]`` /
    ``mw.weights[layer_id][W.v4_*]`` (W enum keys, no string lookups).

    Compatibility envelope:
      * V4 ckpt key naming is ``layers.{i}.…`` (no ``model.`` prefix); V4 W
        constants live under the ``v4.…`` namespace so per-block FP8 / per-group
        FP4 quant subclasses can scope ``support()`` to V4 only.
      * Per-layer module set is heterogeneous — driven by the layer's
        ``compress_ratio`` from ``config.attn_config.layer_compress_ratios`` and
        whether the layer's router is hash (``layer_id < num_hash_layers``) or
        noaux_tc (``layer_id >= num_hash_layers``).
    """

    def _process_meta(self, meta_dict, weight_keys):  # type: ignore[override]
        # V4 has no LoRA q-projection split (we use ``wq_a`` / ``wq_b`` directly)
        # and no e_score_correction_bias (uses noaux_tc gate.bias on layers ≥ num_hash_layers).
        self.q_use_lora = False
        self.has_e_score_correction_bias = False
        # Per-layer attention type schedule + hash-router count, both attached to
        # ``self`` so ``_get_hf_layer_weight_info`` can branch per layer.
        self._compress_ratios = list(
            self.model_config.attn_config.layer_compress_ratios
        )
        # Pad/truncate to num_layers for safety.
        if len(self._compress_ratios) < self._num_layers:
            self._compress_ratios = self._compress_ratios + [0] * (
                self._num_layers - len(self._compress_ratios)
            )
        else:
            self._compress_ratios = self._compress_ratios[: self._num_layers]
        self._num_hash_layers = int(self.model_config.num_hash_layers)

    def _compress_ratio(self, layer_id: int) -> int:
        if layer_id < 0 or layer_id >= len(self._compress_ratios):
            return 0
        return int(self._compress_ratios[layer_id])

    _LAYER_PREFIX: str = "layers.{i}"

    def _key(self, suffix: str) -> str:
        return f"{self._LAYER_PREFIX}.{suffix}"

    # ------------------------------------------------------------------
    # Per-layer descriptor builders
    # ------------------------------------------------------------------

    def _build_attn_norms(self, layer_id: int) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.v4_attn_norm,
                [CkptWeightInfo(self._key("attn_norm.weight"), identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_attn_q_norm,
                [CkptWeightInfo(self._key("attn.q_norm.weight"), identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_attn_kv_norm,
                [CkptWeightInfo(self._key("attn.kv_norm.weight"), identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_attn_sink,
                [CkptWeightInfo(self._key("attn.attn_sink"), identity)],
                identity,
                data_type=torch.float32,
            ),
        ]

    def _v4_attn_cfg(self) -> AttnConfig:
        return AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )

    def _build_attn_dense_fp8(self, layer_id: int) -> List[WeightModule]:
        cfg = self._v4_attn_cfg()
        # Each AttnAtomicWeight here will be wrapped by V4PerBlockFp8Weight
        # in to_quant_weight_info() — that subclass auto-derives the .scale
        # ckpt key from the .weight ckpt key using V4's suffix convention.
        return [
            AttnAtomicWeight(
                W.v4_attn_wq_a_w,
                [CkptWeightInfo(self._key("attn.wq_a.weight"), identity)],
                identity,
                config=cfg,
            ),
            AttnAtomicWeight(
                W.v4_attn_wq_b_w,
                [CkptWeightInfo(self._key("attn.wq_b.weight"), identity)],
                identity,
                config=cfg,
            ),
            AttnAtomicWeight(
                W.v4_attn_wkv_w,
                [CkptWeightInfo(self._key("attn.wkv.weight"), identity)],
                identity,
                config=cfg,
            ),
            AttnAtomicWeight(
                W.v4_attn_wo_a_w,
                [CkptWeightInfo(self._key("attn.wo_a.weight"), identity)],
                identity,
                config=cfg,
            ),
            AttnAtomicWeight(
                W.v4_attn_wo_b_w,
                [CkptWeightInfo(self._key("attn.wo_b.weight"), identity)],
                identity,
                config=cfg,
            ),
        ]

    def _build_compressor(
        self, layer_id: int, ratio: int, ckpt_prefix: str, inner: bool = False
    ) -> List[WeightModule]:
        # ``ckpt_prefix`` excludes the ``layers.{i}.`` template piece.
        # e.g. "attn.compressor" or "attn.indexer.compressor".
        wkv_name = W.v4_indexer_compressor_wkv if inner else W.v4_compressor_wkv
        wgate_name = W.v4_indexer_compressor_wgate if inner else W.v4_compressor_wgate
        norm_name = W.v4_indexer_compressor_norm if inner else W.v4_compressor_norm
        ape_name = W.v4_indexer_compressor_ape if inner else W.v4_compressor_ape
        return [
            # wkv / wgate are BF16 in V4 checkpoints and consumed by BF16
            # tensor-core GEMMs in the compressor. Keep them BF16 at load time
            # so module init does not do BF16 -> FP32 -> BF16 churn.
            AtomicWeight(
                wkv_name,
                [CkptWeightInfo(self._key(f"{ckpt_prefix}.wkv.weight"), identity)],
                identity,
                data_type=torch.bfloat16,
            ),
            AtomicWeight(
                wgate_name,
                [CkptWeightInfo(self._key(f"{ckpt_prefix}.wgate.weight"), identity)],
                identity,
                data_type=torch.bfloat16,
            ),
            AtomicWeight(
                norm_name,
                [CkptWeightInfo(self._key(f"{ckpt_prefix}.norm.weight"), identity)],
                identity,
            ),
            AtomicWeight(
                ape_name,
                [CkptWeightInfo(self._key(f"{ckpt_prefix}.ape"), identity)],
                identity,
                data_type=torch.float32,
            ),
        ]

    def _build_indexer(self, layer_id: int) -> List[WeightModule]:
        cfg = self._v4_attn_cfg()
        return [
            AttnAtomicWeight(
                W.v4_indexer_wq_b_w,
                [CkptWeightInfo(self._key("attn.indexer.wq_b.weight"), identity)],
                identity,
                config=cfg,
            ),
            AtomicWeight(
                W.v4_indexer_weights_proj_w,
                [
                    CkptWeightInfo(
                        self._key("attn.indexer.weights_proj.weight"), identity
                    )
                ],
                identity,
            ),
        ]

    def _build_hc_residual(self, layer_id: int) -> List[WeightModule]:
        out: List[WeightModule] = []
        for tag in ("attn", "ffn"):
            for sub, w_name in [
                ("base", getattr(W, f"v4_hc_{tag}_base")),
                ("fn", getattr(W, f"v4_hc_{tag}_fn")),
                ("scale", getattr(W, f"v4_hc_{tag}_scale")),
            ]:
                out.append(
                    AtomicWeight(
                        w_name,
                        [CkptWeightInfo(self._key(f"hc_{tag}_{sub}"), identity)],
                        identity,
                        data_type=torch.float32,
                    )
                )
        return out

    def _build_router(self, layer_id: int) -> List[WeightModule]:
        out: List[WeightModule] = [
            AtomicWeight(
                W.v4_router_w,
                [CkptWeightInfo(self._key("ffn.gate.weight"), identity)],
                identity,
            ),
        ]
        if layer_id < self._num_hash_layers:
            out.append(
                AtomicWeight(
                    W.v4_router_tid2eid,
                    [CkptWeightInfo(self._key("ffn.gate.tid2eid"), identity)],
                    identity,
                    data_type=torch.int32,
                )
            )
        else:
            out.append(
                AtomicWeight(
                    W.v4_router_bias,
                    [CkptWeightInfo(self._key("ffn.gate.bias"), identity)],
                    identity,
                    data_type=torch.float32,
                )
            )
        return out

    def _build_shared_expert(self, layer_id: int) -> List[WeightModule]:
        cfg = self._v4_attn_cfg()
        return [
            AttnAtomicWeight(
                W.v4_shared_w13_w,
                [
                    CkptWeightInfo(
                        self._key("ffn.shared_experts.w1.weight"),
                        identity,
                    ),
                    CkptWeightInfo(
                        self._key("ffn.shared_experts.w3.weight"),
                        identity,
                    ),
                ],
                concat_0,
                config=cfg,
            ),
            AttnAtomicWeight(
                W.v4_shared_w2_w,
                [CkptWeightInfo(self._key("ffn.shared_experts.w2.weight"), identity)],
                identity,
                config=cfg,
            ),
        ]

    def _build_routed_experts_fp4(self, layer_id: int) -> List[WeightModule]:
        moe_cfg = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._moe_align_size,
        )
        # V4 routed experts ship as I8-packed FP4 (e2m1) + UE8M0 group-32
        # scale, both consumed natively by DeepGEMM's fp8_fp4_gemm_nt.  The
        # framework's quant_config (Fp8BlockWiseQuantConfig from ckpt's
        # quantization_config.quant_method=fp8) doesn't describe this FP4
        # routed-expert scheme — there's no ModelOptFp4Config to trigger
        # V4PerGroupFp4Weight.support().  Instead we declare BOTH the .weight
        # (int8 packed FP4) AND the .scale (UE8M0) as separate MoeAtomicWeight
        # entries so the framework loader treats them as two ordinary stacked
        # MoE tensors per expert.  No postprocess transform is needed — the
        # ckpt layout is already DeepGEMM-consumable.  Note: data_type must
        # match the safetensors I8 dtype (torch.int8); torch.uint8 also has
        # 1 byte/element and the bytes are bit-identical, but downstream
        # consumers (QuantizedLinear factory + DeepGEMM kPackedFP4 path)
        # branch on dtype, and a uint8 cast will silently drop the FP4
        # routed-expert linears into a slower BF16 dequant fallback.
        out: List[WeightModule] = []
        for sub_w_name, sub_s_name, sub in [
            (W.v4_routed_w1_w, W.v4_routed_w1_s, "w1"),
            (W.v4_routed_w2_w, W.v4_routed_w2_s, "w2"),
            (W.v4_routed_w3_w, W.v4_routed_w3_s, "w3"),
        ]:
            out.append(
                MoeAtomicWeight(
                    sub_w_name,
                    [
                        CkptWeightInfo(
                            self._key(f"ffn.experts.{{expert_id}}.{sub}.weight"),
                            identity,
                        )
                    ],
                    stack_,
                    config=moe_cfg,
                    data_type=torch.int8,
                )
            )
            out.append(
                MoeAtomicWeight(
                    sub_s_name,
                    [
                        CkptWeightInfo(
                            self._key(f"ffn.experts.{{expert_id}}.{sub}.scale"),
                            identity,
                        )
                    ],
                    stack_,
                    config=moe_cfg,
                    data_type=torch.float8_e8m0fnu,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Top-level entry points
    # ------------------------------------------------------------------

    def _get_hf_layer_weight_info(self, layer_id: int) -> List[WeightModule]:  # type: ignore[override]
        weights: List[WeightModule] = []
        # 1. Attention norms + sink (always present)
        weights += self._build_attn_norms(layer_id)
        # 2. Dense MQA attention FP8 weights (always present)
        weights += self._build_attn_dense_fp8(layer_id)

        # 3. Outer compressor (CSA + HCA, ratio in {4, 128})
        ratio = self._compress_ratio(layer_id)
        if ratio in (4, 128):
            weights += self._build_compressor(
                layer_id, ratio, ckpt_prefix="attn.compressor"
            )

        # 4. Indexer (CSA only — ratio == 4)
        if ratio == 4:
            weights += self._build_indexer(layer_id)
            # indexer's inner compressor — separate W keys to avoid colliding
            # with the outer compressor.
            weights += self._build_compressor(
                layer_id,
                ratio=4,
                ckpt_prefix="attn.indexer.compressor",
                inner=True,
            )

        # 5. mHC residual (always)
        weights += self._build_hc_residual(layer_id)

        # 6. FFN norm (always)
        weights.append(
            AtomicWeight(
                W.v4_ffn_norm,
                [CkptWeightInfo(self._key("ffn_norm.weight"), identity)],
                identity,
            )
        )

        # 7. Router (hash on first num_hash_layers, noaux_tc otherwise)
        weights += self._build_router(layer_id)

        # 8. Shared expert (FP8)
        weights += self._build_shared_expert(layer_id)

        # 9. Routed experts (FP4 — per-expert via MoeAtomicWeight)
        weights += self._build_routed_experts_fp4(layer_id)

        return weights

    def _get_weight_info(self) -> ModelWeightInfo:
        layer_weights: List[List[WeightModule]] = [
            self._get_hf_layer_weight_info(layer_id)
            for layer_id in range(self._num_layers)
        ]
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("embed.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("head.weight", identity)],
                identity,
                # Respect ``enable_fp32_lm_head``: prior code hardcoded
                # ``data_type=torch.float32`` here, which ignored
                # ``ENABLE_FP32_LM_HEAD=0`` / ``config.enable_fp32_lm_head
                # = False`` and pushed lm_head into FP32 storage. Downstream
                # the C++ ``PyWrappedModel::forwardPostLayers`` mm upcast
                # ``last_hidden`` to match, dispatching to a f32 simt sgemm
                # (``cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32`` — CUDA
                # cores, ~2.3 ms / decode step at bs=128 / vocab=129280 on
                # B300) instead of the bf16 tensor-core path (~0.1 ms).
                # ``None`` falls through to ``load_config.compute_dtype`` in
                # AtomicWeight._load_raw_tensor (i.e. bf16 for V4), keeping
                # the ckpt dtype.
                data_type=torch.float32 if self.enable_fp32_lm_head else None,
            ),
            AtomicWeight(
                W.v4_hc_head_base,
                [CkptWeightInfo("hc_head_base", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.v4_hc_head_fn,
                [CkptWeightInfo("hc_head_fn", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.v4_hc_head_scale,
                [CkptWeightInfo("hc_head_scale", identity)],
                identity,
                data_type=torch.float32,
            ),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV4(DeepSeekV2):
    """DeepSeek-V4 entry point.

    M0: parse HF config and register. Engine instantiation will fail at
    `_create_python_model` until M2 lands the HCA-only forward path.
    """

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.attn_config.head_num = 0
        config.attn_config.kv_head_num = 0
        config.attn_config.size_per_head = 0
        config.num_layers = 0
        config.inter_size = 0
        config.vocab_size = 129280
        config.max_seq_len = 8192
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"
        DeepSeekV4._from_hf(config, ckpt_path)
        return config


    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        _refresh_dsv4_kv_cache_specs(model_config)
        # DSV4 uses 7 independent BlockPools (one per cache group). Declare this
        # explicitly so CacheConfigCreator routes to HybridPoolConfigCreator
        # without inspecting internal spec tags or layer_compress_ratios.
        model_config.hybrid_attention_config.enable_hybrid_attention = True
        model_config.hybrid_attention_config.enable_independent_kv_cache_pools = True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model

        self.py_model = DeepSeekV4Model(
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
    def _from_hf(config: ModelConfig, ckpt_path: str):  # noqa: C901  (acceptably long)
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        # ---- basic geometry ----
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get("num_key_value_heads", 1)

        # V4 is MQA with monolithic head_dim — NOT split into nope+rope.
        # Only the LAST `qk_rope_head_dim` dims of each head get RoPE.
        head_dim = config_json["head_dim"]
        rope_dim = config_json["qk_rope_head_dim"]
        config.attn_config.size_per_head = head_dim
        config.attn_config.nope_head_dim = head_dim - rope_dim  # for code that splits
        config.attn_config.rope_head_dim = rope_dim
        config.attn_config.v_head_dim = head_dim
        config.attn_config.use_mla = False  # V4 uses MQA, not MLA

        q_lora_rank = config_json.get("q_lora_rank")
        config.attn_config.q_lora_rank = int(q_lora_rank) if q_lora_rank else 0
        config.attn_config.kv_lora_rank = 0  # V4 has no KV LoRA

        # ---- RoPE: main base + separate compressed base ----
        config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
        config.attn_config.rope_config.dim = rope_dim
        config.attn_config.rope_config.offset = (
            head_dim - rope_dim
        )  # partial RoPE on tail
        config.attn_config.rope_config.style = 0  # interleaved by default per V4
        config.attn_config.compress_rope_theta = float(
            config_json.get("compress_rope_theta", 160000)
        )

        rope_scaling = config_json.get("rope_scaling")
        if rope_scaling is not None:
            scaling_factor = rope_scaling["factor"]
            config.attn_config.rope_config.scale = scaling_factor
            config.attn_config.rope_config.factor1 = float(
                rope_scaling.get("beta_slow", 1)
            )
            config.attn_config.rope_config.factor2 = float(
                rope_scaling.get("beta_fast", 32)
            )
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
            # V4 config.json doesn't carry mscale fields like V3; fall back to factor-of-2
            mscale = rope_scaling.get("mscale", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 1.0)
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            config.attn_config.rope_config.mscale = yarn_get_mscale(
                scaling_factor, mscale
            ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale

        # ---- per-layer attention schedule ----
        compress_ratios = config_json["compress_ratios"]
        # Drop the trailing MTP entry (always 0) so the schedule matches main layers.
        # Length should be num_layers (MTP layers count separately).
        if len(compress_ratios) == config.num_layers + config_json.get(
            "num_nextn_predict_layers", 0
        ):
            main_ratios = compress_ratios[: config.num_layers]
        else:
            main_ratios = compress_ratios
        config.attn_config.layer_compress_ratios = list(main_ratios)

        # ---- output projection / SWA / sink ----
        config.attn_config.o_groups = int(config_json["o_groups"])
        config.attn_config.o_lora_rank = int(config_json["o_lora_rank"])
        config.attn_config.sliding_window = int(config_json.get("sliding_window", 0))

        # ---- sparse indexer (CSA) ----
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_dim = int(config_json["index_head_dim"])
        config.attn_config.indexer_head_num = int(config_json["index_n_heads"])
        config.attn_config.indexer_topk = int(config_json["index_topk"])
        config.kv_cache_specs = _build_dsv4_kv_cache_specs(config)
        config.kv_cache_spec_descs = _build_dsv4_kv_cache_spec_descs(
            config.kv_cache_specs
        )

        # ---- MoE ----
        scoring_func = config_json.get("scoring_func", "softmax")
        if scoring_func == "softmax":
            config.scoring_func = SCORING_FUNC_SOFTMAX
        elif scoring_func == "sigmoid":
            config.scoring_func = SCORING_FUNC_SIGMOID
        elif scoring_func == "sqrtsoftplus":
            config.scoring_func = SCORING_FUNC_SQRT_SOFTPLUS
        else:
            raise ValueError(f"Unknown V4 scoring_func: {scoring_func}")

        config.routed_scaling_factor = config_json["routed_scaling_factor"]
        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["n_routed_experts"]
        moe_intermediate_size = config_json["moe_intermediate_size"]
        # V4 drops n_group/topk_group constraints; explicitly zero them so any
        # downstream code falls back to plain top-k.
        config.moe_n_group = 0
        config.moe_topk_group = 0
        config.has_moe_norm = config_json.get("norm_topk_prob", False)
        config.moe_style = 2  # shared + expert
        n_shared_experts = config_json["n_shared_experts"]
        config.inter_size = n_shared_experts * moe_intermediate_size
        # Every layer is MoE in V4 (compress_ratios doesn't include dense replacement)
        config.moe_layer_index = list(range(config.num_layers))

        # ---- V4-only model-level fields ----
        config.hc_mult = int(config_json.get("hc_mult", 1))
        config.hc_sinkhorn_iters = int(config_json.get("hc_sinkhorn_iters", 0))
        config.hc_eps = float(config_json.get("hc_eps", 1e-6))
        config.swiglu_limit = float(config_json.get("swiglu_limit", 0.0))
        config.num_hash_layers = int(config_json.get("num_hash_layers", 0))

        config.config_dtype = config_json.get("torch_dtype", None)

        # Keep lm_head in its checkpoint dtype (bf16).  The framework default
        # ``enable_fp32_lm_head=True`` calls _fix_fp32_lm_head which overrides
        # the AtomicWeight data_type to fp32 — useful for older models that
        # need fp32 logits accumulation, but doubles RAM (1 GiB → 2 GiB).
        # V4 sampling runs on bf16 logits like V3.
        config.enable_fp32_lm_head = False

        logging.info(
            "DeepSeek-V4 config loaded: layers=%d hidden=%d head_num=%d head_dim=%d "
            "rope_dim=%d q_lora=%d compress_ratios=%s o_groups=%d sliding=%d "
            "n_experts=%d num_hash_layers=%d hc_mult=%d swiglu_limit=%g "
            "scoring_func=%d compress_rope_theta=%g",
            config.num_layers,
            config.hidden_size,
            config.attn_config.head_num,
            config.attn_config.size_per_head,
            config.attn_config.rope_head_dim,
            config.attn_config.q_lora_rank,
            list(config.attn_config.layer_compress_ratios)[:8] + ["..."],
            config.attn_config.o_groups,
            config.attn_config.sliding_window,
            config.expert_num,
            config.num_hash_layers,
            config.hc_mult,
            config.swiglu_limit,
            config.scoring_func,
            config.attn_config.compress_rope_theta,
        )

    @staticmethod
    def get_weight_cls():
        return DeepSeekV4Weight


class DeepSeekV4MtpWeight(DeepSeekV4Weight, DeepSeekV3MtpWeight):
    """MTP draft weight loader for the V4 `mtp.{i}.*` checkpoint layout."""

    _LAYER_PREFIX: str = "mtp.{i}"

    def _process_meta(self, meta_dict, weight_keys):  # type: ignore[override]
        super()._process_meta(meta_dict, weight_keys)
        self._num_hash_layers = 0

    def _get_weight_info(self) -> ModelWeightInfo:
        # Per-layer entries are inherited from DeepSeekV4Weight as-is — the
        # MTP draft is a single regular V4 Block.  All MTP-only tensors
        # (norm, hc_head_*, enorm, hnorm, e_proj, h_proj) live under the
        # ``mtp.0.*`` ckpt prefix and are loaded as GLOBALS, mapped onto
        # the same W keys the main-model V4Transformer reads:
        #   * ``mtp.0.norm.weight``     → W.final_ln_gamma
        #   * ``mtp.0.hc_head_*``       → W.v4_hc_head_*
        # so the main V4Transformer ctor can be reused with no MTP-specific
        # branches.  ``enorm/hnorm/e_proj/h_proj`` are MTP-only fusion
        # weights and stay under their dedicated W.v4_mtp_* tags; the
        # ``DeepSeekV4MtpModel`` subclass picks them up from
        # ``mw.global_weights`` and assembles the mHC fuse stage.
        layer_weights: List[List[WeightModule]] = [
            self._get_hf_layer_weight_info(layer_id)
            for layer_id in range(self._num_layers)
        ]
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("embed.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("mtp.0.norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("head.weight", identity)],
                identity,
                # MTP loads alongside the main model in the same process;
                # keep its lm_head copy in bf16 to halve the per-process
                # memory footprint (the FP32 cast happens lazily on the
                # rare standalone test path).  Production routes the
                # framework lm_head through the main model's tensor.
                data_type=torch.bfloat16,
            ),
            AtomicWeight(
                W.v4_hc_head_base,
                [CkptWeightInfo("mtp.0.hc_head_base", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.v4_hc_head_fn,
                [CkptWeightInfo("mtp.0.hc_head_fn", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.v4_hc_head_scale,
                [CkptWeightInfo("mtp.0.hc_head_scale", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.v4_mtp_enorm,
                [CkptWeightInfo("mtp.0.enorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_mtp_hnorm,
                [CkptWeightInfo("mtp.0.hnorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_mtp_e_proj_w,
                [CkptWeightInfo("mtp.0.e_proj.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_mtp_e_proj_s,
                [CkptWeightInfo("mtp.0.e_proj.scale", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_mtp_h_proj_w,
                [CkptWeightInfo("mtp.0.h_proj.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.v4_mtp_h_proj_s,
                [CkptWeightInfo("mtp.0.h_proj.scale", identity)],
                identity,
            ),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV4Mtp(DeepSeekV4, DeepSeekV3Mtp):

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.num_layers = 1
        config.attn_config.layer_compress_ratios = [0]
        config.kv_cache_specs = _build_dsv4_kv_cache_specs(config)
        config.kv_cache_spec_descs = _build_dsv4_kv_cache_spec_descs(
            config.kv_cache_specs
        )
        config.moe_layer_index = list(range(config.num_layers))
        config.reverse_e_h_norm = True
        config.is_mtp = True
        # MTP module shares the same 7-pool topology as the main model.
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.enable_independent_kv_cache_pools = True
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import (
            DeepSeekV4MtpModel,
        )

        self.py_model = DeepSeekV4MtpModel(
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
    def get_weight_cls():
        return DeepSeekV4MtpWeight


register_model("deepseek_v4", DeepSeekV4, ["DeepseekV4ForCausalLM"])
register_model("deepseek_v4_mtp", DeepSeekV4Mtp, ["DeepseekV4ForCausalLMNextN"])
