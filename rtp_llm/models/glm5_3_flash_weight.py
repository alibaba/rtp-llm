"""Checkpoint manifest for the GLM-5.3-Flash text decoder."""

from __future__ import annotations

import functools
from typing import Iterable, List

import torch

from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    LinearAttnConfig,
)
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    WeightModule,
)
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    sp_id,
    transpose,
    zeros,
)


def _merge_conv1d(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) != 3:
        raise ValueError(
            f"GLM-5.3 fused convolution expects 3 tensors, got {len(tensors)}"
        )
    return torch.cat(tensors, dim=0).contiguous()


def _merge_qkv_fa_beta(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) != 5:
        raise ValueError(
            f"GLM-5.3 KDA fused input expects 5 tensors, got {len(tensors)}"
        )
    q, k, v, f_a, beta = tensors
    return torch.cat((q.T, k.T, v.T, f_a.T, beta.T), dim=1).contiguous()


class Glm53FlashWeight(DeepSeekV2Weight):
    """Reuse the mature DeepSeek MLA/MoE transforms with a nested prefix.

    GLM-5.3 KDA and mHC weights are described explicitly. MLA and ordinary
    dense/MoE tensors share the DeepSeek checkpoint layout after inserting
    ``language_model`` below the outer ``model`` prefix.
    """

    MODEL_PREFIX = "model.language_model."
    LAYER_PREFIX = MODEL_PREFIX + "layers.{i}."
    # The checkpoint declares FP8 globally, while these projections are stored
    # as BF16 and have no ``weight_scale_inv`` companion.
    BF16_WEIGHT_NAMES = {
        W.mla_kv_b_w,
        W.mla_kc,
        W.mla_vc,
        W.mla_indexer_qb_w,
        W.mla_indexer_k_w,
        W.mla_indexer_k_norm_w,
        W.mla_indexer_k_norm_b,
        W.mla_indexer_weights_proj_w,
        W.mla_indexer_kpool_gate_w,
        W.mla_indexer_kpool_ape,
        W.linear_attn_out_w,
    }

    def _process_meta(self, meta_dict, weight_keys):
        del meta_dict
        self.q_use_lora = True
        self.has_e_score_correction_bias = any(
            key.endswith("mlp.gate.e_score_correction_bias") for key in weight_keys
        )

    @classmethod
    def _prefix_weight_modules(cls, modules: Iterable[WeightModule]) -> None:
        for module in modules:
            if isinstance(module, CompositeWeight):
                cls._prefix_weight_modules(module.sub_weights.values())
            weights = getattr(module, "weights", None)
            if weights is None:
                continue
            for ckpt in weights:
                if ckpt.name.startswith("model.") and not ckpt.name.startswith(
                    cls.MODEL_PREFIX
                ):
                    ckpt.name = cls.MODEL_PREFIX + ckpt.name[len("model.") :]

    @classmethod
    def _mark_checkpoint_bf16(cls, modules: Iterable[WeightModule]) -> None:
        for module in modules:
            if isinstance(module, CompositeWeight):
                cls._mark_checkpoint_bf16(module.sub_weights.values())
            if module.name in cls.BF16_WEIGHT_NAMES:
                module.skip_quantization = True
                if hasattr(module, "data_type"):
                    module.data_type = torch.bfloat16

    @classmethod
    def _layer_ckpt(cls, suffix: str) -> str:
        return cls.LAYER_PREFIX + suffix

    def _mhc_weights(self) -> List[WeightModule]:
        result: List[WeightModule] = []
        for name, suffix, dtype in (
            (W.v4_hc_attn_fn, "hc_attn_fn", torch.float32),
            (W.v4_hc_attn_base, "hc_attn_base", torch.float32),
            (W.v4_hc_attn_scale, "hc_attn_scale", torch.float32),
            (W.v4_hc_ffn_fn, "hc_ffn_fn", torch.float32),
            (W.v4_hc_ffn_base, "hc_ffn_base", torch.float32),
            (W.v4_hc_ffn_scale, "hc_ffn_scale", torch.float32),
        ):
            result.append(
                AtomicWeight(
                    name,
                    [CkptWeightInfo(self._layer_ckpt(suffix), identity)],
                    identity,
                    data_type=dtype,
                )
            )
        return result

    def _kpool_weights(self) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.mla_indexer_kpool_gate_w,
                [
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.indexer.index_kpool_compress_gate"),
                        identity,
                    )
                ],
                transpose,
            ),
            AtomicWeight(
                W.mla_indexer_kpool_ape,
                [
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.indexer.index_kpool_compress_ape"),
                        identity,
                    )
                ],
                identity,
            ),
        ]

    def _kda_weights(self) -> List[WeightModule]:
        cfg = LinearAttnConfig(self.model_config.linear_attention_config)

        def weight(name, suffix, process_fun, *, data_type=None):
            return LinearAttnAtomicWeight(
                name,
                [CkptWeightInfo(self._layer_ckpt(suffix), identity)],
                process_fun,
                cfg,
                data_type=data_type,
            )

        return [
            LinearAttnAtomicWeight(
                W.linear_attn_qkv_fa_beta_w,
                [
                    CkptWeightInfo(self._layer_ckpt("self_attn.q_proj.weight")),
                    CkptWeightInfo(self._layer_ckpt("self_attn.k_proj.weight")),
                    CkptWeightInfo(self._layer_ckpt("self_attn.v_proj.weight")),
                    CkptWeightInfo(self._layer_ckpt("self_attn.f_a_proj.weight")),
                    CkptWeightInfo(self._layer_ckpt("self_attn.b_proj.weight")),
                ],
                _merge_qkv_fa_beta,
                cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(self._layer_ckpt("self_attn.q_conv1d.weight")),
                    CkptWeightInfo(self._layer_ckpt("self_attn.k_conv1d.weight")),
                    CkptWeightInfo(self._layer_ckpt("self_attn.v_conv1d.weight")),
                ],
                _merge_conv1d,
                cfg,
            ),
            weight(
                W.linear_attn_alog, "self_attn.A_log", identity, data_type=torch.float32
            ),
            weight(
                W.linear_attn_dt_b_kda,
                "self_attn.dt_bias",
                identity,
                data_type=torch.float32,
            ),
            weight(W.linear_attn_f_b_w, "self_attn.f_b_proj.weight", transpose),
            weight(W.linear_attn_g_a_w, "self_attn.g_a_proj.weight", transpose),
            weight(W.linear_attn_g_b_w, "self_attn.g_b_proj.weight", transpose),
            weight(W.linear_attn_norm_w, "self_attn.o_norm.weight", identity),
            weight(W.linear_attn_out_w, "self_attn.o_proj.weight", transpose),
        ]

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_type = self.model_config.hybrid_attention_config.hybrid_attention_types[
            layer_id
        ]
        from rtp_llm.ops import HybridAttentionType

        if layer_type != HybridAttentionType.LINEAR:
            weights = super()._get_hf_layer_weight_info(layer_id)
            self._prefix_weight_modules(weights)
            self._mark_checkpoint_bf16(weights)
            weights.extend(self._mhc_weights())
            weights.extend(self._kpool_weights())
            return weights

        weights: List[WeightModule] = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(self._layer_ckpt("input_layernorm.weight"))],
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(self._layer_ckpt("post_attention_layernorm.weight"))],
            ),
        ]
        kda_weights = self._kda_weights()
        self._mark_checkpoint_bf16(kda_weights)
        weights.extend(kda_weights)
        ffn_weights = super()._get_hf_ffn_layer_weight_info(layer_id)
        self._prefix_weight_modules(ffn_weights)
        weights.extend(ffn_weights)
        weights.extend(self._mhc_weights())
        return weights

    def _create_rope_w(self):
        # Both MLA and indexer have zero RoPE dimensions. Keep the global key
        # because shared attention construction expects it.
        def empty_rope(_):
            return torch.empty((self.model_config.max_seq_len, 0), dtype=torch.float32)

        return AtomicWeight(
            W.rope_cos_sin_cache,
            [],
            process_fun=empty_rope,
            data_type=torch.float32,
        )

    def _get_weight_info(self):
        global_weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.MODEL_PREFIX + "embed_tokens.weight")],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.MODEL_PREFIX + "norm.weight")],
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight")]),
            self._create_rope_w(),
        ]
        layer_weights = [
            self._get_hf_layer_weight_info(layer) for layer in range(self._num_layers)
        ]
        return ModelWeightInfo(weights=global_weights, layer_weights=layer_weights)

    @staticmethod
    def checkpoint_names(weight_info: ModelWeightInfo) -> set[str]:
        """Expand all target-layer checkpoint names for manifest unit tests."""

        result: set[str] = set()

        def visit(modules: Iterable[WeightModule], layer_id=None) -> None:
            for module in modules:
                if isinstance(module, CompositeWeight):
                    visit(module.sub_weights.values(), layer_id)
                for ckpt in getattr(module, "weights", []):
                    name = ckpt.tensor_name(layer_id)
                    if "{expert_id}" not in name:
                        result.add(name)

        visit(weight_info.weights)
        for layer_id, modules in enumerate(weight_info.layer_weights):
            visit(modules, layer_id)
        return result


__all__ = ["Glm53FlashWeight"]
