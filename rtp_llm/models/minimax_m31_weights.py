"""MiniMax-M3.1 text checkpoint mapping for the current RTP-LLM weight loader."""

from __future__ import annotations

import functools
from typing import Any, List

import torch

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, CustomAtomicWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
)

IDX_Q_WEIGHT = "minimax_m31.index_q_weight"
IDX_K_WEIGHT = "minimax_m31.index_k_weight"
IDX_Q_SCALE = "minimax_m31.index_q_scale"
IDX_K_SCALE = "minimax_m31.index_k_scale"
IDX_Q_NORM = "minimax_m31.index_q_norm"
IDX_K_NORM = "minimax_m31.index_k_norm"
RAW_Q_NORM = "minimax_m31.raw_q_norm"
RAW_K_NORM = "minimax_m31.raw_k_norm"


def add_unit_offset(tensors: List[torch.Tensor]) -> torch.Tensor:
    return (tensors[0] + 1.0).contiguous()


def stack_w13_global_scales(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) % 2:
        raise ValueError("NVFP4 w13 scales must be up/gate pairs")
    half = len(tensors) // 2
    up = torch.stack([t.reshape(()) for t in tensors[:half]])
    gate = torch.stack([t.reshape(()) for t in tensors[half:]])
    return torch.stack([up, gate], dim=1).contiguous()


class M31PackedMoeWeight(MoeAtomicWeight):
    """Keep checkpoint NVFP4 bytes/scales intact for the model-owned executor."""

    is_prepacked_nvfp4 = True

    def _postprocess(self, tensor, device, load_config):
        raw = tensor.get(self.name) if isinstance(tensor, dict) else tensor
        return {self.name: raw}


class MiniMaxM31Weight(ModelDeployWeightInfo):
    """Map the released mixed MXFP8/NVFP4 checkpoint without M3 variants."""

    prefix = "language_model."

    def _process_meta(self, meta_dict: Any, weight_keys: List[str]) -> None:
        keys = set(weight_keys)
        for layer in range(self._num_layers):
            root = f"{self.prefix}model.layers.{layer}."
            required = (
                "self_attn.index_q_proj.weight",
                "self_attn.index_q_proj.weight_scale_inv",
                "self_attn.index_k_proj.weight",
                "self_attn.index_k_proj.weight_scale_inv",
            )
            for suffix in required:
                if root + suffix not in keys:
                    raise ValueError(
                        f"MiniMax-M3.1 missing checkpoint tensor {root + suffix}"
                    )
            if layer in self.moe_layer_index_:
                if root + "block_sparse_moe.experts.0.w1.weight_packed" not in keys:
                    raise ValueError(f"MiniMax-M3.1 missing NVFP4 MoE layer {layer}")

    def _get_hf_layer_weight_info(self, layer_id: int):
        root = self.prefix + "model.layers.{i}."
        attn_root = root + "self_attn."
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        result = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(root + "input_layernorm.weight", identity)],
                add_unit_offset,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(attn_root + name + "_proj.weight", identity)
                    for name in ("q", "k", "v")
                ],
                merge_qkv_hf,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(attn_root + "o_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(root + "post_attention_layernorm.weight", identity)],
                add_unit_offset,
            ),
        ]
        for name, key in (
            ("q_norm", W.q_ln_gamma),
            ("k_norm", W.k_ln_gamma),
        ):
            checkpoint_name = attn_root + name + ".weight"
            result.append(
                AttnAtomicWeight(
                    key,
                    [CkptWeightInfo(checkpoint_name, identity)],
                    add_unit_offset,
                    config=attn_config,
                )
            )
        for name, raw_key in (("q_norm", RAW_Q_NORM), ("k_norm", RAW_K_NORM)):
            result.append(
                CustomAtomicWeight(
                    raw_key,
                    [CkptWeightInfo(attn_root + name + ".weight", identity)],
                    identity,
                    data_type=torch.bfloat16,
                    disable_quantization=True,
                )
            )
        for side, weight_key, scale_key, norm_key in (
            ("q", IDX_Q_WEIGHT, IDX_Q_SCALE, IDX_Q_NORM),
            ("k", IDX_K_WEIGHT, IDX_K_SCALE, IDX_K_NORM),
        ):
            base = attn_root + "index_" + side
            result.extend(
                [
                    CustomAtomicWeight(
                        weight_key,
                        [CkptWeightInfo(base + "_proj.weight", identity)],
                        identity,
                        data_type=torch.float8_e4m3fn,
                        disable_quantization=True,
                    ),
                    CustomAtomicWeight(
                        scale_key,
                        [CkptWeightInfo(base + "_proj.weight_scale_inv", identity)],
                        identity,
                        data_type=torch.uint8,
                        disable_quantization=True,
                    ),
                    CustomAtomicWeight(
                        norm_key,
                        [CkptWeightInfo(base + "_norm.weight", identity)],
                        identity,
                        data_type=torch.bfloat16,
                        disable_quantization=True,
                    ),
                ]
            )
        result.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return result

    def _dense_mlp(self, root: str):
        config = FfnConfig(
            align_size=self._align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )
        return FfnWeight(
            sub_weights=[
                FfnAtomicWeight(
                    key,
                    [CkptWeightInfo(root + suffix + ".weight", identity)],
                    functools.partial(
                        transpose_pad, align_size=self._align_size, dim=dim
                    ),
                    config=config,
                )
                for key, suffix, dim in (
                    (W.ffn_w1, "gate_proj", 0),
                    (W.ffn_w2, "down_proj", 1),
                    (W.ffn_w3, "up_proj", 0),
                )
            ],
            config=config,
        )

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        root = self.prefix + "model.layers.{i}."
        if layer_id not in self.moe_layer_index_:
            return [self._dense_mlp(root + "mlp.")]
        moe_root = root + "block_sparse_moe."
        config = MoeConfig(align_size=self._align_size, expert_num=self.expert_num_)
        experts = moe_root + "experts.{expert_id}."
        result = [
            self._dense_mlp(moe_root + "shared_experts."),
            MoeAtomicWeight(
                W.moe_gate,
                [CkptWeightInfo(moe_root + "gate.weight", identity)],
                transpose,
                data_type=torch.float32,
                config=config,
            ),
        ]
        for key, sources, process, dtype in (
            (
                W.moe_w1,
                ("w3.weight_packed", "w1.weight_packed"),
                stack_moe_w1,
                torch.int8,
            ),
            (W.moe_w2, ("w2.weight_packed",), stack_, torch.int8),
            (
                W.moe_s1,
                ("w3.weight_scale", "w1.weight_scale"),
                stack_moe_w1,
                torch.float8_e4m3fn,
            ),
            (W.moe_s2, ("w2.weight_scale",), stack_, torch.float8_e4m3fn),
            (
                W.moe_w1_s2,
                ("w3.weight_global_scale", "w1.weight_global_scale"),
                stack_w13_global_scales,
                torch.float32,
            ),
            (
                W.moe_w2_s2,
                ("w2.weight_global_scale",),
                stack_,
                torch.float32,
            ),
        ):
            result.append(
                M31PackedMoeWeight(
                    key,
                    [CkptWeightInfo(experts + suffix, identity) for suffix in sources],
                    process,
                    data_type=dtype,
                    config=config,
                    disable_quantization=True,
                )
            )
        result.append(
            AtomicWeight(
                W.e_score_correction_b,
                [CkptWeightInfo(moe_root + "e_score_correction_bias", identity)],
                identity,
                data_type=torch.float32,
                disable_quantization=True,
            )
        )
        return result

    def _get_weight_info(self):
        globals = [
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
        layers = [
            self._get_hf_layer_weight_info(layer) for layer in range(self._num_layers)
        ]
        return ModelWeightInfo(weights=globals, layer_weights=layers)
