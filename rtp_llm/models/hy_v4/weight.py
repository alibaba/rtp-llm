"""HY V4 backbone checkpoint descriptors and layout conversions."""

from __future__ import annotations

import functools
from typing import Any, List

import torch

from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight
from rtp_llm.ops import MlaOpsType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    transpose,
    transpose_pad,
)


def _transpose_stacked_gate_up(ts: List[torch.Tensor]) -> torch.Tensor:
    """Convert per-expert checkpoint ``[gate, up]`` to RTP ``[up, gate]``."""
    # ``stack_`` has an uint8-view fallback for FP8 tensors on runtimes where
    # torch.stack does not implement FP8 directly.
    stacked = stack_(ts)
    if stacked.dim() != 3 or stacked.size(1) % 2:
        raise ValueError(
            "HY V4 fused gate_up_proj must split evenly on dimension 1, "
            f"got {tuple(stacked.shape)}"
        )
    half = stacked.size(1) // 2
    return torch.cat((stacked[:, half:, :], stacked[:, :half, :]), dim=1)


def _move_indexer_rope_to_front(
    ts: List[torch.Tensor],
    *,
    output_dim: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Adapt HY V4 indexer output channels from ``[nope, rope]`` to RTP's
    existing ``[rope, nope]`` contract.

    Checkpoint kernels and MXFP8 scales both have one row per output channel,
    so the same merge function keeps them aligned. Coarser per-block/per-tensor
    scales do not distinguish the two half-heads and remain unchanged.
    """
    tensor = identity(ts)
    if tensor.dim() == 0 or tensor.size(0) != output_dim:
        return tensor
    if head_dim != 2 * rope_dim or output_dim % head_dim:
        raise ValueError(
            "invalid HY V4 indexer layout: "
            f"output_dim={output_dim}, head_dim={head_dim}, rope_dim={rope_dim}"
        )
    grouped = tensor.reshape(output_dim // head_dim, head_dim, *tensor.shape[1:])
    grouped = torch.cat(
        (grouped[:, head_dim - rope_dim :], grouped[:, : head_dim - rope_dim]),
        dim=1,
    )
    return grouped.reshape(tensor.shape).contiguous()


class Hy4Weight(DeepSeekV2Weight):
    """Checkpoint mapping for HY V4 backbone weights."""

    has_fused_experts = False
    has_direct_mxfp4_experts = False
    fused_gate_up_suffix = ""
    fused_down_suffix = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Metadata may be processed once for pretrain shards and again for a
        # finetune overlay. Keep the detection state per loader instance and
        # accumulate it across both passes.
        self.q_use_lora = False
        self.has_e_score_correction_bias = False
        self.has_fused_experts = False
        self.has_direct_mxfp4_experts = False
        self.fused_gate_up_suffix = ""
        self.fused_down_suffix = ""
        super().__init__(*args, **kwargs)

    def _process_meta(self, meta_dict: Any, weight_keys: List[str]):
        super()._process_meta(meta_dict, weight_keys)
        self.has_fused_experts = self.has_fused_experts or any(
            ".mlp.experts.gate_up_proj" in key for key in weight_keys
        )
        self.has_direct_mxfp4_experts = self.has_direct_mxfp4_experts or any(
            key.endswith(".mlp.experts.w13_weight") for key in weight_keys
        )
        self.has_fused_experts = self.has_fused_experts or self.has_direct_mxfp4_experts
        if any(key.endswith(".mlp.experts.gate_up_proj.weight") for key in weight_keys):
            self.fused_gate_up_suffix = ".weight"
        if any(key.endswith(".mlp.experts.down_proj.weight") for key in weight_keys):
            self.fused_down_suffix = ".weight"

    def _mla_config(self) -> MlaConfig:
        return MlaConfig(
            head_num=self._head_num,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            ope_head_dim=self.nope_head_dim,
            v_head_dim=self.v_head_dim,
            use_mla=self.model_config.attn_config.use_mla
            and self.model_config.mla_ops_type != MlaOpsType.MHA,
            q_use_lora=self.q_use_lora,
        )

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_weights = super()._get_hf_layer_weight_info(layer_id)
        index_head_dim = self.model_config.attn_config.indexer_head_dim
        rope_dim = self.model_config.attn_config.rope_head_dim
        indexer_output_dims = {
            W.mla_indexer_qb_w: (
                self.model_config.attn_config.indexer_head_num * index_head_dim
            ),
            W.mla_indexer_k_w: index_head_dim,
            W.mla_indexer_k_norm_w: index_head_dim,
            W.mla_indexer_k_norm_b: index_head_dim,
        }
        # The shared Indexer implementation predates HY V4 and expects the
        # RoPE channels first. Adapt only HY V4 checkpoint descriptors here;
        # quant wrappers propagate CkptWeightInfo.merge_fun to both kernel and
        # scale descriptors, so MXFP8 weight/scale rows receive the same swap.
        for weight in layer_weights:
            for component in weight.get_components():
                output_dim = indexer_output_dims.get(component.name)
                if output_dim is None:
                    continue
                merge_fun = functools.partial(
                    _move_indexer_rope_to_front,
                    output_dim=output_dim,
                    head_dim=index_head_dim,
                    rope_dim=rope_dim,
                )
                for ckpt_weight in component.weights:
                    ckpt_weight.merge_fun = merge_fun

        attn_config = self._mla_config()
        layer_weights.extend(
            [
                MlaAttnAtomicWeight(
                    W.attn_gate_w,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.linear_gate.weight", identity
                        )
                    ],
                    transpose,
                    config=attn_config,
                ),
                AtomicWeight(
                    W.hy4_attn_sink,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.learnable_sink_param",
                            identity,
                        )
                    ],
                    identity,
                    data_type=torch.float32,
                ),
            ]
        )
        if self.model_config.enable_ihc:
            for kind in ("attn", "mlp"):
                prefix = f"model.layers.{{i}}.hc_{kind}_layer.hc_pre"
                layer_weights.extend(
                    [
                        AtomicWeight(
                            getattr(W, f"hy4_ihc_{kind}_fn"),
                            [CkptWeightInfo(prefix + ".hc_fn", identity)],
                            identity,
                            data_type=torch.float32,
                        ),
                        AtomicWeight(
                            getattr(W, f"hy4_ihc_{kind}_scale"),
                            [CkptWeightInfo(prefix + ".hc_scale", identity)],
                            identity,
                            data_type=torch.float32,
                        ),
                        AtomicWeight(
                            getattr(W, f"hy4_ihc_{kind}_base"),
                            [CkptWeightInfo(prefix + ".hc_base", identity)],
                            identity,
                            data_type=torch.float32,
                        ),
                    ]
                )
        # Quantization exclusion entries in ModelOpt are concrete per-layer
        # module names. Preserve the layer id on each descriptor so the MXFP8
        # loader can leave excluded BF16 modules (for example linear_gate)
        # untouched without disabling quantization for every layer.
        for weight in layer_weights:
            for component in weight.get_components():
                component.layer_id = layer_id
        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        if layer_id not in self.moe_layer_index_ or not self.has_fused_experts:
            layer_weights = super()._get_hf_ffn_layer_weight_info(layer_id)
            if layer_id in self.moe_layer_index_:
                for weight in layer_weights:
                    for component in weight.get_components():
                        if component.name == W.moe_gate:
                            component.data_type = torch.float32
            return layer_weights

        align_size = self._align_size
        ffn_config = FfnConfig(
            align_size=align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )
        moe_config = MoeConfig(align_size=align_size, expert_num=self.expert_num_)
        direct_mxfp4 = getattr(self, "has_direct_mxfp4_experts", False)
        routed_w2_name = (
            "model.layers.{i}.mlp.experts.w2_weight"
            if direct_mxfp4
            else "model.layers.{i}.mlp.experts.down_proj" + self.fused_down_suffix
        )
        routed_w13_name = (
            "model.layers.{i}.mlp.experts.w13_weight"
            if direct_mxfp4
            else "model.layers.{i}.mlp.experts.gate_up_proj" + self.fused_gate_up_suffix
        )
        layer_weights: List[WeightModule] = [
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_experts.down_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=1),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_experts.up_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [CkptWeightInfo("model.layers.{i}.mlp.gate.weight", identity)],
                        transpose,
                        data_type=torch.float32,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [CkptWeightInfo(routed_w2_name)],
                        stack_,
                        config=moe_config,
                        stacked_ckpt_keys=True,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [CkptWeightInfo(routed_w13_name)],
                        _transpose_stacked_gate_up,
                        config=moe_config,
                        stacked_ckpt_keys=True,
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
                            "model.layers.{i}.mlp.gate.e_score_correction_bias",
                            identity,
                        )
                    ],
                    identity,
                    data_type=torch.float32,
                )
            )
        return layer_weights

    def _get_weight_info(self):
        info = super()._get_weight_info()
        if self.model_config.enable_ihc:
            info.weights.extend(
                [
                    AtomicWeight(
                        W.hy4_ihc_head_fn,
                        [CkptWeightInfo("model.hc_head.hc_head_fn", identity)],
                        identity,
                        data_type=torch.float32,
                    ),
                    AtomicWeight(
                        W.hy4_ihc_head_scale,
                        [CkptWeightInfo("model.hc_head.hc_head_scale", identity)],
                        identity,
                        data_type=torch.float32,
                    ),
                    AtomicWeight(
                        W.hy4_ihc_head_base,
                        [CkptWeightInfo("model.hc_head.hc_head_base", identity)],
                        identity,
                        data_type=torch.float32,
                    ),
                ]
            )
        return info
