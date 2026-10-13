import functools
from typing import List

import torch

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen_v2 import QWenV2Weight
from rtp_llm.ops import HybridAttentionType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    transpose,
    transpose_pad,
)


def squeeze_conv_weight(ts: List[torch.Tensor]) -> torch.Tensor:
    # checkpoint: [dim, 1, kernel]; conv kernels take [dim, kernel]
    return ts[0].squeeze(1)


class Lfm2Weight(QWenV2Weight):
    """LFM2 / LFM2-VL weight loading.

    Text checkpoints live under ``model.language_model.*`` (VL) or
    ``model.*`` (text-only); ``QWenV2Weight._process_meta`` already handles
    the ``language_model.`` prefix detection.

    conv layers (hybrid_attention_types[i] == LINEAR) load
    ``layers.{i}.conv.{in_proj,conv,out}_proj`` plus the shared norms/FFN;
    full-attention layers load the standard GQA weights with per-head
    q/k layernorm (``q_layernorm`` / ``k_layernorm``).

    The multimodal tower/projector tensors are not part of this mapping;
    they are loaded through the multimodal mixin path.
    """

    def _get_weight_info(self):
        return self._get_hf_weight_info()

    def _get_hf_weight_info(self):
        prefix = self.transformer_prefix
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(prefix + "embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(prefix + "embedding_norm.weight", identity)],
                identity,
            ),
        ]

        layer_weights: List[List[WeightModule]] = []
        for layer in range(self._num_layers):
            layer_weights.append(self._get_hf_layer_weight_info(layer))

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_type = self.model_config.hybrid_attention_config.hybrid_attention_types[
            layer_id
        ]
        if layer_type == HybridAttentionType.LINEAR:
            return self._get_conv_layer_weight_info(layer_id)
        return self._get_attn_layer_weight_info(layer_id)

    def _get_attn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        prefix = self.transformer_prefix
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )

        layer_weights: List[WeightModule] = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(prefix + "layers.{i}.operator_norm.weight", identity)],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        prefix + "layers.{i}.self_attn.q_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        prefix + "layers.{i}.self_attn.k_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        prefix + "layers.{i}.self_attn.v_proj.weight", identity
                    ),
                ],
                functools.partial(merge_qkv_hf),
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.q_ln_gamma,
                [
                    CkptWeightInfo(
                        prefix + "layers.{i}.self_attn.q_layernorm.weight", identity
                    )
                ],
                identity,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.k_ln_gamma,
                [
                    CkptWeightInfo(
                        prefix + "layers.{i}.self_attn.k_layernorm.weight", identity
                    )
                ],
                identity,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        prefix + "layers.{i}.self_attn.out_proj.weight", identity
                    )
                ],
                transpose,
                config=attn_config,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(prefix + "layers.{i}.ffn_norm.weight", identity)],
                identity,
            ),
        ]
        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    def _get_conv_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        prefix = self.transformer_prefix

        layer_weights: List[WeightModule] = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(prefix + "layers.{i}.operator_norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.conv_in_proj_w,
                [CkptWeightInfo(prefix + "layers.{i}.conv.in_proj.weight", identity)],
                transpose,
            ),
            AtomicWeight(
                W.conv_w,
                [CkptWeightInfo(prefix + "layers.{i}.conv.conv.weight", identity)],
                squeeze_conv_weight,
            ),
            AtomicWeight(
                W.conv_out_proj_w,
                [CkptWeightInfo(prefix + "layers.{i}.conv.out_proj.weight", identity)],
                transpose,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(prefix + "layers.{i}.ffn_norm.weight", identity)],
                identity,
            ),
        ]
        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        prefix = self.transformer_prefix
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=self._align_size,
            is_moe=False,
        )

        align_size = self._align_size
        return [
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                prefix + "layers.{i}.feed_forward.w1.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                prefix + "layers.{i}.feed_forward.w3.weight",
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
                                prefix + "layers.{i}.feed_forward.w2.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=1),
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            )
        ]
