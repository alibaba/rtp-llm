"""MiMo V2.5 weight mapping.

Three differences from the generic path, each handled explicitly:

1. The ckpt stores QKV as a single fused tensor ``qkv_proj.weight``. The generic FP8
   quantization wrapper (``PerBlockFp8Weight._get_qkv_quant_weight``) assumes Q/K/V are
   three separate tensors and hardcodes ``merge_te_qkv``, which fails to unpack for MiMo,
   so ``MiMoPerBlockFp8Weight`` takes over.
2. ``o_proj`` is BF16 (listed in ``quantization_config.ignored_layers``, no
   ``weight_scale_inv``), so once tagged as MiMo the base quantization wrapper lets it
   through and it takes the plain BF16 path.
3. The KV head count differs per layer (GA=4 / SWA=8) and K != V head dim (192/128),
   while the standard TP split only reads the global ``load_config``. The MiMo-specific
   weight classes therefore override ``_split`` and use the per-layer ``AttnConfig``.

Non-text weights (``visual.*`` / ``audio_encoder.*`` / ``speech_embeddings.*`` /
``model.mtp.*``) are not registered here; the loader filters the ckpt against the
registered list, so they are excluded naturally.
"""

import functools
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    QS_SUFFIX,
    W_SUFFIX,
    PerBlockFp8Weight,
    W8A8Fp8PerBlockAttnAtomicWeight,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    WeightModule,
)
from rtp_llm.ops import HybridAttentionType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    get_sp_tensor_kv_asym,
    identity,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
    zeros,
)

# ---------------------------------------------------------------------------
# FP8 slab-interleaved layout constants
# ---------------------------------------------------------------------------
# The ckpt's fused QKV weight is slab-interleaved: the whole tensor is divided along
# rows into 4 slabs, each slab holding a complete [Q_shard | K_shard | V_shard], and each
# slab is FP8 block-quantized independently (block=[128,128]). The runtime kernel expects
# a section-major [Q|K|V] layout. At TP==4 (the ckpt's native slab count) every rank gets
# exactly one slab, and a slab is already [Q|K|V] section-major, so no conversion is
# needed -- the FP8 weight and scale row ranges can be sliced straight out per slab.
BLOCK_SIZE = 128
QKV_QUANT_SHARDS = 4  # number of slabs in the ckpt, independent of the TP deployment

# Row layout inside a slab for each layer kind (kv_heads -> layout params)
SLAB_LAYOUT: Dict[int, Dict[str, int]] = {
    4: {  # GA layers (kv_heads=4): each slab has Q=16heads*192, K=1head*192, V=1head*128
        "q_rows": 3072,
        "k_rows": 192,
        "v_rows": 128,
        "slab_rows": 3392,
        "slab_scale_rows": 27,  # ceil(3392/128)
    },
    8: {  # SWA layers (kv_heads=8): Q=16heads*192, K=2heads*192, V=2heads*128 per slab
        "q_rows": 3072,
        "k_rows": 384,
        "v_rows": 256,
        "slab_rows": 3712,
        "slab_scale_rows": 29,  # ceil(3712/128)
    },
}


def check_qkv_scale_rows(s: torch.Tensor, kv_heads: int) -> None:
    """Check the total scale row count == slab_scale_rows * 4, so a ckpt with a different
    layout fails here first."""
    layout = SLAB_LAYOUT[kv_heads]
    expected = layout["slab_scale_rows"] * QKV_QUANT_SHARDS
    assert s.shape[0] == expected, (
        f"qkv scale rows {s.shape[0]} != expected {expected} "
        f"(kv_heads={kv_heads}, slab_scale_rows={layout['slab_scale_rows']} * "
        f"{QKV_QUANT_SHARDS}); the ckpt's slab layout may have changed"
    )


def process_mimo_qkv_scale(ts: List[torch.Tensor], kv_heads: int) -> torch.Tensor:
    """process_fn for the qkv scale: check total scale rows == slab_scale_rows * 4.

    value_scale has been moved to o_proj (``_transpose_with_value_scale``), so the qkv
    path does not handle it."""
    s = ts[0]
    check_qkv_scale_rows(s, kv_heads)
    return s


def _tp_bypass(load_config: LoadConfig) -> bool:
    """Same short-circuit condition as ``AtomicWeight._split``: nothing to split on a
    single device."""
    return (
        load_config.tp_size <= 1
        and load_config.dp_size <= 1
        and load_config.ep_size <= 1
    )


def _transpose_with_value_scale(
    ts: List[torch.Tensor], value_scale: Optional[float]
) -> torch.Tensor:
    """Post-load processing for o_proj: transpose, then fold in attention_value_scale.

    ``value_scale`` is whatever the ckpt's config.json carries (parsed in mimo_v25.py);
    the factor is never assumed here. The reference implementation scales V in every layer
    before the KV cache is written, and attention is linear in V, so
    ``s*(softmax(QK)*V)*W_o == (softmax(QK)*V)*(s*W_o)`` -- folding it into o_proj's BF16
    weight keeps the qkv path free of float-domain work and lets the FP8 weights be sliced
    as-is.

    None means the ckpt does not scale V, so the weight is only transposed.
    """
    w = ts[0].t().contiguous()
    return w if value_scale is None else w * value_scale


# ---------------------------------------------------------------------------
# MiMo-tagged weight classes
# ---------------------------------------------------------------------------
# The class attribute is_mimo_v25 is the discriminator used to dispatch the quantization
# wrapper (see model_weight.is_mimo_v25_weight and the exclusion in
# PerBlockFp8Weight.support); it plays the same role as DSV4's is_v4_weight.


class MiMoAttnAtomicWeight(AttnAtomicWeight):
    """MiMo qkv weight (unquantized path, as registered).

    For a quantized ckpt ``MiMoPerBlockFp8Weight`` takes over; the ``_split`` override
    here serves the BF16 ckpt case.
    """

    is_mimo_v25 = True

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        raw = tensor if isinstance(tensor, torch.Tensor) else tensor[self.name]
        if _tp_bypass(load_config):
            return {self.name: raw}
        # After the transpose the layout is [hidden, q+k+v], so slice segment-wise along
        # the last dim. The per-layer kv head count comes from this weight's own
        # AttnConfig rather than the single global value in load_config.
        cfg = self.config
        ts = get_sp_tensor_kv_asym(
            raw,
            head_num=cfg.head_num,
            head_num_kv=cfg.head_num_kv,
            size_per_head=cfg.size_per_head,
            v_size_per_head=cfg.v_size_per_head,
            tp=load_config.tp_size,
            tp_rank=load_config.tp_rank,
        )
        return {self.name: ts.contiguous().clone()}


class MiMoBf16AtomicWeight(AtomicWeight):
    """MiMo's BF16 weight (o_proj), tagged so the FP8 quantization wrapper lets it through
    (the ckpt has no o_proj.weight_scale_inv). Splitting follows the default
    ``W.gpt_style_tp_strategy`` rule."""

    is_mimo_v25 = True


# Both quantized-path weights below are split by MiMoPerBlockFp8Weight._split, which
# slices the FP8 weight and its scale together by slab row range and never delegates to
# its sub-weights. They therefore carry no _split of their own.


class MiMoQkvW8A8KernelWeight(W8A8Fp8PerBlockAttnAtomicWeight):
    """qkv weight on the quantized path (FP8 kernel, [out, in] layout)."""

    is_mimo_v25 = True


class MiMoQkvW8A8ScaleWeight(W8A8Fp8PerBlockAttnAtomicWeight):
    """qkv scale on the quantized path ([rows, in/128] layout)."""

    is_mimo_v25 = True


class MiMoPerBlockFp8Weight(PerBlockFp8Weight):
    """Handles only the MiMo-tagged fused qkv.

    - At TP==4 each rank slices the FP8 weight and scale straight out by slab row range
      (no conversion, no dequant/requant), since a slab is already section-major [Q|K|V].
    - value_scale has been moved to o_proj (``_transpose_with_value_scale``), so the qkv
      path no longer involves any float-domain operation.
    - kernel: the FP8 section-major weight ([out_local, in] layout).
    - scale: the corresponding block-wise scale_inv ([ceil(out_local/128), in/128]).

    All other (FFN / MoE) weights are untagged and still go through the base
    ``PerBlockFp8Weight`` generic path.
    """

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, Fp8BlockWiseQuantConfig
        ):
            return False
        if not getattr(src_weight_info, "is_mimo_v25", False):
            return False
        return src_weight_info.name == W.attn_qkv_w

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        assert src_weight_info.name == W.attn_qkv_w, src_weight_info.name
        self.group_size = quant_config.group_size()
        kernel, scale = self._get_mimo_fused_qkv_pair(src_weight_info)
        sub_weights = {kernel.name: kernel, scale.name: scale}
        # Bypass the base class's name-dispatched if/elif chain: its qkv branch assumes
        # three separate tensors.
        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale

    def _get_mimo_fused_qkv_pair(self, src_weight_info: MiMoAttnAtomicWeight):
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kv_heads = src_weight_info.config.head_num_kv
        kernel = MiMoQkvW8A8KernelWeight(
            W.attn_qkv_w,
            [CkptWeightInfo(w_name + W_SUFFIX, identity)],
            identity,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = MiMoQkvW8A8ScaleWeight(
            W.attn_qkv_s,
            [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
            functools.partial(process_mimo_qkv_scale, kv_heads=kv_heads),
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return kernel, scale

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ):
        """Zero-conversion TP split: at TP==4 each rank slices the FP8 weight and scale
        directly by slab row range, with no dequant/requant. value_scale is already folded
        into o_proj, so there is no float-domain operation here.
        """
        weight_fp8 = (
            tensor.get(self.kernel.name) if isinstance(tensor, dict) else tensor
        )
        scale_inv = tensor.get(self.scale.name) if isinstance(tensor, dict) else None
        assert (
            scale_inv is not None
        ), "MiMoPerBlockFp8Weight._split: scale tensor missing"

        kv_heads = self.kernel.config.head_num_kv
        tp = max(load_config.tp_size, 1)
        tp_rank = load_config.tp_rank if tp > 1 else 0

        # The MiMo V2.5 FP8 ckpt is natively sliced into TP=4 slabs, and slabs cannot be
        # merged or re-split in the FP8 domain: K and V share quantization block 25 and
        # slab_rows is not a multiple of 128.
        assert tp == QKV_QUANT_SHARDS, (
            f"the MiMo V2.5 FP8 ckpt is natively split into TP={QKV_QUANT_SHARDS} slabs; "
            f"slabs cannot be merged or re-split in the FP8 domain, so it must run with "
            f"TP={QKV_QUANT_SHARDS}. Got tp_size={tp}"
        )

        layout = SLAB_LAYOUT[kv_heads]
        slab_rows = layout["slab_rows"]
        slab_scale_rows = layout["slab_scale_rows"]

        # Zero-conversion slice: each rank takes its own slab row range, FP8 as-is
        new_weight = weight_fp8[
            tp_rank * slab_rows : (tp_rank + 1) * slab_rows
        ].contiguous()
        new_scale = scale_inv[
            tp_rank * slab_scale_rows : (tp_rank + 1) * slab_scale_rows
        ].contiguous()

        return {self.kernel.name: new_weight, self.scale.name: new_scale}


# ---------------------------------------------------------------------------
# Main weight mapping class
# ---------------------------------------------------------------------------


class MiMoV25Weight(ModelDeployWeightInfo):
    def __init__(self, prefix: str = None, **kwargs: Any):
        self.prefix = prefix or ""
        self.model_prefix = "model."
        self.bias = False  # attention_bias=false: MiMo has no qkv / o bias
        super().__init__(**kwargs)
        # QK=192 / V=128
        self._v_size_per_head = self.model_config.attn_config.v_size_per_head
        # Folded into o_proj below. Comes from the ckpt's config.json via
        # MiMoV25._parse_basic_config; None there means the ckpt does not scale V.
        self._attention_value_scale = self.model_config.attention_value_scale

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        self.transformer_prefix = self.prefix + self.model_prefix

    def _get_weight_info(self) -> ModelWeightInfo:
        return self._get_hf_weight_info()

    def _get_hf_weight_info(self) -> ModelWeightInfo:
        weights = [
            AtomicWeight(
                W.embedding,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "embed_tokens.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.transformer_prefix + "norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
        ]
        layer_weights: List[List[WeightModule]] = []
        for layer_id in range(self._num_layers):
            layer_weights.append(self._get_hf_layer_weight_info(layer_id))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _is_ga_layer(self, layer_id: int) -> bool:
        types = self.model_config.hybrid_attention_config.hybrid_attention_types
        return types[layer_id] != HybridAttentionType.SLIDING_WINDOW

    def _get_hf_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        swa_cfg = self.model_config.hybrid_attention_config.swa_attention_config
        is_ga = self._is_ga_layer(layer_id)
        # self._head_num_kv cannot be used here: it is a single model-wide value, whereas
        # the two layer kinds need different kv head counts (GA=4 / SWA=8)
        kv_heads = swa_cfg.ga_kv_head_num if is_ga else swa_cfg.swa_kv_head_num

        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,  # 192, QK
            v_size_per_head=self._v_size_per_head,  # 128
            head_num=self._head_num,  # 64
            head_num_kv=kv_heads,
        )
        weights: List[WeightModule] = [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.input_layernorm.weight"
                    )
                ],
                identity,
            ),
            # qkv stays fused and is only transposed; in the FP8 case
            # MiMoPerBlockFp8Weight takes over and the scale is stored in W.attn_qkv_s
            MiMoAttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.self_attn.qkv_proj.weight"
                    )
                ],
                transpose,
                config=attn_config,
            ),
            # o_proj: BF16 (in ignored_layers, no scale), input dim 64*128=8192, taking
            # the unquantized path. The ckpt's attention_value_scale is folded in here.
            MiMoBf16AtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.self_attn.o_proj.weight"
                    )
                ],
                functools.partial(
                    _transpose_with_value_scale,
                    value_scale=self._attention_value_scale,
                ),
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.transformer_prefix
                        + "layers.{i}.post_attention_layernorm.weight"
                    )
                ],
                identity,
            ),
        ]

        if not is_ga:  # only SWA layers have a sink bias, BF16 [64]
            weights.append(
                AtomicWeight(
                    W.attn_sink_bias,
                    [
                        CkptWeightInfo(
                            self.transformer_prefix
                            + "layers.{i}.self_attn.attention_sink_bias"
                        )
                    ],
                    identity,
                )
            )

        weights.extend(self._get_ffn_layer_weight_info(layer_id))
        return weights

    def _get_ffn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        """Layer 0 is dense (inter=16384); layers 1..47 are MoE (256 experts, inter=2048).

        The MoE structure is isomorphic to deepseek_v2 (sigmoid + e_score_correction_bias,
        with moe_layer_index separating dense from MoE, and no shared experts), so the
        mapping follows its implementation."""
        if layer_id in self.moe_layer_index_:
            return self._get_moe_layer_weight_info(layer_id)
        return self._get_dense_ffn_layer_weight_info(layer_id)

    def _get_dense_ffn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        align_size = self._align_size
        ffn_config = FfnConfig(
            align_size=align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )
        return [
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.gate_proj.weight",
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
                                self.transformer_prefix
                                + "layers.{i}.mlp.down_proj.weight",
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
                                self.transformer_prefix
                                + "layers.{i}.mlp.up_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            )
        ]

    def _get_moe_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        moe_config = MoeConfig(
            align_size=self._align_size,
            expert_num=self.expert_num_,
        )
        return [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix + "layers.{i}.mlp.gate.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                                identity,
                            ),
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                                identity,
                            ),
                        ],
                        stack_moe_w1,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            ),
            # noaux_tc routing: the bias only participates in expert selection, while the
            # topk weights use the bias-free sigmoid scores
            AtomicWeight(
                W.e_score_correction_b,
                [
                    CkptWeightInfo(
                        self.transformer_prefix
                        + "layers.{i}.mlp.gate.e_score_correction_bias",
                        identity,
                    )
                ],
                identity,
                data_type=torch.float32,
            ),
        ]
