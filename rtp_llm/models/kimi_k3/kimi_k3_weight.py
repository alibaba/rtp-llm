"""Kimi K3 checkpoint-to-runtime weight manifest.

K3 is a mixed-format checkpoint: routed experts are group-32 MXFP4 (two E2M1
values per byte plus one UE8M0 scale byte per group), most dense tensors are
BF16, and KDA recurrence/short-convolution/output-norm control weights are
FP32.  Treating the nested compressed-tensors config as a model-wide
quantization mode or coercing every non-expert tensor to BF16 would corrupt the
checkpoint, so every exceptional dtype is represented explicitly.
"""

from __future__ import annotations

import functools
from typing import Iterator, List, Optional

import torch

from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    LinearAttnConfig,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CustomAtomicWeight,
    WeightModule,
)
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3RotaryEmbedding,
)
from rtp_llm.ops import HybridAttentionType, MlaOpsType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    ffn_sp_0,
    ffn_sp_neg1,
    identity,
    mla_pad_t,
    sp_0,
    sp_id,
    stack_,
    transpose,
    transpose_slice_k,
    transpose_slice_v,
)


def _merge_conv1d(ts: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate q/k/v depthwise-conv weights along the channel axis.

    Mirrors main's ``kimi_linear`` fused conv layout ``[C_q+C_k+C_v, 1, K]`` so
    the shared ``split_conv1d`` TP strategy and a future rebase both apply.  The
    ``[.,1,K]`` middle dim is preserved here and squeezed at consume time.
    """

    return torch.cat(ts, dim=0)


def _merge_kda_qkvg_fa_beta(ts: List[torch.Tensor]) -> torch.Tensor:
    """Build the global K3 fused projection before heterogeneous TP split."""

    if len(ts) != 6:
        raise ValueError(f"K3 KDA fused projection expects six tensors, got {len(ts)}")
    q, k, v, g, f_a, beta = ts
    return torch.cat((q.T, k.T, v.T, g.T, f_a.T, beta.T), dim=1).contiguous()


def _merge_mla_input_projections(
    ts: List[torch.Tensor], *, tp_size: int, tp_rank: int
) -> torch.Tensor:
    """Pack replicated Q/KV-A and the rank-local output gate into one GEMM B."""

    q_a, kv_a, output_gate = ts
    local_output_gate = output_gate.chunk(tp_size, dim=0)[tp_rank]
    return torch.cat((q_a, kv_a, local_output_gate), dim=0).T.contiguous()


def _unpad_kda_alog(ts: List[torch.Tensor], *, num_heads: int) -> torch.Tensor:
    """Remove the checkpoint's zero alignment tail before TP head sharding.

    The production K3 checkpoint stores ``A_log`` as 128 fp32 values although
    the model has 96 KDA heads.  Values ``[96:128]`` are alignment padding and
    are exactly zero.  Keeping that tail would make TP ranks after rank 0 read
    the wrong logical heads, because ``split_head_linear`` shards by the
    configured 96-head layout.
    """

    tensor = identity(ts)
    if tensor.ndim != 1 or tensor.shape[0] < num_heads:
        raise ValueError(
            "K3 KDA A_log must be a 1-D tensor containing at least "
            f"{num_heads} logical heads, got {tuple(tensor.shape)}"
        )
    padding = tensor[num_heads:]
    if padding.numel() and torch.count_nonzero(padding).item() != 0:
        raise ValueError(
            "K3 KDA A_log alignment padding must be exactly zero, got "
            f"{torch.count_nonzero(padding).item()} non-zero values"
        )
    return tensor[:num_heads].contiguous()


class KimiK3WeightNames:
    """Stable keys stored in :class:`ModelWeights` and consumed by the model."""

    OUTPUT_ATTN_RES_NORM = "kimi_k3.output_attn_res.norm"
    OUTPUT_ATTN_RES_PROJ = "kimi_k3.output_attn_res.proj"

    SELF_ATTN_RES_NORM = "kimi_k3.self_attn_res.norm"
    SELF_ATTN_RES_PROJ = "kimi_k3.self_attn_res.proj"
    MLP_RES_NORM = "kimi_k3.mlp_res.norm"
    MLP_RES_PROJ = "kimi_k3.mlp_res.proj"

    # KDA linear-attention weights migrated to the shared ``W.linear_attn_*``
    # vocabulary (see ``_kda_weights``); no K3-private keys remain for them.

    MOE_GATE = "kimi_k3.moe.gate"
    MOE_CORRECTION_BIAS = "kimi_k3.moe.correction_bias"
    MOE_ROUTED_DOWN = "kimi_k3.moe.routed_down"
    MOE_ROUTED_UP = "kimi_k3.moe.routed_up"
    MOE_ROUTED_NORM = "kimi_k3.moe.routed_norm"
    MOE_SHARED_GATE = "kimi_k3.moe.shared_gate"
    MOE_SHARED_UP = "kimi_k3.moe.shared_up"
    MOE_SHARED_DOWN = "kimi_k3.moe.shared_down"
    MOE_W1_PACKED = "kimi_k3.moe.w1_packed"
    MOE_W1_SCALE = "kimi_k3.moe.w1_scale"
    MOE_W2_PACKED = "kimi_k3.moe.w2_packed"
    MOE_W2_SCALE = "kimi_k3.moe.w2_scale"
    MOE_W3_PACKED = "kimi_k3.moe.w3_packed"
    MOE_W3_SCALE = "kimi_k3.moe.w3_scale"


class _KimiExpertByteWeight(MoeAtomicWeight):
    """EP-select experts while leaving checkpoint-native byte layout untouched."""

    def _get_split_func(self):
        # Expert parallel selection happens in MoeAtomicWeight._load_raw_tensor.
        # K3's first correctness path does not additionally tensor-shard the
        # packed expert matrices; a future native MXFP4 kernel may do so.
        return sp_id


class KimiK3Weight(ModelDeployWeightInfo):
    """Describe every text-model tensor in the Kimi K3 checkpoint."""

    MODEL_PREFIX = "language_model.model."
    LAYER_PREFIX = MODEL_PREFIX + "layers.{i}."

    _COMMON_LAYER_SUFFIXES = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
    )
    _KDA_SUFFIXES = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
        "self_attn.A_log",
        "self_attn.dt_bias",
        "self_attn.f_a_proj.weight",
        "self_attn.f_b_proj.weight",
        "self_attn.b_proj.weight",
        "self_attn.g_proj.weight",
        "self_attn.o_norm.weight",
        "self_attn.o_proj.weight",
    )
    _MLA_SUFFIXES = (
        "self_attn.q_a_proj.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.g_proj.weight",
        "self_attn.o_proj.weight",
    )
    _DENSE_SUFFIXES = (
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    _MOE_SUFFIXES = (
        "block_sparse_moe.gate.weight",
        "block_sparse_moe.gate.e_score_correction_bias",
        "block_sparse_moe.routed_expert_down_proj.weight",
        "block_sparse_moe.routed_expert_up_proj.weight",
        "block_sparse_moe.routed_expert_norm.weight",
        "block_sparse_moe.shared_experts.gate_proj.weight",
        "block_sparse_moe.shared_experts.up_proj.weight",
        "block_sparse_moe.shared_experts.down_proj.weight",
    )

    @classmethod
    def _layer_ckpt(cls, suffix: str) -> str:
        return cls.LAYER_PREFIX + suffix

    @classmethod
    def global_checkpoint_tensor_names(cls) -> tuple[str, ...]:
        return (
            cls.MODEL_PREFIX + "embed_tokens.weight",
            cls.MODEL_PREFIX + "norm.weight",
            cls.MODEL_PREFIX + "output_attn_res_norm.weight",
            cls.MODEL_PREFIX + "output_attn_res_proj.weight",
            "language_model.lm_head.weight",
        )

    @classmethod
    def checkpoint_tensor_names_for_layer(
        cls, model_config, layer_id: int
    ) -> Iterator[str]:
        prefix = cls.LAYER_PREFIX.format(i=layer_id)
        for suffix in cls._COMMON_LAYER_SUFFIXES:
            yield prefix + suffix

        layer_type = model_config.hybrid_attention_config.hybrid_attention_types[
            layer_id
        ]
        attention_suffixes = (
            cls._KDA_SUFFIXES
            if layer_type == HybridAttentionType.LINEAR
            else cls._MLA_SUFFIXES
        )
        for suffix in attention_suffixes:
            yield prefix + suffix

        if layer_id in set(model_config.moe_layer_index):
            for suffix in cls._MOE_SUFFIXES:
                yield prefix + suffix
            expert_prefix = prefix + "block_sparse_moe.experts."
            for expert_id in range(model_config.expert_num):
                for projection in ("w1", "w2", "w3"):
                    yield (f"{expert_prefix}{expert_id}.{projection}.weight_packed")
                    yield f"{expert_prefix}{expert_id}.{projection}.weight_scale"
        else:
            for suffix in cls._DENSE_SUFFIXES:
                yield prefix + suffix

    @classmethod
    def expected_checkpoint_tensor_names(cls, model_config) -> Iterator[str]:
        yield from cls.global_checkpoint_tensor_names()
        for layer_id in range(model_config.num_layers):
            yield from cls.checkpoint_tensor_names_for_layer(model_config, layer_id)

    @classmethod
    def checkpoint_tensor_patterns(cls, model_config) -> set[str]:
        """Return the compact placeholder-based text-checkpoint manifest."""

        patterns = set(cls.global_checkpoint_tensor_names())
        patterns.update(cls._layer_ckpt(s) for s in cls._COMMON_LAYER_SUFFIXES)
        if any(
            layer_type == HybridAttentionType.LINEAR
            for layer_type in model_config.hybrid_attention_config.hybrid_attention_types
        ):
            patterns.update(cls._layer_ckpt(s) for s in cls._KDA_SUFFIXES)
        if any(
            layer_type != HybridAttentionType.LINEAR
            for layer_type in model_config.hybrid_attention_config.hybrid_attention_types
        ):
            patterns.update(cls._layer_ckpt(s) for s in cls._MLA_SUFFIXES)
        if model_config.moe_layer_index:
            patterns.update(cls._layer_ckpt(s) for s in cls._MOE_SUFFIXES)
            for projection in ("w1", "w2", "w3"):
                patterns.add(
                    cls._layer_ckpt(
                        "block_sparse_moe.experts.{expert_id}."
                        f"{projection}.weight_packed"
                    )
                )
                patterns.add(
                    cls._layer_ckpt(
                        "block_sparse_moe.experts.{expert_id}."
                        f"{projection}.weight_scale"
                    )
                )
        if len(model_config.moe_layer_index) < model_config.num_layers:
            patterns.update(cls._layer_ckpt(s) for s in cls._DENSE_SUFFIXES)
        return patterns

    @classmethod
    def _custom(
        cls,
        name: str,
        suffix: str,
        *,
        process_fun=identity,
        split_func=sp_id,
        data_type=None,
    ) -> CustomAtomicWeight:
        return CustomAtomicWeight(
            name,
            [CkptWeightInfo(cls._layer_ckpt(suffix), identity)],
            process_fun=process_fun,
            split_func=split_func,
            data_type=data_type,
        )

    @classmethod
    def _linear(cls, name: str, suffix: str, *, split_func=sp_id) -> CustomAtomicWeight:
        # RTP LinearFactory stores GEMM weights as [in_features,out_features].
        return cls._custom(name, suffix, process_fun=transpose, split_func=split_func)

    def _global_weights(self) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.MODEL_PREFIX + "embed_tokens.weight", identity)],
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("language_model.lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.MODEL_PREFIX + "norm.weight", identity)],
            ),
            CustomAtomicWeight(
                KimiK3WeightNames.OUTPUT_ATTN_RES_NORM,
                [
                    CkptWeightInfo(
                        self.MODEL_PREFIX + "output_attn_res_norm.weight", identity
                    )
                ],
            ),
            CustomAtomicWeight(
                KimiK3WeightNames.OUTPUT_ATTN_RES_PROJ,
                [
                    CkptWeightInfo(
                        self.MODEL_PREFIX + "output_attn_res_proj.weight", identity
                    )
                ],
                process_fun=transpose,
            ),
        ]

    def _common_layer_weights(self) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(self._layer_ckpt("input_layernorm.weight"), identity)],
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self._layer_ckpt("post_attention_layernorm.weight"), identity
                    )
                ],
            ),
            self._custom(
                KimiK3WeightNames.SELF_ATTN_RES_NORM,
                "self_attention_res_norm.weight",
            ),
            self._linear(
                KimiK3WeightNames.SELF_ATTN_RES_PROJ,
                "self_attention_res_proj.weight",
            ),
            self._custom(KimiK3WeightNames.MLP_RES_NORM, "mlp_res_norm.weight"),
            self._linear(KimiK3WeightNames.MLP_RES_PROJ, "mlp_res_proj.weight"),
        ]

    def _kda_weights(self) -> List[WeightModule]:
        """KDA linear-attention weights on the shared ``W.linear_attn_*`` vocab.

        Q/K/V/G, the full forget-gate down projection and the full 96-column
        beta projection fuse into ``linear_attn_qkvg_fa_beta_w``. The three
        depthwise convs fuse into ``linear_attn_conv1d_w``. One K3-specific
        deviation from ``kimi_linear`` is that the checkpoint stores ``A_log``
        as a 128-element
        aligned vector whose first ``num_heads`` entries are logical and whose
        remaining entries are zero padding.  The padding is removed before TP
        head sharding. The output gate is K3's single full-rank projection and
        is sharded by head in the fused weight. Per-weight TP sharding is
        resolved by name via ``LinearAttnAtomicWeight``'s split-strategy table.
        """

        cfg = LinearAttnConfig(self.model_config.linear_attention_config)

        def _w(name, suffix, process_fun, *, data_type=None):
            return LinearAttnAtomicWeight(
                name,
                [CkptWeightInfo(self._layer_ckpt(suffix), identity)],
                process_fun,
                cfg,
                data_type=data_type,
            )

        return [
            LinearAttnAtomicWeight(
                W.linear_attn_qkvg_fa_beta_w,
                [
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.q_proj.weight"), identity
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.k_proj.weight"), identity
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.v_proj.weight"), identity
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.g_proj.weight"), identity
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.f_a_proj.weight"), identity
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.b_proj.weight"), identity
                    ),
                ],
                _merge_kda_qkvg_fa_beta,
                cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.q_conv1d.weight"), identity
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.k_conv1d.weight"), identity
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.v_conv1d.weight"), identity
                    ),
                ],
                _merge_conv1d,
                cfg,
                # The checkpoint tensors are FP32, but the official model
                # converts ShortConvolution parameters to the requested model
                # dtype in ``from_pretrained``.  Inherit the runtime load dtype
                # here as well; forcing FP32 changes the convolution result
                # before KDA Q/K normalization.
            ),
            _w(
                W.linear_attn_alog,
                "self_attn.A_log",
                functools.partial(
                    _unpad_kda_alog,
                    num_heads=cfg.linear_num_value_heads,
                ),
                data_type=torch.float32,
            ),
            _w(
                W.linear_attn_dt_b_kda,
                "self_attn.dt_bias",
                identity,
                data_type=torch.float32,
            ),
            _w(W.linear_attn_f_b_w, "self_attn.f_b_proj.weight", transpose),
            _w(
                W.linear_attn_norm_w,
                "self_attn.o_norm.weight",
                identity,
                data_type=torch.float32,
            ),
            _w(W.linear_attn_out_w, "self_attn.o_proj.weight", transpose),
        ]

    def _mla_config(self) -> MlaConfig:
        attn = self.model_config.attn_config
        return MlaConfig(
            head_num=self._head_num,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            v_head_dim=self.v_head_dim,
            use_mla=attn.use_mla and self.model_config.mla_ops_type != MlaOpsType.MHA,
            q_use_lora=attn.q_lora_rank > 0,
        )

    def _mla_weights(self) -> List[WeightModule]:
        """MLA weights on the shared ``W.mla_*`` vocabulary (DeepSeek-V2 layout).

        Same fused + absorbed layout the generic ``MlaAttention`` +
        ``MlaFlashInfer*`` path consumes, so K3's full-attention layers reuse the
        framework MLA kernels instead of a bespoke einsum.  Two K3 specifics:

        * ``mla.rope_head_dim`` is a *physical* 64-d suffix but carries no
          positional signal (NoPE).  The unconditional RoPE in the shared Impl
          is neutralised by an identity ``W.rope_cos_sin_cache`` (see
          ``_create_rope_w``), not by a code fork.
        * K3's Q-A, KV-A and sigmoid-gate projections share the same input, so
          the loader packs them into one rank-local GEMM weight.
        """

        cfg = self._mla_config()

        def _mla(name, suffix, process_fun):
            return MlaAttnAtomicWeight(
                name,
                [CkptWeightInfo(self._layer_ckpt(suffix), identity)],
                process_fun,
                config=cfg,
            )

        weights: List[WeightModule] = [
            # o_proj: pad the v-only head layout to the fused nope+rope stride
            # the shared kernel writes (rope_head_dim=0 -> no padding columns).
            _mla(
                W.attn_o_w,
                "self_attn.o_proj.weight",
                functools.partial(
                    mla_pad_t,
                    head_num=self._head_num,
                    nope_head_dim=self.v_head_dim,
                    rope_head_dim=0,
                ),
            ),
            _mla(W.mla_kv_b_w, "self_attn.kv_b_proj.weight", transpose),
            _mla(W.mla_kv_a_ln_gamma, "self_attn.kv_a_layernorm.weight", identity),
            _mla(W.mla_q_b_w, "self_attn.q_b_proj.weight", transpose),
            _mla(W.mla_q_a_ln_gamma, "self_attn.q_a_layernorm.weight", identity),
            # Q-A and KV-A are replicated; g_proj contributes this rank's heads.
            MlaAttnAtomicWeight(
                W.mla_fusedqkrope_w,
                [
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.q_a_proj.weight"), identity
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.kv_a_proj_with_mqa.weight"),
                        identity,
                    ),
                    CkptWeightInfo(
                        self._layer_ckpt("self_attn.g_proj.weight"), identity
                    ),
                ],
                functools.partial(
                    _merge_mla_input_projections,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                ),
                config=cfg,
            ),
        ]

        # Absorbed decode weights: slice kv_b into the compressed-cache
        # bmm operands the FlashInfer decode kernel expects.
        if (
            self.model_config.attn_config.use_mla
            and self.model_config.mla_ops_type != MlaOpsType.MHA
        ):
            weights.append(
                _mla(
                    W.mla_kc,
                    "self_attn.kv_b_proj.weight",
                    functools.partial(
                        transpose_slice_k,
                        head_num=self._head_num,
                        nope_head_dim=self.nope_head_dim,
                        v_head_dim=self.v_head_dim,
                        lora_rank=self.kv_lora_rank,
                    ),
                )
            )
            weights.append(
                _mla(
                    W.mla_vc,
                    "self_attn.kv_b_proj.weight",
                    functools.partial(
                        transpose_slice_v,
                        head_num=self._head_num,
                        nope_head_dim=self.nope_head_dim,
                        v_head_dim=self.v_head_dim,
                        lora_rank=self.kv_lora_rank,
                    ),
                )
            )
        return weights

    def _create_rope_w(self) -> Optional[AtomicWeight]:
        """Identity RoPE cache so the shared MLA Impl's unconditional rope is a
        no-op — K3 is NoPE.

        ``get_mla_impl`` unconditionally fetches ``W.rope_cos_sin_cache`` and
        ``NewMlaRotaryEmbeddingOp`` raises if it is ``None``, so the cache must
        exist.  A cache of cos=1, sin=0 makes ``_apply_rope_pos_ids_cos_sin_cache``
        the identity map, giving NoPE with zero change to the shared kernels.
        The layout matches DeepSeek-V2's ``_create_rope_w``:
        ``[max_seq_len, rope_dim]`` float32 with the first half cos, second sin.
        """

        if self.model_config.mla_ops_type == MlaOpsType.MHA:
            return None

        max_seq_len = self.model_config.max_seq_len
        half_rope_dim = self.model_config.attn_config.rope_config.dim // 2

        def _identity_cos_sin(ts: List[torch.Tensor]) -> torch.Tensor:
            cos_cache = torch.ones(max_seq_len, half_rope_dim, dtype=torch.float32)
            sin_cache = torch.zeros(max_seq_len, half_rope_dim, dtype=torch.float32)
            return torch.cat([cos_cache, sin_cache], dim=-1).contiguous()

        return AtomicWeight(
            W.rope_cos_sin_cache,
            [],
            process_fun=_identity_cos_sin,
            data_type=torch.float32,
        )

    def _dense_weights(self) -> List[WeightModule]:
        return [
            self._linear(W.ffn_w1, "mlp.gate_proj.weight", split_func=ffn_sp_neg1),
            self._linear(W.ffn_w3, "mlp.up_proj.weight", split_func=ffn_sp_neg1),
            self._linear(W.ffn_w2, "mlp.down_proj.weight", split_func=ffn_sp_0),
        ]

    def _moe_weights(self) -> List[WeightModule]:
        n = KimiK3WeightNames
        weights: List[WeightModule] = [
            self._linear(n.MOE_GATE, "block_sparse_moe.gate.weight"),
            self._custom(
                n.MOE_CORRECTION_BIAS,
                "block_sparse_moe.gate.e_score_correction_bias",
                data_type=torch.float32,
            ),
            # These latent projections are replicated in the initial EP path.
            # Attention still uses TP; this avoids nibble-alignment coupling
            # between the latent projection and the packed expert matrices.
            self._linear(
                n.MOE_ROUTED_DOWN,
                "block_sparse_moe.routed_expert_down_proj.weight",
            ),
            self._linear(
                n.MOE_ROUTED_UP,
                "block_sparse_moe.routed_expert_up_proj.weight",
            ),
            self._custom(
                n.MOE_ROUTED_NORM,
                "block_sparse_moe.routed_expert_norm.weight",
            ),
            self._linear(
                n.MOE_SHARED_GATE,
                "block_sparse_moe.shared_experts.gate_proj.weight",
                split_func=ffn_sp_neg1,
            ),
            self._linear(
                n.MOE_SHARED_UP,
                "block_sparse_moe.shared_experts.up_proj.weight",
                split_func=ffn_sp_neg1,
            ),
            self._linear(
                n.MOE_SHARED_DOWN,
                "block_sparse_moe.shared_experts.down_proj.weight",
                split_func=ffn_sp_0,
            ),
        ]

        moe_config = MoeConfig(
            expert_num=self.expert_num_, align_size=self._moe_align_size
        )
        for packed_name, scale_name, projection in (
            (n.MOE_W1_PACKED, n.MOE_W1_SCALE, "w1"),
            (n.MOE_W2_PACKED, n.MOE_W2_SCALE, "w2"),
            (n.MOE_W3_PACKED, n.MOE_W3_SCALE, "w3"),
        ):
            checkpoint_prefix = "block_sparse_moe.experts.{expert_id}." f"{projection}."
            weights.append(
                _KimiExpertByteWeight(
                    packed_name,
                    [
                        CkptWeightInfo(
                            self._layer_ckpt(checkpoint_prefix + "weight_packed"),
                            identity,
                        )
                    ],
                    process_fun=stack_,
                    config=moe_config,
                    data_type=torch.uint8,
                )
            )
            weights.append(
                _KimiExpertByteWeight(
                    scale_name,
                    [
                        CkptWeightInfo(
                            self._layer_ckpt(checkpoint_prefix + "weight_scale"),
                            identity,
                        )
                    ],
                    process_fun=stack_,
                    config=moe_config,
                    data_type=torch.uint8,
                )
            )
        return weights

    def _get_weight_info(self) -> ModelWeightInfo:
        layer_weights: List[List[WeightModule]] = []
        moe_layers = set(self.model_config.moe_layer_index)
        layer_types = list(
            self.model_config.hybrid_attention_config.hybrid_attention_types
        )
        if len(layer_types) < self._num_layers:
            raise ValueError(
                "Kimi K3 hybrid attention schedule is shorter than num_layers: "
                f"{len(layer_types)} < {self._num_layers}"
            )
        for layer_id, layer_type in enumerate(layer_types[: self._num_layers]):
            weights = self._common_layer_weights()
            weights.extend(
                self._kda_weights()
                if layer_type == HybridAttentionType.LINEAR
                else self._mla_weights()
            )
            weights.extend(
                self._moe_weights() if layer_id in moe_layers else self._dense_weights()
            )
            layer_weights.append(weights)
        return ModelWeightInfo(
            weights=self._global_weights(), layer_weights=layer_weights
        )


class KimiK3Eagle3Weight(KimiK3Weight):
    """Weight manifest for the standalone Kimi K3 EAGLE-3 checkpoint."""

    MODEL_PREFIX = ""
    LAYER_PREFIX = "layers.{i}."

    def _global_weights(self) -> List[WeightModule]:
        return [
            AtomicWeight(W.embedding, [CkptWeightInfo("embed_tokens.weight", identity)]),
            AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight", identity)]),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo("norm.weight", identity)]),
        ]

    def _common_layer_weights(self) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(self._layer_ckpt("input_layernorm.weight"), identity)],
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self._layer_ckpt("post_attention_layernorm.weight"), identity
                    )
                ],
            ),
            AtomicWeight(
                W.eagle3_input_norm_gamma,
                [CkptWeightInfo(self._layer_ckpt("input_layernorm.weight"), identity)],
            ),
            AtomicWeight(
                W.eagle3_fc_norm_gamma,
                [CkptWeightInfo(self._layer_ckpt("hidden_norm.weight"), identity)],
            ),
            CustomAtomicWeight(
                W.eagle3_fc_proj,
                [CkptWeightInfo("fc.weight", identity)],
                process_fun=transpose,
            ),
        ]

    def _dense_weights(self) -> List[WeightModule]:
        return [
            self._linear(W.ffn_w1, "mlp.gate_proj.weight", split_func=ffn_sp_neg1),
            self._linear(W.ffn_w3, "mlp.up_proj.weight", split_func=ffn_sp_neg1),
            self._linear(W.ffn_w2, "mlp.down_proj.weight", split_func=ffn_sp_0),
        ]

    def _create_rope_w(self) -> Optional[AtomicWeight]:
        config = self.model_config

        def _rope_cache(_: List[torch.Tensor]) -> torch.Tensor:
            rotary = DeepseekV3RotaryEmbedding(
                dim=config.attn_config.rope_config.dim,
                max_position_embeddings=config.max_seq_len,
                base=config.attn_config.rope_config.base,
                device="cuda",
            )
            half_dim = config.attn_config.rope_config.dim // 2
            return torch.cat(
                [
                    rotary.cos_cached[:, :half_dim],
                    rotary.sin_cached[:, :half_dim],
                ],
                dim=-1,
            ).float().contiguous()

        return AtomicWeight(
            W.rope_cos_sin_cache,
            [],
            process_fun=_rope_cache,
            data_type=torch.float32,
        )


__all__ = ["KimiK3Weight", "KimiK3WeightNames", "KimiK3Eagle3Weight"]
