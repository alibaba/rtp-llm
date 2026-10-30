import weakref
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.qwen4_exp_config import IndexerGeometry
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
)
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    get_layer_tags,
    get_primary_attention_inputs,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMetadata,
    Qwen35Model,
    _is_cuda_graph_forward,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.pd_publication import (
    prepare_side_cache_store_writers,
    publish_layer_side_caches,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_execution import (
    PLEExecution,
    ple_input_ids,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_state import (
    _PLETargetTransaction,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_transaction import (
    PLETransactionCoordinator,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.modules import FMHAImplBase
from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    ZeroCenteredFusedQKRMSNorm,
)
from rtp_llm.models_py.modules.hybrid.causal_attention import CausalAttention
from rtp_llm.models_py.modules.hybrid.gated_residual import (
    GatedResidual,
    inject_into_residual,
)
from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa import (
    sparse_gqa_impl as _sparse_gqa_backend,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer
from rtp_llm.models_py.modules.qwen4_exp.weight_views import (
    Qwen4LinearFactory,
    mark_projection_for_restart,
    prepare_compute_weights,
)
from rtp_llm.ops import (
    AttentionConfigs,
    HWKernelConfig,
    HybridAttentionType,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class Qwen4ExpAttention(Qwen3NextAttention):
    """Qwen4 full attention plus its checkpoint-backed QSA indexer.

    The restricted production path supports ragged prefix-free prefill,
    ordinary single-token text decode and target verification no wider than one
    QSA compression group. The runtime side context owns the compressed-KV/state
    writes and hands request-local token selections to sparse GQA FMHA.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        model_config: ModelConfig,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional[HWKernelConfig] = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__(
            attn_config,
            parallelism_config,
            weights,
            layernorm_eps,
            quant_config,
            hw_kernel_config=hw_kernel_config,
            linear_factory=Qwen4LinearFactory,
        )
        self.layer_idx = layer_idx
        self.is_mtp_draft = bool(getattr(model_config, "is_mtp", False))
        self.qsa_indexer: Optional[Qwen4ExpQSAIndexer] = None
        if not getattr(model_config, "enable_qwen4_qsa", False):
            return
        self.qsa_rope_config = model_config.attn_config.rope_config

        required = (
            W.qwen4_indexer_qk_proj_w,
            W.qwen4_indexer_q_ln_gamma,
            W.qwen4_indexer_k_ln_gamma,
        )
        missing = [name for name in required if name not in weights]
        if missing:
            raise RuntimeError(
                f"qwen4_exp QSA layer {layer_idx} is missing weights {missing}"
            )
        geometry = IndexerGeometry.from_config(model_config)
        head_dim = geometry.head_dim
        ratio = geometry.compress_ratio
        budget = geometry.budget
        kv_heads = geometry.kv_heads
        heads = int(model_config.attn_config.indexer_head_num)
        if min(heads, kv_heads, head_dim, ratio, budget) <= 0:
            raise RuntimeError(
                f"qwen4_exp QSA layer {layer_idx} has incomplete indexer geometry"
            )
        self.qsa_indexer = Qwen4ExpQSAIndexer(
            weights[W.qwen4_indexer_qk_proj_w],
            weights[W.qwen4_indexer_q_ln_gamma],
            weights[W.qwen4_indexer_k_ln_gamma],
            n_heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            token_budget=budget,
            compress_ratio=ratio,
            norm_eps=layernorm_eps,
            projection_scale=weights.get(
                W.qwen4_indexer_qk_proj_w + ".qwen4_fp8_storage_scale"
            ),
            rope_config=self.qsa_rope_config,
        )

        mark_projection_for_restart(
            weights,
            getattr(self.qsa_indexer, "qk_projection", None),
        )

    def _qsa_selected_indices(
        self,
        hidden_states: torch.Tensor,
        qsa_runtime: Qwen4ExpQSARuntimeContext,
        *,
        projected=None,
    ) -> torch.Tensor:
        """Project once, persist raw keys, then build request-local selections."""
        assert self.qsa_indexer is not None
        raw_draft = (
            bool(getattr(self, "is_mtp_draft", False))
            and bool(qsa_runtime.main_inputs.is_prefill)
            and bool(getattr(qsa_runtime.main_inputs, "is_cuda_graph", False))
            and not bool(qsa_runtime.main_inputs.is_target_verify)
            and type(self.qsa_indexer) is Qwen4ExpQSAIndexer
            and (self.qsa_indexer.n_heads, self.qsa_indexer.head_dim) == (4, 128)
            and hidden_states.ndim == 2
            and hidden_states.shape[1] == 2560
            and hidden_states.shape[0] in range(4, 33, 4)
            and not hidden_states.requires_grad
            and not any(
                name in self.qsa_indexer.__dict__
                for name in ("project", "project_raw", "split_projected")
            )
        )
        raw_query = (
            (bool(qsa_runtime.main_inputs.is_target_verify) or raw_draft)
            and callable(getattr(self.qsa_indexer, "project_raw", None))
            and hidden_states.is_cuda
            and torch.version.hip is None
            and hidden_states.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(hidden_states.device) == (12, 0)
        )
        if projected is None:
            q, raw_keys = (
                self.qsa_indexer.project_raw(hidden_states)
                if raw_query
                else self.qsa_indexer.project(hidden_states)
            )
        else:
            q, raw_keys = projected
            if not raw_query:
                from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
                    exact_head_rms_norm,
                )

                q = exact_head_rms_norm(
                    q, self.qsa_indexer.q_norm_gamma, self.qsa_indexer.norm_eps
                )
        if bool(qsa_runtime.main_inputs.is_target_verify):
            return qsa_runtime.select_target_verify_tokens(
                q,
                raw_keys,
                indexer=self.qsa_indexer,
                rope_config=self.qsa_rope_config,
                **({"raw_query": True} if raw_query else {}),
            )
        if not bool(qsa_runtime.main_inputs.is_prefill):
            return qsa_runtime.select_decode_tokens(
                q,
                raw_keys,
                indexer=self.qsa_indexer,
                rope_config=self.qsa_rope_config,
            )
        if self.is_mtp_draft and bool(qsa_runtime.main_inputs.is_cuda_graph):
            return qsa_runtime.select_draft_graph_tokens(
                q,
                raw_keys,
                indexer=self.qsa_indexer,
                rope_config=self.qsa_rope_config,
                **({"raw_query": True} if raw_query else {}),
            )
        prefixes = qsa_runtime.main_inputs.prefix_lengths
        has_prefix = bool(prefixes.numel()) and bool(torch.any(prefixes != 0).item())
        if not bool(qsa_runtime.main_inputs.is_target_verify) and has_prefix:
            if self.is_mtp_draft:
                input_lengths = qsa_runtime.main_inputs.input_lengths
                max_input_length = max(int(length) for length in input_lengths.tolist())
                # PD prefix hits retain local draft-prefill cu_kv coordinates
                # even when only one token remains. Publication distinguishes
                # them from post-rejection commits, whose KV ends are absolute.
                pd_prefill = (
                    getattr(qsa_runtime.main_inputs, "cache_store_inputs", None)
                    is not None
                )
                # A first prefill may mix cold requests with short prefix hits.
                # Post-rejection commits always have a nonzero prefix per row.
                has_cold_request = bool(torch.any(prefixes == 0).item())
                if (
                    pd_prefill
                    or has_cold_request
                    or max_input_length > int(self.qsa_indexer.compress_ratio)
                ):
                    return qsa_runtime.select_draft_prefix_reuse_prefill_tokens(
                        q,
                        raw_keys,
                        indexer=self.qsa_indexer,
                        rope_config=self.qsa_rope_config,
                    )
                return qsa_runtime.select_draft_incremental_prefill_tokens(
                    q,
                    raw_keys,
                    indexer=self.qsa_indexer,
                    rope_config=self.qsa_rope_config,
                )
            return qsa_runtime.select_prefix_reuse_prefill_tokens(
                q,
                raw_keys,
                indexer=self.qsa_indexer,
                rope_config=self.qsa_rope_config,
            )
        lengths, rope_cos, rope_sin, indexer_write = (
            qsa_runtime.write_prefill_indexer_cache(
                raw_keys,
                indexer=self.qsa_indexer,
                rope_config=self.qsa_rope_config,
            )
        )

        selections = []
        offset = 0
        prepared_keys = (
            indexer_write.get("block_keys") if isinstance(indexer_write, dict) else None
        )
        for request_idx, seq_len in enumerate(lengths):
            end = offset + seq_len
            selections.append(
                self.qsa_indexer.forward_causal_from_projected(
                    q[offset:end].unsqueeze(0),
                    raw_keys[offset:end].unsqueeze(0),
                    rope_cos[request_idx, :seq_len].unsqueeze(0),
                    rope_sin[request_idx, :seq_len].unsqueeze(0),
                    **(
                        {
                            "precomputed_block_keys": prepared_keys[
                                request_idx : request_idx + 1, : seq_len // 4
                            ]
                        }
                        if prepared_keys is not None
                        else {}
                    ),
                ).squeeze(0)
            )
            offset = end
        return torch.cat(selections, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
        qsa_runtime: Optional[Qwen4ExpQSARuntimeContext] = None,
    ) -> torch.Tensor:
        if self.qsa_indexer is not None:
            if attention_inputs is None:
                raise RuntimeError("qwen4_exp QSA requires attention inputs")
            set_selected = getattr(fmha_impl, "set_selected_indices", None)
            if not callable(set_selected):
                raise RuntimeError(
                    "qwen4_exp QSA requires the sparse GQA FMHA implementation; "
                    "dense fallback is forbidden"
                )
            if qsa_runtime is None:
                raise RuntimeError("qwen4_exp QSA requires its two side-cache regions")
            if (
                qsa_runtime.main_cache is not kv_cache
                or qsa_runtime.main_inputs is not attention_inputs
            ):
                raise RuntimeError("qwen4_exp QSA main cache context is inconsistent")
            rollback_side_cache = getattr(qsa_runtime, "rollback_side_cache", None)
            finalize_side_cache = getattr(qsa_runtime, "finalize_side_cache", None)
            begin_cache_transaction = getattr(
                fmha_impl, "begin_qsa_cache_transaction", None
            )
            main_cache_mutation_started = getattr(
                fmha_impl, "qsa_main_cache_mutation_started", None
            )
            if not callable(rollback_side_cache) or not callable(finalize_side_cache):
                raise RuntimeError(
                    "qwen4_exp QSA runtime must provide side-cache transaction hooks"
                )
            if not callable(begin_cache_transaction) or not callable(
                main_cache_mutation_started
            ):
                raise RuntimeError(
                    "qwen4_exp QSA FMHA must expose its main-cache write phase"
                )
            begin_cache_transaction()
            if (
                self.is_mtp_draft
                and bool(qsa_runtime.main_inputs.is_prefill)
                and not bool(qsa_runtime.main_inputs.is_target_verify)
            ):
                set_mtp_draft_mode = getattr(fmha_impl, "set_mtp_draft_mode", None)
                if not callable(set_mtp_draft_mode):
                    raise RuntimeError(
                        "qwen4_exp QSA MTP draft requires an explicit sparse-GQA "
                        "draft-mode gate"
                    )
                if not bool(getattr(qsa_runtime, "is_mtp_draft", False)):
                    raise RuntimeError(
                        "qwen4_exp QSA MTP draft runtime context is not marked as draft"
                    )
                set_mtp_draft_mode(True)
            validate_before_write = getattr(
                fmha_impl, "validate_qsa_before_side_write", None
            )
            if callable(validate_before_write):
                validate_before_write(
                    qsa_runtime,
                    self.qsa_indexer,
                    hidden_states,
                )
            else:
                # Component-test adapters do not own a production main cache,
                # but still validate every side-cache invariant before the
                # projection. Production SparseGqaFmhaImpl takes the branch
                # above and additionally validates main-cache geometry.
                qsa_runtime.validate_before_projection(
                    indexer=self.qsa_indexer,
                    token_count=int(hidden_states.shape[0]),
                    device=hidden_states.device,
                )
            try:
                from rtp_llm.models_py.modules.base.common.fp8_linear import (
                    project_fp8_pair,
                )

                prepared = None
                if (
                    type(self.qsa_indexer) is Qwen4ExpQSAIndexer
                    and hidden_states.ndim == 2
                    and (
                        1 <= hidden_states.shape[0] <= 32
                        or 4093 <= hidden_states.shape[0] <= 32768
                    )
                    and not any(
                        name in self.qsa_indexer.__dict__
                        for name in ("project", "project_raw", "split_projected")
                    )
                ):
                    prepared = project_fp8_pair(
                        hidden_states,
                        self.gate,
                        self.qsa_indexer.qk_projection,
                    )
                if prepared is None:
                    gate = self.gate(hidden_states)
                    selected = self._qsa_selected_indices(hidden_states, qsa_runtime)
                else:
                    gate, projected_qk = prepared
                    selected = self._qsa_selected_indices(
                        hidden_states,
                        qsa_runtime,
                        projected=self.qsa_indexer.split_projected(projected_qk),
                    )
                trusted_setter = getattr(
                    fmha_impl, "set_trusted_selected_indices", None
                )
                if (
                    type(self) is Qwen4ExpAttention
                    and type(self.qsa_indexer) is Qwen4ExpQSAIndexer
                    and type(fmha_impl) is _sparse_gqa_backend.SparseGqaFmhaImpl
                    and bool(qsa_runtime.main_inputs.is_prefill)
                    and not bool(qsa_runtime.main_inputs.is_target_verify)
                    and not bool(qsa_runtime.main_inputs.is_cuda_graph)
                    and "_qsa_selected_indices" not in self.__dict__
                    and "forward_causal_from_projected" not in self.qsa_indexer.__dict__
                    and callable(trusted_setter)
                ):
                    trusted_setter(selected)
                else:
                    set_selected(selected)
                fused_gate_forward = getattr(
                    fmha_impl, "forward_with_output_gate", None
                )
                if callable(fused_gate_forward):
                    from rtp_llm.models_py.distributed.collective_torch import (
                        Group,
                        all_reduce,
                    )

                    qkv = self.qkv_proj(hidden_states)
                    norm_gate_forward = getattr(
                        fmha_impl, "forward_with_qk_norm_and_output_gate", None
                    )
                    if callable(norm_gate_forward):
                        attention = norm_gate_forward(
                            qkv, kv_cache, self.layer_idx, gate, self.qk_fuse_norm
                        )
                    else:
                        if self.qk_fuse_norm is not None:
                            qkv = self.qk_fuse_norm(qkv)
                        attention = fused_gate_forward(
                            qkv, kv_cache, self.layer_idx, gate
                        )
                    attention = attention.reshape(
                        *hidden_states.shape[:-1], -1
                    ).contiguous()
                    output = self.o_proj(attention)
                    if self.tp_size > 1:
                        output = getattr(self, "tp_all_reduce", all_reduce)(
                            output, group=Group.TP
                        )
                else:
                    output = CausalAttention.forward(
                        self, hidden_states, fmha_impl, kv_cache, gate
                    )
            except BaseException:
                if main_cache_mutation_started():
                    # Once the fused main writer has started, rolling back only
                    # the side pools would make the two cache views disagree.
                    # Preserve both and let the request-level exception take
                    # the fail-stop path; reused blocks are overwritten later.
                    finalize_side_cache()
                else:
                    rollback_side_cache()
                raise
            finalize_side_cache()
            return output
        return super().forward(
            hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta
        )


class Qwen4ExpDecoderLayer(Qwen3NextDecoderLayer):
    """Qwen3.8-Flash-Next layer: gated residual units instead of pre/post norms.

    Operates on the ``[..., hc_mult * hidden]`` stream. Each unit both collapses
    the stream for its sublayer and normalizes it, so there is no separate
    ``input_layernorm`` / ``post_attention_layernorm``.
    """

    _linear_factory = Qwen4LinearFactory
    _moe_layer_cls = Qwen4MoeLayer

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional[HWKernelConfig] = None,
    ) -> None:
        self._execution_role = getattr(parallelism_config, "role_type", "PDFUSION")
        super().__init__(
            config,
            parallelism_config,
            weights,
            layer_idx,
            moe_config,
            max_generate_batch_size,
            enable_cuda_graph,
            hw_kernel_config,
        )
        if self.layer_type == HybridAttentionType.LINEAR:
            from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
                Qwen4Sm120GdnPrefill,
            )

            native = self.self_attn.prefill_gdn
            self.self_attn.prefill_gdn = Qwen4Sm120GdnPrefill(
                native.linear_attn_config, native.parallelism_config, native.weights
            )
            from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton import (
                Qwen4PrefillGatedNorm,
            )

            gdn_norm = getattr(self.self_attn, "norm", None)
            if gdn_norm is not None:
                self.self_attn.norm = Qwen4PrefillGatedNorm(
                    gdn_norm, role=self._execution_role
                )
        if hasattr(self.mlp, "select_topk"):
            from rtp_llm.models_py.modules.qwen4_exp.moe.moe_topk_triton import (
                Qwen4ExpMoeTopk,
            )

            self.mlp.select_topk = Qwen4ExpMoeTopk(self.mlp.select_topk)
            self.mlp.prepare_shared_expert_overlap()
            from rtp_llm.models_py.modules.qwen4_exp.moe.marlin_nvfp4 import (
                make_decode_marlin_executor,
            )

            self.mlp.fused_moe.fused_experts = make_decode_marlin_executor(
                self.mlp.fused_moe.fused_experts,
                self._execution_role,
                max_generate_batch_size,
            )
        if self.layer_type != HybridAttentionType.LINEAR and getattr(
            config, "enable_qwen4_qsa", False
        ):
            self.self_attn = Qwen4ExpAttention(
                config.getAttentionConfigs(parallelism_config.get_attn_tp_size()),
                parallelism_config,
                weights,
                config.layernorm_eps,
                config,
                self._attention_quant_config(config, weights),
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )
        from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import (
            install_target_graph_all_reduce,
        )

        install_target_graph_all_reduce(self.self_attn, self.mlp)
        # Full-attention layers only; linear-attention layers have no qk norm.
        # The checkpoint's q/k gamma is stored raw for this reason, so the fused
        # version would silently drop upstream's +1 -- see norm.py.
        fused = getattr(self.self_attn, "qk_fuse_norm", None)
        if fused is not None:
            self.self_attn.qk_fuse_norm = ZeroCenteredFusedQKRMSNorm.replacing(fused)

    def _attention_quant_config(self, config, weights):
        if bool(getattr(config, "_qwen4_full_fp8_nvfp4", False)):
            from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig

            return Fp8BlockWiseQuantConfig(is_quanted=False)
        return super()._attention_quant_config(config, weights)

    def _build_residual_modules(
        self, config: ModelConfig, weights: Dict[str, torch.Tensor]
    ) -> None:
        self.attn_hyper_connection = GatedResidual(
            weights[W.qwen4_hc_attn_norm],
            weights[W.qwen4_hc_attn_mix_down],
            weights[W.qwen4_hc_attn_mix_up],
            weights[W.qwen4_hc_attn_inject],
            hc_mult=config.hc_mult,
            norm_eps=config.layernorm_eps,
            role=getattr(self, "_execution_role", "PDFUSION"),
            weight_scales=tuple(
                weights.get(name + ".qwen4_fp8_storage_scale")
                for name in (
                    W.qwen4_hc_attn_mix_down,
                    W.qwen4_hc_attn_mix_up,
                    W.qwen4_hc_attn_inject,
                )
            ),
        )
        self.mlp_hyper_connection = GatedResidual(
            weights[W.qwen4_hc_mlp_norm],
            weights[W.qwen4_hc_mlp_mix_down],
            weights[W.qwen4_hc_mlp_mix_up],
            weights[W.qwen4_hc_mlp_inject],
            hc_mult=config.hc_mult,
            norm_eps=config.layernorm_eps,
            role=getattr(self, "_execution_role", "PDFUSION"),
            weight_scales=tuple(
                weights.get(name + ".qwen4_fp8_storage_scale")
                for name in (
                    W.qwen4_hc_mlp_mix_down,
                    W.qwen4_hc_mlp_mix_up,
                    W.qwen4_hc_mlp_inject,
                )
            ),
        )
        for unit in (self.attn_hyper_connection, self.mlp_hyper_connection):
            for projection in (
                unit.down_projection,
                unit.up_projection,
                unit.inject_projection,
            ):
                mark_projection_for_restart(weights, projection)
        for unit, down_name, inject_name in (
            (
                self.attn_hyper_connection,
                W.qwen4_hc_attn_mix_down,
                W.qwen4_hc_attn_inject,
            ),
            (
                self.mlp_hyper_connection,
                W.qwen4_hc_mlp_mix_down,
                W.qwen4_hc_mlp_inject,
            ),
        ):
            merged = unit.merged_down_inject
            if merged is None:
                continue
            rank = int(unit.mix_down.shape[0])
            unit.mix_down = merged[:rank]
            unit.inject = merged[rank : rank + unit.hc_mult]
            # Views are also used by captured merged GEMMs. Updating them
            # in place requires restarting, before either execution path
            # can observe a partially changed projection.
            unit.mix_down._rtp_requires_restart_for_update = True
            unit.inject._rtp_requires_restart_for_update = True
            weights[down_name] = unit.mix_down
            weights[inject_name] = unit.inject

    def forward(
        self,
        hyper_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
        qsa_runtime: Optional[Qwen4ExpQSARuntimeContext] = None,
    ) -> torch.Tensor:
        base, output, gate = self.forward_deferred(
            hyper_states,
            fmha_impl,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            attn_meta=attn_meta,
            qsa_runtime=qsa_runtime,
        )
        return inject_into_residual(base, output, gate)

    def forward_deferred(
        self,
        hyper_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
        qsa_runtime: Optional[Qwen4ExpQSARuntimeContext] = None,
        pending_injection=None,
    ) -> tuple:
        if pending_injection is not None:
            previous_output, previous_gate = pending_injection
            hidden_states, hyper_states, inject_weights = (
                self.attn_hyper_connection.forward_with_injection(
                    hyper_states, previous_output, previous_gate
                )
            )
        else:
            hidden_states, hyper_states, inject_weights = self.attn_hyper_connection(
                hyper_states
            )
        attn_kwargs = dict(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            attn_meta=attn_meta,
        )
        if qsa_runtime is not None:
            attn_kwargs["qsa_runtime"] = qsa_runtime
        hidden_states = self.self_attn(**attn_kwargs)
        if isinstance(self.mlp_hyper_connection, GatedResidual):
            hidden_states, hyper_states, inject_weights = (
                self.mlp_hyper_connection.forward_with_injection(
                    hyper_states, hidden_states, inject_weights
                )
            )
        else:
            hyper_states = inject_into_residual(
                hyper_states, hidden_states, inject_weights
            )
            hidden_states, hyper_states, inject_weights = self.mlp_hyper_connection(
                hyper_states
            )
        hidden_states = self.mlp(hidden_states)
        return hyper_states, hidden_states, inject_weights


class Qwen4ExpModel(Qwen35Model):
    """Qwen3.8-Flash-Next.

    Differs from :class:`Qwen35Model` only in the residual stream: the embedding
    is broadcast into ``hc_mult`` branches before the layer loop, every layer
    reads and writes that wide stream through its gated residual units, and the
    global mixer collapses it back while doubling as the final norm.
    """

    _decoder_layer_cls = Qwen4ExpDecoderLayer
    # Regions that are not a layer's attention region. Every one must be listed or
    # `_attention_tag` sees more than one candidate and raises -- full-attention
    # layers own three regions once the indexer is enabled.
    _side_region_tags = frozenset(
        {PLE_STATE_TAG, PLE_NGRAM_CTX_TAG, INDEXER_KV_TAG, INDEXER_STATE_TAG}
    )

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        from rtp_llm.model_loader.fp8_weight import FP8_STORAGE_SCALE_SUFFIX
        from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear

        config = model_config
        compute_weights = (
            prepare_compute_weights(weights)
            if getattr(config, "_qwen4_full_fp8_nvfp4", False)
            else weights
        )
        super().__init__(
            model_config,
            parallelism_config,
            compute_weights,
            moe_config,
            max_generate_batch_size,
            fmha_config,
            py_hw_kernel_config,
            device_resource_config,
        )
        if getattr(config, "_qwen4_full_fp8_nvfp4", False):
            self._fp8_lm_head = Fp8Linear(
                weights.global_weights[W.lm_head],
                weights.global_weights[W.lm_head + FP8_STORAGE_SCALE_SUFFIX],
            )
        if getattr(self.config, "_qwen4_full_fp8_nvfp4", False):
            from rtp_llm.models_py.modules.base.common.fp8_embedding import (
                Fp8VocabEmbedding,
            )

            self.embed_tokens = Fp8VocabEmbedding(
                self.embed_tokens.weight,
                self.embed_tokens.tp_size,
                torch.device("cuda", torch.cuda.current_device()),
                weight_scale=weights.global_weights.get(
                    W.embedding + FP8_STORAGE_SCALE_SUFFIX
                ),
            )
        if getattr(self.config, "_qwen4_full_fp8_nvfp4", False):
            self.embed_tokens.prepare_byte_gather(
                role=getattr(parallelism_config, "role_type", "PDFUSION")
            )
        self.ple_layers = nn.ModuleDict()
        self._ple_target_transaction: Optional[_PLETargetTransaction] = None
        self._ple_graph_target_transactions: Dict[int, _PLETargetTransaction] = {}
        self._capture_mtp_target_hidden = bool(getattr(self.config, "is_mtp", False))
        if getattr(self.config, "enable_qwen4_ple", False):
            self._build_ple_layers()
        from .qwen4_exp_runtime.target_head_graph import TargetHeadGraph

        self._target_head_graph = TargetHeadGraph(
            enabled=hasattr(self, "_fp8_lm_head")
            and bool(self.ple_layers)
            and not getattr(self.config, "is_mtp", False)
        )
        from rtp_llm.models_py.modules.qwen4_exp.head_ipc_gather import (
            prepare_head_ipc_gather,
        )

        self._head_ipc_gather = prepare_head_ipc_gather(
            config, parallelism_config, getattr(self, "_fp8_lm_head", None)
        )
        from rtp_llm.models_py.modules.qwen4_exp.cpu_target_status import (
            prepare_cpu_target_status,
        )

        self._cpu_target_status = prepare_cpu_target_status(config, parallelism_config)

    def _ple_coordinator(self) -> PLETransactionCoordinator:
        coordinator = self.__dict__.get("_ple_transaction_coordinator")
        if coordinator is None:
            owner = weakref.ref(self)

            def fault(phase: str) -> None:
                model = owner()
                if model is not None:
                    model._maybe_inject_speculative_fault(phase)

            coordinator = PLETransactionCoordinator(self.ple_layers, fault)
            self._ple_transaction_coordinator = coordinator
        return coordinator

    def _ple_execution(self) -> PLEExecution:
        execution = self.__dict__.get("_ple_execution_runtime")
        if execution is None or execution.kv_cache is not self.kv_cache:
            execution = PLEExecution(
                self.ple_layers,
                self.kv_cache,
                self.parallelism_config,
                self.config.special_tokens.eos_token_id,
                self._ple_coordinator(),
            )
            self._ple_execution_runtime = execution
        return execution

    @property
    def _ple_target_transaction(self):
        return self._ple_coordinator().active_transaction

    @_ple_target_transaction.setter
    def _ple_target_transaction(self, value):
        self._ple_coordinator().active_transaction = value

    @property
    def _ple_graph_target_transactions(self):
        return self._ple_coordinator().captured_transactions

    @_ple_graph_target_transactions.setter
    def _ple_graph_target_transactions(self, value):
        self._ple_coordinator().captured_transactions = value

    def resource_usage(self) -> dict[str, int]:
        totals = {}
        tables = {}
        for ple in self.ple_layers.values():
            embedding = ple.ple_embedding
            for shard in embedding.shards:
                if shard.device.type == "cpu":
                    storage = shard.untyped_storage()
                    tables[storage.data_ptr()] = (storage.nbytes(), shard.is_pinned())
            for name, value in embedding.resource_usage().items():
                if name != "cpu_table_bytes":
                    totals[name] = totals.get(name, 0) + value
        totals["cpu_table_bytes"] = sum(size for size, _ in tables.values())
        totals["cpu_pinned_table_bytes"] = sum(
            size for size, pinned in tables.values() if pinned
        )
        totals["ple_layer_count"] = len(self.ple_layers)
        return totals

    def close(self) -> None:
        # C++ destroys graph runners before this hook. Drain all consumers before
        # releasing mapped CPU or graph-owned lookup storage.
        devices = {
            ple.ple_embedding.vocab_sizes.device
            for ple in self.ple_layers.values()
            if ple.ple_embedding.vocab_sizes.is_cuda
        }
        for device in devices:
            torch.cuda.synchronize(device)
        coordinator = self.__dict__.get("_ple_transaction_coordinator")
        if coordinator is not None:
            coordinator.close()
        for ple in self.ple_layers.values():
            ple.ple_embedding.close()
        self._ple_execution_runtime = None
        head_graph = self.__dict__.get("_target_head_graph")
        if head_graph is not None:
            head_graph.clear_activation()
            head_graph.outputs.clear()
        head_ipc = self.__dict__.get("_head_ipc_gather")
        if head_ipc is not None:
            head_ipc.close()
        status = self.__dict__.get("_cpu_target_status")
        if status is not None:
            status.close()
        self._closed = True

    def compute_logits(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if not hasattr(self, "_fp8_lm_head"):
            return None
        from rtp_llm.models_py.modules.base.common.vocab_parallel_logits import (
            gather_vocab_parallel_logits,
        )

        workspace = self.__dict__.get("_head_ipc_gather")
        return gather_vocab_parallel_logits(
            self._compute_local_logits(hidden),
            int(self.parallelism_config.tp_size),
            try_gather=workspace.try_gather if workspace is not None else None,
        )

    def _compute_local_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        head_graph = self.__dict__.get("_target_head_graph")
        logits = head_graph.take(hidden) if head_graph is not None else None
        return (
            self._fp8_lm_head(hidden.to(torch.bfloat16)) if logits is None else logits
        )

    def initialize(self, init_resource) -> bool:
        initialized = super().initialize(init_resource)
        # The target and the draft both export their wide post-block residual:
        # target -> first draft step, draft -> subsequent draft steps.  Draft
        # configs carry is_mtp even if their wrapper does not inherit the
        # target's speculative flag.
        self._capture_mtp_target_hidden = bool(
            init_resource.is_speculative or getattr(self.config, "is_mtp", False)
        )
        return initialized

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        hidden = getattr(self, "_mtp_target_hidden_states", None)
        if hidden is None:
            return None
        if hidden.dim() != 2:
            raise RuntimeError(
                "qwen4_exp MTP target hidden state must be a packed 2-D tensor"
            )
        requested = int(num_tokens)
        if requested < 0:
            requested = int(hidden.shape[0])
        if requested > int(hidden.shape[0]):
            raise RuntimeError(
                "qwen4_exp requested MTP target hidden rows exceed the last "
                f"forward: requested={requested}, available={hidden.shape[0]}"
            )
        return hidden[:requested]

    def _try_fp8_hyper_embedding(self, inputs):
        # A target owns the HC repetition. Draft models consume the ordinary
        # H-wide embedding through their separate input projection.
        if type(self) is not Qwen4ExpModel:
            return None
        from rtp_llm.models_py.modules.base.common.fp8_embedding import (
            Fp8VocabEmbedding,
        )

        embedding = getattr(self, "embed_tokens", None)
        if not isinstance(embedding, Fp8VocabEmbedding) or embedding.tp_size != 2:
            return None
        if (
            torch.version.hip is not None
            or torch.cuda.get_device_capability(embedding.device) != (12, 0)
            or "forward" in embedding.__dict__
            or "word_embedding" in self.__dict__
            or embedding._forward_pre_hooks
            or embedding._forward_hooks
            or torch.nn.modules.module._global_forward_pre_hooks
            or torch.nn.modules.module._global_forward_hooks
        ):
            return None
        multimodal = getattr(inputs, "multimodal_inputs", None)
        if multimodal is not None:
            for name in ("multimodal_features", "mm_features_locs"):
                value = getattr(multimodal, name, None)
                if value is not None and len(value):
                    return None
        embedding_inputs = getattr(inputs, "embedding_inputs", None)
        mask = getattr(embedding_inputs, "text_tokens_mask", None)
        return embedding.forward_hyper(
            inputs.input_ids, branches=self.hc_mult, text_tokens_mask=mask
        )

    def _initial_hyper_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = int(self.config.hidden_size)
        if int(hidden_states.shape[-1]) != expected:
            raise RuntimeError(
                "qwen4_exp token embedding width mismatch: "
                f"expected {expected}, got {hidden_states.shape[-1]}"
            )
        return hidden_states.repeat(*(1,) * (hidden_states.dim() - 1), self.hc_mult)

    def _build_ple_layers(self) -> None:
        layer_ids = list(getattr(self.config, "_qwen4_ple_layer_ids", []))
        total_shards = int(getattr(self.config, "_qwen4_split_ngram_parts", 0))
        ngram_size = int(getattr(self.config, "_qwen4_ngram_size", 0))
        # Weight descriptors split with the effective attention TP topology;
        # consume the same view when validating and assembling local shards.
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        tp_rank = int(self.parallelism_config.get_attn_tp_rank())
        if not layer_ids or total_shards <= 0 or ngram_size <= 1:
            raise RuntimeError("qwen4_exp PLE configuration is incomplete")
        if tp_size < 1 or total_shards % tp_size:
            raise RuntimeError(
                "qwen4_exp PLE requires an even TP shard split, got "
                f"total_shards={total_shards}, tp_size={tp_size}"
            )
        per_rank = total_shards // tp_size
        expected_shards = set(range(tp_rank * per_rank, (tp_rank + 1) * per_rank))
        prefix = W.qwen4_ple_ngram_shards + "."

        for upstream_layer_id in layer_ids:
            layer_idx = int(upstream_layer_id) - 1
            if not 0 <= layer_idx < len(self.layers):
                raise RuntimeError(f"qwen4_exp PLE layer {layer_idx} is out of range")
            layer_weights = self.weight.weights[layer_idx]
            shard_pairs = sorted(
                (
                    (int(name[len(prefix) :]), tensor)
                    for name, tensor in layer_weights.items()
                    if name.startswith(prefix)
                ),
                key=lambda item: item[0],
            )
            shard_indices = [index for index, _ in shard_pairs]
            if set(shard_indices) != expected_shards:
                raise RuntimeError(
                    f"qwen4_exp PLE layer {layer_idx} expected local shards "
                    f"{sorted(expected_shards)}, got {shard_indices}"
                )
            required = (
                W.qwen4_ple_ngram_vocab_sizes,
                W.qwen4_ple_ngram_offsets,
                W.qwen4_ple_multipliers,
                W.qwen4_ple_key_proj_w,
                W.qwen4_ple_value_proj_w,
                W.qwen4_ple_conv_w,
                W.qwen4_ple_norm_key_gamma,
                W.qwen4_ple_norm_query_gamma,
                W.qwen4_ple_norm_conv_gamma,
            )
            missing = [name for name in required if name not in layer_weights]
            if missing:
                raise RuntimeError(
                    f"qwen4_exp PLE layer {layer_idx} is missing weights {missing}"
                )
            embedding = Qwen4ExpNGramEmbedding(
                [tensor for _, tensor in shard_pairs],
                layer_weights[W.qwen4_ple_ngram_vocab_sizes],
                layer_weights[W.qwen4_ple_ngram_offsets],
                layer_weights[W.qwen4_ple_multipliers],
                ngram_size=ngram_size,
                eos_token_id=int(self.config.special_tokens.eos_token_id),
                shard_indices=shard_indices,
                total_shards=total_shards,
                distributed_reduce=tp_size > 1,
                weight_scale=layer_weights.get(W.qwen4_ple_ngram_scale),
                prefetch_budget_mb=getattr(self.config, "qwen4_ple_prefetch_mb", 256),
                inline_cpu_uva_graph=(
                    shard_pairs[0][1].device.type == "cpu" and torch.version.hip is None
                ),
            )
            # Retain the same pinned owners in the weight/export inventory;
            # release pageable copies after this initialization scope finishes.
            for (index, _), shard in zip(shard_pairs, embedding.shards):
                layer_weights[prefix + str(index)] = shard
                storage = getattr(self.weight, "_fp8_storage_owner", self.weight)
                storage.weights[layer_idx][prefix + str(index)] = shard
            conv_weight = layer_weights[W.qwen4_ple_conv_w]
            self.ple_layers[str(layer_idx)] = Qwen4ExpPLELayer(
                embedding,
                layer_weights[W.qwen4_ple_key_proj_w],
                layer_weights[W.qwen4_ple_value_proj_w],
                conv_weight,
                layer_weights[W.qwen4_ple_norm_key_gamma],
                layer_weights[W.qwen4_ple_norm_query_gamma],
                layer_weights[W.qwen4_ple_norm_conv_gamma],
                hc_mult=self.config.hc_mult,
                hidden_size=self.config.hidden_size,
                conv_kernel_size=int(conv_weight.shape[-1]),
                projection_scales=tuple(
                    layer_weights.get(name + ".qwen4_fp8_storage_scale")
                    for name in (W.qwen4_ple_key_proj_w, W.qwen4_ple_value_proj_w)
                ),
                norm_eps=self.config.layernorm_eps,
            )

            unit = self.ple_layers[str(layer_idx)]
            mark_projection_for_restart(layer_weights, unit.key_projection)
            mark_projection_for_restart(layer_weights, unit.value_projection)

    def _maybe_inject_speculative_fault(self, phase: str) -> None:
        """Invoke the private transaction test hook with actual phase and rank."""
        hook = getattr(self, "_speculative_test_hook", None)
        if hook is not None:
            hook(phase, int(self.parallelism_config.get_attn_tp_rank()))

    def on_cuda_graph_warmup(self, inputs: PyModelInputs) -> None:
        """Release eager-only state after each graph warmup forward."""
        if get_primary_attention_inputs(inputs).is_target_verify:
            self._ple_coordinator().clear_speculative_target_graph_capture()

    def on_cuda_graph_capture(self, graph_key: int, inputs: PyModelInputs) -> None:
        """Retain model-owned tensors for this captured graph instance."""
        if not get_primary_attention_inputs(inputs).is_target_verify:
            return
        from rtp_llm.models_py.distributed.flashinfer_all_reduce import (
            get_flashinfer_allreduce,
        )

        communicator = get_flashinfer_allreduce()
        if communicator is not None:
            communicator.finish_graph_capture()
        return self._ple_coordinator().save_speculative_target_graph_capture(graph_key)

    def on_cuda_graph_replay(self, graph_key: int, inputs: PyModelInputs) -> None:
        """Activate the model state belonging to the graph that just replayed."""
        if not get_primary_attention_inputs(inputs).is_target_verify:
            return
        self._ple_coordinator().activate_speculative_target_graph_replay(graph_key)
        head_graph = self.__dict__.get("_target_head_graph")
        if head_graph is not None:
            head_graph.activate(graph_key)

    def has_speculative_target_status_reducer(self) -> bool:
        return self.__dict__.get("_cpu_target_status") is not None

    def reduce_speculative_target_prepare_status(self, local_ok: bool) -> bool:
        status = self.__dict__.get("_cpu_target_status")
        if status is None:
            raise RuntimeError("CPU target status was not initialized collectively")
        return status.reduce(local_ok)

    def prepare_speculative_target_commit(self, accept_len: torch.Tensor) -> None:
        return self._ple_coordinator().prepare_speculative_target_commit(accept_len)

    def finish_speculative_target_commit(self, commit: bool) -> None:
        return self._ple_coordinator().finish_speculative_target_commit(commit)

    def finalize_speculative_target_commit(self) -> None:
        return self._ple_coordinator().finalize_speculative_target_commit()

    def _start_eager_ple_prefetch(self, inputs: PyModelInputs, attention_inputs):
        if not self.ple_layers:
            return {}
        return self._ple_execution()._start_eager_ple_prefetch(inputs, attention_inputs)

    def _apply_ple(
        self,
        layer_idx: int,
        hyper_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_inputs: Any,
        prefetch=None,
    ) -> torch.Tensor:
        if str(layer_idx) not in self.ple_layers:
            return hyper_states
        return self._ple_execution()._apply_ple(
            layer_idx, hyper_states, input_ids, attention_inputs, prefetch
        )

    def _attention_tag(self, layer_idx: int) -> str:
        """Tag of the layer's attention region, excluding the side regions.

        A layer can own several regions, so neither ``get_layer_cache(idx)`` nor
        ``select_attention_inputs_for_layer`` can resolve attention for it: the
        former needs a sole group and the latter would hand ``self_attn`` a list.
        """
        tags = [
            t
            for t in get_layer_tags(self.kv_cache, layer_idx)
            if t not in self._side_region_tags
        ]
        if len(tags) != 1:
            raise RuntimeError(
                f"layer {layer_idx} must own exactly one attention region, got {tags}"
            )
        return tags[0]

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        """The attention regions only; side regions need no FMHA impl.

        The base implementation collects every tag of every non-linear layer,
        which for this model would include ``indexer_kv`` / ``indexer_state``
        and make ``prepare_fmha_impl`` build impls for pools that are not
        attention at all.
        """
        if self.kv_cache is None:
            return None
        tags: list[str] = []
        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_type == HybridAttentionType.LINEAR:
                continue
            tag = self._attention_tag(layer_idx)
            if tag not in tags:
                tags.append(tag)
        return tags

    def _layer_fmha_impl(self, decoder_layer, fmha_impl, layer_idx: int):
        if decoder_layer.layer_type == HybridAttentionType.LINEAR:
            return None
        if not isinstance(fmha_impl, Mapping):
            return fmha_impl
        tag = self._attention_tag(layer_idx)
        try:
            return fmha_impl[tag]
        except KeyError as error:
            raise RuntimeError(
                f"FMHA impl for tag {tag!r} is missing; "
                f"available tags={list(fmha_impl)}"
            ) from error

    def _qsa_runtime_context(
        self,
        layer_idx: int,
        main_cache: LayerKVCache,
        main_inputs: PyAttentionInputs,
        attention_inputs: Any,
    ) -> Qwen4ExpQSARuntimeContext:
        if self.kv_cache is None or not isinstance(attention_inputs, Mapping):
            raise RuntimeError(
                "qwen4_exp QSA requires tag-local main/indexer cache inputs"
            )
        return Qwen4ExpQSARuntimeContext(
            main_cache=main_cache,
            main_inputs=main_inputs,
            indexer_kv_cache=self.kv_cache.get_layer_cache(layer_idx, INDEXER_KV_TAG),
            indexer_kv_inputs=select_attention_inputs_for_tag(
                attention_inputs, INDEXER_KV_TAG
            ),
            indexer_state_cache=self.kv_cache.get_layer_cache(
                layer_idx, INDEXER_STATE_TAG
            ),
            indexer_state_inputs=select_attention_inputs_for_tag(
                attention_inputs, INDEXER_STATE_TAG
            ),
            is_mtp_draft=bool(getattr(self.config, "is_mtp", False)),
        )

    def _build_output_head(
        self, model_config: ModelConfig, weights: ModelWeights
    ) -> None:
        self.hc_mult = model_config.hc_mult
        self.hyper_connection_mixer = GatedResidual(
            weights.get_global_weight(W.qwen4_hc_mixer_norm),
            weights.get_global_weight(W.qwen4_hc_mixer_mix_down),
            weights.get_global_weight(W.qwen4_hc_mixer_mix_up),
            None,
            hc_mult=model_config.hc_mult,
            norm_eps=model_config.layernorm_eps,
            role=getattr(self.parallelism_config, "role_type", "PDFUSION"),
            weight_scales=(
                weights.global_weights.get(
                    W.qwen4_hc_mixer_mix_down + ".qwen4_fp8_storage_scale"
                ),
                weights.global_weights.get(
                    W.qwen4_hc_mixer_mix_up + ".qwen4_fp8_storage_scale"
                ),
                None,
            ),
        )

        for projection in (
            self.hyper_connection_mixer.down_projection,
            self.hyper_connection_mixer.up_projection,
        ):
            mark_projection_for_restart(weights.global_weights, projection)

    def _ple_input_ids(self, inputs: PyModelInputs) -> torch.Tensor:
        tokens = getattr(getattr(self, "config", None), "special_tokens", None)
        return ple_input_ids(inputs, getattr(tokens, "eos_token_id", None))

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if self.__dict__.get("_closed", False):
            raise RuntimeError("Qwen4 model is closed")
        from rtp_llm.models_py.distributed.flashinfer_all_reduce import (
            get_flashinfer_allreduce,
        )

        communicator = get_flashinfer_allreduce()
        prior_capture = None
        if communicator is not None:
            groups = get_attention_inputs_value(inputs)
            groups = groups.values() if isinstance(groups, dict) else (groups,)
            prior_capture = communicator.allow_ipc_capture
            communicator.allow_ipc_capture = not bool(
                getattr(self.config, "is_mtp", False)
            ) and any(bool(getattr(a, "is_target_verify", False)) for a in groups)
        head_graph = self.__dict__.get("_target_head_graph")
        if head_graph is not None:
            head_graph.clear_activation()
        try:
            outputs = self._forward_with_ple_prefetch(inputs, fmha_impl)
            transaction = (
                self._ple_target_transaction
                if head_graph is not None and head_graph.enabled
                else None
            )
            if transaction is not None:
                groups = get_attention_inputs_value(inputs)
                values = (
                    tuple(groups.values()) if isinstance(groups, Mapping) else (groups,)
                )
                if values and all(
                    value.is_cuda_graph and value.is_target_verify for value in values
                ):
                    head_graph.stage(
                        self._fp8_lm_head, outputs.hidden_states, transaction.batch_size
                    )
            return outputs
        finally:
            if communicator is not None:
                communicator.allow_ipc_capture = prior_capture
            for ple in self.ple_layers.values():
                embedding = getattr(ple, "ple_embedding", None)
                finish = getattr(embedding, "finish_prefetch", None)
                if callable(finish):
                    finish()

    def _forward_with_ple_prefetch(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        if (
            self.ple_layers
            and getattr(self, "_ple_target_transaction", None) is not None
        ):
            raise RuntimeError(
                "qwen4_exp PLE target transaction must be finalized or rolled "
                "back before the next forward"
            )
        attention_inputs = get_attention_inputs_value(inputs)
        has_side_caches = bool(self.ple_layers) or any(
            getattr(getattr(layer, "self_attn", None), "qsa_indexer", None) is not None
            for layer in self.layers
        )
        side_cache_writers = (
            prepare_side_cache_store_writers(
                attention_inputs, self.kv_cache, layer_count=len(self.layers)
            )
            if has_side_caches
            else {}
        )
        # vLLM starts the PLE side-stream lookup after token embedding. An
        # earlier pinned-host gather can occupy every SM while vocabulary
        # lookup waits for it, despite the independent CUDA streams.
        prefetch_after_embedding = False
        if type(self) is Qwen4ExpModel and self.ple_layers:
            # Native grouped inputs are a tag mapping. Use the same PLE
            # context group as _start_eager_ple_prefetch, not the mapping
            # itself or an unrelated attention group's execution mode.
            ple_context_inputs = select_attention_inputs_for_tag(
                attention_inputs, PLE_NGRAM_CTX_TAG
            )
            prefetch_after_embedding = bool(
                ple_context_inputs.is_prefill and not ple_context_inputs.is_cuda_graph
            )
        ple_prefetch = (
            {}
            if prefetch_after_embedding
            else self._start_eager_ple_prefetch(inputs, attention_inputs)
        )
        prepared_hyper = self._try_fp8_hyper_embedding(inputs)
        hidden_states = (
            self.word_embedding(inputs) if prepared_hyper is None else prepared_hyper
        )
        if prefetch_after_embedding:
            ple_prefetch = self._start_eager_ple_prefetch(inputs, attention_inputs)

        is_cuda_graph = _is_cuda_graph_forward(inputs, fmha_impl)
        attn_meta = self._build_attn_meta(inputs, hidden_states.device, is_cuda_graph)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        # The target tiles token embeddings here.  The MTP draft overrides the
        # hook because its input projection has already produced one distinct
        # hidden vector per hyper-connection branch.
        hyper_states = (
            self._initial_hyper_states(hidden_states)
            if prepared_hyper is None
            else prepared_hyper
        )

        attention_inputs = get_attention_inputs_value(inputs)
        ple_input_ids = inputs.input_ids
        if self.ple_layers:
            ple_input_ids = self._ple_input_ids(inputs)
        pending_injection = None
        defer_residual = (
            not is_cuda_graph
            and hyper_states.is_cuda
            and torch.version.hip is None
            and hyper_states.ndim == 2
            and hyper_states.shape[1] == 10240
            and 4093 <= hyper_states.shape[0] <= 12288
            and hyper_states.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(hyper_states.device) == (12, 0)
        )
        for i, decoder_layer in enumerate(self.layers):
            if str(i) in self.ple_layers:
                if pending_injection is not None:
                    hyper_states = inject_into_residual(
                        hyper_states, *pending_injection
                    )
                    pending_injection = None
                if self.kv_cache is None:
                    raise RuntimeError(
                        "qwen4_exp PLE requires its two state cache regions"
                    )
                hyper_states = self._apply_ple(
                    i,
                    hyper_states,
                    ple_input_ids,
                    attention_inputs,
                    **({"prefetch": ple_prefetch[i]} if i in ple_prefetch else {}),
                )
            if self.kv_cache is None:
                layer_kv_cache = None
                layer_attention_inputs = attention_inputs
            else:
                tag = self._attention_tag(i)
                layer_kv_cache = self.kv_cache.get_layer_cache(i, tag)
                layer_attention_inputs = select_attention_inputs_for_tag(
                    attention_inputs, tag
                )
            layer_fmha_impl = self._layer_fmha_impl(decoder_layer, fmha_impl, i)
            # The engine rebuilds equivalent attention-input objects per batch,
            # so bind this layer to the exact object its sparse-GQA impl was
            # constructed with. The QSA runtime context, the impl and the
            # attention module all validate that identity before writing.
            impl_main_inputs = getattr(layer_fmha_impl, "attn_inputs", None)
            if impl_main_inputs is not None:
                layer_attention_inputs = impl_main_inputs
            qsa_runtime = None
            if getattr(getattr(decoder_layer, "self_attn", None), "qsa_indexer", None):
                if layer_kv_cache is None or layer_attention_inputs is None:
                    raise RuntimeError("qwen4_exp QSA requires its main cache region")
                qsa_runtime = self._qsa_runtime_context(
                    i,
                    layer_kv_cache,
                    layer_attention_inputs,
                    attention_inputs,
                )
            layer_kwargs = dict(
                kv_cache=layer_kv_cache,
                attention_inputs=layer_attention_inputs,
                attn_meta=attn_meta,
                qsa_runtime=qsa_runtime,
            )
            if defer_residual and isinstance(decoder_layer, Qwen4ExpDecoderLayer):
                hyper_states, pending_output, pending_gate = (
                    decoder_layer.forward_deferred(
                        hyper_states,
                        layer_fmha_impl,
                        pending_injection=pending_injection,
                        **layer_kwargs,
                    )
                )
                pending_injection = (pending_output, pending_gate)
            else:
                if pending_injection is not None:
                    hyper_states = inject_into_residual(
                        hyper_states, *pending_injection
                    )
                    pending_injection = None
                hyper_states = decoder_layer(
                    hyper_states, layer_fmha_impl, **layer_kwargs
                )
            publish_layer_side_caches(self.kv_cache, i, side_cache_writers)

        # PLE and MTP/global-mixer consumers observe the materialized BF16 stream.
        if pending_injection is not None:
            hyper_states = inject_into_residual(hyper_states, *pending_injection)

        if getattr(self, "_capture_mtp_target_hidden", False):
            # The pre-collapse stream is the MTP input. Return it as a
            # first-class output: graph replay does not rerun Python, so a
            # mutable model attribute alone would expose the capture-time rows.
            self._mtp_target_hidden_states = hyper_states
        else:
            self._mtp_target_hidden_states = None
        hidden_states, _, _ = self.hyper_connection_mixer(hyper_states)
        if self._mtp_target_hidden_states is not None:
            return PyModelOutputs(hidden_states, self._mtp_target_hidden_states)
        return PyModelOutputs(hidden_states)
