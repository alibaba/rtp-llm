import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
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
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMetadata,
    Qwen35Model,
    _is_cuda_graph_forward,
)
from rtp_llm.models_py.modules import FMHAImplBase
from rtp_llm.models_py.modules.hybrid.causal_attention import CausalAttention
from rtp_llm.models_py.modules.qwen4_exp.gated_residual import (
    Qwen4ExpGatedResidual,
    inject_into_residual,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer import Qwen4ExpQSAIndexer
from rtp_llm.models_py.modules.qwen4_exp.norm import Qwen4ExpFusedQKRMSNorm
from rtp_llm.models_py.modules.qwen4_exp.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa_runtime import Qwen4ExpQSARuntimeContext
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


@dataclass
class _PLETargetLayerStage:
    layer_idx: int
    page_size: int
    prefixes: torch.Tensor
    state_pool: torch.Tensor
    ctx_pool: torch.Tensor
    state_inputs: PyAttentionInputs
    ctx_inputs: PyAttentionInputs
    initial_state: torch.Tensor
    candidate_state_inputs: torch.Tensor
    initial_context: torch.Tensor
    candidate_ids: torch.Tensor


@dataclass
class _PLEPreparedWrite:
    state_pool: torch.Tensor
    state_blocks: torch.Tensor
    state_values: torch.Tensor
    original_state: torch.Tensor
    ctx_pool: torch.Tensor
    ctx_blocks: torch.Tensor
    ctx_values: torch.Tensor
    original_context: torch.Tensor


@dataclass
class _PLETargetTransaction:
    batch_size: int
    query_len: int
    prefixes: torch.Tensor
    expected_layers: frozenset[int]
    stages: Dict[int, _PLETargetLayerStage]
    prepared_writes: Optional[list[_PLEPreparedWrite]] = None
    commit_started: bool = False
    tentative_committed: bool = False


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
        head_dim = int(getattr(model_config, "_qwen4_indexer_head_dim", 0))
        ratio = int(getattr(model_config, "_qwen4_indexer_compress_ratio", 0))
        budget = int(getattr(model_config, "_qwen4_indexer_budget", 0))
        kv_heads = int(getattr(model_config, "_qwen4_indexer_kv_heads", 0))
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
        )

    def _qsa_selected_indices(
        self,
        hidden_states: torch.Tensor,
        qsa_runtime: Qwen4ExpQSARuntimeContext,
    ) -> torch.Tensor:
        """Project once, persist raw keys, then build request-local selections."""
        assert self.qsa_indexer is not None
        q, raw_keys = self.qsa_indexer.project(hidden_states)
        if bool(qsa_runtime.main_inputs.is_target_verify):
            return qsa_runtime.select_target_verify_tokens(
                q,
                raw_keys,
                indexer=self.qsa_indexer,
                rope_config=self.qsa_rope_config,
            )
        if not bool(qsa_runtime.main_inputs.is_prefill):
            return qsa_runtime.select_decode_tokens(
                q,
                raw_keys,
                indexer=self.qsa_indexer,
                rope_config=self.qsa_rope_config,
            )
        prefixes = qsa_runtime.main_inputs.prefix_lengths
        if (
            self.is_mtp_draft
            and not bool(qsa_runtime.main_inputs.is_target_verify)
            and prefixes.numel()
            and bool(torch.any(prefixes != 0).item())
        ):
            return qsa_runtime.select_draft_incremental_prefill_tokens(
                q,
                raw_keys,
                indexer=self.qsa_indexer,
                rope_config=self.qsa_rope_config,
            )
        lengths, rope_cos, rope_sin, _ = qsa_runtime.write_prefill_indexer_cache(
            raw_keys,
            indexer=self.qsa_indexer,
            rope_config=self.qsa_rope_config,
        )

        selections = []
        offset = 0
        for request_idx, seq_len in enumerate(lengths):
            end = offset + seq_len
            selections.append(
                self.qsa_indexer.forward_causal_from_projected(
                    q[offset:end].unsqueeze(0),
                    raw_keys[offset:end].unsqueeze(0),
                    rope_cos[request_idx, :seq_len].unsqueeze(0),
                    rope_sin[request_idx, :seq_len].unsqueeze(0),
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
                gate = self.gate(hidden_states)
                set_selected(self._qsa_selected_indices(hidden_states, qsa_runtime))
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
        if self.layer_type != HybridAttentionType.LINEAR and getattr(
            config, "enable_qwen4_qsa", False
        ):
            self.self_attn = Qwen4ExpAttention(
                config.getAttentionConfigs(parallelism_config.get_attn_tp_size()),
                parallelism_config,
                weights,
                config.layernorm_eps,
                config,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )
        # Full-attention layers only; linear-attention layers have no qk norm.
        # The checkpoint's q/k gamma is stored raw for this reason, so the fused
        # version would silently drop upstream's +1 -- see norm.py.
        fused = getattr(self.self_attn, "qk_fuse_norm", None)
        if fused is not None:
            self.self_attn.qk_fuse_norm = Qwen4ExpFusedQKRMSNorm.replacing(fused)

    def _build_residual_modules(
        self, config: ModelConfig, weights: Dict[str, torch.Tensor]
    ) -> None:
        self.attn_hyper_connection = Qwen4ExpGatedResidual(
            weights[W.qwen4_hc_attn_norm],
            weights[W.qwen4_hc_attn_mix_down],
            weights[W.qwen4_hc_attn_mix_up],
            weights[W.qwen4_hc_attn_inject],
            hc_mult=config.hc_mult,
            norm_eps=config.layernorm_eps,
        )
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(
            weights[W.qwen4_hc_mlp_norm],
            weights[W.qwen4_hc_mlp_mix_down],
            weights[W.qwen4_hc_mlp_mix_up],
            weights[W.qwen4_hc_mlp_inject],
            hc_mult=config.hc_mult,
            norm_eps=config.layernorm_eps,
        )

    def forward(
        self,
        hyper_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
        qsa_runtime: Optional[Qwen4ExpQSARuntimeContext] = None,
    ) -> torch.Tensor:
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
        hyper_states = inject_into_residual(hyper_states, hidden_states, inject_weights)

        hidden_states, hyper_states, inject_weights = self.mlp_hyper_connection(
            hyper_states
        )
        hidden_states = self.mlp(hidden_states)
        return inject_into_residual(hyper_states, hidden_states, inject_weights)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ple_layers = nn.ModuleDict()
        self._ple_target_transaction: Optional[_PLETargetTransaction] = None
        self._capture_mtp_target_hidden = bool(getattr(self.config, "is_mtp", False))
        if getattr(self.config, "enable_qwen4_ple", False):
            self._build_ple_layers()

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
        if tp_size <= 1 or total_shards % tp_size:
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
                distributed_reduce=True,
            )
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
                norm_eps=self.config.layernorm_eps,
            )

    @staticmethod
    def _fixed_state_pool(
        cache: LayerKVCache, dtype: torch.dtype, entries: int, width: int
    ) -> torch.Tensor:
        base = cache.kv_cache_base
        if base is None or base.dim() != 2 or not base.is_contiguous():
            raise RuntimeError(f"PLE cache {cache.tag!r} must be a contiguous 2-D pool")
        if base.dtype != dtype:
            raise RuntimeError(
                f"PLE cache {cache.tag!r} has dtype {base.dtype}, expected {dtype}"
            )
        payload_bytes = entries * width * torch.empty((), dtype=dtype).element_size()
        raw = base.view(torch.uint8)
        if int(raw.shape[1]) != payload_bytes:
            raise RuntimeError(
                f"PLE cache {cache.tag!r} row has {raw.shape[1]} bytes, "
                f"expected {payload_bytes}"
            )
        return raw.view(dtype).view(int(raw.shape[0]), entries, width)

    @staticmethod
    def _physical_blocks(
        attention_inputs: PyAttentionInputs,
        tag: str,
        logical_pages: torch.Tensor,
        pool_rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve request-local logical pages through a HybridPool block table."""
        table = getattr(attention_inputs, "kv_cache_block_id_device", None)
        if table is None or table.numel() == 0:
            table = getattr(attention_inputs, "kv_cache_block_id", None)
        if table is None or table.numel() == 0 or table.dim() != 2:
            raise RuntimeError(f"PLE cache {tag!r} needs a 2-D physical block table")
        if table.dtype != torch.int32:
            raise RuntimeError(
                f"PLE cache {tag!r} physical block table must be int32, "
                f"got {table.dtype}"
            )
        if logical_pages.dim() != 1 or int(table.shape[0]) != int(
            logical_pages.numel()
        ):
            raise RuntimeError(f"PLE cache {tag!r} block-table batch is inconsistent")
        pages = logical_pages.to(device=table.device, dtype=torch.long)
        if not pages.numel() or bool((pages < 0).any().item()):
            raise RuntimeError(f"PLE cache {tag!r} has an invalid logical page")
        if bool((pages >= int(table.shape[1])).any().item()):
            raise RuntimeError(
                f"PLE cache {tag!r} logical page exceeds its block table"
            )
        blocks = table.gather(1, pages.unsqueeze(1)).squeeze(1).to(torch.long)
        if bool((blocks <= 0).any().item()):
            raise RuntimeError(f"PLE cache {tag!r} has an unallocated logical page")
        if bool((blocks >= pool_rows).any().item()):
            raise RuntimeError(f"PLE cache {tag!r} physical block id exceeds its pool")
        return blocks.to(device=device)

    def _validate_ple_mode(
        self,
        attention_inputs: PyAttentionInputs,
        *,
        allow_target_verify: bool = False,
    ) -> None:
        is_target_verify = bool(getattr(attention_inputs, "is_target_verify", False))
        if is_target_verify and not allow_target_verify:
            raise RuntimeError("qwen4_exp PLE does not support target-verify/MTP yet")
        if getattr(attention_inputs, "is_cuda_graph", False):
            raise RuntimeError("qwen4_exp PLE does not support CUDA Graph yet")
        if getattr(attention_inputs, "is_s_padded", False):
            raise RuntimeError("qwen4_exp PLE does not support padded execution yet")
        if getattr(attention_inputs, "context_parallel_info", None) is not None:
            raise RuntimeError("qwen4_exp PLE does not support context parallelism yet")
        cp = self.parallelism_config.prefill_cp_config
        if cp.is_enabled() or cp.is_prefill_enabled():
            raise RuntimeError("qwen4_exp PLE does not support prefill CP yet")
        if getattr(attention_inputs, "cache_store_inputs", None) is not None:
            raise RuntimeError("qwen4_exp PLE does not support PD cache-store yet")
        prefixes = getattr(attention_inputs, "prefix_lengths", None)
        if (
            not is_target_verify
            and prefixes is not None
            and prefixes.numel()
            and bool((prefixes != 0).any().item())
        ):
            raise RuntimeError("qwen4_exp PLE does not support prefix reuse yet")

    @staticmethod
    def _same_metadata(
        lhs: PyAttentionInputs, rhs: PyAttentionInputs, name: str
    ) -> bool:
        left = getattr(lhs, name, None)
        right = getattr(rhs, name, None)
        if left is None or right is None:
            return left is right
        return (
            tuple(left.shape) == tuple(right.shape)
            and left.dtype == right.dtype
            and left.device == right.device
            and bool(torch.equal(left, right))
        )

    def _stage_ple_target_verify(
        self,
        *,
        layer_idx: int,
        ple: Qwen4ExpPLELayer,
        hyper_states: torch.Tensor,
        ids: torch.Tensor,
        state_inputs: PyAttentionInputs,
        ctx_inputs: PyAttentionInputs,
        state_pool: torch.Tensor,
        ctx_pool: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        """Compute all target rows while leaving the persistent PLE pools intact."""
        if not bool(state_inputs.is_prefill):
            raise RuntimeError(
                "qwen4_exp PLE target verification must use context-style prefill"
            )
        lengths = state_inputs.input_lengths
        prefixes = state_inputs.prefix_lengths
        sequence_lengths = state_inputs.sequence_lengths
        if (
            lengths.dim() != 1
            or lengths.dtype not in (torch.int32, torch.int64)
            or not int(lengths.numel())
        ):
            raise RuntimeError(
                "qwen4_exp PLE target verify requires integer input_lengths [B]"
            )
        batch_size = int(lengths.numel())
        query_len = int(lengths[0].item())
        if query_len <= 0 or bool(torch.any(lengths != query_len).item()):
            raise RuntimeError(
                "qwen4_exp PLE target verify requires one uniform gamma+1 width"
            )
        if sequence_lengths.numel() != 0:
            raise RuntimeError(
                "qwen4_exp PLE target verify requires empty sequence_lengths"
            )
        if (
            prefixes.dim() != 1
            or prefixes.dtype != torch.int32
            or int(prefixes.numel()) != batch_size
        ):
            raise RuntimeError(
                "qwen4_exp PLE target verify requires int32 prefix_lengths [B]"
            )
        prefixes_device = prefixes.to(
            device=hyper_states.device, dtype=torch.long, non_blocking=True
        ).contiguous()
        if bool(torch.any(prefixes_device <= 0).item()):
            raise RuntimeError(
                "qwen4_exp PLE target verify requires a non-empty committed history"
            )
        if int(ids.numel()) != batch_size * query_len:
            raise RuntimeError(
                "qwen4_exp PLE target verify input lengths do not partition "
                f"the {ids.numel()} packed ids"
            )
        if (
            state_pool.device != hyper_states.device
            or ctx_pool.device != hyper_states.device
        ):
            raise RuntimeError(
                "qwen4_exp PLE target projection and side pools must share a device"
            )

        read_pages = torch.div(prefixes_device - 1, page_size, rounding_mode="floor")
        state_read_blocks = self._physical_blocks(
            state_inputs,
            PLE_STATE_TAG,
            read_pages,
            int(state_pool.shape[0]),
            state_pool.device,
        )
        ctx_read_blocks = self._physical_blocks(
            ctx_inputs,
            PLE_NGRAM_CTX_TAG,
            read_pages,
            int(ctx_pool.shape[0]),
            ctx_pool.device,
        )
        initial_state = state_pool.index_select(0, state_read_blocks)
        initial_context = ctx_pool.index_select(0, ctx_read_blocks)
        candidate_ids = ids.reshape(batch_size, query_len)
        history = torch.cat([initial_context, candidate_ids], dim=1)
        output, candidate_state_inputs = ple.decode_chunk(
            hyper_states.reshape(batch_size, query_len, -1),
            history,
            initial_state,
        )

        stage = _PLETargetLayerStage(
            layer_idx=layer_idx,
            page_size=page_size,
            prefixes=prefixes_device,
            state_pool=state_pool,
            ctx_pool=ctx_pool,
            state_inputs=state_inputs,
            ctx_inputs=ctx_inputs,
            initial_state=initial_state,
            candidate_state_inputs=candidate_state_inputs,
            initial_context=initial_context,
            candidate_ids=candidate_ids,
        )
        transaction = getattr(self, "_ple_target_transaction", None)
        if transaction is None:
            expected_layers = frozenset(int(key) for key in self.ple_layers.keys())
            transaction = _PLETargetTransaction(
                batch_size=batch_size,
                query_len=query_len,
                prefixes=prefixes_device,
                expected_layers=expected_layers,
                stages={},
            )
            self._ple_target_transaction = transaction
        else:
            if (
                transaction.prepared_writes is not None
                or transaction.commit_started
                or transaction.tentative_committed
            ):
                raise RuntimeError("qwen4_exp PLE target transaction was not finalized")
            if (
                transaction.batch_size != batch_size
                or transaction.query_len != query_len
                or transaction.prefixes.device != prefixes_device.device
                or not bool(torch.equal(transaction.prefixes, prefixes_device))
            ):
                raise RuntimeError(
                    "qwen4_exp PLE layers disagree about target-verify geometry"
                )
        if layer_idx not in transaction.expected_layers:
            raise RuntimeError(
                f"qwen4_exp PLE target transaction received unexpected layer {layer_idx}"
            )
        if layer_idx in transaction.stages:
            raise RuntimeError(
                f"qwen4_exp PLE target layer {layer_idx} was staged twice"
            )
        transaction.stages[layer_idx] = stage
        return hyper_states + output.reshape_as(hyper_states)

    @staticmethod
    def _synchronize_ple_transaction(transaction: _PLETargetTransaction) -> None:
        if transaction.prefixes.is_cuda:
            torch.cuda.synchronize(transaction.prefixes.device)

    def prepare_speculative_target_commit(self, accept_len: torch.Tensor) -> None:
        """Select accepted PLE snapshots and prepare undo without writing pools."""
        transaction = getattr(self, "_ple_target_transaction", None)
        if transaction is None:
            if len(getattr(self, "ple_layers", ())):
                raise RuntimeError(
                    "qwen4_exp PLE target commit has no staged transaction"
                )
            # QSA-only or PLE-disabled qwen4 targets need no PLE side-state commit.
            return
        if (
            transaction.prepared_writes is not None
            or transaction.commit_started
            or transaction.tentative_committed
        ):
            raise RuntimeError("qwen4_exp PLE target transaction is already prepared")
        if set(transaction.stages) != set(transaction.expected_layers):
            missing = sorted(transaction.expected_layers.difference(transaction.stages))
            extra = sorted(
                set(transaction.stages).difference(transaction.expected_layers)
            )
            raise RuntimeError(
                "qwen4_exp PLE target transaction has incomplete layer coverage: "
                f"missing={missing}, extra={extra}"
            )
        if (
            not isinstance(accept_len, torch.Tensor)
            or accept_len.dim() != 1
            or accept_len.dtype != torch.int32
            or int(accept_len.numel()) != transaction.batch_size
            or accept_len.device != transaction.prefixes.device
        ):
            raise RuntimeError(
                "qwen4_exp PLE target commit requires device-local int32 accept_len [B]"
            )
        if bool(
            torch.any((accept_len < 1) | (accept_len > transaction.query_len)).item()
        ):
            raise RuntimeError(
                "qwen4_exp PLE target accept_len must be in "
                f"[1, {transaction.query_len}]"
            )

        accepted = accept_len.to(torch.long)
        pending = []
        destination_rows: set[tuple[int, int]] = set()
        for layer_idx in sorted(transaction.stages):
            stage = transaction.stages[layer_idx]
            state_len = int(stage.initial_state.shape[1])
            state_timeline = torch.cat(
                [stage.initial_state, stage.candidate_state_inputs], dim=1
            )
            state_indices = accepted.unsqueeze(1) + torch.arange(
                state_len, device=accepted.device, dtype=torch.long
            ).unsqueeze(0)
            state_values = state_timeline.gather(
                1,
                state_indices.unsqueeze(-1).expand(
                    -1, -1, int(state_timeline.shape[-1])
                ),
            )

            context_len = int(stage.initial_context.shape[1])
            context_timeline = torch.cat(
                [stage.initial_context, stage.candidate_ids], dim=1
            )
            context_indices = accepted.unsqueeze(1) + torch.arange(
                context_len, device=accepted.device, dtype=torch.long
            ).unsqueeze(0)
            context_values = context_timeline.gather(1, context_indices)

            final_positions = stage.prefixes + accepted - 1
            final_pages = torch.div(
                final_positions, stage.page_size, rounding_mode="floor"
            )
            state_blocks = self._physical_blocks(
                stage.state_inputs,
                PLE_STATE_TAG,
                final_pages,
                int(stage.state_pool.shape[0]),
                stage.state_pool.device,
            )
            ctx_blocks = self._physical_blocks(
                stage.ctx_inputs,
                PLE_NGRAM_CTX_TAG,
                final_pages,
                int(stage.ctx_pool.shape[0]),
                stage.ctx_pool.device,
            )
            pending.append(
                (
                    stage,
                    state_blocks,
                    state_values,
                    ctx_blocks,
                    context_values,
                )
            )

            for pool, blocks, tag in (
                (stage.state_pool, state_blocks, PLE_STATE_TAG),
                (stage.ctx_pool, ctx_blocks, PLE_NGRAM_CTX_TAG),
            ):
                pool_ptr = int(pool.data_ptr())
                for block in blocks.tolist():
                    row = (pool_ptr, int(block))
                    if row in destination_rows:
                        raise RuntimeError(
                            f"qwen4_exp PLE cache {tag!r} has a duplicate target "
                            "destination physical row"
                        )
                    destination_rows.add(row)

        # Every layer, destination and selected value has been validated. Only
        # now allocate the immutable undo snapshots; this phase still performs
        # no persistent-pool writes.
        prepared_writes = []
        for stage, state_blocks, state_values, ctx_blocks, context_values in pending:
            prepared_writes.append(
                _PLEPreparedWrite(
                    state_pool=stage.state_pool,
                    state_blocks=state_blocks,
                    state_values=state_values,
                    original_state=stage.state_pool.index_select(
                        0, state_blocks
                    ).clone(),
                    ctx_pool=stage.ctx_pool,
                    ctx_blocks=ctx_blocks,
                    ctx_values=context_values,
                    original_context=stage.ctx_pool.index_select(0, ctx_blocks).clone(),
                )
            )
        # Gather/clone launches above are part of prepare, not commit.  Complete
        # them before publishing the plan so the C++ prepare consensus observes
        # any asynchronous failure on this rank before persistent writes begin.
        self._synchronize_ple_transaction(transaction)
        transaction.prepared_writes = prepared_writes

    def finish_speculative_target_commit(self, commit: bool) -> None:
        """Tentatively apply a prepared PLE plan or idempotently roll it back."""
        transaction = getattr(self, "_ple_target_transaction", None)
        if transaction is None:
            return
        if not commit:
            if transaction.commit_started:
                assert transaction.prepared_writes is not None
                for write in transaction.prepared_writes:
                    write.state_pool.index_copy_(
                        0, write.state_blocks, write.original_state
                    )
                    write.ctx_pool.index_copy_(
                        0, write.ctx_blocks, write.original_context
                    )
            # Also synchronize an uncommitted transaction: target staging and a
            # failed prepare may have queued work even though no pool write was
            # launched.  Do not clear its lifetime until that work is settled.
            self._synchronize_ple_transaction(transaction)
            self._ple_target_transaction = None
            return

        if transaction.prepared_writes is None:
            raise RuntimeError("qwen4_exp PLE target transaction was not prepared")
        if transaction.commit_started:
            raise RuntimeError("qwen4_exp PLE target transaction was already committed")
        # Mark before the first copy so finish(false) restores every destination
        # even if a later launch or synchronization reports an error.
        transaction.commit_started = True
        for write in transaction.prepared_writes:
            write.state_pool.index_copy_(0, write.state_blocks, write.state_values)
            write.ctx_pool.index_copy_(0, write.ctx_blocks, write.ctx_values)
        self._synchronize_ple_transaction(transaction)
        transaction.tentative_committed = True

    def finalize_speculative_target_commit(self) -> None:
        """Release undo state after C++ confirms every TP rank committed."""
        transaction = getattr(self, "_ple_target_transaction", None)
        if transaction is None:
            return
        if not transaction.tentative_committed:
            raise RuntimeError(
                "qwen4_exp PLE target transaction cannot finalize before commit"
            )
        self._ple_target_transaction = None

    def _apply_ple(
        self,
        layer_idx: int,
        hyper_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_inputs: Any,
    ) -> torch.Tensor:
        ple = self.ple_layers[str(layer_idx)]
        state_inputs = select_attention_inputs_for_tag(attention_inputs, PLE_STATE_TAG)
        ctx_inputs = select_attention_inputs_for_tag(
            attention_inputs, PLE_NGRAM_CTX_TAG
        )
        self._validate_ple_mode(state_inputs, allow_target_verify=True)
        self._validate_ple_mode(ctx_inputs, allow_target_verify=True)
        is_target_verify = bool(state_inputs.is_target_verify)
        if (
            is_target_verify != bool(ctx_inputs.is_target_verify)
            or bool(state_inputs.is_prefill) != bool(ctx_inputs.is_prefill)
            or any(
                not self._same_metadata(state_inputs, ctx_inputs, name)
                for name in (
                    "input_lengths",
                    "prefix_lengths",
                    "sequence_lengths",
                    "cu_seqlens_device",
                    "cu_kv_seqlens_device",
                    "combo_position_ids",
                )
            )
        ):
            raise RuntimeError("qwen4_exp PLE state-region metadata is inconsistent")
        state_cache = self.kv_cache.get_layer_cache(layer_idx, PLE_STATE_TAG)
        ctx_cache = self.kv_cache.get_layer_cache(layer_idx, PLE_NGRAM_CTX_TAG)
        if state_cache.seq_size_per_block != ctx_cache.seq_size_per_block:
            raise RuntimeError("qwen4_exp PLE state-region page sizes are inconsistent")
        if hyper_states.dtype != torch.bfloat16:
            raise RuntimeError(
                "qwen4_exp PLE currently requires BF16 activations, "
                f"got {hyper_states.dtype}"
            )
        state_pool = self._fixed_state_pool(
            state_cache,
            torch.bfloat16,
            ple.short_conv_state_len,
            ple.hc_hidden_size,
        )
        ctx_pool = self._fixed_state_pool(
            ctx_cache, torch.int64, ple.ple_embedding.context_len, 1
        ).squeeze(-1)
        ids = input_ids.reshape(-1).to(device=hyper_states.device, dtype=torch.long)
        if hyper_states.dim() != 2 or int(hyper_states.shape[0]) != int(ids.numel()):
            raise RuntimeError("qwen4_exp PLE expects packed [tokens, hc_hidden] input")
        eos = ple.ple_embedding.eos_token_id
        context_len = ple.ple_embedding.context_len
        page_size = int(state_cache.seq_size_per_block)
        if page_size <= 0:
            raise RuntimeError("qwen4_exp PLE cache page size must be positive")

        if is_target_verify:
            return self._stage_ple_target_verify(
                layer_idx=layer_idx,
                ple=ple,
                hyper_states=hyper_states,
                ids=ids,
                state_inputs=state_inputs,
                ctx_inputs=ctx_inputs,
                state_pool=state_pool,
                ctx_pool=ctx_pool,
                page_size=page_size,
            )

        if state_inputs.is_prefill:
            lengths = [int(x) for x in state_inputs.input_lengths.tolist()]
            if any(length <= 0 for length in lengths):
                raise RuntimeError("qwen4_exp PLE requires positive prefill lengths")
            if sum(lengths) != int(ids.numel()):
                raise RuntimeError(
                    "qwen4_exp PLE packed prefill lengths are inconsistent"
                )
            terminal_pages = torch.tensor(
                [(length - 1) // page_size for length in lengths],
                dtype=torch.long,
            )
            state_blocks = self._physical_blocks(
                state_inputs,
                PLE_STATE_TAG,
                terminal_pages,
                int(state_pool.shape[0]),
                state_pool.device,
            )
            ctx_blocks = self._physical_blocks(
                ctx_inputs,
                PLE_NGRAM_CTX_TAG,
                terminal_pages,
                int(ctx_pool.shape[0]),
                ctx_pool.device,
            )
            outputs, states, contexts = [], [], []
            offset = 0
            for length in lengths:
                seq_ids = ids[offset : offset + length].view(1, length)
                history = torch.cat(
                    [seq_ids.new_full((1, context_len), eos), seq_ids], dim=1
                )
                output, state = ple.prefill(
                    hyper_states[offset : offset + length].view(1, length, -1),
                    history,
                )
                outputs.append(output.squeeze(0))
                states.append(state.squeeze(0))
                contexts.append(history[0, -context_len:])
                offset += length
            state_pool.index_copy_(
                0, state_blocks, torch.stack(states).to(state_pool.dtype)
            )
            ctx_pool.index_copy_(
                0, ctx_blocks, torch.stack(contexts).to(ctx_pool.dtype)
            )
            return hyper_states + torch.cat(outputs, dim=0)

        lengths = state_inputs.input_lengths
        sequence_lengths = state_inputs.sequence_lengths
        batch = int(sequence_lengths.numel())
        if batch == 0 or int(ids.numel()) != batch or int(lengths.numel()) != batch:
            raise RuntimeError("qwen4_exp PLE supports single-token decode only")
        if sequence_lengths.dim() != 1 or sequence_lengths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise RuntimeError(
                "qwen4_exp PLE decode sequence_lengths must be a 1-D integer tensor"
            )
        sequence_lengths = sequence_lengths.to(dtype=torch.long)
        if bool((sequence_lengths <= 0).any().item()):
            raise RuntimeError(
                "qwen4_exp PLE decode requires a non-empty prefill history"
            )
        read_pages = torch.div(sequence_lengths - 1, page_size, rounding_mode="floor")
        write_pages = torch.div(sequence_lengths, page_size, rounding_mode="floor")
        state_read_blocks = self._physical_blocks(
            state_inputs,
            PLE_STATE_TAG,
            read_pages,
            int(state_pool.shape[0]),
            state_pool.device,
        )
        ctx_read_blocks = self._physical_blocks(
            ctx_inputs,
            PLE_NGRAM_CTX_TAG,
            read_pages,
            int(ctx_pool.shape[0]),
            ctx_pool.device,
        )
        state_write_blocks = self._physical_blocks(
            state_inputs,
            PLE_STATE_TAG,
            write_pages,
            int(state_pool.shape[0]),
            state_pool.device,
        )
        ctx_write_blocks = self._physical_blocks(
            ctx_inputs,
            PLE_NGRAM_CTX_TAG,
            write_pages,
            int(ctx_pool.shape[0]),
            ctx_pool.device,
        )
        conv_buffer = state_pool.index_select(0, state_read_blocks)
        context = ctx_pool.index_select(0, ctx_read_blocks)
        history = torch.cat([context, ids.view(batch, 1)], dim=1)
        output, new_state = ple.decode_step(
            hyper_states.view(batch, 1, -1), history, conv_buffer
        )
        state_pool.index_copy_(0, state_write_blocks, new_state.to(state_pool.dtype))
        ctx_pool.index_copy_(
            0,
            ctx_write_blocks,
            history[:, -context_len:].to(ctx_pool.dtype),
        )
        return hyper_states + output.squeeze(1)

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
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            weights.get_global_weight(W.qwen4_hc_mixer_norm),
            weights.get_global_weight(W.qwen4_hc_mixer_mix_down),
            weights.get_global_weight(W.qwen4_hc_mixer_mix_up),
            None,
            hc_mult=model_config.hc_mult,
            norm_eps=model_config.layernorm_eps,
        )

    def _ple_input_ids(self, inputs: PyModelInputs) -> torch.Tensor:
        """Return text ids while rejecting unverified multimodal PLE semantics."""
        input_ids = inputs.input_ids
        embedding_inputs = getattr(inputs, "embedding_inputs", None)
        text_mask = getattr(embedding_inputs, "text_tokens_mask", None)
        if text_mask is not None and text_mask.numel():
            if text_mask.numel() != input_ids.numel():
                raise RuntimeError(
                    "qwen4_exp PLE text token mask size does not match input ids: "
                    f"{text_mask.numel()} != {input_ids.numel()}"
                )
            raise RuntimeError(
                "qwen4_exp PLE does not support multimodal inputs yet; "
                "text_tokens_mask must be absent"
            )
        multimodal_inputs = getattr(inputs, "multimodal_inputs", None)
        features = getattr(multimodal_inputs, "multimodal_features", None)
        if features is not None and len(features):
            raise RuntimeError("qwen4_exp PLE does not support multimodal inputs yet")
        return input_ids

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if (
            self.ple_layers
            and getattr(self, "_ple_target_transaction", None) is not None
        ):
            raise RuntimeError(
                "qwen4_exp PLE target transaction must be finalized or rolled "
                "back before the next forward"
            )
        hidden_states = self.word_embedding(inputs)

        is_cuda_graph = _is_cuda_graph_forward(inputs, fmha_impl)
        attn_meta = self._build_attn_meta(inputs, hidden_states.device, is_cuda_graph)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        # The target tiles token embeddings here.  The MTP draft overrides the
        # hook because its input projection has already produced one distinct
        # hidden vector per hyper-connection branch.
        hyper_states = self._initial_hyper_states(hidden_states)

        attention_inputs = get_attention_inputs_value(inputs)
        ple_input_ids = inputs.input_ids
        if self.ple_layers:
            ple_input_ids = self._ple_input_ids(inputs)
        # Temporary numerical diagnostics: per-layer residual-stream norms.
        # Enabled only when RTP_LLM_QWEN4_LAYER_STATS is set; logs the first
        # few forwards so layer-by-layer magnitudes can be inspected.
        stats_enabled = os.environ.get(
            "RTP_LLM_QWEN4_LAYER_STATS", ""
        ).strip().lower() in ("1", "true", "yes")
        stats_forward = getattr(self, "_layer_stats_calls", 0)
        stats_this_forward = stats_enabled and stats_forward < 3
        if stats_this_forward:
            self._layer_stats_calls = stats_forward + 1
            logging.warning(
                "[qwen4-layer-stats] forward=%d embed_stream=%.4f",
                stats_forward,
                float(hyper_states.float().norm()),
            )
        for i, decoder_layer in enumerate(self.layers):
            layer_in_norm = None
            if stats_this_forward:
                layer_in_norm = float(hyper_states.float().norm())
            if str(i) in self.ple_layers:
                if self.kv_cache is None:
                    raise RuntimeError(
                        "qwen4_exp PLE requires its two state cache regions"
                    )
                before_ple = hyper_states
                hyper_states = self._apply_ple(
                    i, hyper_states, ple_input_ids, attention_inputs
                )
                if stats_this_forward:
                    logging.warning(
                        "[qwen4-layer-stats] forward=%d layer=%d ple_delta=%.4f",
                        stats_forward,
                        i,
                        float((hyper_states - before_ple).float().norm()),
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
            hyper_states = decoder_layer(
                hyper_states,
                layer_fmha_impl,
                kv_cache=layer_kv_cache,
                attention_inputs=layer_attention_inputs,
                attn_meta=attn_meta,
                qsa_runtime=qsa_runtime,
            )
            if layer_in_norm is not None:
                logging.warning(
                    "[qwen4-layer-stats] forward=%d layer=%d in=%.4f out=%.4f",
                    stats_forward,
                    i,
                    layer_in_norm,
                    float(hyper_states.float().norm()),
                )

        if getattr(self, "_capture_mtp_target_hidden", False):
            # Keep the exact pre-collapse tensor alive until MtpExecutor asks
            # for it immediately after this forward.  Qwen4 currently rejects
            # speculative CP and CUDA Graph, so a dynamic tensor is sufficient
            # and does not advertise the fixed-buffer CP capability.
            self._mtp_target_hidden_states = hyper_states
        else:
            self._mtp_target_hidden_states = None
        hidden_states, _, _ = self.hyper_connection_mixer(hyper_states)
        if stats_this_forward:
            logging.warning(
                "[qwen4-layer-stats] forward=%d stream=%.4f final=%.4f",
                stats_forward,
                float(hyper_states.float().norm()),
                float(hidden_states.float().norm()),
            )
        return PyModelOutputs(hidden_states)
