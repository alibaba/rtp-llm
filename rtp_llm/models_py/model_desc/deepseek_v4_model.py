"""DeepSeek-V4 integration into RTP-LLM's `GptModelBase` framework.

Shape mirrors `qwen3.py`: the Model owns weights (via ``self.v4``, which
holds embed/layers/norm/head) and dispatches prefill / decode. The
per-layer loops themselves live in sibling packages:

  * :mod:`rtp_llm.models_py.modules.dsv4.prefill` — prefill forward,
    block-table builder, CP metadata binding, PD cache_store writer
  * :mod:`rtp_llm.models_py.modules.dsv4.decode.forward` — decode
    forward, eager metadata builder, paged pool spec derivation

``DeepSeekV4Model`` is a thin orchestrator: ``forward`` dispatches
prefill/decode and handles the per-request split; everything else is
a free-function call into one of those two packages.

  - `forward(PyModelInputs) -> PyModelOutputs` returns [total_tokens, hidden_dim]
    pre-lm-head hidden states; engine applies lm_head + sampling externally.
  - KV pools are the framework BlockPools (``self.kv_cache``). The handle is
    read on every forward and threaded as ``kv_cache=`` to each layer; per-
    request block tables come from ``attn_inputs.kv_cache_kernel_block_id_device_by_group``
    via ``kv_cache_utils.build_block_tables``.

Weight loading: `_initialize_impl` reads `self.weight` (a `ModelWeights`
populated by the framework's fastsafetensors loader via
`DeepSeekV4Weight._get_hf_layer_weight_info`) and hands it directly to
`V4Transformer(args, mw=self.weight)`. Each dsv4 sub-module's factory
mode reads tensors from `mw.global_weights[W.*]` and
`mw.weights[layer_id][W.v4_*]` (W tag enum, no string keys).
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv4.chunk_env import (
    DSV4_CHUNK_TOKENS_ENV,
    dsv4_global_chunk_tokens_configured,
)
from rtp_llm.models_py.modules.dsv4.decode.forward import (
    build_paged_pool_specs,
    forward_decode,
)
from rtp_llm.models_py.modules.dsv4.moe.moe_layer import (
    chunked_moe_enabled,
    cp_padded_tokens_per_rank_bound,
    moe_chunk_tokens_from_env,
    resolve_moe_max_tokens_per_rank,
)
from rtp_llm.models_py.modules.dsv4.prefill.forward import forward_prefill
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer


def _materialize_meta_buffers(module: torch.nn.Module, device: str) -> int:
    """Walk the module tree and replace any meta-device buffer with a
    zeroed real-device tensor of the same shape/dtype. Factory-mode
    construction builds V4Transformer under `torch.device("meta")` to
    skip placeholder allocations; this pass re-materializes the
    non-parameter buffers (kv_cache, kv_state, score_state, etc.) so
    forward() can read and write them."""
    count = 0
    for mod in module.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is None:
                continue
            if buf.device.type == "meta":
                mod._buffers[name] = torch.zeros(
                    buf.shape,
                    dtype=buf.dtype,
                    device=device,
                )
                count += 1
    return count


from rtp_llm.ops.compute_ops import PyModelInitResources, PyModelInputs, PyModelOutputs


def _is_decode_fmha(fmha_impl: Any) -> bool:
    """True when ``fmha_impl`` is one of the dsv4 captured decode impls.
    Imports are lazy so missing FP8 builds don't bring the module down."""
    if fmha_impl is None:
        return False
    from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
        DSv4DecodeFmhaImpl,
    )

    decode_types: tuple = (DSv4DecodeFmhaImpl,)
    try:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )

        decode_types = (DSv4DecodeFmhaImpl, DSv4DecodeFmhaImplFP8)
    except ImportError:
        pass
    return isinstance(fmha_impl, decode_types)


@dataclass(frozen=True)
class Dsv4MtpHiddenBufferSpec:
    token_capacity: int
    hc_dim: int


class Dsv4SharedRuntimeBufferStore:

    _instance: Optional["Dsv4SharedRuntimeBufferStore"] = None
    _mtp_hidden_requested = False

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        mtp_hidden_enabled: bool,
        mtp_hidden: Optional[Dsv4MtpHiddenBufferSpec],
    ) -> None:
        device = torch.device(device)
        self._mtp_hidden_enabled = bool(mtp_hidden_enabled)
        self._mtp_hidden_token_capacity = (
            int(mtp_hidden.token_capacity) if mtp_hidden is not None else 0
        )
        self._mtp_hidden_hc_dim = (
            int(mtp_hidden.hc_dim) if mtp_hidden is not None else 0
        )
        self._mtp_hidden_storage = (
            torch.empty(
                self._mtp_hidden_token_capacity,
                self._mtp_hidden_hc_dim,
                dtype=dtype,
                device=device,
            )
            if self._mtp_hidden_enabled
            else None
        )
        self._subscribers: list[torch.nn.Module] = []

    @classmethod
    def instance(cls) -> "Dsv4SharedRuntimeBufferStore":
        assert cls._instance is not None, "Dsv4SharedRuntimeBufferStore is not bound"
        return cls._instance

    @classmethod
    def _reset_for_test(cls) -> None:
        cls._instance = None
        cls._mtp_hidden_requested = False

    @classmethod
    def mtp_hidden_requested(cls) -> bool:
        return cls._mtp_hidden_requested or (
            cls._instance is not None and cls._instance.mtp_hidden_enabled
        )

    @classmethod
    def enable_mtp_hidden(cls) -> None:
        if cls._instance is not None and not cls._instance.mtp_hidden_enabled:
            raise RuntimeError(
                "Dsv4SharedRuntimeBufferStore: cannot enable MTP hidden buffer "
                "after shared storage has been allocated"
            )
        cls._mtp_hidden_requested = True

    @classmethod
    def get_or_create(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        mtp_hidden: Optional[Dsv4MtpHiddenBufferSpec] = None,
    ) -> "Dsv4SharedRuntimeBufferStore":
        if cls._instance is None:
            cls._instance = cls(
                device=device,
                dtype=dtype,
                mtp_hidden_enabled=cls._mtp_hidden_requested,
                mtp_hidden=mtp_hidden,
            )
        else:
            cls._instance._validate_request(
                mtp_hidden_token_capacity=(
                    int(mtp_hidden.token_capacity) if mtp_hidden is not None else 0
                ),
            )
        return cls._instance

    @property
    def mtp_hidden_enabled(self) -> bool:
        return self._mtp_hidden_enabled

    def bind(
        self,
        module: torch.nn.Module,
    ) -> Optional[torch.Tensor]:
        if not any(existing is module for existing in self._subscribers):
            self._subscribers.append(module)
        self._sync_subscribers()
        return self._views()

    def _validate_request(
        self,
        *,
        mtp_hidden_token_capacity: int,
    ) -> None:
        if mtp_hidden_token_capacity > self._mtp_hidden_token_capacity:
            raise RuntimeError(
                "Dsv4SharedRuntimeBufferStore: cannot grow MTP hidden capacity "
                f"from {self._mtp_hidden_token_capacity} to "
                f"{mtp_hidden_token_capacity} after first bind"
            )

    def _views(self) -> Optional[torch.Tensor]:
        return self._mtp_hidden_storage

    def _sync_subscribers(self) -> None:
        mtp_hidden_buffer = self._views()
        for module in self._subscribers:
            module._bind_runtime_buffers(mtp_hidden_buffer)


def _args_from_model_config(
    model_config: ModelConfig, max_generate_batch_size: int = 4
) -> V4Args:
    from rtp_llm.ops import KvCacheDataType

    attn_config = model_config.attn_config
    rope_config = attn_config.rope_config
    fp8_kv_cache = attn_config.kv_cache_dtype == KvCacheDataType.FP8
    return V4Args(
        vocab_size=model_config.vocab_size,
        dim=model_config.hidden_size,
        n_heads=attn_config.head_num,
        n_layers=model_config.num_layers,
        n_mtp_layers=0,  # MTP wiring lands later; main-layer inference first.
        q_lora_rank=attn_config.q_lora_rank,
        head_dim=attn_config.size_per_head,
        rope_head_dim=attn_config.rope_head_dim,
        o_groups=attn_config.o_groups,
        o_lora_rank=attn_config.o_lora_rank,
        window_size=attn_config.sliding_window,
        compress_ratios=list(attn_config.layer_compress_ratios)[
            : model_config.num_layers
        ],
        rope_theta=float(rope_config.base),
        compress_rope_theta=float(attn_config.compress_rope_theta),
        rope_factor=float(rope_config.scale) if rope_config.scale else 1.0,
        beta_fast=int(rope_config.factor2) if rope_config.factor2 else 32,
        beta_slow=int(rope_config.factor1) if rope_config.factor1 else 1,
        original_seq_len=int(rope_config.max_pos) if rope_config.max_pos else 0,
        index_n_heads=attn_config.indexer_head_num,
        index_head_dim=attn_config.indexer_head_dim,
        index_topk=attn_config.indexer_topk,
        moe_inter_dim=(
            model_config.inter_size // max(1, model_config.moe_k or 1)
            if False
            else model_config.inter_size // 1
        ),
        n_routed_experts=model_config.expert_num,
        n_shared_experts=1,
        n_activated_experts=model_config.moe_k,
        score_func={0: "softmax", 1: "sigmoid", 2: "sqrtsoftplus"}[
            model_config.scoring_func
        ],
        route_scale=float(model_config.routed_scaling_factor),
        swiglu_limit=float(model_config.swiglu_limit),
        n_hash_layers=int(model_config.num_hash_layers),
        hc_mult=int(model_config.hc_mult),
        hc_sinkhorn_iters=int(model_config.hc_sinkhorn_iters),
        hc_eps=float(model_config.hc_eps),
        norm_eps=float(model_config.layernorm_eps),
        max_batch_size=max_generate_batch_size,  # from framework, supports concurrent requests
        max_seq_len=int(model_config.max_seq_len) or 4096,
        # Mega MoE sizes its symm-mem dispatch buffer from this bound.
        # max_seq_len is the safest per-rank upper bound (one long prefill
        # fully on one rank) — the buffer is allocated once and reused.
        max_tokens_per_rank=int(model_config.max_seq_len) or 4096,
        fp8_kv_cache=fp8_kv_cache,
    )


class DeepSeekV4Model(GptModelBase):
    """Framework-facing model: owns a V4Transformer, feeds framework IO into it."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        # Build V4Transformer with matching args.
        args = _args_from_model_config(model_config, max_generate_batch_size)
        self._max_generate_batch_size = int(max_generate_batch_size)
        assert self._max_generate_batch_size > 0, (
            "max_generate_batch_size must be positive, "
            f"got {self._max_generate_batch_size}"
        )
        self._gen_num_per_cycle = int(model_config.gen_num_per_cycle)
        # MoE inter dim from V4 config: explicit (not inter_size which in RTP-LLM
        # is n_shared_experts * moe_intermediate_size for DeepSeek). Use moe_config
        # if available; else read from config's hidden_size-derived fallback.
        if (
            moe_config is not None
            and getattr(moe_config, "moe_inter_padding_size", 0) > 0
        ):
            args.moe_inter_dim = int(moe_config.moe_inter_padding_size)
        else:
            # V4-Flash = 2048. config.inter_size = n_shared * 2048 = 2048 (since n_shared=1).
            args.moe_inter_dim = int(model_config.inter_size) or args.moe_inter_dim

        # S7 scaffold: thread the framework's parallelism config into V4Args.
        # No behavior change at TP=1; the fields are read by future patches
        # (see docs/dsv4/parallel_design.md) when sharding lands per-module.
        #
        # When CP is enabled (``prefill_cp_config.is_enabled()``), RTP-LLM
        # REPURPOSES the TP process group as the CP group — every attn
        # module must see ``tp_size = 1`` so it does NOT shard heads, and
        # the sequence-axis work instead runs via CP.  The helper
        # ``ParallelismConfig.get_attn_tp_size()`` already returns 1 in
        # that case; use it instead of raw ``tp_size``.
        if parallelism_config is not None:
            if hasattr(parallelism_config, "get_attn_tp_size"):
                args.tp_size = int(parallelism_config.get_attn_tp_size() or 1)
                args.tp_rank = int(parallelism_config.get_attn_tp_rank() or 0)
            else:
                args.tp_size = int(getattr(parallelism_config, "tp_size", 1) or 1)
                args.tp_rank = int(getattr(parallelism_config, "tp_rank", 0) or 0)
            args.ep_size = int(getattr(parallelism_config, "ep_size", 1) or 1)
            args.ep_rank = int(getattr(parallelism_config, "ep_rank", 0) or 0)
            args.dp_size = int(getattr(parallelism_config, "dp_size", 1) or 1)
            args.dp_rank = int(getattr(parallelism_config, "dp_rank", 0) or 0)
            args.world_size = int(getattr(parallelism_config, "world_size", 1) or 1)
            args.world_rank = int(getattr(parallelism_config, "world_rank", 0) or 0)
        if args.world_size > 1:
            logging.info(
                "[DeepSeekV4Model] parallelism: world=%d tp=%d/%d ep=%d/%d dp=%d/%d "
                "(scaffold only — V4 sharding not yet implemented; see "
                "docs/dsv4/parallel_design.md)",
                args.world_size,
                args.tp_rank,
                args.tp_size,
                args.ep_rank,
                args.ep_size,
                args.dp_rank,
                args.dp_size,
            )

        # CP-aware Mega MoE buffer sizing.  CP first reduces the rank-local
        # sequence bound, then chunked MoE caps the per-forward routed/shared
        # expert workspace to a scheduler-style token chunk: allocate by max
        # batched/chunked tokens, not by the full 1M context length.
        cp_size = 1
        if (
            parallelism_config is not None
            and getattr(parallelism_config, "prefill_cp_config", None) is not None
        ):
            try:
                if parallelism_config.prefill_cp_config.is_enabled():
                    cp_size = int(getattr(parallelism_config, "tp_size", 1) or 1)
            except Exception:  # pyi-only stub or non-CP build
                pass
        if cp_size > 1:
            cp_tokens_per_rank_bound = min(
                args.max_tokens_per_rank,
                max(
                    cp_padded_tokens_per_rank_bound(args.max_seq_len, cp_size),
                    4096,
                ),
            )
            if cp_tokens_per_rank_bound != args.max_tokens_per_rank:
                logging.info(
                    "[DeepSeekV4Model] CP=%d: max_tokens_per_rank %d -> %d "
                    "(Mega MoE per-rank symm-mem buffer)",
                    cp_size,
                    args.max_tokens_per_rank,
                    cp_tokens_per_rank_bound,
                )
                args.max_tokens_per_rank = cp_tokens_per_rank_bound
        self._prefill_cp_size = int(cp_size)
        self._is_speculative = False
        self._is_decode_role = False
        self._shared_runtime_buffers: Optional[Dsv4SharedRuntimeBufferStore] = None

        logging.info(
            "[DeepSeekV4Model] V4Args: n_layers=%d n_heads=%d head_dim=%d q_lora=%d "
            "o_groups=%d n_experts=%d n_act=%d moe_inter=%d win=%d hc_mult=%d "
            "compress_ratios[:8]=%s score=%s route_scale=%g swiglu_limit=%g",
            args.n_layers,
            args.n_heads,
            args.head_dim,
            args.q_lora_rank,
            args.o_groups,
            args.n_routed_experts,
            args.n_activated_experts,
            args.moe_inter_dim,
            args.window_size,
            args.hc_mult,
            list(args.compress_ratios)[:8],
            args.score_func,
            args.route_scale,
            args.swiglu_limit,
        )
        self._v4_args = args
        # Surface FP8 KV-cache flag at the Model level so
        # ``prepare_fmha_impl`` can dispatch BF16 / FP8 decode FMHA impls
        # without re-reading attn_config.
        self.fp8_kv_cache = bool(args.fp8_kv_cache)

        # Defer V4Transformer construction to `initialize()` where each
        # dsv4 sub-module reads its weights directly from the framework's
        # ``ModelWeights`` via ``mw.global_weights[W.*]`` /
        # ``mw.weights[layer_id][W.v4_*]``.
        self.v4: Optional[V4Transformer] = None

        self._materialized = False
        self._ckpt_path: str = model_config.ckpt_path

        # Optional on-demand timeline capture. Set DSV4_PROFILE_TRACE=/path/trace.json
        # and touch /tmp/dsv4_profile_trigger to capture the NEXT forward only.
        self._profile_path = os.environ.get("DSV4_PROFILE_TRACE")
        self._profile_trigger = "/tmp/dsv4_profile_trigger"
        self._profile_done = False
        if self._profile_path:
            logging.info(
                "[DeepSeekV4Model] timeline trigger: `touch %s` — "
                "next forward captures to %s",
                self._profile_trigger,
                self._profile_path,
            )

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        try:
            return self._initialize_impl(init_resource)
        except BaseException as e:
            import traceback

            logging.error(
                "[DeepSeekV4Model] initialize() raised %s: %s\nTraceback:\n%s",
                type(e).__name__,
                e,
                traceback.format_exc(),
            )
            raise

    def _resolve_shared_token_capacity(self) -> int:
        if self._is_decode_role:
            return resolve_moe_max_tokens_per_rank(
                max_seq_len=int(self._v4_args.max_seq_len),
                current_max_tokens_per_rank=int(self._v4_args.max_tokens_per_rank),
                cp_size=1,
                max_generate_batch_size=int(self._max_generate_batch_size),
                is_decode_role=True,
                is_speculative=self._is_speculative,
                gen_num_per_cycle=self._gen_num_per_cycle,
            )
        cp_size = int(self._prefill_cp_size)
        if cp_size > 1:
            return (
                cp_padded_tokens_per_rank_bound(int(self._v4_args.max_seq_len), cp_size)
                * self._max_context_batch_size
            )
        return self._v4_args.max_seq_len * self._max_context_batch_size

    def _resolve_mtp_hidden_token_capacity(self) -> int:
        return self._resolve_shared_token_capacity()

    def _resolve_prefill_q_token_capacity(self) -> int:
        return self._resolve_shared_token_capacity()

    def _resolve_mtp_last_hidden_token_capacity(self) -> Optional[int]:
        return None

    def _resolve_prefill_ws_gather_widths(self) -> Tuple[int, int, int]:
        """Per-row element counts for the three concurrent CP gather roles —
        main CSA/HCA compressor, nested indexer compressor, SWA ``kv_full`` —
        sized off ``V4Args`` STATIC dims, NOT the runtime layer compositions.

        Under the overlap orchestrator + the SWA side stream, up to three
        gathers can be in flight concurrently within one layer, so each role
        owns a dedicated workspace sub-region (a shared one would alias).

        Widths are the protocol-level UPPER BOUND for each role, independent
        of how many CSA/HCA layers the current model happens to instantiate:
          * main: ``2 * 2 * args.head_dim`` (the widest fused projection from
            ``CompressorFP8.start_prefill`` — CSA uses ``coff=2``, HCA uses
            ``coff=1``; we always pre-reserve the CSA size). fp32 elements
            (compressor uses fp32 fused gather).
          * indexer: ``2 * 2 * args.index_head_dim`` (nested indexer
            compressor is CSA-only → ``coff=2``). fp32 elements.
          * swa: ``args.head_dim`` (the KV per-head dim seen after
            ``fused_rmsnorm_rope``; see ``kv_full.reshape(-1, self.head_dim)``
            in ``AttentionFP8``). bf16 elements (SWA's only gather dtype).

        Whether a model actually USES a role on some layer is irrelevant for
        sizing — the union buffer is ``max(q_bytes, 2*main+2*idx+2*swa)`` and
        q dominates in practice, so over-reserving costs nothing while keeping
        the union BYTE-IDENTICAL across the main forward and the MTP draft
        forward. That identity is what lets the caching allocator hand the
        same block back to the draft at the main→draft boundary — the whole
        reason the per-forward workspace exists. Doing it any other way would
        re-introduce the allocator fragmentation we built this to avoid.
        """
        head_dim = int(self._v4_args.head_dim)
        index_head_dim = int(self._v4_args.index_head_dim)
        main_w = 2 * 2 * head_dim
        idx_w = 2 * 2 * index_head_dim
        swa_w = head_dim
        return main_w, idx_w, swa_w

    def _bind_runtime_buffers(self, device: torch.device) -> None:
        assert self.v4 is not None
        mtp_hidden = None
        mtp_last_hidden_capacity = None
        if Dsv4SharedRuntimeBufferStore.mtp_hidden_requested():
            mtp_hidden = Dsv4MtpHiddenBufferSpec(
                token_capacity=self._resolve_mtp_hidden_token_capacity(),
                hc_dim=int(self._v4_args.hc_mult) * int(self._v4_args.dim),
            )
            if self._is_speculative:
                mtp_last_hidden_capacity = (
                    self._resolve_mtp_last_hidden_token_capacity()
                )

        # Per-forward prefill workspace dims (max-sized; consumed by
        # ``forward_layers`` to build a per-forward ``PrefillWorkspace``). These
        # are plain ints — config, not buffers — so they carry no cross-forward
        # lifetime. CP gather/restore region is sized only when CP is active.
        cp_size = int(self._prefill_cp_size)
        q_rows = int(self._resolve_prefill_q_token_capacity())
        q_dim = int(self._v4_args.n_heads) * int(self._v4_args.head_dim)
        if cp_size > 1:
            full_rows = q_rows * cp_size
            main_w, idx_w, swa_w = self._resolve_prefill_ws_gather_widths()
        else:
            full_rows = 0
            main_w = 0
            idx_w = 0
            swa_w = 0
        self.v4._bind_prefill_workspace_dims(
            q_rows, q_dim, full_rows, main_w, idx_w, swa_w
        )

        self._shared_runtime_buffers = Dsv4SharedRuntimeBufferStore.get_or_create(
            device=device,
            dtype=torch.bfloat16,
            mtp_hidden=mtp_hidden,
        )
        self._shared_runtime_buffers.bind(self.v4)

        if mtp_last_hidden_capacity is not None:
            self.v4._allocate_mtp_last_hidden_buffer(
                device,
                mtp_last_hidden_capacity,
            )

    def _initialize_impl(self, init_resource: PyModelInitResources) -> bool:
        # Called by the engine after construction and before forward.
        super().initialize(init_resource)
        if self._materialized:
            return True

        device = (
            next(iter(self.weight.global_weights.values())).device
            if self.weight.global_weights
            else "cuda:0"
        )
        device_str = str(device)

        self._is_speculative = bool(init_resource.is_speculative)
        self._is_decode_role = bool(init_resource.is_decode_role)
        self._max_context_batch_size = init_resource.max_context_batch_size
        self._v4_args.is_decode_role = self._is_decode_role
        runtime_resolved_max_tokens_per_rank = resolve_moe_max_tokens_per_rank(
            max_seq_len=int(self._v4_args.max_seq_len),
            current_max_tokens_per_rank=int(self._v4_args.max_tokens_per_rank),
            cp_size=1,
            max_generate_batch_size=int(self._max_generate_batch_size),
            is_decode_role=self._is_decode_role,
            is_speculative=self._is_speculative,
            gen_num_per_cycle=self._gen_num_per_cycle,
        )
        if runtime_resolved_max_tokens_per_rank != self._v4_args.max_tokens_per_rank:
            chunk_tokens_env_for_log = (
                DSV4_CHUNK_TOKENS_ENV
                if dsv4_global_chunk_tokens_configured()
                else "DSV4_MOE_CHUNK_TOKENS"
            )
            chunk_tokens_for_log = -1
            if not self._is_decode_role and (
                dsv4_global_chunk_tokens_configured()
                or os.environ.get("DSV4_MOE_CHUNK_PREFILL", "1") != "0"
            ):
                chunk_tokens_for_log = moe_chunk_tokens_from_env()
            logging.info(
                "[DeepSeekV4Model] runtime MoE token budget: "
                "max_tokens_per_rank %d -> %d (%s=%d, role=%s, "
                "speculative=%s, gen_num_per_cycle=%d)",
                self._v4_args.max_tokens_per_rank,
                runtime_resolved_max_tokens_per_rank,
                chunk_tokens_env_for_log,
                chunk_tokens_for_log,
                "DECODE" if self._is_decode_role else "PREFILL",
                self._is_speculative,
                self._gen_num_per_cycle,
            )
            self._v4_args.max_tokens_per_rank = runtime_resolved_max_tokens_per_rank

        # ``self.weight`` is a framework ``ModelWeights`` populated by the
        # ``DeepSeekV4Weight`` descriptor (see ``rtp_llm/models/deepseek_v4.py``)
        # via the fastsafetensors loader.  Each dsv4 sub-module's factory
        # mode reads tensors directly from ``mw.global_weights[W.*]`` and
        # ``mw.weights[layer_id][W.v4_*]`` — see the per-module __init__
        # for the W tags consumed.
        logging.info(
            "[DeepSeekV4Model] building V4Transformer via factory "
            "(layers=%d, globals=%d) ...",
            len(self.weight.weights),
            len(self.weight.global_weights),
        )
        # Construct under meta device so the nn.Embedding / nn.Linear /
        # QuantizedLinear placeholder allocations inside the module tree
        # skip real RAM. Each module's factory branch then reassigns
        # `.weight`/`.scale` to the cuda tensors from the weights dict.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                self.v4 = V4Transformer(self._v4_args, mw=self.weight)
        finally:
            torch.set_default_dtype(prev_dtype)

        # Recompute RoPE cache on real device (precompute_freqs_cis under
        # meta context yields zeros; we need real values).
        for layer in self.v4.layers:
            layer.attn.reset_rope_cache(device=device_str)

        # Subclass hook: lift any model-level weights (e.g. MTP fusion
        # norms / projections) off the ModelWeights wrapper before we
        # discard it.  Default impl is a no-op.
        self._load_extra_weights(self.weight)

        # Drop the ModelWeights wrapper — per-tensor refs are now held
        # only by the V4Transformer modules (or were popped during
        # MoE._setup_mega_moe / per-expert routed init).
        del self.weight

        if torch.cuda.is_available() and device_str.startswith("cuda"):
            gpu_mem_gb = torch.cuda.memory_allocated(torch.device(device_str)) / 1024**3
            logging.info("[DeepSeekV4Model] GPU mem after load: %.1f GB", gpu_mem_gb)

        # Pre-warm the TileLang sparse_attn kernel before the C++ engine
        # creates CudaGraphRunner.  The first call triggers JIT compilation;
        # doing it here (outside graph capture) caches the compiled kernel so
        # subsequent calls inside CUDA graph capture hit the cache and skip
        # JIT — which would otherwise abort via __unexpected (noexcept violation).
        if device_str.startswith("cuda"):
            from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _tl_kernels

            first_attn = self.v4.layers[0].attn
            _tl_kernels.prewarm(
                first_attn.n_heads,
                first_attn.head_dim,
                first_attn.softmax_scale,
                device_str,
            )

            # Pre-warm the v4_indexer_score Triton kernel for the SAME
            # constexpr config the live decode (and CSA prefill) call will
            # use, so JIT compile never happens inside CUDA graph capture
            # (which silently corrupts the captured graph and kills the
            # rank). The kernel keys on (S, T, H, D, BLOCK_S, BLOCK_T,
            # COMPRESS_RATIO, APPLY_MASK) — match every one of those.
            #
            # Decode call: q_len=S=1, T = max_seq_len // ratio, no mask.
            # CSA prefill (also captured if cuda-graph prefill ever lands):
            # same (H, D, ratio, T) but with mask. We compile both APPLY_MASK
            # variants here.
            import torch as _torch

            from rtp_llm.models_py.modules.dsv4._indexer_score_triton import (
                v4_indexer_score as _v4_idx,
            )
            from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
                _run_triton_warmup_launch_with_retry,
            )

            if len(self.v4.layers) > 2:
                _idx = self.v4.layers[2].attn.indexer  # first CSA layer (ratio=4)
                _H = int(_idx.n_heads)
                _D = int(_idx.head_dim)
                _ratio = max(int(_idx.compress_ratio), 1)
                _T_dec = int(self._v4_args.max_seq_len) // _ratio
                _q_dec = _torch.zeros(
                    (1, 1, _H, _D), dtype=_torch.bfloat16, device=device_str
                )
                _kv_dec = _torch.zeros(
                    (1, _T_dec, _D), dtype=_torch.bfloat16, device=device_str
                )
                _w_dec = _torch.zeros(
                    (1, 1, _H), dtype=_torch.bfloat16, device=device_str
                )
                # Decode shape (no mask).
                _run_triton_warmup_launch_with_retry(
                    "DSV4IndexerScore",
                    f"decode shape S=1 T={_T_dec} H={_H} D={_D}",
                    lambda: _v4_idx(
                        _q_dec,
                        _kv_dec,
                        _w_dec,
                        q_pos=None,
                        compress_ratio=1,
                    ),
                    device=_torch.device(device_str),
                )
                # Prefill mask variant — only matters if CSA prefill goes through
                # graph capture. Cheap to prewarm regardless.
                _q_pos_dec = _torch.zeros((1, 1), dtype=_torch.int32, device=device_str)
                _run_triton_warmup_launch_with_retry(
                    "DSV4IndexerScore",
                    f"prefill-mask shape S=1 T={_T_dec} H={_H} D={_D} ratio={_ratio}",
                    lambda: _v4_idx(
                        _q_dec,
                        _kv_dec,
                        _w_dec,
                        q_pos=_q_pos_dec,
                        compress_ratio=_ratio,
                    ),
                    device=_torch.device(device_str),
                )

            if os.environ.get("DSV4_PREWARM_FLASH_MLA_SWA", "1") != "0":
                try:
                    from flash_mla import flash_mla_sparse_fwd as _flash_mla_sparse_fwd

                    _swa_attn = self.v4.layers[0].attn
                    _H_swa = int(_swa_attn.n_heads)
                    _D_swa = int(_swa_attn.head_dim)
                    _W_swa = int(_swa_attn.window_size)
                    _q_swa = _torch.zeros(
                        (2, _H_swa, _D_swa), dtype=_torch.bfloat16, device=device_str
                    )
                    _kv_swa = _torch.zeros(
                        (5, 1, _D_swa), dtype=_torch.bfloat16, device=device_str
                    )
                    _idx_swa = _torch.full(
                        (2, 1, _W_swa), -1, dtype=_torch.int32, device=device_str
                    )
                    _cp_rank_swa = 0
                    try:
                        import torch.distributed as _dist

                        if _dist.is_available() and _dist.is_initialized():
                            _cp_rank_swa = int(_dist.get_rank())
                    except Exception:
                        _cp_rank_swa = 0
                    _first_topk_len_swa = min(_cp_rank_swa + 1, 5, _W_swa)
                    _idx_swa[0, 0, :_first_topk_len_swa] = _torch.arange(
                        _first_topk_len_swa, dtype=_torch.int32, device=device_str
                    )
                    _idx_swa[1, 0, :5] = _torch.arange(
                        5, dtype=_torch.int32, device=device_str
                    )
                    _topk_len_swa = _torch.tensor(
                        [_first_topk_len_swa, 5], dtype=_torch.int32, device=device_str
                    )
                    _flash_mla_sparse_fwd(
                        q=_q_swa,
                        kv=_kv_swa,
                        indices=_idx_swa,
                        sm_scale=float(_swa_attn.softmax_scale),
                        attn_sink=_swa_attn.attn_sink,
                        topk_length=_topk_len_swa,
                    )
                    logging.info("[DeepSeekV4Model] flash_mla SWA kv_full prewarm done")
                except Exception:
                    logging.exception(
                        "[DeepSeekV4Model] flash_mla SWA kv_full prewarm failed"
                    )
                    raise

            if os.environ.get("DSV4_PREWARM_MEGA_MOE", "1") != "0":
                try:
                    import torch.distributed as _dist

                    _strategy = self.v4.layers[0].ffn._strategy
                    if getattr(_strategy, "name", "") == "mega":
                        if _dist.is_available() and _dist.is_initialized():
                            _dist.barrier()
                        _cfg = _strategy.cfg
                        _x_moe = _torch.zeros(
                            (1, int(_cfg.dim)), dtype=_torch.bfloat16, device=device_str
                        )
                        _w_moe = _torch.zeros(
                            (1, int(_cfg.n_activated_experts)),
                            dtype=_torch.float32,
                            device=device_str,
                        )
                        _w_moe[:, 0] = 1.0
                        _start = int(_cfg.local_expert_start)
                        _end = max(int(_cfg.local_expert_end), _start + 1)
                        _idx_vals = (
                            _torch.arange(
                                int(_cfg.n_activated_experts),
                                dtype=_torch.int64,
                                device=device_str,
                            )
                            % (_end - _start)
                        ) + _start
                        _idx_moe = _idx_vals.unsqueeze(0).contiguous()
                        _strategy(_x_moe, _w_moe, _idx_moe)
                        _torch.cuda.synchronize()
                        if _dist.is_available() and _dist.is_initialized():
                            _dist.barrier()
                        logging.info("[DeepSeekV4Model] MegaMoE prewarm done")
                except Exception:
                    logging.exception("[DeepSeekV4Model] MegaMoE prewarm failed")
                    raise

            try:
                from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
                    _collect_dsv4_batched_fp8_einsum_shapes,
                    _collect_dsv4_dense_gemm_shapes,
                    _collect_dsv4_fp8_mqa_logits_shapes,
                    _collect_dsv4_mhc_head_fused_shapes,
                    _collect_dsv4_mhc_prenorm_shapes,
                    resolve_dense_gemm_warmup_max_m,
                    warmup_batched_fp8_einsum_jit,
                    warmup_compressor_combine_branch_kernels,
                    warmup_dense_gemm_jit,
                    warmup_dsv4_fp8_swa_slot_dequant_jit,
                    warmup_fp8_mqa_logits_jit,
                    warmup_mhc_head_fused_jit,
                    warmup_mhc_prenorm_gemm_jit,
                )

                _jit_device = _torch.device(device_str)
                _fixed_region_cp_size = 1
                _fixed_region_prefill_sliced = False
                _fixed_region_cp_config = getattr(
                    self.parallelism_config, "prefill_cp_config", None
                )
                if _fixed_region_cp_config is not None and bool(
                    getattr(_fixed_region_cp_config, "kv_cache_sharded", False)
                ):
                    if self._is_decode_role:
                        try:
                            _decode_prefill_cp = bool(
                                _fixed_region_cp_config.is_prefill_enabled()
                            )
                        except Exception:
                            _decode_prefill_cp = False
                        if _decode_prefill_cp:
                            _fixed_region_cp_size = max(
                                int(
                                    getattr(
                                        _fixed_region_cp_config, "prefill_cp_size", 0
                                    )
                                    or 1
                                ),
                                1,
                            )
                    else:
                        _fixed_region_cp_size = max(
                            int(getattr(self.parallelism_config, "tp_size", 1) or 1), 1
                        )
                        _fixed_region_prefill_sliced = _fixed_region_cp_size > 1
                warmup_compressor_combine_branch_kernels(
                    v4=self.v4,
                    v4_args=self._v4_args,
                    device=_jit_device,
                    gen_num_per_cycle=self._gen_num_per_cycle,
                    fixed_region_cp_size=_fixed_region_cp_size,
                    fixed_region_prefill_sliced=_fixed_region_prefill_sliced,
                )
                _dense_shapes = _collect_dsv4_dense_gemm_shapes(self)
                _dense_gemm_prefill_chunk_size = 0
                if not self._is_decode_role and chunked_moe_enabled():
                    _dense_gemm_prefill_chunk_size = max(
                        int(moe_chunk_tokens_from_env()), 0
                    )
                _prefill_cp_config = getattr(
                    self.parallelism_config, "prefill_cp_config", None
                )
                _prefill_cp_enabled = False
                if _prefill_cp_config is not None:
                    try:
                        _prefill_cp_enabled = bool(_prefill_cp_config.is_enabled())
                    except Exception:
                        _prefill_cp_enabled = False
                _prefill_kv_cache_sharded = bool(
                    getattr(_prefill_cp_config, "kv_cache_sharded", False)
                )
                _prefill_cp_size = (
                    int(getattr(self.parallelism_config, "tp_size", 1) or 1)
                    if _prefill_cp_enabled
                    else 1
                )
                _dense_gemm_max_m = resolve_dense_gemm_warmup_max_m(
                    max_seq_len=int(self._v4_args.max_seq_len),
                    max_batch_size=int(self._v4_args.max_batch_size),
                    role_type_name="DECODE" if self._is_decode_role else "PREFILL",
                    prefill_chunk_size=_dense_gemm_prefill_chunk_size,
                    max_tokens_per_rank=int(self._v4_args.max_tokens_per_rank),
                    is_speculative=self._is_speculative,
                    gen_num_per_cycle=self._gen_num_per_cycle,
                    cp_size=_prefill_cp_size,
                    cp_enabled=_prefill_cp_enabled,
                )
                logging.info(
                    "[DeepSeekV4Model] DenseGEMM JIT warmup max_m=%d "
                    "prefill_chunk_size=%d max_tokens_per_rank=%d "
                    "role=%s cp_enabled=%s cp_size=%d",
                    _dense_gemm_max_m,
                    _dense_gemm_prefill_chunk_size,
                    int(self._v4_args.max_tokens_per_rank),
                    "DECODE" if self._is_decode_role else "PREFILL",
                    _prefill_cp_enabled,
                    _prefill_cp_size,
                )
                warmup_dense_gemm_jit(
                    _dense_shapes,
                    max_m=_dense_gemm_max_m,
                    device=_jit_device,
                )
                _batched_fp8_einsum_shapes = _collect_dsv4_batched_fp8_einsum_shapes(
                    self.v4
                )
                warmup_batched_fp8_einsum_jit(
                    _batched_fp8_einsum_shapes,
                    max_m=_dense_gemm_max_m,
                    device=_jit_device,
                )
                _mhc_prenorm_shapes = _collect_dsv4_mhc_prenorm_shapes(self.v4)
                warmup_mhc_prenorm_gemm_jit(
                    _mhc_prenorm_shapes,
                    max_m=_dense_gemm_max_m,
                    device=_jit_device,
                )
                _mhc_head_fused_shapes = _collect_dsv4_mhc_head_fused_shapes(self.v4)
                warmup_mhc_head_fused_jit(
                    _mhc_head_fused_shapes,
                    device=_jit_device,
                )
                if (
                    self.fp8_kv_cache
                    and not self._is_decode_role
                    and _prefill_cp_enabled
                    and _prefill_cp_size > 1
                    and _prefill_kv_cache_sharded
                ):
                    warmup_dsv4_fp8_swa_slot_dequant_jit(
                        kv_cache=self.kv_cache,
                        cp_size=_prefill_cp_size,
                        device=_jit_device,
                    )
                _fp8_mqa_logits_shapes = _collect_dsv4_fp8_mqa_logits_shapes(self.v4)
                warmup_fp8_mqa_logits_jit(
                    _fp8_mqa_logits_shapes,
                    device=_jit_device,
                )
                logging.info("[DeepSeekV4Model] kernel JIT prewarm done")
            except Exception:
                logging.exception("[DeepSeekV4Model] kernel JIT prewarm failed")
                raise
            _torch.cuda.synchronize()

        self._bind_runtime_buffers(torch.device(device_str))
        logging.info(
            "[DeepSeekV4Model] bound runtime buffers: prefill_ws_q_tokens=%d "
            "(per-forward) mtp_hidden_enabled=%s",
            self._resolve_prefill_q_token_capacity(),
            self._shared_runtime_buffers.mtp_hidden_enabled,
        )

        self._materialized = True

        return True

    def _should_capture_cuda_graph(self, attn: Any, is_target_verify: bool) -> bool:
        """Default: capture decode + verify, skip plain prefill.  MTP
        overrides to capture all cudagraph requests because the draft's
        post-verify multi-token batch arrives with ``is_prefill=True``
        but is functionally a multi-token decode."""
        return not (bool(attn.is_prefill) and not is_target_verify)

    def _load_extra_weights(self, weights: ModelWeights) -> None:
        """Subclass hook for loading model-level (non-Block) tensors off
        the framework's ``ModelWeights`` before it gets dropped.  Default
        no-op; ``DeepSeekV4MtpModel`` overrides to bind enorm/hnorm/
        e_proj/h_proj from the MTP-only globals."""
        return None

    def _prepare_decode_hidden(
        self,
        input_ids: torch.Tensor,
        meta: Any,
    ) -> torch.Tensor:
        """Build the per-token ``[B, q_len, hc, dim]`` hidden tensor that
        feeds the layer loop on the decode path.  Default impl is the
        regular embed+repeat that the main model uses; ``DeepSeekV4MtpModel``
        overrides with the e_proj/h_proj fusion stage."""
        B = meta.batch_size
        q_len = meta.q_len_per_req
        h = self.v4.embed(input_ids).view(B, q_len, -1)
        return h.unsqueeze(2).repeat(1, 1, self.v4.hc_mult, 1)

    def _prepare_prefill_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Build the flat ``[T_total, hc, dim]`` hidden tensor that feeds
        the layer loop on the prefill path.  Default = embed+repeat."""
        h = self.v4.embed(input_ids)
        return h.unsqueeze(-2).repeat(1, self.v4.hc_mult, 1)

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        """Return a ``DSv4DecodeFmhaImpl`` for decode CUDA-graph capture; None otherwise.

        Prefill runs eagerly (no graph). Decode uses its own sparse/compressed
        attention; the impl owns persistent metadata buffers updated in place
        by ``prepare_cuda_graph`` between replays."""
        if not is_cuda_graph:
            return None

        attn = inputs.attention_inputs
        # V4 captures CUDA graphs for decode AND target verify — but not
        # plain prefill. Verify carries ``is_prefill==True`` (C++ MtpExecutor
        # routes verify-split through PyWrappedModel's prefill path), so
        # rejecting on ``is_prefill`` alone would also reject verify.
        # Reject only when prefill AND not verify.  Subclasses (MTP) may
        # opt in to all-cudagraph capture by overriding
        # ``_should_capture_cuda_graph``.
        if attn is None:
            return None
        is_target_verify = bool(getattr(attn, "is_target_verify", False))
        if not self._should_capture_cuda_graph(attn, is_target_verify):
            return None

        if self.kv_cache is None:
            logging.warning(
                "[DeepSeekV4Model] prepare_fmha_impl: kv_cache not yet allocated "
                "(warmup phase); skipping CUDA-graph impl creation."
            )
            return None

        if is_target_verify and not self.fp8_kv_cache:
            # Per REFORMAT_FINAL.md A2: BF16 verify intentionally unsupported
            # in this scope (BF16 decode attention still has q_len==1
            # assumptions). The eager path's assert is the load-bearing one;
            # under cudagraph we just refuse the impl.
            return None

        if self.fp8_kv_cache:
            from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
                DSv4DecodeFmhaImplConfigFP8 as _DecodeFmhaImplConfig,
            )
            from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
                DSv4DecodeFmhaImplFP8 as _DecodeFmhaImpl,
            )
        else:
            from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
                DSv4DecodeFmhaImpl as _DecodeFmhaImpl,
            )
            from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
                DSv4DecodeFmhaImplConfig as _DecodeFmhaImplConfig,
            )

        batch_size = (
            int(attn.input_lengths.size(0)) if attn.input_lengths.numel() > 0 else 1
        )
        # Per-graph q_len comes from ``input_lengths[0]`` whenever the
        # batch carries it. Three flows pass through here:
        #   * main decode capture: ``input_lengths=[1,1,…]`` → q_len=1
        #   * main verify capture: ``input_lengths=[gen+1]`` → q_len=gen+1
        #   * MTP draft capture (single-token / multi-token):
        #     ``input_lengths=[1]`` or ``[gen+1]`` respectively
        # The C++ CudaGraphRunner captures one graph per (batch, q_len).
        q_len = int(attn.input_lengths[0]) if attn.input_lengths.numel() > 0 else 1
        device = self.v4.embed.weight.device

        paged_pool_specs = build_paged_pool_specs(
            self.kv_cache, self.v4, max_seq_len=int(self._v4_args.max_seq_len)
        )
        # Snapshot framework's group ordering — CUDA-graph replay path
        # inside the impl's ``prepare`` has no live kv_cache, so carry
        # the list in the config. Position IS the group id.
        group_region_names_snapshot = (
            [int(t) for t in (self.kv_cache.group_region_names or [])]
            if self.kv_cache is not None
            else []
        )

        if self.kv_cache is None:
            raise RuntimeError(
                "DSV4 prepare_decode_metadata: self.kv_cache is None; "
                "C++ KVCacheManager must propagate KVCache before forward."
            )
        cfg_kwargs = dict(
            max_batch_size=batch_size,
            q_len=q_len,
            window_size=int(self._v4_args.window_size),
            head_dim=int(self._v4_args.head_dim),
            max_seq_len=int(self._v4_args.max_seq_len),
            compress_ratios=list(self._v4_args.compress_ratios)[
                : self._v4_args.n_layers
            ],
            index_topk=int(self._v4_args.index_topk),
            paged_pool_specs=paged_pool_specs,
            group_region_names=group_region_names_snapshot,
        )
        cfg = _DecodeFmhaImplConfig(**cfg_kwargs)
        impl = _DecodeFmhaImpl(
            cfg,
            device=device,
            attn_inputs=attn,
        )
        # Phase F: pool views resolved on demand in Attention — no
        # per-layer descriptor cache to stash on metadata.
        return impl

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        """
        num_tokens >= 0: return the explicit row count without reading
            _mtp_hidden_valid_tokens. This is CUDA-graph safe because replay does
            not update Python attributes.
        num_tokens < 0: return the last non-graph-written row count. This is only
            for CP prefill, where the C++ global token count has been restored
            but the buffer intentionally stores rank-local rows.
        """
        if self.v4 is None:
            raise RuntimeError("DeepSeekV4Model: v4 transformer not initialized")
        buf = self.v4._mtp_hidden_buffer
        if buf is None:
            return None
        requested = int(num_tokens)
        if requested < 0:
            assert (
                not self._is_decode_role
            ), "decode MTP hidden reads must pass row count"
            requested = int(self.v4._mtp_hidden_valid_tokens)
            assert requested > 0, "MTP hidden buffer has no written rows"
        assert requested <= buf.size(0), (
            "DeepSeekV4Model: requested MTP hidden states exceed buffer capacity: "
            f"requested={requested}, capacity={buf.size(0)}"
        )
        return buf[:requested]

    def get_mtp_last_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        if self.v4 is None:
            raise RuntimeError("DeepSeekV4Model: v4 transformer not initialized")
        assert not self._is_decode_role, "decode MTP last-hidden reads are unsupported"
        buf = self.v4._mtp_last_hidden_buffer
        if buf is None:
            return None
        requested = int(num_tokens)
        if requested < 0:
            requested = int(self.v4._mtp_last_hidden_valid_tokens)
        assert requested <= buf.size(0), (
            "DeepSeekV4Model: requested MTP last hidden states exceed buffer capacity: "
            f"requested={requested}, capacity={buf.size(0)}"
        )
        assert requested <= int(self.v4._mtp_last_hidden_valid_tokens), (
            "DeepSeekV4Model: requested MTP last hidden states exceed rows written "
            f"by the previous forward: requested={requested}, "
            f"valid={self.v4._mtp_last_hidden_valid_tokens}"
        )
        return buf[:requested]

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        """qwen3-style dispatcher — per-arm orchestration lives in the
        prefill / decode runtime modules.

        ``self.kv_cache`` may be None during warmup (NormalExecutor builds
        the PyWrappedModel with cache_manager==nullptr); only the prefill
        path needs to tolerate this — warmup never enters decode.
        """
        if self.kv_cache is None:
            # Warmup-only PyWrappedModel: NormalExecutor builds it with
            # cache_manager==nullptr, so init_resources carries no kv_cache.
            logging.warning(
                "[DeepSeekV4Model] forward() with kv_cache=None — warmup only"
            )
            T = max(inputs.input_ids.numel(), 1)
            device = self.v4.embed.weight.device
            return PyModelOutputs(
                torch.zeros(T, self._v4_args.dim, dtype=torch.bfloat16, device=device)
            )
        attn = inputs.attention_inputs

        # Subclass-overridable hidden-state preparation hooks.  When a
        # subclass (e.g. ``DeepSeekV4MtpModel``) overrides
        # ``_prepare_decode_hidden`` / ``_prepare_prefill_hidden``, the
        # forward helpers splice the override in front of the layer
        # loop while leaving the rest of the body untouched.  Defaults
        # do exactly the embed+expand the main path always did.
        cls = type(self)
        prep_decode = (
            self._prepare_decode_hidden
            if cls._prepare_decode_hidden is not DeepSeekV4Model._prepare_decode_hidden
            else None
        )
        prep_prefill = (
            self._prepare_prefill_hidden
            if cls._prepare_prefill_hidden
            is not DeepSeekV4Model._prepare_prefill_hidden
            else None
        )

        if _is_decode_fmha(fmha_impl) or bool(getattr(attn, "is_target_verify", False)):
            if bool(getattr(attn, "is_target_verify", False)):
                assert bool(
                    getattr(self.v4, "fp8_kv_cache", False)
                ), "target verify requires fp8 kv cache"
            return forward_decode(
                self.v4,
                self.kv_cache,
                self._v4_args,
                inputs,
                fmha_impl,
                prepare_hidden_fn=prep_decode,
            )
        elif attn.is_prefill:
            return forward_prefill(
                self.v4,
                self.kv_cache,
                self.parallelism_config,
                inputs,
                prepare_hidden_fn=prep_prefill,
            )
        else:
            return forward_decode(
                self.v4,
                self.kv_cache,
                self._v4_args,
                inputs,
                fmha_impl,
                prepare_hidden_fn=prep_decode,
            )
