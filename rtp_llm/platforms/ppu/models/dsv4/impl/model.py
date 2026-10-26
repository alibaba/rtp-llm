"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.model_desc.deepseek_v4_model import (
    _materialize_meta_buffers,
    _is_decode_fmha,
    Dsv4MtpHiddenBufferSpec,
    Dsv4SharedRuntimeBufferStore,
    _resolve_dsv4_moe_strategy,
    _resolve_dsv4_moe_inter_dim,
)
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
from rtp_llm.config.cuda_graph import (
    CudaGraphSelectionMode,
    GenerationPrefillCudaGraphUnsupportedBackend,
)
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.platforms.ppu.models.dsv4.impl.chunk_env import (
    DSV4_CHUNK_TOKENS_ENV,
    chunked_moe_enabled,
    dsv4_global_chunk_tokens_configured,
    moe_chunk_tokens_from_env,
)
from rtp_llm.platforms.ppu.models.dsv4.impl.decode.forward import (
    build_paged_pool_specs,
    forward_decode,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import primary_attention_inputs
from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import (
    Dsv4ProviderCapability,
    build_dsv4_decode_metadata,
    resolve_dsv4_platform_provider,
)
from rtp_llm.platforms.ppu.models.dsv4.impl.prefill.forward import forward_prefill
from rtp_llm.platforms.ppu.models.dsv4.impl.transformer import V4Args, V4Transformer
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    cp_padded_tokens_per_rank_bound,
    resolve_moe_max_tokens_per_rank,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.warmup import model_warm_up_enabled
from rtp_llm.ops.compute_ops import PyModelInitResources, PyModelInputs, PyModelOutputs
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model as _BaseDeepSeekV4Model

def _env_flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")

def _args_from_model_config(
    model_config: ModelConfig,
    max_generate_batch_size: int = 4,
    moe_config=None,
) -> V4Args:
    from rtp_llm.ops import KvCacheDataType, RoleType

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
        kernel_tokens_per_block=attn_config.kernel_tokens_per_block,
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
        moe_inter_dim=_resolve_dsv4_moe_inter_dim(model_config),
        n_routed_experts=model_config.expert_num,
        n_physical_experts=int(
            model_config.eplb_config.phy_exp_num(model_config.expert_num)
        ),
        n_shared_experts=int(model_config.n_shared_experts),
        moe_strategy=_resolve_dsv4_moe_strategy(moe_config),
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
        # A scheduler-admitted prefill batch can exceed one request's max_seq_len.
        # CP, decode and chunk limits are applied after this initial batch bound.
        max_tokens_per_rank=int(
            model_config.moe_prefill_max_tokens_per_rank
            if model_config.moe_prefill_max_tokens_per_rank is not None
            else (int(model_config.max_seq_len) or 4096)
        ),
        fp8_kv_cache=fp8_kv_cache,
    )

class DeepSeekV4Model(_BaseDeepSeekV4Model):
    _captures_aux_hidden = True

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
        platform_provider=None,
    ):
        if platform_provider is None:
            platform_provider = resolve_dsv4_platform_provider(
                {Dsv4ProviderCapability.BLOCK, Dsv4ProviderCapability.TRANSFORMER}
            )
        self._platform_provider = platform_provider
        self._execution_options = getattr(platform_provider, "execution_options", None)
        super(_BaseDeepSeekV4Model, self).__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )

        # Build V4Transformer with matching args.
        args = _args_from_model_config(
            model_config,
            max_generate_batch_size,
            moe_config=moe_config,
        )
        self._max_generate_batch_size = int(max_generate_batch_size)
        assert self._max_generate_batch_size > 0, (
            "max_generate_batch_size must be positive, "
            f"got {self._max_generate_batch_size}"
        )
        self._gen_num_per_cycle = int(model_config.gen_num_per_cycle)
        # Python-only DSpARK config, populated on the target model by the
        # speculative-engine setup. ``getattr`` keeps older ModelConfig
        # bindings and all non-DSpARK paths unchanged.
        self._capture_aux_hidden_layer_ids = tuple(
            int(layer_id)
            for layer_id in (
                getattr(model_config, "capture_aux_hidden_layer_ids", None) or ()
            )
        )
        if self._capture_aux_hidden_layer_ids:
            # DSpARK: the captured aux features ride the shared MTP hidden
            # buffer instead of a dedicated output channel. The factory sets
            # the capture ids on both the target and the draft config, so
            # whichever model binds the store first derives the same row
            # width in _bind_runtime_buffers; only the target actually
            # captures (see _captures_aux_hidden).
            Dsv4SharedRuntimeBufferStore.enable_mtp_hidden()
        # MoE inter dim from V4 config: explicit (not inter_size which in RTP-LLM
        # is n_shared_experts * moe_intermediate_size for DeepSeek). Use moe_config
        # if available; else read from config's hidden_size-derived fallback.
        if (
            moe_config is not None
            and getattr(moe_config, "moe_inter_padding_size", 0) > 0
        ):
            args.moe_inter_dim = int(moe_config.moe_inter_padding_size)
        else:
            args.moe_inter_dim = _resolve_dsv4_moe_inter_dim(model_config)

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

        # Resolve CP padding at initialize(), once the scheduler's maximum
        # context batch size is available. Each request is padded separately.
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
        self._init_capture_context(
            self._capture_canonical_layer,
            self._capture_canonical_final,
        )

    def _initialize_commit_only(
        self, init_resource: PyModelInitResources, device_str: str
    ) -> bool:
        """Initialize an attention-only DSpARK PREFILL commit model."""
        logging.info(
            "[DeepSeekV4Model] building DSpARK commit-only transformer "
            "(layers=%d, globals=%d)",
            len(self.weight.weights),
            len(self.weight.global_weights),
        )
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                self.v4 = V4Transformer(self._v4_args, mw=self.weight)
        finally:
            torch.set_default_dtype(prev_dtype)

        for layer in self.v4.layers:
            layer.attn.init_rope_cache(device=device_str)

        self._load_extra_weights(self.weight)
        del self.weight
        self._bind_runtime_buffers(torch.device(device_str))
        self._materialized = True
        return True

    def _initialize_impl(self, init_resource: PyModelInitResources) -> bool:
        # Called by the engine after construction and before forward.
        super(_BaseDeepSeekV4Model, self).initialize(init_resource)
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
            cp_size=self._prefill_cp_size,
            max_generate_batch_size=int(self._max_generate_batch_size),
            max_context_batch_size=self._max_context_batch_size,
            is_decode_role=self._is_decode_role,
            is_speculative=self._is_speculative,
            gen_num_per_cycle=self._gen_num_per_cycle,
            chunking_enabled=chunked_moe_enabled(self._execution_options),
            chunk_tokens=moe_chunk_tokens_from_env(options=self._execution_options),
        )
        if runtime_resolved_max_tokens_per_rank != self._v4_args.max_tokens_per_rank:
            chunk_tokens_env_for_log = (
                DSV4_CHUNK_TOKENS_ENV
                if dsv4_global_chunk_tokens_configured(self._execution_options)
                else "DSV4_MOE_CHUNK_TOKENS"
            )
            chunk_tokens_for_log = -1
            if not self._is_decode_role and (
                dsv4_global_chunk_tokens_configured(self._execution_options)
                or chunked_moe_enabled(self._execution_options)
            ):
                chunk_tokens_for_log = moe_chunk_tokens_from_env(
                    options=self._execution_options
                )
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

        if bool(getattr(self._v4_args, "commit_only", False)):
            return self._initialize_commit_only(init_resource, device_str)

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
                self.v4 = self._platform_provider.build_transformer(
                    V4Transformer,
                    self._v4_args,
                    self.weight,
                    platform_provider=self._platform_provider,
                )
        finally:
            torch.set_default_dtype(prev_dtype)
        if self._captures_aux_hidden:
            self.v4.set_aux_hidden_capture_layer_ids(self._capture_aux_hidden_layer_ids)

        # Recompute RoPE on the real device and prebuild the compressors'
        # shared cos_sin_cache before runtime memory allocation starts.
        for layer in self.v4.layers:
            layer.attn.init_rope_cache(device=device_str)

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
        model_warm_up = model_warm_up_enabled()
        if device_str.startswith("cuda") and model_warm_up:
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

            if (
                _env_flag_enabled("DSV4_INDEXER_SCORE_WARMUP", True)
                and len(self.v4.layers) > 2
                and self.v4.layers[2].attn.indexer is not None
            ):
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
            elif not _env_flag_enabled("DSV4_INDEXER_SCORE_WARMUP", True):
                logging.info(
                    "[DeepSeekV4Model] skip DSV4IndexerScore Triton warmup: "
                    "DSV4_INDEXER_SCORE_WARMUP=0"
                )

            try:
                from rtp_llm.models_py.utils.flash_mla_legacy import (
                    flash_mla_sparse_fwd as _flash_mla_sparse_fwd,
                )

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

            try:
                import torch.distributed as _dist

                _moe_layer = self.v4.layers[0].ffn
                if getattr(_moe_layer, "strategy_name", None) in (
                    "mega_moe",
                    "mega_moe_se",
                ):
                    _executor = _moe_layer.fused_moe.fused_experts
                    if _dist.is_available() and _dist.is_initialized():
                        _dist.barrier()
                    _cfg = _executor.cfg
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
                    _executor(_x_moe, _w_moe, _idx_moe)
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
                    warmup_prefill_cp_metadata_jit,
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
                if not self._is_decode_role and chunked_moe_enabled(
                    self._execution_options
                ):
                    _dense_gemm_prefill_chunk_size = max(
                        int(moe_chunk_tokens_from_env(options=self._execution_options)),
                        0,
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
                warmup_prefill_cp_metadata_jit(
                    is_decode_role=self._is_decode_role,
                    cp_enabled=_prefill_cp_enabled,
                    cp_size=_prefill_cp_size,
                    max_batch_size=max(
                        int(self._max_context_batch_size),
                        int(self._max_generate_batch_size),
                        1,
                    ),
                    fp8_kv_cache=self.fp8_kv_cache,
                    kv_cache_sharded=_prefill_kv_cache_sharded,
                    device=_jit_device,
                )
                _dense_gemm_max_m = resolve_dense_gemm_warmup_max_m(
                    max_seq_len=int(self._v4_args.max_seq_len),
                    max_batch_size=int(self._v4_args.max_batch_size),
                    role_type_name=(
                        RoleType.DECODE if self._is_decode_role else RoleType.PREFILL
                    ),
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
                        max_batch_size=max(
                            int(self._max_context_batch_size),
                            int(self._max_generate_batch_size),
                            1,
                        ),
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
        h = self.v4._embed(input_ids).view(B, q_len, -1)
        return h.unsqueeze(2).repeat(1, 1, self.v4.hc_mult, 1)

    def _prepare_prefill_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Build the flat ``[T_total, hc, dim]`` hidden tensor that feeds
        the layer loop on the prefill path.  Default = embed+repeat."""
        h = self.v4._embed(input_ids)
        return h.unsqueeze(-2).repeat(1, self.v4.hc_mult, 1)

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ) -> Any:
        """Return a ``DSv4DecodeFmhaImpl`` for decode CUDA-graph capture; None otherwise.

        Prefill runs eagerly (no graph). Decode uses its own sparse/compressed
        attention; the impl owns persistent metadata buffers updated in place
        by ``prepare_cuda_graph`` between replays."""
        if cuda_graph_selection_mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH:
            # DSV4 captures only decode/target-verify graphs; generation-prefill
            # capture would otherwise receive a decode impl, so degrade via the
            # factory's unsupported-backend channel instead of crashing on the
            # extra keyword.
            raise GenerationPrefillCudaGraphUnsupportedBackend(
                "DeepSeek V4 does not support generation prefill CUDA graphs"
            )
        if not is_cuda_graph:
            return None

        # ``attention_inputs`` is a ``{tag: PyAttentionInputs}`` mapping for the
        # multi-group DSV4 cache; the capture gate only reads group-invariant
        # fields, so take the primary entry.
        attn = primary_attention_inputs(inputs.attention_inputs, self.kv_cache)
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
        # the list in the config. Position IS the topology group id.
        group_tags_snapshot = (
            [str(tag) for tag in (self.kv_cache.group_tags or [])]
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
            group_tags=group_tags_snapshot,
        )
        cfg = _DecodeFmhaImplConfig(**cfg_kwargs)
        impl = build_dsv4_decode_metadata(
            _DecodeFmhaImpl,
            cfg,
            platform_provider=self._platform_provider,
            device=device,
            attn_inputs=attn,
        )
        # Phase F: pool views resolved on demand in Attention — no
        # per-layer descriptor cache to stash on metadata.
        return impl

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
            hidden = torch.zeros(
                T, self._v4_args.dim, dtype=torch.bfloat16, device=device
            )
            return PyModelOutputs(hidden)
        # Group-invariant view of ``PyModelInputs.attention_inputs``; the
        # per-tag block tables are resolved inside forward_prefill /
        # forward_decode.
        attn = primary_attention_inputs(inputs.attention_inputs, self.kv_cache)
        if attn is None:
            raise RuntimeError(
                "DeepSeekV4Model.forward: PyModelInputs carries no attention inputs"
            )

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
            with _profiler.moe_record_function_scope():
                capture = self.capture_context(inputs.capture_hidden_states)
            return forward_prefill(
                self.v4,
                self.kv_cache,
                self.parallelism_config,
                inputs,
                prepare_hidden_fn=prep_prefill,
                capture_context=capture,
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
