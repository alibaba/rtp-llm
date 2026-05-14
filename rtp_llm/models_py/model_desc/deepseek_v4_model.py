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
from typing import Any, Dict, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv4.decode.forward import (
    build_paged_pool_specs,
    forward_decode,
)
from rtp_llm.models_py.modules.dsv4.moe.moe_layer import (
    moe_chunk_tokens_from_env,
    resolve_moe_max_tokens_per_rank,
)
from rtp_llm.models_py.modules.dsv4.prefill.forward import forward_prefill
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.ops import RoleType


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
        original_max_tokens_per_rank = args.max_tokens_per_rank
        if cp_size > 1:
            cp_tokens_per_rank_bound = min(
                args.max_tokens_per_rank,
                max(args.max_seq_len // cp_size, 4096),
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

        # Resolve role_type from parallelism_config (mirrored from
        # py_env_configs.role_config in engine_config.setup_engine_config),
        # avoiding the prior os.environ["ROLE_TYPE"] env-side-channel
        # dependency. Pass the string name to resolve_moe_max_tokens_per_rank
        # so it does not fall back to env.
        role_type_enum = (
            getattr(parallelism_config, "role_type", RoleType.PDFUSION)
            if parallelism_config is not None
            else RoleType.PDFUSION
        )
        role_type_name = (
            role_type_enum.name
            if hasattr(role_type_enum, "name")
            else str(role_type_enum)
        )

        resolved_max_tokens_per_rank = resolve_moe_max_tokens_per_rank(
            max_seq_len=args.max_seq_len,
            current_max_tokens_per_rank=args.max_tokens_per_rank,
            cp_size=cp_size,
            max_generate_batch_size=max_generate_batch_size,
            role_type=role_type_name,
        )
        if resolved_max_tokens_per_rank != args.max_tokens_per_rank:
            chunk_tokens_for_log = -1
            if (
                role_type_name.upper() != "DECODE"
                and os.environ.get("DSV4_MOE_CHUNK_PREFILL", "1") != "0"
            ):
                chunk_tokens_for_log = moe_chunk_tokens_from_env()
            logging.info(
                "[DeepSeekV4Model] MoE token budget: max_tokens_per_rank %d -> %d "
                "(DSV4_MOE_CHUNK_TOKENS=%d, original=%d, CP=%d, role=%s)",
                args.max_tokens_per_rank,
                resolved_max_tokens_per_rank,
                chunk_tokens_for_log,
                original_max_tokens_per_rank,
                cp_size,
                role_type_name,
            )
            args.max_tokens_per_rank = resolved_max_tokens_per_rank

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
            _w_dec = _torch.zeros((1, 1, _H), dtype=_torch.bfloat16, device=device_str)
            # Decode shape (no mask).
            _v4_idx(_q_dec, _kv_dec, _w_dec, q_pos=None, compress_ratio=1)
            # Prefill mask variant — only matters if CSA prefill goes through
            # graph capture. Cheap to prewarm regardless.
            _v4_idx(
                _q_dec,
                _kv_dec,
                _w_dec,
                q_pos=_torch.zeros((1, 1), dtype=_torch.int32, device=device_str),
                compress_ratio=_ratio,
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
                    logging.exception("[DeepSeekV4Model] flash_mla SWA kv_full prewarm failed")
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
                    _collect_dsv4_dense_gemm_shapes,
                    warmup_compressor_combine_branch_kernels,
                    warmup_dense_gemm_jit,
                )

                _jit_device = _torch.device(device_str)
                warmup_compressor_combine_branch_kernels(
                    v4=self.v4,
                    v4_args=self._v4_args,
                    device=_jit_device,
                )
                _dense_shapes = _collect_dsv4_dense_gemm_shapes(self.v4)
                warmup_dense_gemm_jit(
                    _dense_shapes,
                    max_m=int(self._v4_args.max_seq_len),
                    device=_jit_device,
                )
                logging.info("[DeepSeekV4Model] kernel JIT prewarm done")
            except Exception:
                logging.exception("[DeepSeekV4Model] kernel JIT prewarm failed")
                raise
            _torch.cuda.synchronize()

        self._materialized = True

        # qwen3-style: no per-Attention KV wiring needed — ``self.kv_cache``
        # is read on every ``forward`` call and threaded through as a
        # kwarg. Framework may still allocate ``self.kv_cache`` after
        # ``initialize()`` returns; the first forward will pick it up.

        return True

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
        # V4 only captures CUDA graphs for decode — prefill runs eagerly.
        # Mirrors attn_factory.py: PREFILL_MLA_IMPS if is_prefill else DECODE_MLA_IMPS.
        if attn is None or bool(attn.is_prefill):
            return None

        if self.kv_cache is None:
            logging.warning(
                "[DeepSeekV4Model] prepare_fmha_impl: kv_cache not yet allocated "
                "(warmup phase); skipping CUDA-graph impl creation."
            )
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

        cfg = _DecodeFmhaImplConfig(
            max_batch_size=batch_size,
            q_len=1,
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
        impl = _DecodeFmhaImpl(
            cfg,
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
        if inputs.attention_inputs.is_prefill:
            return forward_prefill(
                self.v4, self.kv_cache, self.parallelism_config, inputs
            )
        return forward_decode(self.v4, self.kv_cache, self._v4_args, inputs, fmha_impl)
