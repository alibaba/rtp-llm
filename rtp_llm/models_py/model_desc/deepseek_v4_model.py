"""DeepSeek-V4 integration into RTP-LLM's `GptModelBase` framework.

Wraps the standalone `V4Transformer` (rtp_llm/models_py/modules/dsv4/transformer.py)
so the C++ engine can drive it:
  - `forward(PyModelInputs) -> PyModelOutputs` returns [total_tokens, hidden_dim]
    pre-lm-head hidden states; engine applies lm_head + sampling externally.
  - Per-layer KV cache kept INSIDE the Attention modules (mock, register_buffer).
    Framework's BlockPool-backed KV cache is NOT wired yet (M4 work); this wrapper
    ignores the framework's kv_cache and bookkeeps its own `start_pos` from
    `attention_inputs.is_prefill` + sequence lengths.

Weight loading happens in `initialize()`: we bypass the framework's AtomicWeight
flow (V4-Flash has 67k+ tensors in ~250 packed FP4/FP8 shards) and directly call
`load_v4_safetensors` against the checkpoint path stored in `model_config`.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.models_py.modules.dsv4.weight_loader import (
    load_v4_safetensors,
    load_v4_weights_dict,
)


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
    mc: ModelConfig, max_generate_batch_size: int = 4
) -> V4Args:
    ac = mc.attn_config
    rc = ac.rope_config
    return V4Args(
        vocab_size=mc.vocab_size,
        dim=mc.hidden_size,
        n_heads=ac.head_num,
        n_layers=mc.num_layers,
        n_mtp_layers=0,  # MTP wiring lands later; main-layer inference first.
        q_lora_rank=ac.q_lora_rank,
        head_dim=ac.size_per_head,
        rope_head_dim=ac.rope_head_dim,
        o_groups=ac.o_groups,
        o_lora_rank=ac.o_lora_rank,
        window_size=ac.sliding_window,
        compress_ratios=list(ac.layer_compress_ratios)[: mc.num_layers],
        rope_theta=float(rc.base),
        compress_rope_theta=float(ac.compress_rope_theta),
        rope_factor=float(rc.scale) if rc.scale else 1.0,
        beta_fast=int(rc.factor2) if rc.factor2 else 32,
        beta_slow=int(rc.factor1) if rc.factor1 else 1,
        original_seq_len=int(rc.max_pos) if rc.max_pos else 0,
        index_n_heads=ac.indexer_head_num,
        index_head_dim=ac.indexer_head_dim,
        index_topk=ac.indexer_topk,
        moe_inter_dim=(
            mc.inter_size // max(1, mc.moe_k or 1) if False else mc.inter_size // 1
        ),
        n_routed_experts=mc.expert_num,
        n_shared_experts=1,
        n_activated_experts=mc.moe_k,
        score_func={0: "softmax", 1: "sigmoid", 2: "sqrtsoftplus"}[mc.scoring_func],
        route_scale=float(mc.routed_scaling_factor),
        swiglu_limit=float(mc.swiglu_limit),
        n_hash_layers=int(mc.num_hash_layers),
        hc_mult=int(mc.hc_mult),
        hc_sinkhorn_iters=int(mc.hc_sinkhorn_iters),
        hc_eps=float(mc.hc_eps),
        norm_eps=float(mc.layernorm_eps),
        max_batch_size=max_generate_batch_size,  # from framework, supports concurrent requests
        max_seq_len=int(mc.max_seq_len) or 4096,
        # Mega MoE sizes its symm-mem dispatch buffer from this bound.
        # max_seq_len is the safest per-rank upper bound (one long prefill
        # fully on one rank) — the buffer is allocated once and reused.
        max_tokens_per_rank=int(mc.max_seq_len) or 4096,
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
        pc = parallelism_config
        if pc is not None:
            if hasattr(pc, "get_attn_tp_size"):
                args.tp_size = int(pc.get_attn_tp_size() or 1)
                args.tp_rank = int(pc.get_attn_tp_rank() or 0)
            else:
                args.tp_size = int(getattr(pc, "tp_size", 1) or 1)
                args.tp_rank = int(getattr(pc, "tp_rank", 0) or 0)
            args.ep_size = int(getattr(pc, "ep_size", 1) or 1)
            args.ep_rank = int(getattr(pc, "ep_rank", 0) or 0)
            args.dp_size = int(getattr(pc, "dp_size", 1) or 1)
            args.dp_rank = int(getattr(pc, "dp_rank", 0) or 0)
            args.world_size = int(getattr(pc, "world_size", 1) or 1)
            args.world_rank = int(getattr(pc, "world_rank", 0) or 0)
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

        # CP-aware Mega MoE buffer sizing.  When CP is on, each rank only
        # sees ``max_seq_len / cp_size`` prefill tokens (zigzag split), so
        # the per-rank symm-mem dispatch buffer can be cp_size× smaller.
        # Without this, a 64k+CP=4 prefill tries to allocate a per-rank
        # buffer sized for the full 64k which aborts in
        # ``CUDASymmetricMemory::~CUDASymmetricMemory`` during V4Transformer
        # construction (per-rank symm region limit + ~190 GB BF16 footprint
        # split across 4 ranks leaves no headroom).  RTP-LLM's CP convention
        # repurposes the TP group as the CP group (cp_size == raw tp_size,
        # ``get_attn_tp_size()`` returns 1), so use the raw pc.tp_size here.
        cp_size = 1
        if pc is not None and getattr(pc, "prefill_cp_config", None) is not None:
            try:
                if pc.prefill_cp_config.is_enabled():
                    cp_size = int(getattr(pc, "tp_size", 1) or 1)
            except Exception:  # pyi-only stub or non-CP build
                pass
        if cp_size > 1:
            new_bound = max(args.max_seq_len // cp_size, 4096)
            logging.info(
                "[DeepSeekV4Model] CP=%d: max_tokens_per_rank %d -> %d "
                "(Mega MoE per-rank symm-mem buffer)",
                cp_size,
                args.max_tokens_per_rank,
                new_bound,
            )
            args.max_tokens_per_rank = new_bound

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

        # Factory-path flow (S2+): we defer construction to `initialize()`
        # where the weights dict is loaded from the ckpt directly to the
        # target device, then V4Transformer modules are built bound to
        # real ckpt tensors via LinearFactory (see dsv4/weight_loader.py
        # and attention.py).  `self.v4` is assigned in `initialize()`.
        self.v4: Optional[V4Transformer] = None

        self._materialized = False
        self._ckpt_path: str = model_config.ckpt_path
        # Running cursor for prefill-then-decode — the only batching we support
        # until framework KV cache lands.
        self._running_pos: int = 0

        # FP8 KV cache toggle: read from attn_config.kv_cache_dtype, which the
        # framework sets to KvCacheDataType.FP8 when ``--fp8_kv_cache 1`` is
        # supplied (see ``ModelConfig._set_dtype_and_check`` in
        # rtp_llm/config/model_config.py).  When set, the first prefill triggers
        # ``Attention.enable_fp8_kv_cache`` on every layer, switching subsequent
        # decodes to the FlashMLA ``is_fp8_kvcache=True`` sparse path.
        # Override via ``DSV4_FP8_KV=0`` to force-disable for debugging.
        self._fp8_kv_requested = False
        try:
            from rtp_llm.config.model_config import KvCacheDataType  # local import

            kv_dtype = getattr(model_config.attn_config, "kv_cache_dtype", None)
            self._fp8_kv_requested = kv_dtype == KvCacheDataType.FP8
        except Exception:
            pass
        if os.environ.get("DSV4_FP8_KV", "1") == "0":
            self._fp8_kv_requested = False
        self._fp8_kv_armed = False  # set True after the first prefill enables FP8

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

        logging.info(
            "[DeepSeekV4Model] loading ckpt dict from %s to %s "
            "(ep_size=%d ep_rank=%d) ...",
            self._ckpt_path,
            device_str,
            self._v4_args.ep_size,
            self._v4_args.ep_rank,
        )
        weights = load_v4_weights_dict(
            self._ckpt_path,
            device=device_str,
            ep_size=self._v4_args.ep_size,
            ep_rank=self._v4_args.ep_rank,
            n_routed_experts=self._v4_args.n_routed_experts,
        )
        logging.info("[DeepSeekV4Model] loaded %d tensors from ckpt", len(weights))

        logging.info("[DeepSeekV4Model] building V4Transformer via factory ...")
        # Construct under meta device so the nn.Embedding / nn.Linear /
        # QuantizedLinear placeholder allocations inside the module tree
        # skip real RAM. Each module's factory branch then reassigns
        # `.weight`/`.scale` to the cuda tensors from the weights dict.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                self.v4 = V4Transformer(self._v4_args, weights=weights)
        finally:
            torch.set_default_dtype(prev_dtype)

        n = _materialize_meta_buffers(self.v4, device_str)
        logging.info(
            "[DeepSeekV4Model] materialized %d meta buffers on %s", n, device_str
        )

        # Recompute RoPE cache on real device (precompute_freqs_cis under
        # meta context yields zeros; we need real values).
        for layer in self.v4.layers:
            layer.attn.reset_rope_cache(device=device_str)

        # Drop the dict's references to release any residual CPU copies.
        del weights

        if torch.cuda.is_available() and device_str.startswith("cuda"):
            mem = torch.cuda.memory_allocated(torch.device(device_str)) / 1024**3
            logging.info("[DeepSeekV4Model] GPU mem after load: %.1f GB", mem)

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

        if self._fp8_kv_requested:
            logging.info(
                "[DeepSeekV4Model] --fp8_kv_cache=1 — first prefill will arm "
                "Attention.enable_fp8_kv_cache() on every layer, switching "
                "subsequent decodes to the FlashMLA FP8 sparse path."
            )

        self._materialized = True
        self._running_pos = 0

        # Wire framework KV cache — note: self.kv_cache may not be set yet
        # (framework allocates it after initialize() returns). We defer wiring
        # to the first forward() call via _framework_kv_pending.
        self._framework_kv_pending = True
        if self.kv_cache is not None:
            self._wire_framework_kv_cache(device_str)
            self._framework_kv_pending = False
            logging.info(
                "[DeepSeekV4Model] framework KV cache wired for %d layers",
                len(self.v4.layers),
            )

        return True

    def _reset_compressor_state(self):
        """Reset compressor/indexer state for a new prefill request.

        When using framework KV cache, the kv_cache tensor persists across
        requests (it's a framework-managed buffer, not a per-request register_buffer).
        The compressor's kv_state/score_state and kv_cache must be zeroed on each
        new prefill to avoid stale data from previous requests corrupting output.

        NOTE: Do NOT set compressor.kv_cache = None here. The gather path
        needs the buffer reference to exist so it can copy prefix-cache data
        into it BEFORE forward() runs. The underlying storage is already
        zeroed by attn.kv_cache.zero_() (compressor.kv_cache is a view).
        """
        for layer in self.v4.layers:
            attn = layer.attn
            # Reset kv_cache (SWA + compressed KV) — also zeros compressor.kv_cache view
            attn.kv_cache.zero_()
            # Reset compressor state
            if attn.compressor is not None:
                attn.compressor.kv_state.zero_()
                attn.compressor.score_state.fill_(float("-inf"))
            # Reset indexer state
            if attn.indexer is not None:
                attn.indexer.kv_cache.zero_()
                attn.indexer.compressor.kv_state.zero_()
                attn.indexer.compressor.score_state.fill_(float("-inf"))

    def _wire_framework_kv_cache(self, device: str):
        """Wire framework BlockPool for 7-pool gather/scatter.

        DSV4 uses HybridPoolKVCacheAllocator with 7 independent BlockPools.
        Each pool has its own [total_blocks, stride_bytes] uint8 tensor.
        We use get_raw_pool_tensor(layer_id, attn_type) to get each pool's tensor.
        """
        self._framework_kv_enabled = True
        if self.kv_cache is None:
            logging.warning("[DeepSeekV4Model] no framework kv_cache, skipping wiring")
            return

        # Build per-layer per-pool tensor cache from flat field
        flat = getattr(self.kv_cache, "kv_cache_base_by_layer_attn_flat", None)
        attn_type_count = 8
        self._fw_pool_tensors = []
        # Fingerprint the underlying pool storage so we can detect when the
        # framework rebuilds the KV cache (warmup→real transition: NormalEngine
        # first allocates a stub allocator with block_num=1 to measure GPU
        # memory, then tears it down and creates the real one with the
        # measured block_num). Without re-wiring, _fw_pool_tensors keeps
        # pointing at the freed warmup tensors and gather hits IndexError
        # because the warmup CSA_KV/INDEXER_KV pools only had 1 block each.
        self._wired_kv_cache_id = id(self.kv_cache)
        self._wired_first_data_ptr = None
        if flat is not None and len(flat) > 0:
            num_layers = len(flat) // attn_type_count
            for i in range(len(self.v4.layers)):
                slots = [None] * attn_type_count
                if i < num_layers:
                    for a in range(attn_type_count):
                        t = flat[i * attn_type_count + a]
                        if t is not None and t.numel() > 0:
                            slots[a] = t
                            if self._wired_first_data_ptr is None:
                                self._wired_first_data_ptr = t.data_ptr()
                self._fw_pool_tensors.append(slots)
        else:
            logging.warning(
                "[DeepSeekV4Model] kv_cache_base_by_layer_attn_flat empty, gather/scatter disabled"
            )
            return

        # Build per-layer pool list from compress_ratio
        # _layer_pools: all pools for scatter (write all 7 pools)
        # _layer_gather_pools: only paged pools for gather (skip SWA/State which
        #   get overwritten during decode, making cached data unreliable)
        # _layer_reuse_gather_pools: paged + state pools for prefill reuse gather
        #   (restores compressor overlap carry, but skips SWA ring buffer which
        #   holds data at the wrong position)
        self._layer_pools = []
        self._layer_gather_pools = []
        self._layer_reuse_gather_pools = []
        # Phase 2B-3: STATE-only pool list per layer for the decode hot-path,
        # used when paged read+write keep KV pools fresh and we only need to
        # gather/scatter the compressor/indexer state pools (4/5/6).
        self._layer_state_pools = []
        for i, layer in enumerate(self.v4.layers):
            ratio = layer.attn.compress_ratio
            if ratio == 4:
                self._layer_pools.append([7, 1, 3, 4, 5])
                self._layer_gather_pools.append([1, 3])  # CSA_KV, INDEXER_KV only
                self._layer_reuse_gather_pools.append([1, 3, 4, 5])  # + STATE pools
                self._layer_state_pools.append([4, 5])  # INDEXER_STATE + CSA_STATE
            elif ratio == 128:
                self._layer_pools.append([7, 2, 6])
                self._layer_gather_pools.append([2])  # HCA_KV only
                self._layer_reuse_gather_pools.append([2, 6])  # + STATE pool
                self._layer_state_pools.append([6])  # HCA_STATE only
            else:
                self._layer_pools.append([7])
                self._layer_gather_pools.append([])  # SWA-only, nothing to gather
                self._layer_reuse_gather_pools.append([])  # SWA-only
                self._layer_state_pools.append([])  # SWA-only has no state

        # Debug: log shapes for layer 0 and layer 2
        for dbg in [0, 2]:
            if dbg < len(self._fw_pool_tensors):
                ne = {
                    j: self._fw_pool_tensors[dbg][j].shape
                    for j in range(attn_type_count)
                    if self._fw_pool_tensors[dbg][j] is not None
                }
                logging.info("[DeepSeekV4Model] _fw_pool_tensors[%d]: %s", dbg, ne)

        spb = (
            self.kv_cache.seq_size_per_block
            if self.kv_cache.seq_size_per_block > 0
            else 256
        )
        logging.info(
            "[DeepSeekV4Model] framework 7-pool KV wired: %d layers, seq_size_per_block=%d",
            len(self.v4.layers),
            spb,
        )

        # Phase 0 layout probe — print actual stride/blocks/entries for every
        # pool of layer 0 and the first compress_ratio==4 / ==128 layers, so
        # we can verify that "256 tokens/block, fixed 2 blocks/req for state
        # & SWA, shared variable-length pool for CSA/HCA/INDEXER KV" holds
        # against the actual framework allocator before writing paged ops.
        self._dump_pool_layout(spb)

        # Phase 2: build the per-layer PoolDescriptor cache for the paged
        # decode path. ``self._layer_pool_descs[layer_idx][attn_type]``
        # gives the typed view + entries_per_block at decode call time.
        # Source-of-truth = framework pool stride (seen by build_pool_descriptor).
        from rtp_llm.models_py.modules.dsv4.decode.pool_layout import (
            build_layer_pool_descriptors,
        )

        idx_hd = 0
        for layer in self.v4.layers:
            if layer.attn.indexer is not None:
                idx_hd = layer.attn.indexer.head_dim
                break
        self._layer_pool_descs = []
        for li, layer in enumerate(self.v4.layers):
            self._layer_pool_descs.append(
                build_layer_pool_descriptors(
                    flat_pools=flat,
                    layer_idx=li,
                    head_dim=layer.attn.head_dim,
                    indexer_head_dim=idx_hd,
                    compress_ratio=layer.attn.compress_ratio,
                    attn_type_count=attn_type_count,
                )
            )
        logging.info(
            "[DeepSeekV4Model] Phase 2: built per-layer PoolDescriptors for %d layers",
            len(self._layer_pool_descs),
        )

    def _dump_pool_layout(self, tokens_per_block: int) -> None:
        """One-shot dump of per-attn_type pool layout for the new paged decode
        path. Logs (total_blocks, stride_bytes, expected_entries_per_block,
        bytes_per_entry, slack_bytes) for layer 0 and one CSA/HCA exemplar."""
        # attn_type → (vector dtype, vector dim, expected entries per 256-token block)
        # KV vectors:
        #   1=CSA_KV     bf16, head_dim,    256 / 4   = 64
        #   2=HCA_KV     bf16, head_dim,    256 / 128 = 2
        #   3=INDEXER_KV bf16, idx_head,    256 / 4   = 64   (indexer has compress_ratio=4)
        #   7=SWA_KV     bf16, head_dim,    256       (cyclic, fixed 2 blocks/req → 512 cap)
        # State vectors (paired kv_state ‖ score_state, fp32):
        #   4=INDEXER_STATE  fp32, 2*coff_idx*idx_head   slots = coff_idx*ratio_idx
        #   5=CSA_STATE      fp32, 2*coff*head_dim       slots = coff*ratio (overlap=True coff=2)
        #   6=HCA_STATE      fp32, 2*head_dim            slots = ratio (overlap=False coff=1)
        attn_type_names = {
            1: "CSA_KV",
            2: "HCA_KV",
            3: "INDEXER_KV",
            4: "INDEXER_STATE",
            5: "CSA_STATE",
            6: "HCA_STATE",
            7: "SWA_KV",
        }

        def _describe_layer(layer_idx: int) -> None:
            attn = self.v4.layers[layer_idx].attn
            ratio = attn.compress_ratio
            hd = attn.head_dim
            idx_hd = attn.indexer.head_dim if attn.indexer is not None else 0
            for at in range(1, 8):
                t = self._get_pool_tensor(at, layer_idx)
                if t is None or t.numel() == 0:
                    logging.info(
                        "[layout] L%02d ratio=%-3d %-13s : <empty>",
                        layer_idx,
                        ratio,
                        attn_type_names[at],
                    )
                    continue
                num_blocks, stride_bytes = int(t.shape[0]), int(t.shape[1])
                # Expected geometry per attn_type
                if at == 1:  # CSA_KV
                    bpe, exp_eb = hd * 2, max(1, tokens_per_block // 4)
                elif at == 2:  # HCA_KV
                    bpe, exp_eb = hd * 2, max(1, tokens_per_block // 128)
                elif at == 3:  # INDEXER_KV
                    bpe, exp_eb = idx_hd * 2, max(1, tokens_per_block // 4)
                elif at == 7:  # SWA_KV
                    bpe, exp_eb = hd * 2, tokens_per_block
                elif at == 5:  # CSA_STATE  (overlap=True, coff=2)
                    bpe, exp_eb = (
                        2 * 2 * hd
                    ) * 4, 4  # 4 state slots/block × 2 blocks/req = 8 slots
                elif at == 6:  # HCA_STATE  (overlap=False, coff=1)
                    bpe, exp_eb = (
                        2 * hd
                    ) * 4, 8  # 8 state slots/block × 2 blocks/req = 16 slots
                elif at == 4:  # INDEXER_STATE (overlap=True for idx ratio=4, coff=2)
                    bpe, exp_eb = (2 * 2 * idx_hd) * 4, 4
                else:
                    bpe, exp_eb = 0, 0
                eb_from_stride = stride_bytes // bpe if bpe > 0 else -1
                slack = stride_bytes - eb_from_stride * bpe if bpe > 0 else 0
                logging.info(
                    "[layout] L%02d ratio=%-3d %-13s blocks=%-7d stride=%-7dB "
                    "bytes/entry=%-5d entries=%-5d (expected=%d) slack=%dB",
                    layer_idx,
                    ratio,
                    attn_type_names[at],
                    num_blocks,
                    stride_bytes,
                    bpe,
                    eb_from_stride,
                    exp_eb,
                    slack,
                )

        # First layer of each ratio class for compactness
        seen: set[int] = set()
        for li, layer in enumerate(self.v4.layers):
            r = layer.attn.compress_ratio
            if r in seen:
                continue
            seen.add(r)
            _describe_layer(li)
            if len(seen) >= 3:
                break

    def _gather_kv_pool(
        self, fw_tensor, block_ids_2d, python_buf, entries_per_block, head_dim, B
    ):
        """Gather paged KV from BlockPool into python_buf[B, T, head_dim] (bf16).

        fw_tensor is [total_blocks, stride_elems] uint8. Each page has stride_elems bytes.
        Vectorized: one index_select + reshape instead of per-block Python loops.
        """
        if fw_tensor is None or fw_tensor.numel() == 0:
            return
        bytes_per_entry = head_dim * 2  # bf16
        page_bytes_avail = fw_tensor.size(1)
        capacity = min(entries_per_block, page_bytes_avail // bytes_per_entry)
        useful_bytes = capacity * bytes_per_entry

        # Move block_ids to CPU once to avoid per-element GPU sync
        bids_cpu = block_ids_2d[:B].cpu()
        T = python_buf.size(1)

        for b in range(B):
            row = bids_cpu[b]
            for k in range(row.size(0)):
                bid = int(row[k].item())
                if bid <= 0:
                    continue
                entry_start = k * capacity
                entry_end = min(entry_start + capacity, T)
                n = entry_end - entry_start
                if n <= 0:
                    break
                page_bf16 = (
                    fw_tensor[bid, : n * bytes_per_entry]
                    .view(torch.bfloat16)
                    .view(n, head_dim)
                )
                python_buf[b, entry_start:entry_end] = page_bf16

    def _scatter_kv_pool(
        self, fw_tensor, block_ids_2d, python_buf, entries_per_block, head_dim, B
    ):
        """Scatter python_buf[B, T, head_dim] (bf16) back to BlockPool pages."""
        if fw_tensor is None or fw_tensor.numel() == 0:
            return
        bytes_per_entry = head_dim * 2
        page_bytes_avail = fw_tensor.size(1)
        capacity = min(entries_per_block, page_bytes_avail // bytes_per_entry)
        T = python_buf.size(1)

        bids_cpu = block_ids_2d[:B].cpu()
        for b in range(B):
            row = bids_cpu[b]
            for k in range(row.size(0)):
                bid = int(row[k].item())
                if bid <= 0:
                    continue
                entry_start = k * capacity
                entry_end = min(entry_start + capacity, T)
                n = entry_end - entry_start
                if n <= 0:
                    break
                data = (
                    python_buf[b, entry_start:entry_end]
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(-1)
                )
                fw_tensor[bid, : n * bytes_per_entry] = data

    def _gather_state_pool(
        self,
        fw_tensor,
        block_ids_2d,
        kv_state,
        score_state,
        entries_per_block,
        state_dim,
        B,
    ):
        """Gather fixed-allocation state from BlockPool into kv_state + score_state (fp32)."""
        if fw_tensor is None or fw_tensor.numel() == 0:
            return
        half_dim = state_dim // 2
        bids_cpu = block_ids_2d[:B].cpu()
        max_blks = min(2, bids_cpu.size(1))
        state_rows = kv_state.size(1)
        for b in range(B):
            for blk_idx in range(max_blks):
                bid = int(bids_cpu[b, blk_idx].item())
                if bid <= 0:
                    continue
                page_fp32 = (
                    fw_tensor[bid, : entries_per_block * state_dim * 4]
                    .view(torch.float32)
                    .view(entries_per_block, state_dim)
                )
                row_start = blk_idx * entries_per_block
                n = min(entries_per_block, state_rows - row_start)
                if n <= 0:
                    break
                kv_state[b, row_start : row_start + n] = page_fp32[:n, :half_dim]
                score_state[b, row_start : row_start + n] = page_fp32[:n, half_dim:]

    def _scatter_state_pool(
        self,
        fw_tensor,
        block_ids_2d,
        kv_state,
        score_state,
        entries_per_block,
        state_dim,
        B,
    ):
        """Scatter kv_state + score_state (fp32) back to BlockPool pages."""
        if fw_tensor is None or fw_tensor.numel() == 0:
            return
        half_dim = state_dim // 2
        bids_cpu = block_ids_2d[:B].cpu()
        max_blks = min(2, bids_cpu.size(1))
        state_rows = kv_state.size(1)
        for b in range(B):
            for blk_idx in range(max_blks):
                bid = int(bids_cpu[b, blk_idx].item())
                if bid <= 0:
                    continue
                page_fp32 = (
                    fw_tensor[bid, : entries_per_block * state_dim * 4]
                    .view(torch.float32)
                    .view(entries_per_block, state_dim)
                )
                row_start = blk_idx * entries_per_block
                n = min(entries_per_block, state_rows - row_start)
                if n <= 0:
                    break
                page_fp32[:n, :half_dim] = kv_state[b, row_start : row_start + n]
                page_fp32[:n, half_dim:] = score_state[b, row_start : row_start + n]

    def _get_pool_tensor(self, attn_type: int, layer_idx: int):
        """Look up per-pool BlockPool tensor."""
        if not hasattr(self, "_fw_pool_tensors") or not self._fw_pool_tensors:
            return None
        if layer_idx >= len(self._fw_pool_tensors):
            return None
        slots = self._fw_pool_tensors[layer_idx]
        if attn_type >= len(slots):
            return None
        return slots[attn_type]

    def _gather_all_layers(
        self,
        attn_inputs,
        B=1,
        batch_offset=0,
        paged_only=False,
        reuse_gather=False,
        state_only=False,
    ):
        """Gather cached KV from BlockPool pages into each layer's buffers.

        batch_offset: row offset into by_group block_ids. In prefill loop,
        request b uses batch_offset=b so we read the correct block_ids row.

        paged_only: if True, only gather paged pools (CSA_KV, HCA_KV, INDEXER_KV).
        Skip SWA and State pools because their fixed blocks get overwritten
        during decode, making cached data unreliable for prefix cache reuse.
        Continuation prefill recomputes them. Decode must set paged_only=False.

        reuse_gather: if True, gather paged + state pools (skip SWA).
        Used for prefill reuse to restore compressor overlap carry state.
        """
        if not hasattr(self, "_fw_pool_tensors") or not self._fw_pool_tensors:
            return

        by_group = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device_by_group", None
        )
        if by_group is None or len(by_group) == 0:
            return

        # attn_type → group_id (0-indexed): CSA_KV=1→0, HCA_KV=2→1, ...SWA_KV=7→6
        attn_type_to_gid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

        for i, layer in enumerate(self.v4.layers):
            attn_mod = layer.attn
            win = attn_mod.window_size
            hd = attn_mod.head_dim

            # Select pool list based on gather mode
            if state_only:
                pools = self._layer_state_pools[i]
            elif reuse_gather:
                pools = self._layer_reuse_gather_pools[i]
            elif paged_only:
                pools = self._layer_gather_pools[i]
            else:
                pools = self._layer_pools[i]
            for attn_type in pools:
                gid = attn_type_to_gid.get(attn_type)
                if gid is None or gid >= len(by_group):
                    continue
                block_ids = by_group[gid]  # [total_requests, max_blocks]
                if block_ids is None or block_ids.numel() == 0:
                    continue
                bid_slice = block_ids[batch_offset : batch_offset + B]
                fw_t = self._get_pool_tensor(attn_type, i)

                if attn_type == 7:  # SWA_KV
                    swa_buf = attn_mod.kv_cache[:B, :win]
                    self._gather_kv_pool(fw_t, bid_slice, swa_buf, 256, hd, B)
                elif attn_type == 1:  # CSA_KV
                    if (
                        attn_mod.compressor is not None
                        and attn_mod.compressor.kv_cache is not None
                    ):
                        self._gather_kv_pool(
                            fw_t, bid_slice, attn_mod.compressor.kv_cache[:B], 64, hd, B
                        )
                elif attn_type == 2:  # HCA_KV
                    if (
                        attn_mod.compressor is not None
                        and attn_mod.compressor.kv_cache is not None
                    ):
                        self._gather_kv_pool(
                            fw_t, bid_slice, attn_mod.compressor.kv_cache[:B], 2, hd, B
                        )
                elif attn_type == 3:  # INDEXER_KV
                    if attn_mod.indexer is not None:
                        idx_hd = attn_mod.indexer.head_dim
                        self._gather_kv_pool(
                            fw_t,
                            bid_slice,
                            attn_mod.indexer.kv_cache[:B],
                            64,
                            idx_hd,
                            B,
                        )
                elif attn_type == 4:  # INDEXER_STATE
                    if attn_mod.indexer is not None:
                        comp = attn_mod.indexer.compressor
                        idx_hd = attn_mod.indexer.head_dim
                        self._gather_state_pool(
                            fw_t,
                            bid_slice,
                            comp.kv_state[:B],
                            comp.score_state[:B],
                            4,
                            4 * idx_hd,
                            B,
                        )
                elif attn_type == 5:  # CSA_STATE
                    if attn_mod.compressor is not None:
                        self._gather_state_pool(
                            fw_t,
                            bid_slice,
                            attn_mod.compressor.kv_state[:B],
                            attn_mod.compressor.score_state[:B],
                            4,
                            4 * hd,
                            B,
                        )
                elif attn_type == 6:  # HCA_STATE
                    if attn_mod.compressor is not None:
                        self._gather_state_pool(
                            fw_t,
                            bid_slice,
                            attn_mod.compressor.kv_state[:B],
                            attn_mod.compressor.score_state[:B],
                            8,
                            2 * hd,
                            B,
                        )

    def _scatter_all_layers(self, attn_inputs, B=1, batch_offset=0, state_only=False):
        """Scatter STATE pools from each layer's buffers back to BlockPool.

        Phase D (partial): KV pools (attn_types 1/2/3/7) are now sole-written
        by Attention.forward's prefill paged dual-write — verified byte-equal
        by test_phase_b_prefill_dual_write. This scatter iterates STATE pools
        only (attn_types 4/5/6); ``state_only`` is vestigial and kept for API
        stability. B.3 will view-ify the STATE buffers and retire this method.

        batch_offset: row offset into by_group block_ids (same as gather).
        """
        if not hasattr(self, "_fw_pool_tensors") or not self._fw_pool_tensors:
            return

        by_group = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device_by_group", None
        )
        if by_group is None or len(by_group) == 0:
            return

        attn_type_to_gid = {4: 3, 5: 4, 6: 5}  # STATE only: 4/5/6 → gid 3/4/5

        for i, layer in enumerate(self.v4.layers):
            attn_mod = layer.attn
            hd = attn_mod.head_dim

            # STATE-only loop — KV scatter retired.
            for attn_type in self._layer_state_pools[i]:
                gid = attn_type_to_gid.get(attn_type)
                if gid is None or gid >= len(by_group):
                    continue
                block_ids = by_group[gid]
                if block_ids is None or block_ids.numel() == 0:
                    continue
                bid_slice = block_ids[batch_offset : batch_offset + B]
                fw_t = self._get_pool_tensor(attn_type, i)

                if attn_type == 4:  # INDEXER_STATE
                    if attn_mod.indexer is not None:
                        comp = attn_mod.indexer.compressor
                        idx_hd = attn_mod.indexer.head_dim
                        self._scatter_state_pool(
                            fw_t,
                            bid_slice,
                            comp.kv_state[:B],
                            comp.score_state[:B],
                            4,
                            4 * idx_hd,
                            B,
                        )
                elif attn_type == 5:  # CSA_STATE
                    if attn_mod.compressor is not None:
                        self._scatter_state_pool(
                            fw_t,
                            bid_slice,
                            attn_mod.compressor.kv_state[:B],
                            attn_mod.compressor.score_state[:B],
                            4,
                            4 * hd,
                            B,
                        )
                elif attn_type == 6:  # HCA_STATE
                    if attn_mod.compressor is not None:
                        self._scatter_state_pool(
                            fw_t,
                            bid_slice,
                            attn_mod.compressor.kv_state[:B],
                            attn_mod.compressor.score_state[:B],
                            8,
                            2 * hd,
                            B,
                        )

    def _bind_prefill_paged_ctx(self, attn_inputs, batch_offset: int = 0) -> bool:
        """Phase B: bind per-layer paged dual-write ctx on every Attention.

        Returns True if ctx was bound (caller must call
        ``_clear_prefill_paged_ctx`` after the forward).

        Mirror of ``_build_paged_pool_specs_for_phase2`` but for prefill:
        we only need SWA_KV right now (CSA/HCA/INDEXER dual-write will be
        layered on in Phase B.2).
        """
        if not getattr(self, "_layer_pool_descs", None):
            return False
        if attn_inputs is None:
            return False
        by_group = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device_by_group", None
        )
        if by_group is None or len(by_group) == 0:
            return False
        from rtp_llm.models_py.modules.dsv4.decode.pool_layout import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )

        # attn_type → group_id (same mapping as gather/scatter):
        # CSA_KV=1→0, HCA_KV=2→1, INDEXER_KV=3→2, SWA_KV=7→6.
        attn_type_to_gid = {CSA_KV: 0, HCA_KV: 1, INDEXER_KV: 2, SWA_KV: 6}
        bt_by_type: Dict[int, Any] = {}
        for at, gid in attn_type_to_gid.items():
            if gid >= len(by_group):
                continue
            bt_all = by_group[gid]
            if bt_all is None or bt_all.numel() == 0:
                continue
            # [1, max_blocks_per_req] for this request's row.
            bt_by_type[at] = bt_all[batch_offset : batch_offset + 1]
        if not bt_by_type:
            return False
        for li, layer in enumerate(self.v4.layers):
            layer_desc = (
                self._layer_pool_descs[li] if li < len(self._layer_pool_descs) else {}
            )
            layer.attn.set_prefill_paged_ctx(layer_desc, bt_by_type)
        return True

    def _clear_prefill_paged_ctx(self) -> None:
        for layer in self.v4.layers:
            layer.attn.set_prefill_paged_ctx(None, None)

    def _write_pd_cache_store(self, attn) -> None:
        """Register prefill-side KV blocks with cache_store for PD separation.

        DSV4 has up to 5 pools per layer (CSA layer: SWA_KV, CSA_KV,
        INDEXER_KV, INDEXER_STATE, CSA_STATE; HCA layer: SWA_KV, HCA_KV,
        HCA_STATE; SWA-only layer: SWA_KV). Each pool has its own layout
        (head_dim × entries_per_block × dtype) and its own block_id table.
        The shared write_cache_store path assumes one pool per layer, so
        we call it once per (layer × pool), passing the pool-specific
        block_id table, raw [num_blocks, stride_bytes] tensor, group_id
        (used in the "_g{gid}" cache key suffix that decode mirrors in
        DecodeRpcServer) and per-pool stride.
        """
        cache_store_inputs = getattr(attn, "cache_store_inputs", None)
        if cache_store_inputs is None:
            return
        by_group_host = getattr(attn, "kv_cache_kernel_block_id_host_by_group", None)
        if not by_group_host:
            return
        if not getattr(self, "_layer_pools", None):
            return
        if self.kv_cache is None:
            return

        # attn_type → group_id mirrors DSV4ConfigCreator.cc pool_attn_types
        # (CSA_KV=1→0, HCA_KV=2→1, INDEXER_KV=3→2, INDEXER_STATE=4→3,
        # CSA_STATE=5→4, HCA_STATE=6→5, SWA_KV=7→6).
        attn_type_to_gid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

        import rtp_llm.ops.compute_ops as compute_ops
        from rtp_llm.ops.compute_ops import KVCacheAttnType

        attn_type_enum_by_int = {
            1: KVCacheAttnType.CSA_KV,
            2: KVCacheAttnType.HCA_KV,
            3: KVCacheAttnType.INDEXER_KV,
            4: KVCacheAttnType.INDEXER_STATE,
            5: KVCacheAttnType.CSA_STATE,
            6: KVCacheAttnType.HCA_STATE,
            7: KVCacheAttnType.SWA_KV,
        }

        for layer_idx in range(len(self.v4.layers)):
            if layer_idx >= len(self._layer_pools):
                continue
            for attn_type_int in self._layer_pools[layer_idx]:
                gid = attn_type_to_gid.get(attn_type_int)
                attn_type_enum = attn_type_enum_by_int.get(attn_type_int)
                if gid is None or attn_type_enum is None or gid >= len(by_group_host):
                    continue
                block_ids_2d = by_group_host[gid]
                if block_ids_2d is None or block_ids_2d.numel() == 0:
                    continue
                # LINEAR pools (state pools 4/5/6 + SWA_KV 7) have only 2
                # valid slots per request (DSV4 linear_fixed_cap=2). The
                # framework's per-pool block_id tensor is padded to the
                # FULL group's max_blocks_num, so slice to the first 2
                # entries — otherwise ExecOps reads the zero-padding past
                # the valid slots, registering wrong block_ids and using
                # cache_keys[max-2..max-1] which decode (which sees a
                # 2-entry block_ids vector, indices 0/1) won't match.
                if attn_type_int in (4, 5, 6, 7):
                    block_ids_2d = block_ids_2d[:, :2].contiguous()
                try:
                    layer_kv = self.kv_cache.get_layer_cache(layer_idx, attn_type_enum)
                except Exception:
                    continue
                pool_t = layer_kv.kv_cache_base
                if pool_t is None or pool_t.numel() == 0 or pool_t.dim() != 2:
                    continue
                # pool_t.size(1) is in ELEMENTS (kv_block_stride_elems per
                # MemoryLayoutStrategy::processKVTensor reshape). State pools
                # use fp32 (element_size=4) so per-block bytes is 4× the
                # element count; paged pools use uint8 (element_size=1) so
                # the count is already in bytes. Either way, multiply by
                # element_size() to get the per-block stride in bytes —
                # which matches what convertIndexToBuffer reports on decode.
                stride_bytes = int(pool_t.size(1)) * int(pool_t.element_size())
                if stride_bytes <= 0:
                    continue
                compute_ops.write_cache_store(
                    attn.input_lengths,
                    attn.prefix_lengths,
                    block_ids_2d,
                    cache_store_inputs,
                    layer_kv,
                    group_id=gid,
                    kv_block_stride_bytes=stride_bytes,
                )

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

        from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImpl,
            DSv4DecodeFmhaImplConfig,
        )

        bs = int(attn.input_lengths.size(0)) if attn.input_lengths.numel() > 0 else 1
        device = next(self.v4.parameters()).device

        # Phase 2: derive paged_pool_specs from wired PoolDescriptors so the
        # impl pre-allocates the right block_table / slot_mapping buffers.
        # We only wire the pools whose decode WRITE path has been migrated
        # so far — this turn just SWA. Other entries can be added piecewise.
        paged_pool_specs = self._build_paged_pool_specs_for_phase2()

        cfg = DSv4DecodeFmhaImplConfig(
            max_batch_size=bs,
            q_len=1,
            window_size=int(self._v4_args.window_size),
            head_dim=int(self._v4_args.head_dim),
            max_seq_len=int(self._v4_args.max_seq_len),
            compress_ratios=list(self._v4_args.compress_ratios)[
                : self._v4_args.n_layers
            ],
            index_topk=int(self._v4_args.index_topk),
            paged_pool_specs=paged_pool_specs,
        )
        impl = DSv4DecodeFmhaImpl(
            cfg,
            device=device,
            attn_inputs=attn,
        )
        # Stash the per-layer pool descriptors on the metadata so
        # ``Attention.forward_decode`` can find its pool view at the
        # right layer index without going through the model.
        if getattr(self, "_layer_pool_descs", None):
            impl.metadata.layer_pool_descs = self._layer_pool_descs
        return impl

    def _build_paged_pool_specs_for_phase2(self):
        """Per-attn_type ``(entries_per_block, max_blocks_per_req)`` for the
        decode-FMHA impl's metadata pre-allocation.

        Phase 2 only wires SWA writes. We over-allocate ``max_blocks_per_req``
        slightly so prepare() never has to grow buffers (would void the
        forbid_realloc guarantee). Entries-per-block is read from the
        actual framework allocation via PoolDescriptor (single source of
        truth — see POOL_LAYOUT.md).
        """
        from rtp_llm.models_py.modules.dsv4.decode.pool_layout import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )

        if not getattr(self, "_layer_pool_descs", None):
            return {}
        # Per "不动原则" pools 3-6 get fixed 2 blocks/req. SWA is a
        # 128-entry ring → 2 blocks (256 entries/block) is plenty.
        # HCA/INDEXER are intrinsically lossy under this allocation
        # for long seqs — accepted by design.
        specs: Dict[int, Tuple[int, int]] = {}
        first = self._layer_pool_descs[0] if self._layer_pool_descs else {}
        # CSA_KV is read-only in decode (frozen prefill data) but needed
        # for Phase 2B-2b dual-pool paged read path.
        for at in (SWA_KV, HCA_KV, INDEXER_KV, CSA_KV):
            for descs in self._layer_pool_descs:
                if at in descs:
                    specs[at] = (descs[at].entries_per_block, 2)
                    break
            _ = first  # silence linter — kept for potential future use
        return specs

    def _forward_decode(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        """Phase 2 batched decode arm. Builds DSv4DecodeAttnMetadata from
        the framework's per-request sequence_lengths, then runs through
        ``V4Transformer.forward_decode``. Prefill path is untouched.
        """
        from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
            build_decode_metadata,
        )
        from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImpl,
        )

        attn = inputs.attention_inputs
        param_dev = next(self.v4.parameters()).device

        # input_ids: framework packs as flat [T_total] for B requests, q_len=1.
        input_ids = inputs.input_ids
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.device != param_dev:
            input_ids = input_ids.to(param_dev)

        if isinstance(fmha_impl, DSv4DecodeFmhaImpl):
            # CUDA-graph path: metadata was populated either in __init__
            # (initial dtype-check forward) or by C++ ``prepare_cuda_graph``
            # before each replay. Reading attn.sequence_lengths here would
            # trigger a CPU→CUDA copy that's illegal during stream capture.
            meta = fmha_impl.metadata
        else:
            # Eager path: build metadata inline from sequence_lengths.
            # sequence_lengths[r] = absolute pos of NEW token's predecessor
            # (per NormalModelInputGatherer.cc:255).
            seq_lens_d = attn.sequence_lengths
            if seq_lens_d.device.type == "cpu":
                seq_lens_d = seq_lens_d.to(param_dev)
            start_pos = seq_lens_d.to(torch.int32)
            B = start_pos.shape[0]
            if B == 0:
                return PyModelOutputs(
                    torch.zeros(
                        (0, self._v4_args.dim),
                        dtype=torch.bfloat16,
                        device=param_dev,
                    ),
                )
            # Clamp for warmup safety (probe at max_seq_len then decode).
            max_s = self._v4_args.max_seq_len
            start_pos = torch.clamp(start_pos, min=0, max=max(0, max_s - 1))

            # Phase 2 paged: pull per-attn_type block_tables from
            # attn_inputs and feed build_decode_metadata; eager allocates
            # fresh per step (no graph capture, so no forbid_realloc).
            paged_specs = self._build_paged_pool_specs_for_phase2()
            paged_bts: Dict[int, Any] = {}
            paged_ebs: Dict[int, int] = {}
            if paged_specs:
                from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
                    ATTN_TYPE_TO_GROUP_ID,
                )

                by_group = getattr(
                    attn, "kv_cache_kernel_block_id_device_by_group", None
                )
                if by_group is not None and len(by_group) > 0:
                    for at, (eb, _) in paged_specs.items():
                        gid = ATTN_TYPE_TO_GROUP_ID.get(at)
                        if gid is None or gid >= len(by_group):
                            continue
                        bt = by_group[gid]
                        if bt is None or bt.numel() == 0:
                            continue
                        paged_bts[at] = bt
                        paged_ebs[at] = eb

            meta = build_decode_metadata(
                start_pos=start_pos,
                q_len=1,
                window_size=self._v4_args.window_size,
                head_dim=self._v4_args.head_dim,
                max_seq_len=max_s,
                compress_ratios=list(self._v4_args.compress_ratios)[
                    : self._v4_args.n_layers
                ],
                index_topk=self._v4_args.index_topk,
                device=param_dev,
                paged_block_tables=paged_bts or None,
                paged_pool_entries_per_block=paged_ebs or None,
            )
            if getattr(self, "_layer_pool_descs", None):
                meta.layer_pool_descs = self._layer_pool_descs

        hidden = self.v4.forward_decode(input_ids, meta)  # [B, dim] packed
        return PyModelOutputs(hidden)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        # pybind11's object_api operator() is noexcept — a Python exception
        # would trigger __unexpected → abort, hiding the real error. Log the
        # traceback explicitly before re-raising so the cause survives the
        # noexcept boundary.
        try:
            return self._forward_impl(inputs, fmha_impl)
        except BaseException as e:
            import traceback

            logging.error(
                "[DeepSeekV4Model] forward() raised %s: %s\nTraceback:\n%s",
                type(e).__name__,
                e,
                traceback.format_exc(),
            )
            raise

    def _forward_impl(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        # Framework sends flat [total_tokens] input_ids with cu_seqlens packing.
        # Use attention_inputs fields to properly handle batched requests:
        #   Prefill: input_lengths[b], prefix_lengths[b], cu_seqlens
        #   Decode:  sequence_lengths[b] (per-batch current position)
        # Prefill and decode are NEVER mixed (framework guarantee).
        input_ids: torch.Tensor = inputs.input_ids  # flat [total_tokens]
        attn = inputs.attention_inputs
        is_prefill = bool(attn.is_prefill) if attn is not None else True

        # Decode arm.  Two paths share ``_forward_decode``:
        #   * fmha_impl is a DSv4DecodeFmhaImpl → CUDA-graph replay with
        #     persistent decode metadata.
        #   * fmha_impl is None (eager / --enable_cuda_graph 0) → build
        #     decode metadata inline inside ``_forward_decode``.
        # Eager mode previously fell through to the unified prefill
        # ``forward()`` path with q_len=1, which silently bypassed
        # ``Attention.forward_decode`` (and thus the FP8/FlashMLA decode
        # ops).  Routing both paths through ``_forward_decode`` keeps
        # graph and eager behavior in lock-step so capture-time and
        # replay-time semantics match.
        if (
            not is_prefill
            and attn is not None
            and attn.sequence_lengths is not None
            and attn.sequence_lengths.numel() > 0
        ):
            return self._forward_decode(inputs, fmha_impl)

        # Deferred wiring: framework allocates kv_cache after initialize(),
        # so we wire on first forward() when kv_cache becomes available.
        if getattr(self, "_framework_kv_pending", False) and self.kv_cache is not None:
            device_str = str(input_ids.device)
            self._wire_framework_kv_cache(device_str)
            self._framework_kv_pending = False
            logging.info(
                "[DeepSeekV4Model] framework KV cache wired (deferred) for %d layers",
                len(self.v4.layers),
            )

        # Re-wire if the framework rebuilt the KV cache (warmup→real
        # transition). NormalEngine first allocates a stub allocator with
        # block_num=1 to measure GPU memory, then tears it down and creates
        # the real allocator with the measured block_num. The stub's pool
        # tensors get freed; without re-wiring, gather hits IndexError on
        # the dead warmup tensors.
        if self.kv_cache is not None and getattr(self, "_framework_kv_enabled", False):
            cur_id = id(self.kv_cache)
            cur_first_ptr = None
            flat_now = getattr(self.kv_cache, "kv_cache_base_by_layer_attn_flat", None)
            if flat_now is not None:
                for t in flat_now:
                    if t is not None and t.numel() > 0:
                        cur_first_ptr = t.data_ptr()
                        break
            wired_id = getattr(self, "_wired_kv_cache_id", None)
            wired_ptr = getattr(self, "_wired_first_data_ptr", None)
            if cur_id != wired_id or (
                cur_first_ptr is not None
                and wired_ptr is not None
                and cur_first_ptr != wired_ptr
            ):
                logging.info(
                    "[DeepSeekV4Model] kv_cache changed (id %s→%s, first_ptr %s→%s); re-wiring",
                    wired_id,
                    cur_id,
                    wired_ptr,
                    cur_first_ptr,
                )
                self._wire_framework_kv_cache(str(input_ids.device))

        # --- Context Parallel setup (before forward loop) ---
        # Must be before self.v4(...) — the transformer reads self.v4._cp_info
        # at forward time, so a stale CP context (e.g. from warmup, where
        # padded_seq_len comes from max_seq_len) would mismatch the current
        # request's chunk_length and trip the assertion in build_cp_context.
        pc_cfg = getattr(self, "parallelism_config", None)
        cp_enabled = (
            pc_cfg is not None
            and getattr(pc_cfg, "prefill_cp_config", None) is not None
            and pc_cfg.prefill_cp_config.is_enabled()
            and is_prefill
            and attn is not None
            and getattr(attn, "context_parallel_info", None) is not None
        )
        if cp_enabled:
            self.v4.set_cp_info(
                cp_info=attn.context_parallel_info,
                cp_size=int(pc_cfg.tp_size),
                cp_rank=int(pc_cfg.tp_rank),
            )
        else:
            self.v4.set_cp_info(None, 1, 0)

        if is_prefill:
            # Prefill: split by cu_seqlens, process each request separately
            # DSV4 sparse attention doesn't support cu_seqlens batched prefill
            input_lengths_t = attn.input_lengths if attn is not None else None
            prefix_lengths_t = attn.prefix_lengths if attn is not None else None
            n_prefill = input_lengths_t.size(0) if input_lengths_t is not None else 1

            all_hidden = []
            offset = 0
            for b in range(n_prefill):
                # Reset compressor state before each prefill to prevent leakage
                self._reset_compressor_state()
                if input_lengths_t is not None:
                    inp_len = int(input_lengths_t[b].item())
                    prefix_len = (
                        int(prefix_lengths_t[b].item())
                        if prefix_lengths_t is not None and prefix_lengths_t.numel() > b
                        else 0
                    )
                else:
                    inp_len = input_ids.size(0)
                    prefix_len = 0

                batch_ids = input_ids[offset : offset + inp_len].unsqueeze(
                    0
                )  # [1, inp_len]
                start_pos = prefix_len  # reuse offset

                # Gather from BlockPool if reuse hit
                if (
                    getattr(self, "_framework_kv_enabled", False)
                    and attn is not None
                    and prefix_len > 0
                ):
                    self._gather_all_layers(
                        attn, B=1, batch_offset=b, reuse_gather=True
                    )
                    # After gather, State pools restore kv_state/score_state.
                    # For CSA (overlap=True), kv_state[:, :ratio] holds the
                    # overlap carry from the prefix tail — keep it so the
                    # compressor can inject it into window 0's overlap slots.
                    # Only zero the partial-window portion (offset onwards).
                    for layer in self.v4.layers:
                        a = layer.attn
                        if a.compressor is not None:
                            r = a.compressor.compress_ratio
                            off = r if a.compressor.overlap else 0
                            a.compressor.kv_state[:1, off:].zero_()
                            a.compressor.score_state[:1, off:].fill_(float("-inf"))
                        if a.indexer is not None:
                            ic = a.indexer.compressor
                            ir = ic.compress_ratio
                            ioff = ir if ic.overlap else 0
                            ic.kv_state[:1, ioff:].zero_()
                            ic.score_state[:1, ioff:].fill_(float("-inf"))

                # Phase B: bind prefill paged dual-write ctx per-layer so
                # Attention.forward mirrors the SWA ring-buffer into the
                # framework BlockPool. Additive alongside _scatter_all_layers.
                paged_bound = self._bind_prefill_paged_ctx(attn, batch_offset=b)

                h = self.v4(batch_ids, start_pos=start_pos, apply_lm_head=False)
                all_hidden.append(h)

                if paged_bound:
                    self._clear_prefill_paged_ctx()

                # Scatter to BlockPool after forward
                if getattr(self, "_framework_kv_enabled", False) and attn is not None:
                    self._scatter_all_layers(attn, B=1, batch_offset=b)

                self._running_pos = prefix_len + inp_len
                offset += inp_len

            # PD separation: register all (layer × pool) block_ids in
            # cache_store so the decode side can fetch them. Done after the
            # whole prefill loop because the cache_store hand-off iterates
            # all batches internally via context_batch_size.
            if getattr(self, "_framework_kv_enabled", False) and attn is not None:
                self._write_pd_cache_store(attn)

            hidden = (
                torch.cat(all_hidden, dim=1) if len(all_hidden) > 1 else all_hidden[0]
            )
            reuse_len = (
                int(prefix_lengths_t[0].item())
                if prefix_lengths_t is not None and prefix_lengths_t.numel() > 0
                else 0
            )

        else:
            # Decode: B requests each 1 token
            # Use sequence_lengths as per-batch positions (tensor [B])
            seq_lens = attn.sequence_lengths if attn is not None else None

            if seq_lens is not None and seq_lens.numel() > 0:
                B = seq_lens.size(0)
                start_pos = seq_lens.to(
                    input_ids.device
                )  # tensor [B] for batched decode
            else:
                B = input_ids.size(0)
                start_pos = self._running_pos

            if input_ids.dim() == 1:
                input_ids = input_ids[:B].unsqueeze(1)  # [B, 1]

            # Paged decode read+write is the steady state: KV pools are kept
            # fresh by paged dual-write each step and read directly via
            # paged_topk_translator, and compressor/indexer state lives in
            # ``compressor.kv_state`` across steps. The per-step gather/scatter
            # round-trip is therefore unnecessary on this path.
            hidden = self.v4(input_ids, start_pos=start_pos, apply_lm_head=False)

            self._running_pos += 1
            reuse_len = 0

        # Return flat [total_tokens, hidden_dim] for engine
        flat = hidden.reshape(-1, hidden.size(-1))
        return PyModelOutputs(flat)
