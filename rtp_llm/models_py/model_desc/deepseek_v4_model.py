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
from typing import Any, Optional

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


def _args_from_model_config(mc: ModelConfig) -> V4Args:
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
        max_batch_size=1,
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
        args = _args_from_model_config(model_config)
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
                cp_size, args.max_tokens_per_rank, new_bound,
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

        self._materialized = True
        self._running_pos = 0

        # Wire framework KV cache if enabled
        if (
            os.environ.get("DSV4_USE_FRAMEWORK_KV", "0") == "1"
            and self.kv_cache is not None
        ):
            self._wire_framework_kv_cache(device_str)
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
        """
        for layer in self.v4.layers:
            attn = layer.attn
            # Reset kv_cache (SWA + compressed KV)
            attn.kv_cache.zero_()
            # Reset compressor state
            if attn.compressor is not None:
                attn.compressor.kv_state.zero_()
                attn.compressor.score_state.fill_(float("-inf"))
                attn.compressor.kv_cache = None  # force rebind
            # Reset indexer state
            if attn.indexer is not None:
                attn.indexer.kv_cache.zero_()
                attn.indexer.compressor.kv_state.zero_()
                attn.indexer.compressor.score_state.fill_(float("-inf"))
                attn.indexer.compressor.kv_cache = None

    def _wire_framework_kv_cache(self, device: str):
        """Store references to framework BlockPool tensors for gather/scatter.

        Does NOT do flat view — keeps register_buffer as working memory.
        In forward(), we gather from BlockPool pages into flat buffers before
        computation, and scatter back after computation. This ensures:
        - reuse cache works (cached data is in specific block_ids)
        - PD separation works (cache store transfers BlockPool raw bytes)
        - concurrent requests work (each request has independent block_ids)
        """
        self._framework_kv_enabled = True
        self._fw_layer_tensors = []  # per-layer BlockPool tensor references
        if self.kv_cache is None or not self.kv_cache.kv_cache_base_by_layer:
            logging.warning("[DeepSeekV4Model] no framework kv_cache, skipping wiring")
            return

        for i in range(len(self.v4.layers)):
            if i < len(self.kv_cache.kv_cache_base_by_layer):
                self._fw_layer_tensors.append(self.kv_cache.kv_cache_base_by_layer[i])
            else:
                self._fw_layer_tensors.append(None)

        logging.info(
            "[DeepSeekV4Model] framework BlockPool KV wired for %d layers "
            "(gather/scatter mode, seq_size_per_block=%d)",
            len(self._fw_layer_tensors),
            self.kv_cache.seq_size_per_block if self.kv_cache else 0,
        )

    def _gather_all_layers(self, attn_inputs):
        """Gather cached KV from BlockPool pages into each layer's flat buffer.

        For each layer, uses block_ids to copy data from BlockPool pages into
        the flat [B, kv_cache_size, head_dim] register_buffer that attention reads.
        """
        if not hasattr(self, "_fw_layer_tensors") or not self._fw_layer_tensors:
            return
        from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer

        spb = (
            self.kv_cache.seq_size_per_block
            if self.kv_cache.seq_size_per_block > 0
            else 256
        )

        for i, layer in enumerate(self.v4.layers):
            fw_tensor = (
                self._fw_layer_tensors[i] if i < len(self._fw_layer_tensors) else None
            )
            if fw_tensor is None or fw_tensor.numel() == 0:
                continue

            gid = select_block_map_for_layer(attn_inputs, i)
            if gid is None:
                continue

            block_ids = attn_inputs.kv_cache_kernel_block_id_device
            if block_ids is None or block_ids.numel() == 0:
                continue

            attn_mod = layer.attn
            kv_cache = attn_mod.kv_cache  # [B, kv_cache_size, head_dim]
            head_dim = attn_mod.head_dim
            batch_block_ids = block_ids[0]  # B=1

            for b_idx in range(batch_block_ids.size(0)):
                bid = int(batch_block_ids[b_idx].item())
                if bid < 0:
                    continue
                start = b_idx * spb
                end = min(start + spb, kv_cache.size(1))
                n = end - start
                if n <= 0:
                    continue
                page = fw_tensor[bid]
                kv_cache[0, start:end] = page[: n * head_dim].view(n, head_dim)

    def _scatter_all_layers(self, attn_inputs):
        """Scatter updated KV from each layer's flat buffer back to BlockPool pages.

        After forward computation, copies the updated KV data from flat buffers
        back to the BlockPool pages so the framework can manage/transfer them.
        """
        if not hasattr(self, "_fw_layer_tensors") or not self._fw_layer_tensors:
            return
        from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer

        spb = (
            self.kv_cache.seq_size_per_block
            if self.kv_cache.seq_size_per_block > 0
            else 256
        )

        for i, layer in enumerate(self.v4.layers):
            fw_tensor = (
                self._fw_layer_tensors[i] if i < len(self._fw_layer_tensors) else None
            )
            if fw_tensor is None or fw_tensor.numel() == 0:
                continue

            gid = select_block_map_for_layer(attn_inputs, i)
            if gid is None:
                continue

            block_ids = attn_inputs.kv_cache_kernel_block_id_device
            if block_ids is None or block_ids.numel() == 0:
                continue

            attn_mod = layer.attn
            kv_cache = attn_mod.kv_cache
            head_dim = attn_mod.head_dim
            batch_block_ids = block_ids[0]

            for b_idx in range(batch_block_ids.size(0)):
                bid = int(batch_block_ids[b_idx].item())
                if bid < 0:
                    continue
                start = b_idx * spb
                end = min(start + spb, kv_cache.size(1))
                n = end - start
                if n <= 0:
                    continue
                fw_tensor[bid, : n * head_dim] = kv_cache[0, start:end].reshape(-1)

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        """V4 uses its own sparse/compressed attention internally — no standard FMHA
        backend fits (64 Q heads × 1 KV head × head_dim=512 with CSA/HCA/SWA per-layer
        variants). Return None so the framework's fmha machinery is bypassed."""
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        # Framework expects flat [total_tokens] input_ids (cu_seqlens-packed batching).
        # We support B=1 only; extract as [1, S].
        input_ids: torch.Tensor = inputs.input_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, S]

        attn = inputs.attention_inputs
        is_prefill = (
            bool(attn.is_prefill) if attn is not None else (input_ids.size(1) > 1)
        )

        use_framework_kv = os.environ.get("DSV4_USE_FRAMEWORK_KV", "0") == "1"

        # When framework KV cache is active, use prefix_lengths from attention_inputs
        # to determine reuse offset. The framework reuses cached KV blocks for the
        # prefix and only sends suffix tokens for computation. start_pos must reflect
        # the reuse offset so attention reads cached KV correctly.
        reuse_len = 0
        if use_framework_kv and is_prefill and attn is not None:
            prefix_lengths = getattr(attn, "prefix_lengths", None)
            if prefix_lengths is not None and prefix_lengths.numel() > 0:
                reuse_len = int(prefix_lengths[0].item())

        if is_prefill:
            start_pos = reuse_len  # 0 if no reuse, reuse_len if cache hit
        else:
            start_pos = self._running_pos

        # --- Context Parallel (prefill) — scaffold integration ----------
        #
        # When CP is enabled, the framework zigzag-splits the prefill
        # tokens across the TP group (TP is repurposed as the CP group
        # in RTP-LLM's CP design).  Each rank sees a rank-local slice
        # plus an ``attention_inputs.context_parallel_info`` struct with
        # the restore metadata.
        #
        # For V4 we run the attention/FFN/MoE path RANK-LOCAL (so MoE
        # EP dispatch naturally sees only 1/cp_size of the tokens per
        # rank, avoiding the 4× duplicated-dispatch bug that an adapter-
        # level input-gather would cause).  Only the S-pooling ops
        # (compressor, indexer) need the full sequence; they do their
        # own all-gather internally (see ``Compressor`` and ``Indexer``
        # forward, S8 CP scaffold).  The hidden passed back to the
        # engine is therefore rank-local by construction — no exit
        # gather / scatter needed here.
        #
        # What this scaffold does NOT do yet:
        #   * per-rank-Q attention sharding (attention still runs on
        #     full sequence → no compute speedup, only memory win)
        #   * paged KV cache CP writes (V4 keeps a mock per-module
        #     kv_cache; each rank holds the full KV, written from the
        #     gathered compressor output)
        # See docs/dsv4/parallel_design.md §S8 for the staged plan.
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
            # Bind CP metadata onto V4Transformer so nested modules
            # (Compressor / Indexer) can pick it up without threading
            # yet another argument through every forward signature.
            self.v4.set_cp_info(
                cp_info=attn.context_parallel_info,
                cp_size=int(pc_cfg.tp_size),
                cp_rank=int(pc_cfg.tp_rank),
            )
            if os.environ.get("DSV4_LOG_CP", "0") == "1":
                cpi = attn.context_parallel_info
                logging.info(
                    "[DeepSeekV4Model] CP forward: input_ids.shape=%s "
                    "cp_size=%d cp_rank=%d padding_mask.shape=%s "
                    "restore_indices.shape=%s is_prefill=%s start_pos=%d",
                    tuple(input_ids.shape),
                    int(pc_cfg.tp_size),
                    int(pc_cfg.tp_rank),
                    tuple(cpi.prefill_qkv_padding_mask.shape),
                    tuple(cpi.prefill_qkv_restore_indice.shape),
                    is_prefill,
                    start_pos,
                )
        else:
            self.v4.set_cp_info(None, 1, 0)

        # Warmup safety: framework warmup probes with prefill(max_seq_len)
        # followed by decode(1), pushing start_pos out of our freqs_cis /
        # kv_cache range.  Under DP+EP, short-circuiting the forward to
        # a zero tensor would desync the DeepEP dispatch collective; so
        # we instead clamp start_pos to the last valid slot. The warmup
        # output is discarded anyway.
        max_s = self._v4_args.max_seq_len
        S_local = input_ids.size(1)
        if start_pos + max(S_local, 1) > max_s:
            start_pos = max(0, max_s - max(S_local, 1))

        # Empty-batch handling (DP rank with zero local tokens):
        # V4's attention/mHC/indexer contains many ops that crash on S=0
        # (``.view``, ``.unflatten``, ``einsum`` on zero-element tensors,
        # plus ``F.softplus`` returning "unknown parameter type").  We
        # can't short-circuit the whole forward because DeepEP dispatch
        # is a collective that ALL ranks must enter.  Instead, pad S=0
        # → S=1 with a dummy token, run through the full layer stack
        # (including DeepEP), then discard the dummy's output.
        pad_empty = S_local == 0
        if pad_empty:
            # Create the dummy on the model's device (not input_ids.device
            # — the framework may pass a CPU tensor for empty inputs).
            param_dev = next(self.v4.parameters()).device
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=param_dev)
            S_local = 1

        # On-demand timeline capture for exactly one forward when trigger file exists.
        should_capture = (
            self._profile_path
            and not self._profile_done
            and os.path.exists(self._profile_trigger)
        )
        # Framework KV gather/scatter: before forward, gather cached KV from
        # BlockPool pages into flat buffers; after forward, scatter back.
        fw_active = (
            use_framework_kv
            and hasattr(self, "_framework_kv_enabled")
            and self._framework_kv_enabled
            and attn is not None
        )

        if fw_active and reuse_len > 0:
            # Gather cached KV from BlockPool for reused prefix
            self._gather_all_layers(attn)

        if should_capture:
            try:
                os.remove(self._profile_trigger)
            except OSError:
                pass
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                with torch.profiler.record_function(
                    f"V4_forward_prefill={is_prefill}_S={input_ids.size(1)}"
                ):
                    hidden = self.v4(
                        input_ids, start_pos=start_pos, apply_lm_head=False
                    )
                    torch.cuda.synchronize()
            prof.export_chrome_trace(self._profile_path)
            logging.info(
                "[DeepSeekV4Model] timeline exported: %s (%d events)",
                self._profile_path,
                len(prof.key_averages()),
            )
            self._profile_done = True
        else:
            hidden = self.v4(input_ids, start_pos=start_pos, apply_lm_head=False)

        # After forward: scatter updated KV back to BlockPool pages
        if fw_active:
            self._scatter_all_layers(attn)

        # Advance running position.  Under CP the rank-local ``S`` is
        # ``chunk_length`` (~ S_full / cp_size) — bookkeeping ``_running_pos``
        # against it would leave decode reading the wrong kv_cache slots.
        # Use the REAL prefill length from ``cp_info.prefill_actual_input_lengths_cpu``
        # when CP is enabled, else fall back to ``S``.
        if is_prefill:
            if cp_enabled:
                # prefill_actual_input_lengths_cpu: int32 [num_prefill_streams]
                # V4 is B=1, single stream.
                actual_lens = (
                    attn.context_parallel_info.prefill_actual_input_lengths_cpu
                )
                self._running_pos = int(actual_lens[-1].item())
            else:
                # Total position = reuse_len (cached prefix) + suffix tokens sent
                self._running_pos = reuse_len + input_ids.size(1)
        else:
            self._running_pos += input_ids.size(1)

        # Discard the dummy token we padded in for the empty-batch case.
        if pad_empty:
            hidden = hidden[:, :0]

        # Return flat [total_tokens, hidden_dim] for engine
        flat = hidden.reshape(-1, hidden.size(-1))
        if os.environ.get("DSV4_LOG_CP", "0") == "1":
            logging.info(
                "[DeepSeekV4Model] forward return: hidden.shape=%s flat.shape=%s "
                "cp_enabled=%s is_prefill=%s",
                tuple(hidden.shape),
                tuple(flat.shape),
                cp_enabled,
                is_prefill,
            )
        return PyModelOutputs(flat)
