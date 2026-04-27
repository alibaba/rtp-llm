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

        # Wire framework KV cache if enabled — note: self.kv_cache may not
        # be set yet (framework allocates it after initialize() returns).
        # We defer wiring to the first forward() call via _framework_kv_pending.
        self._framework_kv_pending = os.environ.get("DSV4_USE_FRAMEWORK_KV", "0") == "1"
        if self._framework_kv_pending and self.kv_cache is not None:
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
        if flat is not None and len(flat) > 0:
            num_layers = len(flat) // attn_type_count
            for i in range(len(self.v4.layers)):
                slots = [None] * attn_type_count
                if i < num_layers:
                    for a in range(attn_type_count):
                        t = flat[i * attn_type_count + a]
                        if t is not None and t.numel() > 0:
                            slots[a] = t
                self._fw_pool_tensors.append(slots)
        else:
            logging.warning(
                "[DeepSeekV4Model] kv_cache_base_by_layer_attn_flat empty, gather/scatter disabled"
            )
            return

        # Build per-layer pool list from compress_ratio
        self._layer_pools = []
        for i, layer in enumerate(self.v4.layers):
            ratio = layer.attn.compress_ratio
            if ratio == 4:
                self._layer_pools.append([7, 1, 3, 4, 5])
            elif ratio == 128:
                self._layer_pools.append([7, 2, 6])
            else:
                self._layer_pools.append([7])

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

    def _gather_all_layers(self, attn_inputs, B=1, batch_offset=0):
        """Gather cached KV from BlockPool pages into each layer's buffers.

        batch_offset: row offset into by_group block_ids. In prefill loop,
        request b uses batch_offset=b so we read the correct block_ids row.
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

            for attn_type in self._layer_pools[i]:
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

    def _scatter_all_layers(self, attn_inputs, B=1, batch_offset=0):
        """Scatter updated KV from each layer's buffers back to BlockPool pages.

        batch_offset: row offset into by_group block_ids (same as gather).
        """
        if not hasattr(self, "_fw_pool_tensors") or not self._fw_pool_tensors:
            return

        by_group = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device_by_group", None
        )
        if by_group is None or len(by_group) == 0:
            return

        attn_type_to_gid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}

        for i, layer in enumerate(self.v4.layers):
            attn_mod = layer.attn
            win = attn_mod.window_size
            hd = attn_mod.head_dim

            for attn_type in self._layer_pools[i]:
                gid = attn_type_to_gid.get(attn_type)
                if gid is None or gid >= len(by_group):
                    continue
                block_ids = by_group[gid]
                if block_ids is None or block_ids.numel() == 0:
                    continue
                bid_slice = block_ids[batch_offset : batch_offset + B]
                fw_t = self._get_pool_tensor(attn_type, i)

                if attn_type == 7:  # SWA_KV
                    swa_buf = attn_mod.kv_cache[:B, :win]
                    self._scatter_kv_pool(fw_t, bid_slice, swa_buf, 256, hd, B)
                elif attn_type == 1:  # CSA_KV
                    if (
                        attn_mod.compressor is not None
                        and attn_mod.compressor.kv_cache is not None
                    ):
                        self._scatter_kv_pool(
                            fw_t, bid_slice, attn_mod.compressor.kv_cache[:B], 64, hd, B
                        )
                elif attn_type == 2:  # HCA_KV
                    if (
                        attn_mod.compressor is not None
                        and attn_mod.compressor.kv_cache is not None
                    ):
                        self._scatter_kv_pool(
                            fw_t, bid_slice, attn_mod.compressor.kv_cache[:B], 2, hd, B
                        )
                elif attn_type == 3:  # INDEXER_KV
                    if attn_mod.indexer is not None:
                        idx_hd = attn_mod.indexer.head_dim
                        self._scatter_kv_pool(
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

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        """V4 uses its own sparse/compressed attention internally — no standard FMHA
        backend fits (64 Q heads × 1 KV head × head_dim=512 with CSA/HCA/SWA per-layer
        variants). Return None so the framework's fmha machinery is bypassed."""
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        # Framework sends flat [total_tokens] input_ids with cu_seqlens packing.
        # Use attention_inputs fields to properly handle batched requests:
        #   Prefill: input_lengths[b], prefix_lengths[b], cu_seqlens
        #   Decode:  sequence_lengths[b] (per-batch current position)
        # Prefill and decode are NEVER mixed (framework guarantee).
        input_ids: torch.Tensor = inputs.input_ids  # flat [total_tokens]
        attn = inputs.attention_inputs
        is_prefill = bool(attn.is_prefill) if attn is not None else True

        use_framework_kv = os.environ.get("DSV4_USE_FRAMEWORK_KV", "0") == "1"

        # Deferred wiring: framework allocates kv_cache after initialize(),
        # so we wire on first forward() when kv_cache becomes available.
        if (
            use_framework_kv
            and getattr(self, "_framework_kv_pending", False)
            and self.kv_cache is not None
        ):
            device_str = str(input_ids.device)
            self._wire_framework_kv_cache(device_str)
            self._framework_kv_pending = False
            logging.info(
                "[DeepSeekV4Model] framework KV cache wired (deferred) for %d layers",
                len(self.v4.layers),
            )

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
                    use_framework_kv
                    and hasattr(self, "_framework_kv_enabled")
                    and self._framework_kv_enabled
                    and attn is not None
                    and prefix_len > 0
                ):
                    self._gather_all_layers(attn, B=1, batch_offset=b)

                h = self.v4(batch_ids, start_pos=start_pos, apply_lm_head=False)
                all_hidden.append(h)

                # Scatter to BlockPool after forward
                if (
                    use_framework_kv
                    and hasattr(self, "_framework_kv_enabled")
                    and self._framework_kv_enabled
                    and attn is not None
                ):
                    self._scatter_all_layers(attn, B=1, batch_offset=b)

                self._running_pos = prefix_len + inp_len
                offset += inp_len

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

            if use_framework_kv and seq_lens is not None and seq_lens.numel() > 0:
                B = seq_lens.size(0)
                start_pos = seq_lens.to(
                    input_ids.device
                )  # tensor [B] for batched decode
            else:
                B = input_ids.size(0)
                start_pos = self._running_pos

            if input_ids.dim() == 1:
                input_ids = input_ids[:B].unsqueeze(1)  # [B, 1]

            # Gather KV from BlockPool before decode
            if (
                use_framework_kv
                and hasattr(self, "_framework_kv_enabled")
                and self._framework_kv_enabled
                and attn is not None
            ):
                self._gather_all_layers(attn, B=B)

            hidden = self.v4(input_ids, start_pos=start_pos, apply_lm_head=False)

            # Scatter KV back to BlockPool after decode
            if (
                use_framework_kv
                and hasattr(self, "_framework_kv_enabled")
                and self._framework_kv_enabled
                and attn is not None
            ):
                self._scatter_all_layers(attn, B=B)

            self._running_pos += 1
            reuse_len = 0

        # --- Context Parallel setup (before forward loop) ---
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

        # Return flat [total_tokens, hidden_dim] for engine
        flat = hidden.reshape(-1, hidden.size(-1))
        return PyModelOutputs(flat)
