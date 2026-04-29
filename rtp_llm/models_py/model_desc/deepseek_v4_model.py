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
from typing import Any, Dict, List, Optional, Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.models_py.modules.dsv4.weight_loader import (
    load_v4_safetensors,
    load_v4_weights_dict,
)


class _RequestSlotAllocator:
    """Maps framework request identifiers (first-block-id) to stable rows
    in DSv4's per-layer KV cache pool.

    DSv4 keeps its own [N_concurrent, T, D] per-layer KV cache pool (this
    class chooses N_concurrent at the model level). For each framework
    request the first block_id is a stable per-request key. We assign a
    fresh pool row to every new block_id and remember it — every later
    decode step for the same request resolves to the same row. Block_ids
    not seen in the current batch keep their row reserved (LRU eviction
    only when the pool is full).

    Block_id reuse: the framework recycles block_ids after a request
    finishes. To distinguish "new request happens to get a recycled
    block_id" from "decode step of the same request", the caller MUST
    pass ``is_prefill``. On prefill we always treat every block_id as
    a fresh request — drop any stale mapping and assign a new slot, then
    return the slot in ``slots_needing_reset`` so the stash pool entry
    can be cleared before the prefill writes to it.
    """

    def __init__(self, num_slots: int):
        self.num_slots = int(num_slots)
        self._block_to_slot: Dict[int, int] = {}
        self._lru: List[int] = []  # block_ids, MRU at the end
        # Free-slot pool. Tracks which slot indices in [0, num_slots) are
        # currently NOT mapped to a block_id. Picking the next slot via
        # ``len(self._block_to_slot)`` was a latent collision bug: after
        # freeing a low slot and adding a new mapping, ``len(dict)`` could
        # wrap to point at an already-occupied high slot, silently mapping
        # two block_ids to the same row.
        self._free_slots: List[int] = list(range(self.num_slots))

    def _free_slot(self, bid: int) -> None:
        """Drop the mapping for ``bid`` so its slot is available for reuse.
        Caller is responsible for re-allocating + bookkeeping."""
        if bid in self._block_to_slot:
            slot = self._block_to_slot.pop(bid)
            try:
                self._lru.remove(bid)
            except ValueError:
                pass
            self._free_slots.append(slot)

    def _allocate_new_slot(self, bid: int, active: set) -> int:
        if self._free_slots:
            slot = self._free_slots.pop(0)
        else:
            # Evict LRU row that's NOT in the current batch.
            evict_bid: Optional[int] = None
            for cand in self._lru:
                if cand not in active:
                    evict_bid = cand
                    break
            if evict_bid is None:
                raise RuntimeError(
                    f"_RequestSlotAllocator: no LRU candidate to evict "
                    f"(num_slots={self.num_slots}, batch={len(active)})"
                )
            self._lru.remove(evict_bid)
            slot = self._block_to_slot.pop(evict_bid)
        self._block_to_slot[bid] = slot
        self._lru.append(bid)
        return slot

    def get_slots(
        self, block_ids: List[int], is_prefill: bool
    ) -> Tuple[List[int], List[int]]:
        """Return ``(slots, slots_needing_reset)``.

        On ``is_prefill=True``: every block_id is treated as a brand-new
        request. Any prior mapping for the same block_id is dropped,
        a fresh slot is allocated, and the slot is added to
        ``slots_needing_reset`` (the caller must clear the stash entry
        for that slot before prefill writes to the live buffer row).

        On ``is_prefill=False`` (decode): every block_id MUST already
        have a mapping established by the prefill step; a missing mapping
        indicates the decode arrived without a preceding prefill (a real
        bug we want to surface early), so we fall back to allocating a
        slot but warn loudly via the empty-reset list (the caller can
        skip-or-zero based on policy).
        """
        out: List[int] = []
        needs_reset: List[int] = []
        active = set(block_ids)
        for bid in block_ids:
            if is_prefill:
                # Drop any stale mapping for this block_id (e.g., the
                # framework recycled the block from a finished request).
                # Free the slot first so _allocate_new_slot can reuse it
                # if no eviction is needed.
                self._free_slot(bid)
                slot = self._allocate_new_slot(bid, active)
                needs_reset.append(slot)
            else:
                if bid in self._block_to_slot:
                    slot = self._block_to_slot[bid]
                    try:
                        self._lru.remove(bid)
                    except ValueError:
                        pass
                    self._lru.append(bid)
                else:
                    # Defensive: decode without a prior prefill mapping —
                    # treat as a fresh request. This shouldn't normally
                    # happen but is safer than a KeyError that crashes
                    # the engine; surface in the log.
                    slot = self._allocate_new_slot(bid, active)
                    needs_reset.append(slot)
            out.append(slot)
        return out, needs_reset


class _RequestStateStash:
    """Per-request stash for every ``[max_batch_size, ...]`` buffer in V4Transformer.

    DSv4's KV state is spread across multiple per-layer buffers all sized
    ``[max_batch_size, ...]`` and indexed by the current batch slot:
    ``Attention.kv_cache``, ``Attention.kv_cache_fp8``, ``Compressor.kv_state``,
    ``Compressor.score_state``, ``Indexer.kv_cache``, the indexer's nested
    ``Compressor.kv_state``/``score_state``, etc. Without framework
    PagedAttention wired up (M4 work), the slot index has no per-request
    affinity: prefill (B=1) always writes to slot 0, then concurrent decode
    at BS>1 reads slots ``[0..bsz-1]`` expecting per-request data — but only
    slot 0 was populated by the most recent prefill.

    This stash maintains per-request shadow rows in a parallel pool of size
    ``num_slots`` (chosen >= concurrency_limit). For each forward call we
    gather the shadow rows for the active requests into batch positions
    ``[0..bsz-1]`` of every per-batch-row buffer, run the forward, then
    scatter the updated batch rows back to the shadow rows. The scatter +
    gather are CUDA tensor ops driven by a ``slot_indices`` tensor at a
    fixed address — both eager-mode safe and CUDA-graph-safe (the captured
    graph reads from the same tensor address every replay; only the tensor
    contents change between replays via ``prepare_cuda_graph``).

    The stash is enabled only when ``num_slots > max_batch_size`` (i.e.,
    pool is bigger than max batch). For the ``num_slots == max_batch_size``
    case (single-request use cases that never overlap), we skip the
    gather/scatter entirely.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        max_batch_size: int,
        num_slots: int,
        device: torch.device,
    ):
        self.max_batch_size = int(max_batch_size)
        self.num_slots = int(num_slots)
        self.device = device
        # Stash pool layout: ``num_slots`` real per-request rows + ``max_batch_size``
        # scratch rows for graph-capture padding. The captured graph at fixed BS=k
        # always gathers/scatters ALL k rows; for replays where the actual batch
        # is smaller (bsz < k), padding entries in ``slot_indices`` MUST point at
        # distinct rows — otherwise ``index_copy_(stash, slot_indices, kv_cache[:k])``
        # makes the graph write multiple rows of kv_cache into the same stash
        # slot, clobbering whichever active request happens to share that index.
        # The scratch tail gives every padding position its own throwaway slot.
        self._pool_size = int(num_slots) + int(max_batch_size)
        # Discover [max_batch_size, ...] buffers across the model. ``named_buffers``
        # walks every nn.Module, including Compressor / Indexer's nested
        # Compressor — so we capture every per-batch-row buffer in one pass.
        # We dedupe by storage data_ptr so views (compressor.kv_cache → a slice
        # of attention.kv_cache) don't get a redundant stash.
        # Per-buffer ``init_val`` is the fill we use to "reset" both the
        # stash slot (when it's freshly allocated) and the live buffer row
        # (right before prefill, see ``reset_batch_rows``). For zero-init
        # buffers it's 0.0; for ``Compressor.score_state`` it's -inf —
        # those are the only two patterns DSv4 uses today.
        self._buffers: List[Tuple[str, torch.Tensor, torch.Tensor, float]] = []
        seen_storage: Dict[int, str] = {}
        for name, buf in transformer.named_buffers():
            if buf is None or buf.dim() < 1 or buf.shape[0] != max_batch_size:
                continue
            ptr = buf.data_ptr()
            if ptr in seen_storage:
                continue
            seen_storage[ptr] = name
            stash_shape = (self._pool_size,) + tuple(buf.shape[1:])
            stash = torch.zeros(
                stash_shape,
                dtype=buf.dtype,
                device=buf.device,
            )
            # If the buffer was initialized to a non-zero sentinel (e.g.,
            # Compressor.score_state is full of -inf), match it so newly
            # allocated stash rows look identical to a fresh batch slot.
            init_val = 0.0
            if buf.numel() > 0:
                init_val = float(buf.flatten()[0].item())
                if init_val != 0.0:
                    stash.fill_(init_val)
            self._buffers.append((name, buf, stash, init_val))
        # slot_indices buffer at a fixed address; updated in place each step.
        # Initialized so the [bsz:max_batch_size] suffix points to the scratch
        # tail (each entry distinct) — captured graph at any BS replays
        # correctly even when actual bsz < captured bsz.
        self.slot_indices = torch.arange(
            num_slots,
            num_slots + max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        # Pre-built host scratch padding for fast slot_indices updates;
        # the active prefix is overwritten per call.
        self._padding_host = torch.arange(
            num_slots,
            num_slots + max_batch_size,
            dtype=torch.int64,
        )

    def update_slot_indices(self, slots: List[int]) -> None:
        """Copy the per-batch slot list into the persistent slot_indices
        tensor (in place — keeps the data_ptr stable for graph capture).

        Pads the suffix ``[bsz:max_batch_size]`` with distinct scratch slots
        so the captured graph's gather/scatter touches a throwaway region
        instead of clobbering active stash slots."""
        if len(slots) == 0:
            return
        bs = len(slots)
        if bs > self.max_batch_size:
            raise RuntimeError(
                f"slot_indices update bsz={bs} exceeds max_batch_size={self.max_batch_size}"
            )
        # Build full [max_batch_size] vector on CPU: real slots + scratch tail.
        host = self._padding_host.clone()
        host[:bs] = torch.tensor(slots, dtype=torch.int64)
        self.slot_indices.copy_(host, non_blocking=True)

    def gather(self, bsz: int) -> None:
        """Restore per-request rows from the stash into batch positions
        ``[0..bsz-1]`` of every tracked buffer.

        ``index_select(out=...)`` writes into a pre-existing storage so
        no allocation happens on the capture stream; safe inside a
        captured CUDA graph."""
        if bsz <= 0:
            return
        idx = self.slot_indices.narrow(0, 0, bsz)
        for _name, buf, stash, _init in self._buffers:
            torch.index_select(stash, 0, idx, out=buf.narrow(0, 0, bsz))

    def scatter(self, bsz: int) -> None:
        """Save batch positions ``[0..bsz-1]`` back into the stash slots.

        ``index_copy_`` is in-place on the stash tensor — no allocation,
        safe inside a captured CUDA graph."""
        if bsz <= 0:
            return
        idx = self.slot_indices.narrow(0, 0, bsz)
        for _name, buf, stash, _init in self._buffers:
            stash.index_copy_(0, idx, buf.narrow(0, 0, bsz))

    def reset_batch_rows(self, bsz: int) -> None:
        """Reset live buffer rows ``[0..bsz-1]`` to each buffer's initial
        value (zeros for KV / FP8 KV / KV state / index KV; -inf for
        ``Compressor.score_state``).

        Used in place of ``gather`` at PREFILL time — prefill is supposed
        to start from a clean state. If we gathered instead, a stash slot
        carrying stale data from a prior request that REUSED the same
        block_id (warmup → real, finished → recycled) would seed batch
        row 0 with garbage at any positions prefill does not overwrite
        (e.g., ``Compressor.kv_state[r, ratio:]`` and
        ``Compressor.score_state[r, ratio:]`` only get touched on the
        boundary path; the rest of the row would carry stale finite
        score_state values, breaking the softmax weighting in decode).

        Eager-only — prefill never runs under CUDA graph capture for V4."""
        if bsz <= 0:
            return
        for _name, buf, _stash, init_val in self._buffers:
            row_view = buf.narrow(0, 0, bsz)
            if init_val == 0.0:
                row_view.zero_()
            else:
                row_view.fill_(init_val)

    def reset_stash_slots(self, slots: List[int]) -> None:
        """Reset the given stash slots to each buffer's initial value.

        Called by the slot allocator when assigning a slot to a brand-new
        block_id (or when reusing a slot for a recycled block_id) to
        guarantee the stash pool entry looks identical to a fresh
        allocation. Without this, the next decode step's gather would
        pull contaminated data into the live buffer row.

        Eager-only — slot allocation only happens at prefill time."""
        if not slots:
            return
        idx = torch.tensor(slots, dtype=torch.int64, device=self.device)
        for _name, _buf, stash, init_val in self._buffers:
            sel = stash.index_select(0, idx)
            if init_val == 0.0:
                sel.zero_()
            else:
                sel.fill_(init_val)
            stash.index_copy_(0, idx, sel)


def _extract_first_block_ids(attn_inputs) -> List[int]:
    """Read the first block-id of every active request from attn_inputs.

    The framework provides ``kv_cache_block_id_host`` whose shape varies by
    code path: ``[batch_size, max_blocks_per_seq]`` from PyWrappedModel,
    occasionally ``[layers, batch_size, max_blocks_per_seq]`` from layered
    KV cache plumbing, and an empty / undefined tensor during warmup probes.
    The first block-id of every active request is a stable per-request
    identifier (a block stays allocated to a request for its lifetime).

    Returns ``[]`` if the tensor isn't usable (warmup probe, no real
    request) — callers must treat this as "stash management not applicable".
    """
    # PRIORITY: kv_cache_kernel_block_id_host — this is the field the C++
    # side refreshes every CUDA-graph replay (cuda_graph_runner.cc:219
    # ``stridedCopyHost``). The physical ``kv_cache_block_id_host`` is
    # bound at capture time and NEVER refreshed during replay (cache-store
    # ops run outside the captured graph; see cuda_graph_runner.cc:81),
    # so reading it during replay returns the CAPTURE-TIME first request's
    # block_ids — which is exactly the cross-request KV leakage symptom we
    # were seeing (request A's prefill captures block_id=N → every replay
    # routes to slot N regardless of which request is actually running).
    # The kernel block_id table has the same ``[batch, blocks_per_seq]``
    # shape and is just as stable per request.
    t = getattr(attn_inputs, "kv_cache_kernel_block_id_host", None)
    if t is None or t.numel() == 0:
        # Eager path doesn't always populate the kernel field (PyWrappedModel
        # only sets it when the engine binds a real KV cache; warmup probes
        # often don't). Fall back to the physical block_id table.
        t = getattr(attn_inputs, "kv_cache_block_id_host", None)
        if t is None or t.numel() == 0:
            return []
    if t.dim() == 3:
        # [layers, batch_size, max_blocks_per_seq] — collapse the layer
        # dim by taking layer 0 (per-request first-block-id is shared
        # across layers for the same request).
        t = t[0]
    if t.dim() < 2 or t.shape[1] == 0:
        return []
    first_col = t[:, 0].contiguous()
    if first_col.device.type != "cpu":
        first_col = first_col.cpu()
    # ``.tolist()`` on a 1-D int tensor gives a flat ``[int, int, ...]``;
    # be defensive in case the shape was actually higher-dim by flattening.
    flat = first_col.flatten().tolist()
    return [int(v) for v in flat]


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

        # Per-request KV stash for concurrent decode.  DSv4 maintains its own
        # ``[max_batch_size, ...]`` per-layer KV buffers indexed by current
        # batch slot; without per-request slot affinity, concurrent BS>1
        # decode reads stale data and produces garbage. The stash is sized at
        # the framework's concurrency limit so each active request gets a
        # stable shadow row that's gather/scattered around every forward.
        # When the framework provides max_concurrent_requests <= 1 (or no
        # batching at all), the stash is a no-op.
        self._stash_num_slots = max(
            int(max_generate_batch_size or 1),
            int(getattr(model_config, "concurrency_limit", 0) or 0),
            1,
        )
        self._stash: Optional[_RequestStateStash] = None
        self._slot_alloc: Optional[_RequestSlotAllocator] = None

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

        # Initialize the per-request KV stash after V4Transformer's per-layer
        # buffers exist on the real device.  We size the pool at the larger
        # of (a) max_generate_batch_size and (b) concurrency_limit — each
        # active request needs its own shadow row.
        if device_str.startswith("cuda"):
            self._stash = _RequestStateStash(
                self.v4,
                max_batch_size=int(self._v4_args.max_batch_size),
                num_slots=int(self._stash_num_slots),
                device=torch.device(device_str),
            )
            self._slot_alloc = _RequestSlotAllocator(
                num_slots=int(self._stash_num_slots),
            )
            logging.info(
                "[DeepSeekV4Model] KV stash: tracking %d per-batch-row buffers, "
                "pool size = %d slots (max_batch=%d, concurrency_limit_hint=%d)",
                len(self._stash._buffers),
                self._stash_num_slots,
                int(self._v4_args.max_batch_size),
                self._stash_num_slots,
            )
            if os.environ.get("DSV4_LOG_STASH", "0") == "1":
                for nm, b, s, _init in self._stash._buffers[:10]:
                    logging.info(
                        "[DSv4 stash buffer] %s shape=%s dtype=%s",
                        nm,
                        tuple(b.shape),
                        b.dtype,
                    )
                logging.info(
                    "[DSv4 stash buffer] ... and %d more",
                    max(0, len(self._stash._buffers) - 10),
                )
        if self._fp8_kv_requested:
            logging.info(
                "[DeepSeekV4Model] --fp8_kv_cache=1 — first prefill will arm "
                "Attention.enable_fp8_kv_cache() on every layer, switching "
                "subsequent decodes to the FlashMLA FP8 sparse path."
            )

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

    def _update_slot_indices_for_batch(self, attn_inputs) -> int:
        """Compute per-request stash slots for the current batch and copy them
        into the persistent ``slot_indices`` tensor (in place — keeps the
        data_ptr stable for CUDA graph capture/replay).

        Returns the active batch size ``bsz``.  Called from the eager forward
        path AND from ``DSv4DecodeFmhaImpl.prepare`` (which the graph runner
        invokes between each replay) so the captured graph reads fresh slot
        values from a stable address every step.

        On prefill: any newly-allocated stash slots are cleared to each
        buffer's initial value so the prefill starts from a guaranteed-clean
        per-request state — this fixes block_id-recycling contamination
        where a finished request's stash row would otherwise leak into the
        new request that inherits the same block_id.

        Returns 0 if stash management is not applicable (e.g., warmup probes
        with no block_ids); the caller should skip gather/scatter in that case.
        """
        if self._stash is None or self._slot_alloc is None:
            return 0
        block_ids = _extract_first_block_ids(attn_inputs)
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        if not block_ids:
            if os.environ.get("DSV4_LOG_STASH", "0") == "1":
                t = getattr(attn_inputs, "kv_cache_block_id_host", None)
                k = getattr(attn_inputs, "kv_cache_kernel_block_id_host", None)
                shape = tuple(t.shape) if t is not None else None
                kshape = tuple(k.shape) if k is not None else None
                logging.info(
                    "[DSv4 stash] EARLY-RETURN is_prefill=%s "
                    "kv_cache_block_id_host shape=%s numel=%s "
                    "kv_cache_kernel_block_id_host shape=%s numel=%s",
                    is_prefill,
                    shape,
                    None if t is None else t.numel(),
                    kshape,
                    None if k is None else k.numel(),
                )
            return 0
        slots, slots_needing_reset = self._slot_alloc.get_slots(
            block_ids,
            is_prefill=is_prefill,
        )
        if slots_needing_reset:
            self._stash.reset_stash_slots(slots_needing_reset)
        self._stash.update_slot_indices(slots)
        if os.environ.get("DSV4_LOG_STASH", "0") == "1":
            logging.info(
                "[DSv4 stash] is_prefill=%s bsz=%d block_ids=%s slots=%s reset=%s",
                is_prefill,
                len(block_ids),
                block_ids,
                slots,
                slots_needing_reset,
            )
        return len(slots)

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
        for i, layer in enumerate(self.v4.layers):
            ratio = layer.attn.compress_ratio
            if ratio == 4:
                self._layer_pools.append([7, 1, 3, 4, 5])
                self._layer_gather_pools.append([1, 3])  # CSA_KV, INDEXER_KV only
                self._layer_reuse_gather_pools.append([1, 3, 4, 5])  # + STATE pools
            elif ratio == 128:
                self._layer_pools.append([7, 2, 6])
                self._layer_gather_pools.append([2])  # HCA_KV only
                self._layer_reuse_gather_pools.append([2, 6])  # + STATE pool
            else:
                self._layer_pools.append([7])
                self._layer_gather_pools.append([])  # SWA-only, nothing to gather
                self._layer_reuse_gather_pools.append([])  # SWA-only

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

    def _gather_all_layers(
        self, attn_inputs, B=1, batch_offset=0, paged_only=False, reuse_gather=False
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
            if reuse_gather:
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
        )
        return DSv4DecodeFmhaImpl(
            cfg,
            device=device,
            attn_inputs=attn,
            slot_update_cb=self._update_slot_indices_for_batch,
        )

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
            # Slot indices were ALREADY copied into ``self._stash.slot_indices``
            # by ``prepare`` (called by ``__init__`` and ``prepare_cuda_graph``).
            # We only need bsz here so the gather/scatter narrow to the right
            # prefix of every per-batch-row buffer. Use the captured
            # ``meta.batch_size`` (Python int set inside update_decode_metadata
            # _in_place) — that's frozen into the graph at capture time and
            # selects the same constant on every replay.
            graph_bsz = int(meta.batch_size)
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
            )
            # Eager path: pull slot_indices from the per-batch block_ids and
            # update the persistent slot_indices tensor before gather/scatter.
            graph_bsz = self._update_slot_indices_for_batch(attn)

        # Per-request KV stash gather: copy each request's shadow rows from
        # the stash into batch positions [0..bsz-1] of every per-batch-row
        # buffer (Attention.kv_cache, Compressor.kv_state/score_state, etc.).
        # Both the eager path and the captured graph land here; the graph
        # captures the gather/scatter ops (they read slot_indices from a
        # stable address — contents updated each replay by prepare_cuda_graph).
        if self._stash is not None and graph_bsz > 0:
            self._stash.gather(graph_bsz)

        hidden = self.v4.forward_decode(input_ids, meta)  # [B, dim] packed

        # Scatter the updated batch rows back to the stash so the next forward
        # (possibly a different batch composition) sees fresh per-request data.
        if self._stash is not None and graph_bsz > 0:
            self._stash.scatter(graph_bsz)
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

        # Decode arm — Phase 3: when fmha_impl is a DSv4DecodeFmhaImpl, route
        # to the CUDA-graph replay path with persistent decode metadata.
        if (
            not is_prefill
            and fmha_impl is not None
            and attn is not None
            and attn.sequence_lengths is not None
            and attn.sequence_lengths.numel() > 0
        ):
            return self._forward_decode(inputs, fmha_impl)


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

        # Re-wire if the framework rebuilt the KV cache (warmup→real
        # transition). NormalEngine first allocates a stub allocator with
        # block_num=1 to measure GPU memory, then tears it down and creates
        # the real allocator with the measured block_num. The stub's pool
        # tensors get freed; without re-wiring, gather hits IndexError on
        # the dead warmup tensors.
        if (
            use_framework_kv
            and self.kv_cache is not None
            and getattr(self, "_framework_kv_enabled", False)
        ):
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

            # PD separation: register all (layer × pool) block_ids in
            # cache_store so the decode side can fetch them. Done after the
            # whole prefill loop because the cache_store hand-off iterates
            # all batches internally via context_batch_size.
            if (
                use_framework_kv
                and hasattr(self, "_framework_kv_enabled")
                and self._framework_kv_enabled
                and attn is not None
            ):
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
                # Reset compressor/indexer state for all B slots before gather.
                # New requests joining the batch may have stale data in their slots.
                for layer in self.v4.layers:
                    a = layer.attn
                    if a.compressor is not None:
                        a.compressor.kv_state[:B].zero_()
                        a.compressor.score_state[:B].fill_(float("-inf"))
                    if a.indexer is not None:
                        a.indexer.compressor.kv_state[:B].zero_()
                        a.indexer.compressor.score_state[:B].fill_(float("-inf"))
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
