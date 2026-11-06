"""Single decode-step orchestration for the V4.1 PPU decode machine (D).

Wave-2 rewrite: the orchestrator now consumes L2's REAL surfaces directly —

* weights: :class:`PPUWeightBundle` (:mod:`ppu.weight`) — closing the wave-1
  simplifications. The attention path mirrors ``prefill.run_prefill`` line by
  line: ``wq_a -> q_norm -> wq_b -> RoPE tail`` queries, ``wkv -> kv_norm ->
  RoPE`` SWA rows, the REAL ratio-2 pair-pooling compressor
  (``compressor.wkv``/``wgate`` + ``compressor.norm``, CSA state carried in
  the ``csa_state`` pool), the roped indexer K produced from the compressor
  latent, the indexer query from the ``q_lora`` representation
  (``indexer.wq_b`` + ``weights_proj`` head weights), ``attn_sink`` in the
  sparse attention, and the grouped inverse-RoPE ``wo_a``/``wo_b`` output
  projection (``o_lora``);
* cache: L2's :class:`PPUKVCache` (:mod:`ppu.cache`) through
  :func:`make_decode_cache_seam` — the wave-1 ``LocalDecodeCache`` stub is
  GONE; decode commits KV with the exact same codec calls and pool layout
  prefill writes, so a prefill-seeded cache continues seamlessly.

Per layer (single step, one row per active request plus optional empty-rank
placeholder rows)::

    [Engram lookup -> gated inject] -> mHC pre -> norm -> KV commit (SWA +
    CSA/global for producers) -> indexer top-k (sources compute, consumers
    reuse the source's selection) -> sparse attention (global top-k entries +
    SWA window, attn_sink) -> inverse-RoPE + wo_a/wo_b -> mHC post
    -> mHC pre -> norm -> sqrtsoftplus router (top-6) -> routed MoE + shared
    expert -> mHC post

The mHC chain threads the delayed mix exactly like ``block.py`` /
``prefill.run_prefill``: sublayer N's ``pre`` result is the ``state`` of
sublayer N+1's ``pre`` (including the intra-layer attn->ffn seam, fixed in
wave 2).

Remaining documented simplifications (fail-loud or caller-boundary):
the SWA pool is append-only (L2's cache has no
ring compaction yet — the position->slot mapping is the identity the L1
reader uses, so reads are correct).

Single rank first (DP1/EP per ``ops``): the orchestrator is rank-local and
records per-request owner ranks. Empty-rank placeholder tokens (plan section
3.2) flow through the collective-shaped compute with invalid expert ids
(``-1``) and zero routing weights; they never enter sampling, the token
history or the cache. The DP8 step-level coordination lives in
``dp_coord.py``; fixture load/continue in ``kv_io.py``; the wiring onto
:class:`PPUModelCore` is :func:`wire_decode` (model.py itself is untouched).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.backend import (
    V41QuantizedTensor,
    V41StepContext,
)
from rtp_llm.models_py.modules.dsv41.ppu.cache import CacheGeometry, PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.engram import (
    build_token_windows,
    compute_engram_hash_ids,
    engram_gate_inject,
    wkv_project,
)
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    combine_routed_and_shared,
    pad_topk_for_dispatch,
    shared_expert_forward,
    sqrtsoftplus_gate,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill import (
    _layer_attention,
    _produce_global,
    _quantized_indexer_query,
    _rms_norm,
    _rope_tail,
)
from rtp_llm.models_py.modules.dsv41.ppu.weight import PPUWeightBundle, V41Geometry

__all__ = [
    "DecodeStepOrchestrator",
    "DecodeStepResult",
    "RequestState",
    "PPUCacheSeam",
    "make_decode_cache_seam",
    "wire_decode",
    "cache_geometry_from_v41",
]


# ---------------------------------------------------------------------------
# Cache seam (the single swap point to L2's cache.py)
# ---------------------------------------------------------------------------


def cache_geometry_from_v41(geometry: V41Geometry) -> CacheGeometry:
    """The :class:`CacheGeometry` matching a loaded :class:`V41Geometry`."""
    return CacheGeometry(
        num_layers=geometry.num_layers,
        compress_ratios=geometry.compress_ratios,
        kv_source_layer_ids=geometry.kv_source_layer_ids,
        index_source_layer_ids=geometry.index_source_layer_ids,
        window=geometry.window,
    )


class PPUCacheSeam:
    """Thin adapter over L2's :class:`PPUKVCache`.

    The orchestrator never touches pool tensors directly: every pool access
    goes through this seam, which delegates to the cache's documented public
    API (``alloc`` / ``insert_rows`` / ``slots_at_positions`` /
    ``read_source`` / ``read_csa`` / ``cache_view``). The seam exists so the
    swap point is explicit and a future cache API change fails loudly here
    instead of deep inside the step body.
    """

    def __init__(self, cache: PPUKVCache) -> None:
        self.cache = cache

    # -- lifecycle -----------------------------------------------------------

    def alloc(self, request_id: int, n_tokens: int):
        return self.cache.alloc(request_id, n_tokens)

    def request_length(self, request_id: int) -> int:
        return int(self.cache.request_length(request_id))

    # -- writes (packed rows from ops.encode_kv) ------------------------------

    def insert_swa(
        self,
        layer_id: int,
        request_id: int,
        positions: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        slots = self.cache.slots_at_positions(layer_id, "swa_kv", request_id, positions)
        self.cache.insert_rows(layer_id, "swa_kv", slots, rows)

    # -- reads ----------------------------------------------------------------

    def read_source(self, source_layer_id: int, request_id: int, **kwargs):
        return self.cache.read_source(source_layer_id, request_id, **kwargs)

    def read_csa(self, source_layer_id: int, request_id: int, positions: torch.Tensor):
        return self.cache.read_csa(source_layer_id, request_id, positions)

    def cache_view(self, layer_id: int):
        return self.cache.cache_view(layer_id)

    def descriptors(self, owner_layer: int):
        return self.cache.descriptors(owner_layer)

    # -- fixture passthrough (kv_io builds on these) ---------------------------

    def export_fixture(self, path, meta: Optional[dict] = None) -> dict:
        return self.cache.export_fixture(path, meta)

    @staticmethod
    def load_fixture(path):
        return PPUKVCache.load_fixture(path)


def make_decode_cache_seam(cache: Any = None) -> PPUCacheSeam:
    """The one-place swap point from the wave-1 stub to L2's ``cache.py``.

    ``cache`` must be a :class:`PPUKVCache` (or already a seam). Anything else
    fails loud rather than being silently misused; ``None`` is rejected
    because a fresh cache needs the model geometry — the orchestrator builds
    one itself via :func:`cache_geometry_from_v41`.
    """
    if isinstance(cache, PPUCacheSeam):
        return cache
    if cache is None:
        raise TypeError(
            "make_decode_cache_seam(None) is gone: the wave-1 LocalDecodeCache "
            "stub was replaced by L2's PPUKVCache; pass a PPUKVCache (or let "
            "DecodeStepOrchestrator build a fresh one from the geometry)"
        )
    for required in ("alloc", "read_source", "read_csa", "cache_view", "insert_rows"):
        if not callable(getattr(cache, required, None)):
            raise TypeError(
                f"unsupported cache object {type(cache).__name__}: missing "
                f"{required!r} (expected L2's PPUKVCache)"
            )
    return PPUCacheSeam(cache)


# ---------------------------------------------------------------------------
# Request / step state
# ---------------------------------------------------------------------------


@dataclass
class RequestState:
    """Per-request decode state owned by exactly one DP rank.

    ``history`` carries the request's token ids (prompt + generated) for the
    Engram n-gram windows — never shared across requests.
    """

    request_id: int
    owner_rank: int = 0
    history: list = field(default_factory=list)
    seq_len: int = 0  # tokens whose KV is committed to the pools
    finished: bool = False
    steps_done: int = 0


@dataclass
class DecodeStepResult:
    """One step's outputs: hidden rows for VALID tokens only.

    ``row_request_ids``/``row_positions`` index each hidden row back to its
    request; placeholder rows never appear here (they are excluded from
    sampling by construction).
    """

    hidden: torch.Tensor  # [T_valid, hidden]
    row_request_ids: list
    row_positions: list
    valid_mask: torch.Tensor  # [T_total] over real+placeholder rows
    num_placeholders: int
    logits: Optional[torch.Tensor] = None  # [T_valid, vocab] if lm_head bound
    mtp_target_hidden_states: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


class DecodeStepOrchestrator:
    """Rank-local single decode step over :class:`PPUOps` (contract ops only).

    ``ops`` must be L1's ``PPUOps`` (or an object with the eight contract
    ops); expert weights and Engram shards are bound on it by
    ``bundle.bind(ops)``. ``bundle`` is L2's :class:`PPUWeightBundle` — the
    same surface ``run_prefill`` consumes, so prefill-seeded caches continue
    without any format shim. ``dp_rank``/``dp_size`` parameterize the
    data-parallel role: this rank executes only requests it owns
    (``add_request`` fails loud for a foreign owner), while EP/ENP groups
    ride on ``ops``.
    """

    def __init__(
        self,
        *,
        ops: Any,
        bundle: PPUWeightBundle,
        cache: Optional[PPUKVCache] = None,
        device=None,
        dp_rank: int = 0,
        dp_size: int = 1,
        capture_layer_ids=(),
    ) -> None:
        self.ops = ops
        self.bundle = bundle
        self.geometry: V41Geometry = bundle.geometry
        if cache is None:
            cache = PPUKVCache(
                geometry=cache_geometry_from_v41(self.geometry), device=device
            )
        self.cache = make_decode_cache_seam(cache)
        self.device = (
            torch.device(device) if device is not None else bundle.embedding.device
        )
        self.dp_rank = int(dp_rank)
        self.dp_size = int(dp_size)
        self.capture_layer_ids = tuple(map(int, capture_layer_ids))
        if len(set(self.capture_layer_ids)) != len(self.capture_layer_ids) or any(
            layer < 0 or layer >= self.geometry.num_layers
            for layer in self.capture_layer_ids
        ):
            raise ValueError("target capture layers must be unique valid layer ids")
        if self.geometry.engram_enabled() and bundle.engram_hash_params is None:
            raise RuntimeError(
                "Engram layers require bundle.engram_hash_params; refusing to "
                "run the decode step without them"
            )
        self.requests: Dict[int, RequestState] = {}

    # -- request lifecycle ----------------------------------------------------

    def add_request(
        self,
        request_id: int,
        prompt_token_ids: Sequence[int],
        *,
        owner_rank: Optional[int] = None,
    ) -> RequestState:
        """Register a request on its owner rank (stable routing, dp_coord).

        The request's prompt KV must ALREADY be committed to the cache — by
        prefill on this rank or by a loaded fixture (``kv_io``). The tracked
        cache length must equal the prompt history; a mismatch fails loud
        instead of silently decoding over a half-seeded cache.
        """
        request_id = int(request_id)
        if request_id in self.requests:
            raise ValueError(f"request {request_id} already registered")
        owner = self.dp_rank if owner_rank is None else int(owner_rank)
        if owner != self.dp_rank:
            raise RuntimeError(
                f"request {request_id} is owned by DP rank {owner}, not this "
                f"rank {self.dp_rank}; route it to its owner (dp_coord)"
            )
        state = RequestState(request_id=request_id, owner_rank=owner)
        state.history = [int(t) for t in prompt_token_ids]
        state.seq_len = len(state.history)
        tracked = self.cache.request_length(request_id)
        if tracked != state.seq_len:
            raise RuntimeError(
                f"request {request_id} history has {state.seq_len} tokens but "
                f"the cache tracks {tracked}: run prefill (or load a fixture) "
                "before registering the request for decode"
            )
        self.requests[request_id] = state
        return state

    def mark_finished(self, request_id: int) -> None:
        """Early finish changes only this request's validity (plan 3.2.5)."""
        self.requests[int(request_id)].finished = True

    def resume_request(self, request_id: int, seq_len: int, recent_token_ids):
        """Resume a native scheduler request using its explicit n-gram suffix.

        The framework owns the full history. Do not fabricate prefix tokens
        merely to make len(history) equal the logical cache length.
        """
        request_id, seq_len = int(request_id), int(seq_len)
        recent = [int(token) for token in recent_token_ids]
        required = (
            min(seq_len, self.geometry.engram_max_ngram_size - 1)
            if self.geometry.engram_enabled()
            else 0
        )
        if (
            request_id in self.requests
            or self.cache.request_length(request_id) != seq_len
        ):
            raise ValueError("native request identity/cache length mismatch")
        if (
            len(recent) < required
            or len(recent) > seq_len
            or any(token < 0 for token in recent)
        ):
            raise ValueError("native decode requires the complete recent token history")
        state = RequestState(request_id=request_id, owner_rank=self.dp_rank)
        state.history, state.seq_len = recent, seq_len
        self.requests[request_id] = state
        return state

    def release(self, request_id: int) -> None:
        """Drop this rank's bookkeeping for a request.

        NOTE: L2's ``PPUKVCache`` has no free/evict API yet, so pool blocks
        stay allocated (the same behavior prefill has); eviction is a cache.py
        follow-up, tracked in the L2/L3 ownership log.
        """
        self.requests.pop(int(request_id), None)

    # -- the step --------------------------------------------------------------

    def step(
        self,
        new_token_ids: Mapping[int, int],
        *,
        placeholders: int = 0,
        skip_finished: bool = True,
    ) -> DecodeStepResult:
        """Advance one decode step.

        ``new_token_ids`` maps each unfinished request to its single new
        input token. ``placeholders`` appends empty-rank placeholder rows
        (plan 3.2.3): they keep collective shapes but carry invalid expert
        ids, zero routing weights, and never enter sampling, the token
        history or the cache.
        """
        active: list = []
        for request_id, token in new_token_ids.items():
            request_id = int(request_id)
            if request_id not in self.requests:
                raise KeyError(f"unknown request {request_id}")
            state = self.requests[request_id]
            if state.finished and skip_finished:
                continue
            active.append(request_id)
        tokens = [int(new_token_ids[rid]) for rid in active]
        total = len(active) + int(placeholders)
        if total == 0:
            raise RuntimeError(
                "decode step with zero rows: DP ranks with no local tokens must "
                "pass placeholders so the collectives still run (plan 3.2.2)"
            )

        positions = [self.requests[rid].seq_len for rid in active]
        valid_mask = torch.zeros(total, dtype=torch.bool)
        valid_mask[: len(active)] = True
        input_ids = torch.zeros(total, dtype=torch.int64)
        input_ids[: len(active)] = torch.tensor(tokens, dtype=torch.int64)
        row_positions = torch.full((total,), -1, dtype=torch.int64)
        row_positions[: len(active)] = torch.tensor(positions, dtype=torch.int64)
        row_requests = torch.full((total,), -1, dtype=torch.int64)
        row_requests[: len(active)] = torch.tensor(active, dtype=torch.int64)
        row_owners = torch.full((total,), self.dp_rank, dtype=torch.int64)
        context = V41StepContext(
            positions=row_positions.to(self.device),
            valid_mask=valid_mask.to(self.device),
            request_ids=row_requests.to(self.device),
            owner_ranks=row_owners.to(self.device),
            request_offsets=torch.arange(
                0, total + 1, dtype=torch.int64, device=self.device
            ),
            parallel_context=self,
        )

        # Reserve the new token's pool space (extends every pool's tables).
        handles = {rid: self.cache.alloc(rid, 1) for rid in active}

        hidden = self.bundle.embedding[input_ids.to(self.device)]
        # Reference forward_layers: expand to the hc residual lanes each step.
        h = (
            hidden.unsqueeze(1)
            .expand(total, self.geometry.hc_mult, self.geometry.hidden_size)
            .contiguous()
        )

        # Engram hashes once per step (valid rows only; placeholders dropped).
        hashes = self._engram_hashes(active, tokens, valid_mask)

        # mHC pre-mix chain: layer 0's attention ``pre`` is the step entry;
        # every later sublayer reads its predecessor's ``pre`` result exactly
        # like block.py's delayed-mix wiring (attn pre -> ffn pre -> next
        # layer attn pre -> ...).
        prev_mix = None
        shared_topk: Dict[int, torch.Tensor] = {}
        captured = {}
        for layer_id in range(self.geometry.num_layers):
            if layer_id in self.capture_layer_ids:
                captured[layer_id] = h.mean(dim=-2)
            h, prev_mix = self._forward_layer(
                layer_id,
                h,
                context,
                valid_mask,
                active,
                positions,
                handles,
                hashes=hashes,
                prev_mix=prev_mix,
                shared_topk=shared_topk,
            )

        head = self.ops.mhc(h, None, phase="head", state=prev_mix)
        hidden_out = _rms_norm(
            head.hidden.reshape(total, self.geometry.hidden_size),
            self.bundle.final_norm,
            self.geometry.norm_eps,
        )

        # Commit history / counters only for VALID rows (plan 3.2.3).
        for request_id in active:
            state = self.requests[request_id]
            state.history.append(int(new_token_ids[request_id]))
            state.seq_len += 1
            state.steps_done += 1

        mask = valid_mask.to(hidden_out.device)
        result_rows = hidden_out[mask]
        result = DecodeStepResult(
            hidden=result_rows,
            row_request_ids=list(active),
            row_positions=list(positions),
            valid_mask=valid_mask,
            num_placeholders=int(placeholders),
            mtp_target_hidden_states=(
                torch.cat(
                    [captured[layer] for layer in self.capture_layer_ids], dim=-1
                )[mask]
                if self.capture_layer_ids
                else None
            ),
        )
        lm_head = self.bundle.lm_head
        if lm_head is not None:
            result.logits = torch.nn.functional.linear(
                result_rows.to(lm_head.dtype), lm_head.to(result_rows.device)
            )
        return result

    # -- per-layer body ---------------------------------------------------------

    def _forward_layer(
        self,
        layer_id: int,
        h: torch.Tensor,
        context: V41StepContext,
        valid_mask: torch.Tensor,
        active: Sequence[int],
        positions: Sequence[int],
        handles: Mapping[int, Any],
        *,
        hashes: Optional[torch.Tensor],
        prev_mix: Any,
        shared_topk: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Any]:
        geometry = self.geometry
        layer = self.bundle.layers[layer_id]
        self._current_layer_id = layer_id  # test/debug observability
        freqs = self.bundle.freqs_for(layer_id)
        rope_positions = context.positions.clamp_max(freqs.shape[0] - 1)

        # ---- Engram inject (before the delayed mHC projection) ---------------
        if layer_id in geometry.engram_layer_ids:
            h = self._engram_inject_layer(layer_id, layer, h, hashes, valid_mask)

        # ---- attention sublayer: mHC pre -> norm ------------------------------
        residual = h
        step_attn = self.ops.mhc(
            residual,
            None,
            phase="pre",
            parameters=layer.mhc_parameters("attn"),
            context=context,
            state=prev_mix,
        )
        prev_mix = step_attn
        x_pre = _rms_norm(
            step_attn.hidden.reshape(-1, geometry.hidden_size),
            layer.attn_norm,
            geometry.norm_eps,
        )

        # ---- query path: wq_a -> q_norm -> wq_b -> RoPE tail ------------------
        qr = _rms_norm(layer.wq_a(x_pre), layer.q_norm, geometry.norm_eps)
        q = layer.wq_b(qr).reshape(x_pre.shape[0], geometry.n_heads, geometry.head_dim)
        q = _rope_tail(q, freqs[rope_positions])

        # ---- KV commit (valid rows only) ---------------------------------------
        kv_latent = layer.wkv(x_pre)
        kv_latent = _rms_norm(kv_latent, layer.kv_norm, geometry.norm_eps)
        kv_latent = _rope_tail(kv_latent, freqs[rope_positions])
        packed_swa = self.ops.encode_kv(kv_latent, pool_tag="swa_kv")
        for index, request_id in enumerate(active):
            self.cache.insert_swa(
                layer_id,
                request_id,
                torch.tensor([positions[index]], dtype=torch.int64),
                packed_swa[index : index + 1],
            )

        # ---- global KV / indexer K / CSA state production (producers) ---------
        if layer.is_kv_source:
            # Valid-row views: placeholders never touch the compressed pools.
            keep = valid_mask.to(x_pre.device)
            produce_values = layer.compressor_wkv(x_pre[keep])
            produce_scores = (
                layer.compressor_wgate(x_pre[keep])
                if geometry.ratio(layer_id) == 2
                else None
            )
            _produce_global(
                ops=self.ops,
                layer=layer,
                geometry=geometry,
                bundle=self.bundle,
                values=produce_values,
                scores=produce_scores,
                positions=context.positions[keep],
                request_ids=context.request_ids[keep],
                cache=self.cache.cache,
                slots=dict(handles),
                layer_id=layer_id,
            )

        # ---- indexer top-k: sources compute, consumers reuse ------------------
        if layer.is_index_source:
            keep = valid_mask.to(x_pre.device)
            selected_valid = self._indexer_topk(
                layer, x_pre[keep], qr[keep], active, positions
            )
            shared_topk[layer_id] = selected_valid
        elif geometry.ratio(layer_id):
            source = geometry.index_source_for(layer_id)
            if source is None or source not in shared_topk:
                raise RuntimeError(
                    f"layer {layer_id} has ratio {geometry.ratio(layer_id)} but no "
                    "index source result to reuse; the config is inconsistent"
                )
            selected_valid = shared_topk[source]
        else:
            selected_valid = None

        selected = None
        if selected_valid is not None:
            topk = selected_valid.shape[1]
            selected = torch.full(
                (x_pre.shape[0], topk), -1, dtype=torch.int32, device=x_pre.device
            )
            selected[: selected_valid.shape[0]] = selected_valid.to(selected.device)

        # ---- sparse attention over global top-k entries + the SWA window ------
        # Placeholder rows never enter the attention call: their request id is
        # -1 and L2's workspace builder (and the block tables) index by request
        # id VALUE. They keep a zero attention output instead (the same result
        # a length-0 row yields), and are masked out of sampling downstream.
        keep = valid_mask.to(q.device)
        if int(keep.sum()) == 0:
            # Fully-empty rank (all placeholders): no attention rows at all.
            attn_out = torch.zeros_like(q)
        elif bool(keep.all()):
            attn_out = _layer_attention(
                q,
                selected,
                ops=self.ops,
                cache=self.cache.cache,
                geometry=geometry,
                context=context,
                layer_id=layer_id,
                attn_sink=layer.attn_sink,
                window=geometry.window,
            )
        else:
            n_valid = int(keep.sum())
            valid_context = V41StepContext(
                positions=context.positions[keep],
                valid_mask=torch.ones(
                    n_valid, dtype=torch.bool, device=context.positions.device
                ),
                request_ids=context.request_ids[keep],
                owner_ranks=context.owner_ranks[keep],
                request_offsets=torch.arange(
                    0, n_valid + 1, dtype=torch.int64, device=context.positions.device
                ),
                parallel_context=self,
            )
            attn_valid = _layer_attention(
                q[keep],
                None if selected is None else selected[keep],
                ops=self.ops,
                cache=self.cache.cache,
                geometry=geometry,
                context=valid_context,
                layer_id=layer_id,
                attn_sink=layer.attn_sink,
                window=geometry.window,
            )
            attn_out = torch.zeros_like(q)
            attn_out[keep] = attn_valid

        # ---- output projection: inverse RoPE tail, grouped wo_a, wo_b ---------
        attn_out = _rope_tail(attn_out, freqs[rope_positions], inverse=True)
        tokens = x_pre.shape[0]
        grouped = attn_out.reshape(tokens, geometry.o_groups, -1)
        group_rank = geometry.o_lora_rank
        wo_a_weight = layer.wo_a.weight
        projected = torch.stack(
            [
                F.linear(
                    grouped[:, group].contiguous(),
                    wo_a_weight[
                        group * group_rank : (group + 1) * group_rank
                    ].contiguous(),
                )
                for group in range(geometry.o_groups)
            ],
            dim=1,
        )
        attn_branch = layer.wo_b(projected.reshape(tokens, -1))

        step_attn_post = self.ops.mhc(
            residual,
            attn_branch,
            phase="post",
            parameters=layer.mhc_parameters("attn"),
            context=context,
            state=step_attn,
        )
        h = step_attn_post.hidden

        # ---- FFN sublayer: mHC pre -> norm -> router -> MoE + shared ----------
        residual = h
        step_ffn = self.ops.mhc(
            residual,
            None,
            phase="pre",
            parameters=layer.mhc_parameters("ffn"),
            context=context,
            state=prev_mix,
        )
        prev_mix = step_ffn
        x_ffn = _rms_norm(
            step_ffn.hidden.reshape(-1, geometry.hidden_size),
            layer.ffn_norm,
            geometry.norm_eps,
        )

        ids, route_weights = sqrtsoftplus_gate(
            x_ffn,
            layer.router_weight,
            layer.router_bias,
            top_k=geometry.n_activated_experts,
            routed_scaling_factor=geometry.routed_scaling_factor,
        )
        ids, route_weights = pad_topk_for_dispatch(ids, route_weights)
        if not bool(valid_mask.all()):
            # Placeholder rows: invalid expert ids, zero weights (plan 3.2.3).
            invalid = ~valid_mask.to(ids.device)
            ids = ids.clone()
            route_weights = route_weights.clone()
            ids[invalid] = -1
            route_weights[invalid] = 0.0

        x_bf16 = x_ffn if x_ffn.dtype == torch.bfloat16 else x_ffn.to(torch.bfloat16)
        routed = self.ops.moe(
            x_bf16, ids, route_weights, layer_id=layer_id, context=context
        )
        if layer.shared_w13 is not None:
            shared = shared_expert_forward(
                x_bf16, layer.shared_w13, layer.shared_w2, geometry.swiglu_limit
            )
            ffn_branch = combine_routed_and_shared(routed, shared)
        else:
            ffn_branch = routed
        step_ffn_post = self.ops.mhc(
            residual,
            ffn_branch,
            phase="post",
            parameters=layer.mhc_parameters("ffn"),
            context=context,
            state=step_ffn,
        )
        h = step_ffn_post.hidden
        return h, prev_mix

    # -- Engram -------------------------------------------------------------------

    def _engram_hashes(
        self, active: Sequence[int], tokens: Sequence[int], valid_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """N-gram hashes for the step's VALID rows (placeholders dropped).

        The n-gram window spans the new token plus the request's own
        lookback history; ``build_token_windows`` guarantees no cross-request
        bleed and -1 padding before the sequence starts.
        """
        if not self.geometry.engram_enabled():
            return None
        rows = len(active)
        ngram = self.geometry.engram_max_ngram_size
        if rows == 0:
            return torch.zeros(
                (
                    0,
                    len(self.geometry.engram_layer_ids),
                    (ngram - 1) * self.geometry.engram_n_heads,
                ),
                dtype=torch.int64,
            )
        kept_tokens = torch.tensor([int(t) for t in tokens], dtype=torch.int64)
        kept_lookback = torch.full((rows, ngram - 1), -1, dtype=torch.int64)
        for slot, request_id in enumerate(active):
            state = self.requests[request_id]
            history = state.history[-(ngram - 1) :]
            if history:
                # most-recent-first, exactly what build_token_windows wants
                kept_lookback[slot, : len(history)] = torch.tensor(
                    history[::-1], dtype=torch.int64
                )
        kept_cu = torch.arange(rows + 1, dtype=torch.int64)
        windows = build_token_windows(kept_tokens, kept_cu, kept_lookback)
        return compute_engram_hash_ids(windows, self.bundle.engram_hash_params, None)

    def _engram_group(self, layer_id: int):
        """The ENP lookup group for ``layer_id`` (None = rank-local only).

        L1's ``_resolve_group`` auto-switches to a distributed ENP group when
        ``torch.distributed`` is initialized even without an explicit bound
        group, so an initialized distributed world counts as "collective".
        """
        binding = getattr(self.ops, "_engram", {}).get(int(layer_id))
        if binding and len(binding) > 2 and binding[2] is not None:
            return binding[2]
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return True
        except Exception:  # noqa: BLE001 - fall back to rank-local
            pass
        return None

    def _engram_inject_layer(
        self,
        layer_id: int,
        layer,
        h: torch.Tensor,
        hashes: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Engram lookup + WKV projection + gated inject for one layer.

        Placeholder rows keep zero inject values; the gate is masked by the
        step's validity mask. Mirrors ``run_prefill``'s inject block (the
        token mask also drops the image token, like the reference).

        When an ENP group is bound the lookup is a COLLECTIVE: every rank
        calls it every step even with zero local rows (an empty-rank DP rank
        still participates, plan 3.2.2). Rank-local (group None) skips the
        call for empty row sets.
        """
        geometry = self.geometry
        hash_index = geometry.engram_layer_ids.index(layer_id)
        tokens = h.shape[0]
        gate_mask = valid_mask.to(h.device)
        group = self._engram_group(layer_id)
        rows = None
        if hashes is not None and hashes.shape[0] > 0:
            rows = hashes[:, hash_index]
        elif group is not None:
            rows = torch.zeros(
                (0, (geometry.engram_max_ngram_size - 1) * geometry.engram_n_heads),
                dtype=torch.int64,
            )
        if rows is None:
            value = torch.zeros(
                (tokens, (geometry.hc_mult + 1) * geometry.hidden_size),
                dtype=torch.bfloat16,
                device=h.device,
            )
            return engram_gate_inject(
                h,
                layer.engram_q_weight,
                layer.engram_k_weight,
                value,
                token_mask=gate_mask,
            )
        token_mask = (rows[:, 0] >= 0) & (rows[:, 0] != geometry.image_token_id)
        if group is not None:
            rows = rows.to(self.device)
        embedded = self.ops.engram_lookup(rows, layer_id=int(layer_id))
        # Flatten [T, W, dim] -> [T, W*dim] with an EXPLICIT width: an empty
        # lookup (empty-rank decode still joins the ENP collective) cannot
        # infer the ``-1`` dimension (torch rejects ambiguous 0-element
        # reshapes).
        flat = embedded.reshape(
            embedded.shape[0], embedded.shape[1] * embedded.shape[2]
        )
        kv = wkv_project(
            flat.to(layer.engram_wkv.weight.device), layer.engram_wkv.weight
        )
        # Re-expand onto the full row set (placeholders -> zero inject).
        value = torch.zeros((tokens, kv.shape[1]), dtype=kv.dtype, device=h.device)
        value[: kv.shape[0]] = kv.to(value.device)
        full_mask = torch.zeros(tokens, dtype=torch.bool, device=h.device)
        full_mask[: token_mask.shape[0]] = token_mask.to(h.device)
        return engram_gate_inject(
            h,
            layer.engram_q_weight,
            layer.engram_k_weight,
            value,
            token_mask=full_mask & gate_mask,
        )

    # -- indexer top-k ----------------------------------------------------------

    def _decode_indexer_k(self, layer_id: int, count: int, request_id: int, device):
        """This request's first ``count`` indexer K rows, decoded to BF16.

        The indexer K pool belongs to the layer's KV SOURCE, not the layer
        itself (``prefill._indexer_topk`` reads the source's pool: pure index
        sources -- 24/28/32/36 in the real config -- own no pool of their
        own; the synthetic config hides this because its index sources ARE
        its kv sources).
        """
        geometry = self.geometry
        source = geometry.kv_source_for(layer_id)
        if source is None:
            raise RuntimeError(
                f"index-source layer {layer_id} has no kv-source layer below "
                "it; the config is inconsistent"
            )
        packed = self.cache.read_source(source, request_id, indexer_entries=count)[1]
        decoded = torch.empty(
            (packed.shape[0], geometry.index_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.ops.decode_kv(
            packed,
            pool_tag="indexer_kv",
            descriptor=self.cache.descriptors(source)["indexer_kv"],
            out=decoded,
        )
        return decoded

    def _indexer_query(self, layer, x_pre, qr, positions):
        """Roped indexer Q + prescaled head weights (shared by both scorers)."""
        geometry = self.geometry
        heads = geometry.index_n_heads
        freqs = self.bundle.freqs_for(layer.layer_id)
        index_q = layer.index_wq(qr).reshape(-1, heads, geometry.index_head_dim)
        index_q = _rope_tail(
            index_q,
            freqs[torch.tensor(positions, dtype=torch.int64, device=freqs.device)],
        )
        raw_weights = F.linear(x_pre, layer.index_weights_proj).float()
        head_weights = raw_weights * (geometry.index_head_dim * heads) ** -0.5
        return index_q, head_weights

    def _indexer_topk(self, layer, x_pre, qr, active, positions):
        """Selected compressed entries ``[valid_rows, topk]`` int32 (-1 padded).

        Mirrors ``prefill._indexer_topk`` for the decode shape: one row per
        request, each scored over its OWN visible entries (L1's
        ``index_score`` causality is count-based, so the naive single call
        over concatenated keys would cross requests). The batched formulation
        below scores all rows in ONE contract-op call: fake per-row positions
        widen the count-based causal window to ``[0, offset_r + visible_r)``,
        and the per-request entry-range mask restores the true boundaries
        before the top-k. Equivalence against the per-request reference
        (``_indexer_topk_per_request``) is pinned by ``test_decode.py``.
        """
        geometry = self.geometry
        layer_id = layer.layer_id
        ratio = geometry.ratio(layer_id)
        heads = geometry.index_n_heads
        rows = x_pre.shape[0]
        topk = geometry.index_topk
        device = x_pre.device
        # Visible = the request's COMMITTED entry count (== (pos + 1) // ratio
        # in a real step, where the new token was just committed + alloc'ed).
        visible = [
            self.cache.request_length(request_id) // ratio for request_id in active
        ]

        out = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        keys = []
        offsets = []
        running = 0
        for index, request_id in enumerate(active):
            offsets.append(running)
            decoded = self._decode_indexer_k(
                layer_id, visible[index], request_id, device
            )
            keys.append(decoded)
            running += decoded.shape[0]
        if running == 0:
            return out
        all_keys = torch.cat(keys, dim=0)

        index_q, head_weights = self._indexer_query(layer, x_pre, qr, positions)

        # One batched contract-op call: fake per-row positions make the
        # count-based causal window cover exactly [0, offset_r + visible_r).
        fake = torch.tensor(
            [(offsets[i] + visible[i]) * ratio - 1 for i in range(rows)],
            dtype=torch.int64,
            device=device,
        )
        logits = self.ops.index_score(
            _quantized_indexer_query(index_q),
            V41QuantizedTensor(all_keys, None, "bf16"),
            head_weights,
            layer_id=layer_id,
            candidate_blocks=None,
            key_positions=torch.arange(
                all_keys.shape[0], device=device, dtype=torch.int64
            )
            * ratio,
            context=_PositionsOnlyContext(fake),
            ratio=ratio,
        )

        masked = logits.clone()
        for index in range(rows):
            keep = torch.zeros(all_keys.shape[0], dtype=torch.bool, device=device)
            keep[offsets[index] : offsets[index] + visible[index]] = True
            masked[index] = logits[index].masked_fill(~keep, float("-inf"))
        k_eff = min(topk, all_keys.shape[0])
        values, columns = torch.topk(masked, k_eff, dim=-1)
        # Column j of row r is a GLOBAL key index; shift back to the
        # request-local entry id (0-based, exactly what selected_blocks means).
        shifts = torch.tensor(offsets, dtype=torch.int64, device=device).reshape(
            rows, 1
        )
        entries = columns - shifts
        finite = torch.isfinite(values)
        out[:, :k_eff] = torch.where(
            finite,
            entries.to(torch.int32),
            torch.full_like(entries, -1, dtype=torch.int32),
        ).to(out.device)
        return out

    def _indexer_topk_per_request(self, layer, x_pre, qr, active, positions):
        """Per-request reference scorer (test oracle for the batched path)."""
        geometry = self.geometry
        layer_id = layer.layer_id
        ratio = geometry.ratio(layer_id)
        rows = x_pre.shape[0]
        topk = geometry.index_topk
        device = x_pre.device
        out = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        index_q, head_weights = self._indexer_query(layer, x_pre, qr, positions)
        for index, request_id in enumerate(active):
            visible = self.cache.request_length(request_id) // ratio
            if visible <= 0:
                continue
            decoded = self._decode_indexer_k(layer_id, visible, request_id, device)
            logits = self.ops.index_score(
                _quantized_indexer_query(index_q[index : index + 1]),
                V41QuantizedTensor(decoded, None, "bf16"),
                head_weights[index : index + 1].contiguous(),
                layer_id=layer_id,
                candidate_blocks=None,
                key_positions=torch.arange(visible, dtype=torch.int64) * ratio,
                context=_PositionsOnlyContext(
                    torch.tensor(
                        [int(positions[index])], dtype=torch.int64, device=device
                    )
                ),
                ratio=ratio,
            )
            ids = _topk_with_lengths(
                logits.reshape(1, -1), topk, torch.tensor([visible])
            )
            out[index] = ids[0].to(torch.int32).to(device)
        return out


class _PositionsOnlyContext:
    """Minimal context carrying only ``positions`` (index_score contract)."""

    def __init__(self, positions: torch.Tensor) -> None:
        self.positions = positions


def _topk_with_lengths(
    logits: torch.Tensor, k: int, lengths: torch.Tensor
) -> torch.Tensor:
    """Top-k over the first ``lengths[row]`` entries; -1 pad at the tail.

    Mirrors ``TorchIndexerTopKBackend`` semantics: entries beyond the valid
    length are excluded, and rows with fewer valid entries than ``k`` carry
    ``-1`` in the trailing slots (never in the middle).
    """
    rows, total = logits.shape
    k_eff = min(k, total)
    out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
    if k_eff <= 0 or total <= 0:
        return out
    scores = logits.to(torch.float32)
    arange = torch.arange(total, dtype=torch.int64, device=logits.device).expand(
        rows, total
    )
    limit = lengths.to(torch.int64).to(logits.device).reshape(rows, 1)
    scores = torch.where(arange < limit, scores, torch.full_like(scores, float("-inf")))
    values, indices = torch.topk(scores, k_eff, dim=-1)
    valid = torch.isfinite(values)
    out[:, :k_eff] = torch.where(
        valid, indices.to(torch.int32), torch.full_like(indices, -1, dtype=torch.int32)
    )
    return out


# ---------------------------------------------------------------------------
# Wiring onto PPUModelCore (model.py stays untouched)
# ---------------------------------------------------------------------------


def wire_decode(
    core,
    *,
    dp_rank: int = 0,
    dp_size: int = 1,
) -> DecodeStepOrchestrator:
    """Attach the decode orchestration to a :class:`PPUModelCore`.

    Builds the orchestrator over the core's ops/weights/cache and registers
    ``core.decode_impl`` so ``core.decode(...)`` (and the framework
    ``DeepSeekV41PPUModel.forward`` decode arm) runs the L3 step. Returns the
    orchestrator — the offline drivers keep it for request management.
    """
    orchestrator = DecodeStepOrchestrator(
        ops=core.ops,
        bundle=core.weights,
        cache=getattr(core, "cache", None),
        device=getattr(core, "device", None),
        dp_rank=dp_rank,
        dp_size=dp_size,
    )

    def decode_impl(model_core, input_ids, positions=None, request_ids=None):
        return _framework_decode(orchestrator, input_ids, positions, request_ids)

    core.decode_impl = decode_impl
    return orchestrator


def _framework_decode(orchestrator, input_ids, positions=None, request_ids=None):
    """``core.decode(input_ids, positions=...)`` body (framework contract).

    Rows map to the orchestrator's unfinished requests: either explicitly
    (``request_ids`` per row) or in ascending request-id order. Returns
    ``PyModelOutputs`` over the VALID rows' hidden states.
    """
    ids = input_ids.reshape(-1).tolist()
    if request_ids is None:
        active = sorted(
            rid for rid, state in orchestrator.requests.items() if not state.finished
        )
        if len(active) != len(ids):
            raise RuntimeError(
                f"decode batch has {len(ids)} tokens but {len(active)} unfinished "
                "requests are registered; pass request_ids to disambiguate"
            )
        request_ids = active
    mapping = {int(rid): int(tok) for rid, tok in zip(request_ids, ids)}
    result = orchestrator.step(mapping)
    from rtp_llm.ops import PyModelOutputs

    return PyModelOutputs(result.hidden)
