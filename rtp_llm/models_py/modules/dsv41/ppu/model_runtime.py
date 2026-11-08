"""Framework lifecycle and rank-local resources for shared eager execution."""

import json
import hashlib
import logging
import os
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_tagged_attention_inputs
from rtp_llm.models_py.modules.dsv41.ppu.engram import EngramHashParams, compute_engram_hash_ids
from rtp_llm.models_py.modules.dsv41.ppu.native_model import _NATIVE_RUNNERS, _native_shutdown_group
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import native_cache_protocol
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.dspark_modes import target_loop_geometry
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState, _Overlay, encoder_replay_inputs
from rtp_llm.models_py.modules.dsv41.ppu.weight import _build_freqs_cis
from rtp_llm.models_py.modules.dsv41.prefill.forward import forward_prefill
from rtp_llm.ops.compute_ops import PyModelOutputs


@dataclass(frozen=True)
class _NoBusinessVerifyToken:
    """One fake target call, including calls whose global vote skips compute."""

    call_id: int
    request_ids: tuple
    prefixes: tuple
    lengths: tuple
    reason: str = "collective_placeholder"

    @staticmethod
    def _values(tensor):
        return tuple(map(int, tensor.cpu().reshape(-1).tolist()))

    @classmethod
    def capture(cls, inputs, attn, call_id, width_limit):
        if not bool(inputs.is_fake_stream) or not bool(getattr(attn, "is_target_verify", False)):
            raise ValueError("no-business token requires this call's fake target verify")
        ids = cls._values(inputs.request_id)
        prefixes = cls._values(attn.prefix_lengths)
        lengths = cls._values(attn.input_lengths)
        if (len(ids) != len(prefixes) or len(ids) != len(lengths)
                or len(set(ids)) != len(ids) or sum(lengths) != inputs.input_ids.numel()
                or any(p < 0 for p in prefixes)
                or any(n < 1 or n > width_limit for n in lengths)):
            raise ValueError("fake verify request/prefix/row geometry is inconsistent")
        return cls(call_id, ids, prefixes, lengths)

    def validate(self, request_ids, prefixes, accepted_rows):
        ids, bases, counts = map(self._values, (request_ids, prefixes, accepted_rows))
        if ids != self.request_ids or bases != self.prefixes:
            raise ValueError("stale no-business acceptance: request/order/prefix mismatch")
        # The public fake sampler starts at gamma+1, then the normal stream
        # preview may clip it for max-output/EOS/error. It need not be zero.
        if len(counts) != len(self.lengths) or any(n < 0 or n > width for n, width in zip(counts, self.lengths)):
            raise ValueError("no-business accepted rows exceed this fake verify call")
        return counts

    def evidence(self):
        return {"idle": True, "no_business_call": self.call_id, "reason": self.reason,
                "request_ids": self.request_ids, "prefixes": self.prefixes,
                "verified_rows": self.lengths, "native_rows_published": 0}


class SharedNativeRuntime:
    def __init__(self, model):
        self._model = weakref.ref(model)
        self.geometry = model.shared_weight_views.geometry
        self.policy = SharedPrefillPolicy.from_config(model.shared_weight_views.config)
        self.current = None
        self._freqs = {}
        self._requested, self._observed = lambda: False, lambda: None
        self._stopped = False
        self.observer = None
        self.qkv_streams = None
        self.last_n5_routes = []
        self.last_n5_writes = []
        self.graph_replays = 0
        self.core = SimpleNamespace(geometry=self.geometry)
        self.calls = 0
        self.pending_verify = None
        self.pending_idle_verify = None
        self._idle_verify_calls = 0
        self.device = model.v4.embed.weight.device if model.v4.embed is not None else model.main_proj.weight.device
        self.hash_params = None
        g = self.geometry
        if g.engram_enabled():
            params = EngramHashParams.build(
                layer_ids=g.engram_layer_ids, max_ngram_size=g.engram_max_ngram_size,
                n_heads=g.engram_n_heads, vocab_size=g.engram_vocab_size,
                compressed_vocab_size=g.engram_compressed_vocab_size,
                num_embeddings=g.engram_num_embeddings, token_map=g.engram_token_map,
                pad_token_id=g.engram_pad_token_id,
            )
            self.hash_params = replace(params, **{name: getattr(params, name).to(self.device) for name in ("token_map", "multipliers", "primes", "offsets")})
        if dist.is_initialized():
            _native_shutdown_group()
        _NATIVE_RUNNERS.add(self)
        model.v4._ppu_runtime = self
        for layer in model.v4.layers:
            layer.attn.runtime = self
            if getattr(layer, "ffn", None) is not None:
                layer.ffn.runtime = self

    @property
    def model(self):
        return self._model()

    def freqs_for(self, layer):
        base = self.geometry.rope_theta if self.geometry.ratio(layer) == 0 else self.geometry.compress_rope_theta
        if base not in self._freqs:
            self._freqs[base] = _build_freqs_cis(self.geometry, base, self.device)
        return self._freqs[base]

    def bind_native_stop(self, requested, observed):
        self._requested, self._observed = requested, observed

    def vote(self, count):
        if self._stopped:
            return False
        value = torch.tensor([count, int(self._requested())], dtype=torch.int64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(value)
        count, stopping = value.tolist()
        if stopping:
            self._stopped = True
            self._observed()
            return False
        return count > 0

    def prepare_hashes(self, inputs, idle):
        if self.hash_params is None:
            return
        windows = inputs.engram_token_windows
        if idle:
            windows = torch.full((inputs.input_ids.numel(), self.geometry.engram_max_ngram_size), -1, dtype=torch.int64, device=self.device)
        if windows is None or tuple(windows.shape) != (inputs.input_ids.numel(), self.geometry.engram_max_ngram_size):
            raise ValueError("shared execution requires every scheduler Engram token window")
        windows = windows.to(self.device)
        self.current.window_current_tokens = windows[:, 0]
        if not idle and not torch.equal(windows[:, 0].long(), inputs.input_ids.reshape(-1).long()):
            # CP padding may carry a clamped token; real rows still agree.
            cp = getattr(self.current.attn, "context_parallel_info", None)
            real = windows[:, 0] >= 0 if self.current.prefill and cp is not None else torch.ones_like(windows[:, 0], dtype=torch.bool)
            if not torch.equal(windows[real, 0].long(), inputs.input_ids.reshape(-1)[real].long()):
                raise ValueError("Engram current tokens disagree with actual model input")
        dead = windows == self.geometry.image_token_id
        hashes = compute_engram_hash_ids(windows, self.hash_params, dead_mask=dead)
        mask = (windows[:, 0] >= 0) & ~dead[:, 0]
        if idle:
            mask.zero_()
        for index, layer_id in enumerate(self.geometry.engram_layer_ids):
            layer = self.model.v4.layers[layer_id]
            layer.engram_hashes = hashes[:, index].contiguous()
            layer.engram_token_mask = mask

    def dispatch(self, inputs, fmha_impl=None):
        """Metadata/dispatch only; both complete layer loops remain public."""
        if self.current is not None:
            raise RuntimeError("shared PPU forward is not reentrant")
        if self.pending_verify is not None or self.pending_idle_verify is not None:
            raise RuntimeError("previous shared verify must be accepted or aborted before another forward")
        model = self.model
        rows = int(inputs.input_ids.numel())
        idle = bool(inputs.is_fake_stream)
        attn = as_tagged_attention_inputs(inputs.attention_inputs)
        verify = bool(getattr(attn, "is_target_verify", False))
        packed_verify = verify and bool(getattr(attn, "is_packed_target_verify", False))
        if verify and self.policy.max_rollback_tokens <= 0:
            raise ValueError("target verify requires a frozen speculative rollback capacity")
        idle_token = None
        if idle and verify:
            idle_token = _NoBusinessVerifyToken.capture(inputs, attn, self._idle_verify_calls + 1, self.policy.max_rollback_tokens)
            self._idle_verify_calls += 1
        if not self.vote(0 if idle else rows):
            if idle_token is not None:
                self.pending_idle_verify = replace(idle_token, reason="all_empty_vote")
            return PyModelOutputs(torch.zeros((rows, self.geometry.hidden_size), dtype=torch.bfloat16, device=self.device))
        if model.kv_cache is None:
            if idle_token is not None:
                self.pending_idle_verify = replace(idle_token, reason="no_kv_cache")
            return PyModelOutputs(torch.zeros((rows, self.geometry.hidden_size), dtype=torch.bfloat16, device=self.device))
        if fmha_impl is not None or bool(getattr(inputs.attention_inputs, "is_cuda_graph", False)):
            raise NotImplementedError("shared Graph resources belong to N4/N9")
        prefill = bool(attn.is_prefill) and not model._is_decode_role and not idle and not verify
        cp_info = getattr(attn, "context_parallel_info", None) if prefill else None
        if verify and not idle:
            from rtp_llm.models_py.modules.dsv41.ppu.verify_transaction import VerifyLayout

            layout = VerifyLayout.from_inputs(attn, rows)
            prefixes, lengths = layout.prefixes, layout.lengths
            target_loop_geometry(lengths, packed_verify)
            if max(lengths) > self.policy.max_rollback_tokens or any(p + n > self.geometry.max_seq_len for p, n in zip(prefixes, lengths)):
                raise ValueError("actual target rows exceed frozen rollback/position capacity")
        elif idle:
            # Native fake streams may have no sequence-length metadata. They
            # still execute every ENP/EP collective, without any business pool.
            count = max(rows, 1)
            inputs = _Overlay(inputs, input_ids=torch.zeros(count, dtype=torch.long, device=self.device))
            prefixes, lengths = (0,) * count, (1,) * count
        elif cp_info is not None and model._prefill_cp_size > 1:
            lengths = tuple(map(int, cp_info.prefill_actual_input_lengths_cpu.tolist()))
            prefixes = tuple(map(int, cp_info.prefill_prefix_lengths_cpu.tolist()))
        elif prefill:
            lengths = tuple(map(int, attn.input_lengths.tolist()))
            prefixes = tuple(map(int, attn.prefix_lengths.tolist()))
        else:
            prefixes = tuple(map(int, attn.sequence_lengths.tolist()))
            lengths = (1,) * len(prefixes)
        if not prefixes or len(prefixes) != len(lengths):
            raise ValueError("shared native metadata lacks request lengths")
        totals = tuple(p + n for p, n in zip(prefixes, lengths))
        original_prefixes = prefixes
        restore_replay = None
        if prefill and self.policy.encoder_replay and any(prefixes) and not idle:
            inputs, lengths, restore_replay = encoder_replay_inputs(
                inputs, attn, lengths, prefixes, model._prefill_cp_size,
                int(getattr(model.parallelism_config, "tp_rank", 0)),
            )
            attn = as_tagged_attention_inputs(inputs.attention_inputs)
            prefixes = (0,) * len(lengths)
        from rtp_llm.models_py.modules.dsv41.ppu.verify_transaction import SharedVerifyCache, SharedVerifyTransaction

        cache_type = SharedVerifyCache if verify else SharedNativeCache
        cache = None if idle else cache_type(model.shared_weight_views.config, self.geometry, model.kv_cache, inputs.attention_inputs, totals, device=self.device, prefill=prefill)
        transaction = SharedVerifyTransaction(
            cache, inputs.request_id, layout, inputs.engram_token_windows, inputs.input_ids,
            call_id=self.calls + 1, max_backtracking=self.policy.max_rollback_tokens,
        ) if verify and not idle else None
        state = SharedForwardState(self, inputs, attn, cache, prefill=prefill, idle=idle, prefixes=prefixes, lengths=lengths)
        if restore_replay is not None:
            state.append_starts = dict(enumerate(original_prefixes))
        self.current = state
        state.verify = verify
        validate_state = os.environ.get("RTP_V41_SHARED_VALIDATE_STATE", "0") == "1"
        immutable_before = cache.fingerprint(original_prefixes, global_only=True) if cache is not None and prefill and validate_state and any(original_prefixes) else None
        state.native_state_before = cache.fingerprint(original_prefixes) if cache is not None and not prefill and validate_state else {}
        counts_before = [(layer.attn_norm.calls, layer.ffn_norm.calls, layer.ffn.calls) for layer in model.v4.layers]
        mega_before = sum(model._shared_ops.mhc_execution_stats()["calls"].values())
        try:
            model.v4._propagate_cp_ctx(None)
            self.prepare_hashes(inputs, idle)
            for layer in model.v4.layers:
                layer.attn._shared_attention = state.shared if self.policy.mode == "exact" else None
                layer.attn_hc.state = layer.ffn_hc.state = None
                layer.attn_hc.pre_mix_out = layer.ffn_hc.pre_mix_out = None
            if prefill:
                if model._lazy_mtp_hidden_prefill:
                    model._shared_runtime_buffers.ensure_mtp_hidden_capacity(inputs.input_ids.numel())
                result = forward_prefill(model.v4, model.kv_cache, model.parallelism_config, inputs)
                state.public_returned = True
                if self.policy.mode == "exact" and state.exact_context is None:
                    state.plan_reason = "public_exact_normal_fallback"
                if restore_replay is not None:
                    result = PyModelOutputs(restore_replay(result.hidden_states, state.original_context))
                    if model.v4._mtp_hidden_buffer is not None:
                        source_rows = state.original_context.chunk_length if state.original_context is not None else inputs.input_ids.numel()
                        aux = restore_replay(model.v4._mtp_hidden_buffer[:source_rows], state.original_context)
                        model.v4._mtp_hidden_buffer[:aux.shape[0]].copy_(aux)
                        model.v4._note_aux_hidden_rows(aux.shape[0], False)
            else:
                batch, width = target_loop_geometry(lengths, packed_verify)
                metadata = SimpleNamespace(batch_size=batch, q_len_per_req=width, is_cuda_graph=False)
                hidden = model.v4.forward_decode(inputs.input_ids, metadata, expert_token_counts=inputs.expert_token_counts, kv_cache=model.kv_cache)
                if model.v4.capture_aux_hidden_layer_ids:
                    model.v4._note_aux_hidden_rows(rows, False)
                result = PyModelOutputs(hidden)
            if not idle and cache is not None:
                # Writer sees final live native bytes; no logical-cache export.
                if prefill:
                    cache.publish()
            if idle:
                result = PyModelOutputs(torch.zeros((rows, self.geometry.hidden_size), dtype=torch.bfloat16, device=self.device))
            state.output_finite = bool(torch.isfinite(result.hidden_states).all()) if validate_state else None
            if state.output_finite is False:
                raise RuntimeError("shared native output contains non-finite hidden states")
            state.verify_native_unchanged = None
            if transaction is not None and validate_state:
                state.verify_native_unchanged = cache.native_fingerprint(original_prefixes) == state.native_state_before
                if not state.verify_native_unchanged:
                    raise RuntimeError("verify changed native committed bytes before acceptance")
            if transaction is not None:
                transaction.aux_features = model.get_mtp_target_hidden_states(rows).clone()
                if not bool(torch.isfinite(transaction.aux_features).all()):
                    raise RuntimeError("verify target features contain non-finite values")
            self.calls += 1
            if immutable_before is not None:
                immutable_after = cache.fingerprint(original_prefixes, global_only=True)
                if immutable_after != immutable_before:
                    raise RuntimeError("shared Prefill mutated an immutable Global/Indexer prefix")
                if os.environ.get("RTP_V41_TEST_DSPARK_NATIVE10") == "1":
                    state.reuse_prefix_evidence = {"before": immutable_before, "after": immutable_after}
            state.native_state_after = cache.fingerprint(totals) if cache is not None and prefill and validate_state else {}
            state.normalization = {
                "attention_norm_calls": sum(layer.attn_norm.calls - before[0] for layer, before in zip(model.v4.layers, counts_before)),
                "ffn_norm_calls": sum(layer.ffn_norm.calls - before[1] for layer, before in zip(model.v4.layers, counts_before)),
                "ffn_calls": sum(layer.ffn.calls - before[2] for layer, before in zip(model.v4.layers, counts_before)),
                "mega_transitions": sum(model._shared_ops.mhc_execution_stats()["calls"].values()) - mega_before,
            }
            logging.info("PPU shared %s rank=%d layers=%s prefill_executed=%s profile=%s", "PREFILL" if prefill else "DECODE", dist.get_rank() if dist.is_initialized() else 0, state.executed_layers, prefill, self.policy)
            self.last_n5_routes = state.shared.get("n5_routes", [])
            self.last_n5_writes = list(state.cache.writes) if state.cache is not None else []
            self.report(state)
            if verify:
                self.pending_verify = transaction
                self.pending_idle_verify = idle_token
            return result
        except Exception:
            if transaction is not None:
                transaction.abort()
            raise
        finally:
            for layer in model.v4.layers:
                layer.attn._shared_attention = None
                layer.attn._set_prefill_meta_shared(None)
                if layer.engram is not None:
                    layer.engram.clear()
                    layer.engram_hashes = layer.engram_token_mask = None
                layer.attn_hc.state = layer.ffn_hc.state = None
                layer.attn_hc.pre_mix_out = layer.ffn_hc.pre_mix_out = None
            model.v4._propagate_cp_ctx(None)
            self.current = None

    def stage_idle_graph_verify(self, inputs, reason):
        attn = as_tagged_attention_inputs(inputs.attention_inputs)
        token = _NoBusinessVerifyToken.capture(
            inputs, attn, self._idle_verify_calls + 1, self.policy.max_rollback_tokens
        )
        self._idle_verify_calls += 1
        self.pending_idle_verify = replace(token, reason=reason)

    def stage_graph_verify(self, inputs, impl, aux_features, *, native_replay=False):
        """Stage the completed verify on the consumer stream for acceptance."""
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import SharedDSparkGraphImpl
        from rtp_llm.models_py.modules.dsv41.ppu.verify_transaction import (
            SharedVerifyCache, SharedVerifyTransaction, VerifyLayout,
        )

        if self.current is not None or self.pending_verify is not None or self.pending_idle_verify is not None:
            raise RuntimeError("another native verification transaction is active")
        if (not isinstance(impl, SharedDSparkGraphImpl) or impl.mode not in ("dense", "packed")
            or impl.runtime is not self.model.shared_graph_runtime):
            raise ValueError("target verify requires this model's captured journal")
        completion = int(impl.cache.completions.item())
        armed = impl.armed_verify
        if native_replay:
            if completion <= impl.staged_completion:
                raise ValueError("target verify journal was already staged")
        elif armed is None or completion != armed[0] + 1 or not torch.equal(armed[1], inputs.request_id):
            raise ValueError("target verify journal does not identify this request and completion")
        attn = as_tagged_attention_inputs(inputs.attention_inputs)
        if not attn.is_target_verify or bool(attn.is_packed_target_verify) != (impl.mode == "packed"):
            raise ValueError("verify inputs do not match captured layout")
        if inputs.is_fake_stream:
            if bool(impl.consumed_lengths.any()):
                raise ValueError("fake verify replay consumed live rows")
            self.stage_idle_graph_verify(inputs, "graph_empty_owner")
            impl.staged_completion = completion
            impl.armed_verify = None
            return
        rows = inputs.input_ids.numel()
        layout = VerifyLayout.from_inputs(attn, rows)
        batch = len(layout.lengths)
        if (not 0 < batch <= impl.batch_size or rows > impl.rows
            or tuple(impl.consumed_lengths.tolist()) != layout.lengths + (0,) * (impl.batch_size - batch)
            or tuple(impl.consumed_prefixes[:batch].tolist()) != layout.prefixes
            or not torch.equal(impl.consumed_ids[:rows], inputs.input_ids)
            or not torch.equal(impl.consumed_windows[:rows], inputs.engram_token_windows)):
            raise ValueError("acceptance input does not identify this completed verify replay")
        if max(layout.lengths) > self.policy.max_rollback_tokens or (
            impl.mode == "dense" and any(n != impl.width for n in layout.lengths)
        ):
            raise ValueError("logical verify lengths exceed captured scope")
        ends = tuple(p + n for p, n in zip(layout.prefixes, layout.lengths))
        if max(ends) > impl.cache.max_seq_len:
            raise ValueError("verify positions exceed captured rotary/cache capacity")
        if aux_features is None or aux_features.shape[0] != rows:
            raise ValueError("Graph target aux rows differ from verify rows")
        cache = SharedVerifyCache(
            self.model.shared_weight_views.config, self.geometry, self.model.kv_cache,
            inputs.attention_inputs, ends, device=self.device,
        )
        tx = SharedVerifyTransaction(
            cache, inputs.request_id, layout, inputs.engram_token_windows, inputs.input_ids,
            call_id=self.calls + 1, max_backtracking=self.policy.max_rollback_tokens,
        )
        try:
            impl.cache.stage(cache, layout)
            tx.aux_features = aux_features.clone()
        except Exception:
            tx.abort()
            raise
        self.calls += 1
        self.pending_verify = tx
        impl.staged_completion = completion
        impl.armed_verify = None

    def finish_speculative_forward(self, request_ids, prefixes, accepted_rows):
        tx = self.pending_verify
        if tx is None:
            token = self.pending_idle_verify
            if token is None:
                raise RuntimeError("no pending shared verify transaction")
            try:
                counts = token.validate(request_ids, prefixes, accepted_rows)
                self.event("accept", {**token.evidence(), "accepted_input_rows": counts})
            finally:
                self.pending_idle_verify = None
            return
        try:
            evidence = tx.finish(request_ids, prefixes, accepted_rows)
            features = torch.cat([
                tx.aux_features[start:start + count]
                for start, count in zip(tx.layout.starts, evidence["accepted_input_rows"])
            ])
            evidence["accepted_feature_sha256"] = hashlib.sha256(features.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            evidence["accepted_feature_rows"] = features.shape[0]
            if os.environ.get("RTP_V41_SHARED_VALIDATE_STATE", "0") == "1":
                evidence["native_state_after"] = tx.cache.native_fingerprint(evidence["accepted_ends"])
            self.event("accept", evidence)
        finally:
            if not tx.closed:
                tx.abort()
            self.pending_verify = None

    def abort_speculative_forward(self):
        tx, self.pending_verify = self.pending_verify, None
        token, self.pending_idle_verify = self.pending_idle_verify, None
        if tx is not None:
            tx.abort()
            self.event("abort", {"call": tx.call_id, "request_ids": tx.request_ids, "native_rows_published": 0})
        elif token is not None:
            self.event("abort", token.evidence())

    def event(self, kind, payload):
        directory = os.environ.get("RTP_V41_SHARED_EVIDENCE_DIR")
        if directory:
            if payload.get("idle") and os.environ.get("RTP_V41_SHARED_COMPACT_IDLE_EVIDENCE") == "1":
                if payload["native_rows_published"] != 0:
                    raise AssertionError("idle target event published native rows")
                if payload["no_business_call"] != 1:
                    return
            Path(directory).mkdir(parents=True, exist_ok=True)
            rank = dist.get_rank() if dist.is_initialized() else 0
            owner = f"idle{payload['no_business_call']:05d}" if "no_business_call" in payload else f"call{self.calls:05d}"
            with (Path(directory) / f"target-rank{rank}-{owner}-{kind}.json").open("x") as writer:
                json.dump({"schema": "v41-ppu-shared-dspark-event-v1", "rank": rank, "event": kind, **payload}, writer, indent=2)

    def report(self, state):
        directory = os.environ.get("RTP_V41_SHARED_EVIDENCE_DIR")
        if not directory:
            return
        if state.idle and os.environ.get("RTP_V41_SHARED_COMPACT_IDLE_EVIDENCE") == "1":
            if state.cache is not None or any(state.routed_rows.values()):
                raise AssertionError("idle target forward owns cache or routed rows")
            return
        Path(directory).mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        report = {
            "schema": "v41-ppu-shared-forward-v1", "rank": rank, "call": self.calls,
            "idle": state.idle, "prefill": state.prefill, "model_layers": self.geometry.num_layers,
            "geometry": {"hidden_size": self.geometry.hidden_size, "hc_mult": self.geometry.hc_mult,
                         "routed_experts": self.geometry.n_routed_experts, "top_k": self.geometry.n_activated_experts},
            "native_config_cache_key_hash_seed": self.model.config.cache_key_hash_seed,
            "model_class": type(self.model).__module__ + "." + type(self.model).__name__,
            "target_verify": getattr(state, "verify", False),
            "packed_target_verify": bool(getattr(state.attn, "is_packed_target_verify", False)),
            "row_layout": {"lengths": state.lengths, "prefixes": state.query_prefixes,
                           "order": "scheduler-request-major", "loop_view": "flat" if bool(getattr(state.attn, "is_packed_target_verify", False)) else "dense"},
            "input_token_ids": state.inputs.input_ids.reshape(-1).tolist(),
            "verify_native_publication": "DEFERRED_UNTIL_FRAMEWORK_ACCEPTANCE" if getattr(state, "verify", False) else "ORDINARY",
            "verify_native_unchanged": getattr(state, "verify_native_unchanged", None),
            "profile": self.policy.__dict__, "cache_protocol": native_cache_protocol(self.model.shared_weight_views.config),
            "yoco": state.evidence(), "mega_mhc": self.model._shared_ops.mhc_execution_stats(),
            "native_pool_bindings": [
                {"layer": owner, "tag": tag, "framework_tag": pool.framework_tag,
                 "entries": pool.entries, "span": pool.span, "native_alias": pool.tensor.data_ptr() == pool.handle.kv_cache_base.data_ptr()}
                for (owner, tag), pool in state.cache.pools.items()
            ] if state.cache is not None else [],
            "native_writes": state.cache.writes if state.cache is not None else [],
            "cache_placement": state.cache.placement_evidence() if state.cache is not None else [],
            "cache_read_lifetime": "source assemblies and SWA fresh rows released after Attention; other gathers call-local",
            "pool_gathers": state.cache.pool_gathers if state.cache is not None else [],
            "current_row_gathers": state.shared.get("current_row_gathers", []),
            "persistent_replicated_shadow": False,
            "n5_routes": state.shared.get("n5_routes", []),
            "normalization": state.normalization,
            "output_finite": state.output_finite,
            "moe": self.model._shared_ops.moe_engine.execution_stats(),
            "routed_active_rows": state.routed_rows,
            "native_state_before": state.native_state_before,
            "native_state_after": state.native_state_after,
            "immutable_global_prefix": "EXACT_BYTES_UNCHANGED" if state.prefill and state.append_starts is not None and state.native_state_after else "NOT_CHECKED_OR_NO_REUSE",
        }
        reuse_evidence = state.prefill and os.environ.get("RTP_V41_TEST_DSPARK_NATIVE10") == "1"
        aux_rows = -1 if reuse_evidence and state.append_starts is not None else int(state.inputs.input_ids.numel())
        aux = self.model.get_mtp_target_hidden_states(aux_rows)
        report["aux_feature_rows"] = int(aux.shape[0]) if aux is not None else 0
        report["aux_feature_sha256"] = hashlib.sha256(aux.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() if aux is not None else None
        if reuse_evidence:
            report["request_ids"] = state.inputs.request_id.reshape(-1).tolist()
            report["reuse_prefix"] = getattr(state, "reuse_prefix_evidence", None)
            if report["reuse_prefix"] is not None:
                context = state.original_context
                real = context.local_is_real
                report["replay_history"] = {
                    "positions": context.global_positions[real].tolist(),
                    "tokens": state.inputs.input_ids.reshape(-1)[real].tolist(),
                }
        # Execution evidence is generated output, not a source mutation.
        path = Path(directory) / f"rank{rank}-call{self.calls:05d}.json"
        with path.open("x") as writer:
            json.dump(report, writer, indent=2)
            writer.write("\n")
