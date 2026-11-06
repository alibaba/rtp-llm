"""Native RTP model execution over framework cache resources and scheduler IO."""

from __future__ import annotations

import logging
import os
import weakref

import torch

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import primary_attention_inputs
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.cp import (
    CPPrefillContext,
    DistributedCPGroup,
    build_zigzag_plan,
)
from rtp_llm.models_py.modules.dsv41.ppu.decode import DecodeStepOrchestrator
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy

logger = logging.getLogger(__name__)
_SHUTDOWN_GROUP = None
_NATIVE_RUNNERS = weakref.WeakSet()


def _native_shutdown_group():
    from datetime import timedelta

    import torch.distributed as dist

    global _SHUTDOWN_GROUP
    if _SHUTDOWN_GROUP is None:
        _SHUTDOWN_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    return _SHUTDOWN_GROUP


def drain_native_parallel():
    """Finish device work on every rank before freeing native/model storage."""
    import torch.distributed as dist

    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier(group=_native_shutdown_group())
        for runner in _NATIVE_RUNNERS:
            if runner.observer is not None:
                runner.observer.close()
        _write_perf_evidence()
        logger.info("V4.1 native role engine device work drained")


def _write_perf_evidence():
    import os

    evidence = os.environ.get("RTP_V41_PERF_EVIDENCE_DIR")
    if not evidence:
        return
    import json
    from pathlib import Path

    import torch.distributed as dist

    from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE

    directory = Path(evidence)
    directory.mkdir(parents=True, exist_ok=True)
    engines = list(PPUMoE._MEGA_ENGINES)
    (directory / f"moe-rank{dist.get_rank()}.json").write_text(
        json.dumps(
            {
                "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
                "decode_graph_replays": sum(r.graph_replays for r in _NATIVE_RUNNERS),
                "model_layers": sorted(
                    {r.core.geometry.num_layers for r in _NATIVE_RUNNERS}
                ),
                "engines": [
                    {"layers": len(engine._weights), **engine.execution_stats()}
                    for engine in engines
                ],
            },
            indent=2,
        )
        + "\n"
    )


def shutdown_native_parallel():
    """Release role communication before Python interpreter finalization."""
    import gc

    import torch.distributed as dist

    if not dist.is_initialized():
        return
    from rtp_llm.models_py.distributed.collective_torch import (
        destroy_distributed_environment,
    )
    from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
    from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE

    # A GPU barrier can still be completing on peers when a faster rank starts
    # destroying IPC buffers. Flush device work first, then coordinate teardown
    # on a CPU group so cleanup never adds work to those GPU streams.
    logger.info("V4.1 native role communication shutdown started")
    global _SHUTDOWN_GROUP
    shutdown_group = _native_shutdown_group()
    try:
        torch.cuda.synchronize()
        dist.barrier(group=shutdown_group)
        logger.info("V4.1 native role device work drained")
        PPUMoE.close_mega_resources()
        with PPUMoE._BUFFER_LOCK:
            PPUMoE._BUFFER_CACHE.clear()
        DeepEPWrapper.shutdown()
        gc.collect()
        dist.barrier(group=shutdown_group)
        logger.info("V4.1 native role communication buffers released")
    finally:
        dist.destroy_process_group(shutdown_group)
        _SHUTDOWN_GROUP = None
    destroy_distributed_environment()
    logger.info("V4.1 native role communication released")


class NativeModelRunner:
    def __init__(self, core, *, dp_rank=0, dp_size=1, cp_rank=0, cp_size=1, group=None):
        self.core = core
        self.dp_rank, self.dp_size = int(dp_rank), int(dp_size)
        self.cp_rank, self.cp_size = int(cp_rank), int(cp_size)
        self.group = group
        self.kv_cache = None
        self.decode_role = False
        self.steps = 0
        self._stop_requested = lambda: False
        self._stop_observed = lambda: None
        self._stopped = False
        self._graph_step_prepared = False
        self.graph_replays = 0
        self._pending_verify = None
        self._pending_idle_verify = False
        self.observer = None
        observation_dir = os.environ.get("RTP_V41_DSPARK_OBSERVE_DIR")
        if observation_dir:
            from rtp_llm.models_py.modules.dsv41.ppu.dspark_observation import (
                create_observer,
            )

            self.observer = create_observer(self, observation_dir)
        _NATIVE_RUNNERS.add(self)

    def bind_native_stop(self, requested, observed):
        self._stop_requested = requested
        self._stop_observed = observed

    def initialize(self, resources):
        self.kv_cache = resources.kv_cache
        self.decode_role = bool(resources.is_decode_role)
        import torch.distributed as dist

        if self.kv_cache is not None and dist.is_available() and dist.is_initialized():
            # Bootstrap control while the role's rendezvous is fully alive;
            # shutdown only uses the established CPU connections.
            _native_shutdown_group()
        return True

    def _work_available(self, count):
        import torch.distributed as dist

        if self._stopped:
            return False
        stopping = bool(self._stop_requested())
        if self.observer is not None:
            local_count, local_stopping = count, stopping
        if dist.is_available() and dist.is_initialized():
            value = torch.tensor(
                [count, int(stopping)], dtype=torch.int64, device=self.core.device
            )
            dist.all_reduce(value, group=self.group)
            count, stopping = value.cpu().tolist()
        if self.observer is not None:
            self.observer.vote(
                local_count, count, local_stopping, stopping, self.observer.warmup
            )
        if stopping:
            self._stopped = True
            self._stop_observed()
            logger.info("V4.1 native model stop vote observed")
            return False
        return count > 0

    def _orchestrator(self, cache):
        return DecodeStepOrchestrator(
            ops=self.core.ops,
            bundle=self.core.weights,
            cache=cache,
            device=self.core.device,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            capture_layer_ids=getattr(self.core, "capture_layer_ids", ()),
        )

    def prepare_graph_step(self, inputs):
        self._graph_step_prepared = False
        count = 0 if bool(inputs.is_fake_stream) else int(inputs.input_ids.numel())
        if self.observer is not None:
            self.observer.warmup = bool(getattr(inputs, "warmup", False))
        available = self._work_available(count)
        self._graph_step_prepared = available
        return available

    def note_graph_replay(self):
        self._graph_step_prepared = False
        self.graph_replays += 1
        if self.graph_replays & (self.graph_replays - 1) == 0:
            logger.info(
                "V4.1 native decode Graph replay dp=%d count=%d layers=%d",
                self.dp_rank,
                self.graph_replays,
                self.core.geometry.num_layers,
            )

    def forward(self, inputs, *, return_aux=False):
        primary = primary_attention_inputs(inputs.attention_inputs, self.kv_cache)
        fake = bool(inputs.is_fake_stream)
        rows = int(inputs.input_ids.numel())
        # Native DP supplies fake streams on idle ranks. Exchange work BEFORE
        # any layer collective; an entirely idle world skips ENP/EP uniformly.
        prepared = self._graph_step_prepared
        self._graph_step_prepared = False
        if self.observer is not None:
            self.observer.warmup = bool(getattr(inputs, "warmup", False))
        if not prepared and not self._work_available(0 if fake else rows):
            if bool(getattr(primary, "is_target_verify", False)):
                self._pending_idle_verify = True
            hidden = self.core.weights.embedding.new_zeros(
                (rows, self.core.geometry.hidden_size)
            )
            aux = self._empty_aux(rows)
            return (hidden, aux) if return_aux else hidden
        if bool(getattr(primary, "is_target_verify", False)):
            return self._verify(inputs, primary, return_aux=return_aux)
        if fake:
            cache = PPUKVCache(self.core.config, device=self.core.device)
            self._orchestrator(cache).step({}, placeholders=max(1, rows))
            hidden = self.core.weights.embedding.new_zeros(
                (rows, self.core.geometry.hidden_size)
            )
            return (hidden, self._empty_aux(rows)) if return_aux else hidden
        if self.decode_role and not bool(inputs.warmup):
            return self._decode(inputs, primary, return_aux=return_aux)
        if bool(primary.is_prefill):
            return self._prefill(inputs, primary, return_aux=return_aux)
        return self._decode(inputs, primary, return_aux=return_aux)

    def _empty_aux(self, rows):
        layers = getattr(self.core, "capture_layer_ids", ())
        return (
            self.core.weights.embedding.new_zeros(
                (rows, len(layers) * self.core.geometry.hidden_size)
            )
            if layers
            else None
        )

    def _verify(self, inputs, primary, *, return_aux=False):
        import torch.distributed as dist

        from rtp_llm.models_py.modules.dsv41.ppu.verify import (
            VerifyLayout,
            VerifyTransaction,
        )

        if self._pending_verify is not None or self._pending_idle_verify:
            self.abort_speculative_forward()
            raise RuntimeError("previous target verify was not accepted or aborted")
        layout = VerifyLayout.from_inputs(primary, inputs.input_ids.numel())
        fake = bool(inputs.is_fake_stream)
        width = max(layout.lengths, default=0) if not fake else 0
        stage_count = torch.tensor(width, dtype=torch.int32, device=self.core.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stage_count, op=dist.ReduceOp.MAX, group=self.group)
        width = int(stage_count)
        hidden = self.core.weights.embedding.new_zeros(
            (layout.rows, self.core.geometry.hidden_size)
        )
        aux = self._empty_aux(layout.rows)
        bridge = (
            None
            if fake
            else NativeCacheBridge(
                self.core.config,
                self.kv_cache,
                inputs.attention_inputs,
                device=self.core.device,
            )
        )
        cache = (
            PPUKVCache(self.core.config, device=self.core.device)
            if fake
            else bridge.import_requests(layout.prefixes)
        )
        orchestrator = self._orchestrator(cache)
        windows = inputs.engram_token_windows
        ids = inputs.input_ids.reshape(-1)
        tx = None
        if not fake:
            tx = VerifyTransaction(bridge, inputs.request_id, layout)
            if self.observer is not None and not bool(inputs.warmup):
                tx.observer = self.observer
                tx.observation = self.observer.begin(
                    "verify",
                    bridge,
                    list(tx.request_ids),
                    layout.prefixes,
                    layout.lengths,
                    cache=cache,
                )
            if self.core.geometry.engram_enabled() and (
                windows is None
                or tuple(windows.shape)
                != (layout.rows, self.core.geometry.engram_max_ngram_size)
            ):
                raise ValueError(
                    "causal verify requires every scheduler Engram token window"
                )
            for request, (prefix, start) in enumerate(
                zip(layout.prefixes, layout.starts)
            ):
                recent = (
                    [
                        int(t)
                        for t in windows[start, 1:].cpu().tolist()[::-1]
                        if int(t) >= 0
                    ]
                    if self.core.geometry.engram_enabled()
                    else []
                )
                orchestrator.resume_request(request, prefix, recent)
        try:
            for stage in range(width):
                active = (
                    [row for row, length in enumerate(layout.lengths) if stage < length]
                    if not fake
                    else []
                )
                tokens = {row: int(ids[layout.starts[row] + stage]) for row in active}
                if self.core.geometry.engram_enabled() and not fake:
                    for row in active:
                        offset = layout.starts[row] + stage
                        expected = [tokens[row]] + orchestrator.requests[row].history[
                            -(self.core.geometry.engram_max_ngram_size - 1) :
                        ][::-1]
                        expected += [-1] * (
                            self.core.geometry.engram_max_ngram_size - len(expected)
                        )
                        if windows[offset].cpu().tolist() != expected:
                            raise ValueError(
                                "verify Engram history contains future/mismatched tokens"
                            )
                result = orchestrator.step(tokens, placeholders=0 if active else 1)
                if tx is not None and tx.observer is not None:
                    tx.observer.compute_stage(
                        tx.observation,
                        result.row_request_ids,
                        result.row_positions,
                        [tokens[row] for row in result.row_request_ids],
                    )
                for index, row in enumerate(active):
                    offset = layout.starts[row] + stage
                    hidden[offset] = result.hidden[index]
                    if aux is not None:
                        aux[offset] = result.mtp_target_hidden_states[index]
                    tx.record(cache, row, layout.prefixes[row] + stage + 1)
        except Exception:
            if tx is not None:
                tx.abort()
            raise
        if tx is not None and tx.observer is not None:
            tx.observer.verified(tx.observation)
        self._pending_verify = tx
        self._pending_idle_verify = fake
        logger.info(
            "PPU target VERIFY dp=%d rows=%d lengths=%s prefixes=%s causal_stages=%d native_writes=deferred",
            self.dp_rank,
            layout.rows,
            layout.lengths,
            layout.prefixes,
            width,
        )
        return (hidden, aux) if return_aux else hidden

    def finish_speculative_forward(self, request_ids, prefixes, accepted_rows):
        tx, self._pending_verify = self._pending_verify, None
        if tx is None:
            # Idle owners have collective work, but no business cache versions.
            if self._pending_idle_verify:
                self._pending_idle_verify = False
                return
            raise RuntimeError("no pending verify round (already finalized or aborted)")
        try:
            tx.finish(request_ids, prefixes, accepted_rows)
            logger.info(
                "PPU target ACCEPT dp=%d request_ids=%s prefixes=%s input_rows=%s",
                self.dp_rank,
                request_ids.cpu().tolist(),
                prefixes.cpu().tolist(),
                accepted_rows.cpu().tolist(),
            )
        finally:
            if not tx.closed:
                tx.abort()

    def abort_speculative_forward(self):
        tx, self._pending_verify = self._pending_verify, None
        self._pending_idle_verify = False
        if tx is not None:
            tx.abort()

    def _decode(self, inputs, primary, *, return_aux=False):
        if self.kv_cache is None:
            raise RuntimeError("native decode requires initialized framework pools")
        lengths = [int(n) for n in primary.sequence_lengths.cpu().tolist()]
        ids = inputs.input_ids.reshape(-1)
        if len(lengths) != ids.numel():
            raise ValueError("native PPU decode requires one token per local request")
        windows = inputs.engram_token_windows
        if self.core.geometry.engram_enabled() and (
            windows is None
            or tuple(windows.shape)
            != (len(lengths), self.core.geometry.engram_max_ngram_size)
        ):
            raise ValueError("native decode is missing scheduler Engram token windows")
        bridge = NativeCacheBridge(
            self.core.config,
            self.kv_cache,
            inputs.attention_inputs,
            device=self.core.device,
        )
        cache = bridge.import_requests(lengths)
        orchestrator = self._orchestrator(cache)
        for row, length in enumerate(lengths):
            if self.core.geometry.engram_enabled():
                window = windows[row].cpu().tolist()
                if int(window[0]) != int(ids[row]):
                    raise ValueError(
                        "scheduler Engram window does not match the decode token"
                    )
                recent = [int(token) for token in window[1:][::-1] if int(token) >= 0]
            else:
                recent = []
            orchestrator.resume_request(row, length, recent)
        result = orchestrator.step(
            {row: int(token) for row, token in enumerate(ids.tolist())}
        )
        updated = [length + 1 for length in lengths]
        bridge.export_requests(cache, updated)
        self.steps += 1
        logger.info(
            "V4.1 native DECODE dp=%d requests=%d cached=%s next=%s prefill_executed=false",
            self.dp_rank,
            len(lengths),
            lengths,
            updated,
        )
        return (
            (result.hidden, result.mtp_target_hidden_states)
            if return_aux
            else result.hidden
        )

    def _prefill(self, inputs, primary, *, return_aux=False):
        cp_info = getattr(primary, "context_parallel_info", None)
        context = None
        group = None
        ids = inputs.input_ids.reshape(-1)
        windows = inputs.engram_token_windows
        if self.cp_size > 1:
            if cp_info is None:
                raise ValueError("native CP prefill lacks context_parallel_info")
            group = DistributedCPGroup(self.group)
            restore = cp_info.prefill_qkv_restore_indice_cpu.cpu().long()
            mask = cp_info.prefill_qkv_padding_mask_cpu.cpu().bool()
            unpad = restore[mask].to(ids.device)
            full_ids = group.all_gather(ids).index_select(0, unpad)
            if windows is not None and windows.numel():
                windows = group.all_gather(windows).index_select(0, unpad)
            lengths = [
                int(n) for n in cp_info.prefill_actual_input_lengths_cpu.tolist()
            ]
            prefixes = [int(n) for n in cp_info.prefill_prefix_lengths_cpu.tolist()]
        else:
            full_ids = ids
            lengths = [int(n) for n in primary.input_lengths.cpu().tolist()]
            prefixes = [int(n) for n in primary.prefix_lengths.cpu().tolist()]
        if len(prefixes) != len(lengths):
            raise ValueError("native prefill prefix/request counts differ")
        cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int64
        )
        lookback = None
        if windows is not None and windows.numel():
            lookback = windows[cu[:-1].to(windows.device).long(), 1:].cpu().long()
        elif self.core.geometry.engram_enabled():
            raise ValueError("native prefill is missing scheduler Engram token windows")
        if group is not None:
            plan = build_zigzag_plan(
                full_ids.cpu(),
                cu,
                cp_size=self.cp_size,
                cp_rank=self.cp_rank,
                prefix_lengths=prefixes,
                full_lookback=lookback,
            )
            if not torch.equal(plan.unpad_restore, unpad.cpu()):
                raise ValueError(
                    "native CP restore mapping differs from the PPU zigzag plan"
                )
            context = CPPrefillContext(plan, group, simulate_moe=False)
            ids = plan.local_input_ids.to(self.core.device)
            positions = plan.local_positions.to(self.core.device)
            requests = plan.local_request_ids.to(self.core.device)
            cu_local = plan.cu_local.to(self.core.device)
        else:
            positions = torch.cat(
                [
                    torch.arange(prefix, prefix + length)
                    for prefix, length in zip(prefixes, lengths)
                ]
            ).to(ids.device)
            requests = torch.repeat_interleave(
                torch.arange(len(lengths)), torch.tensor(lengths)
            ).to(ids.device)
            cu_local = cu.to(ids.device)
        replaying = PrefillPolicy.from_config(self.core.config).encoder_replay and any(
            prefixes
        )
        if replaying:
            history = getattr(inputs, "prefill_replay_token_ids", None)
            totals = [prefix + length for prefix, length in zip(prefixes, lengths)]
            if history is None or history.numel() != sum(totals):
                raise ValueError(
                    "encoder SWA recovery needs complete scheduler token history"
                )
            history = history.reshape(-1).cpu().long()
            if bool(((history < 0) | (history >= self.core.geometry.vocab_size)).any()):
                raise ValueError("encoder replay history contains invalid token ids")
            replay_cu = torch.tensor(
                [0] + list(torch.tensor(totals).cumsum(0).tolist())
            )
            for row, prefix in enumerate(prefixes):
                actual = history[int(replay_cu[row]) + prefix : int(replay_cu[row + 1])]
                expected = full_ids[int(cu[row]) : int(cu[row + 1])].cpu().long()
                if not torch.equal(actual, expected):
                    raise ValueError(
                        "encoder replay history disagrees with the fresh query"
                    )
            lookback = None  # Complete histories start at token zero.
            if group is not None:
                replay_plan = build_zigzag_plan(
                    history,
                    replay_cu,
                    cp_size=self.cp_size,
                    cp_rank=self.cp_rank,
                )
                context = CPPrefillContext(replay_plan, group, simulate_moe=False)
                ids = replay_plan.local_input_ids.to(self.core.device)
                positions = replay_plan.local_positions.to(self.core.device)
                requests = replay_plan.local_request_ids.to(self.core.device)
                cu_local = replay_plan.cu_local.to(self.core.device)
            else:
                ids = history.to(self.core.device)
                positions = torch.cat([torch.arange(total) for total in totals]).to(
                    ids.device
                )
                requests = torch.repeat_interleave(
                    torch.arange(len(totals)), torch.tensor(totals)
                ).to(ids.device)
                cu_local = replay_cu.to(ids.device)
        bridge = None
        if self.kv_cache is None:
            if any(prefixes):
                raise ValueError("native prefix restore requires framework KVCache")
            cache = PPUKVCache(self.core.config, device=self.core.device)
        else:
            bridge = NativeCacheBridge(
                self.core.config,
                self.kv_cache,
                inputs.attention_inputs,
                device=self.core.device,
            )
            cache = bridge.import_requests(prefixes, prefill=True)
        result = self.core.prefill(
            ids,
            positions=positions,
            cu_seqlens=cu_local,
            request_ids=requests,
            lookback_token_ids=lookback,
            cache=cache,
            parallel_context=context,
            allow_ced=(
                getattr(inputs, "need_all_logits", None) is False
                and getattr(inputs, "need_all_hidden_states", None) is False
            ),
            replay_prefix_lengths=prefixes if replaying else None,
        )
        if getattr(self.core, "capture_layer_ids", ()):
            logger.info(
                "PPU target capture_features=true ced_compaction=false reason=complete_draft_features "
                "layers=%s local_rows=%d output_rows=%d cp=%d/%d",
                self.core.capture_layer_ids,
                ids.numel(),
                result.hidden.shape[0],
                self.cp_rank,
                self.cp_size,
            )
        if bridge is not None:
            observation = None
            if self.observer is not None and not bool(inputs.warmup):
                observation = self.observer.begin(
                    "prefill",
                    bridge,
                    self.observer.input_request_ids(inputs, len(prefixes)),
                    prefixes,
                    lengths,
                )
            bridge.export_requests(
                cache,
                [a + b for a, b in zip(prefixes, lengths)],
                publish=not bool(inputs.warmup),
            )
            if self.observer is not None:
                self.observer.finish(observation, "accept", lengths, cache=cache)
        hidden = result.hidden
        aux = getattr(result, "mtp_target_hidden_states", None)
        if replaying:
            hidden = torch.cat(
                [
                    hidden[int(replay_cu[row]) + prefix : int(replay_cu[row + 1])]
                    for row, prefix in enumerate(prefixes)
                ]
            )
            if aux is not None:
                aux = torch.cat(
                    [
                        aux[int(replay_cu[row]) + prefix : int(replay_cu[row + 1])]
                        for row, prefix in enumerate(prefixes)
                    ]
                )
            logger.info(
                "PPU encoder SWA rebuilt cp=%d/%d prefix=%s query_rows=%s "
                "append_rows=%s global_prefix_readonly=true",
                self.cp_rank,
                self.cp_size,
                prefixes,
                totals,
                lengths,
            )
        if context is not None:
            # C++ owns the final CP output gather. Return this rank's rows,
            # including harmless padding, rather than a duplicate full batch.
            indices = torch.empty(plan.chunk_length, dtype=torch.int64)
            for row, split in enumerate(plan.requests):
                sl = slice(split.local_offset, split.local_offset + split.chunk)
                indices[sl] = cu[row] + plan.local_positions[sl] - split.prefix
            hidden = hidden.index_select(0, indices.to(hidden.device))
            if aux is not None:
                aux = aux.index_select(0, indices.to(aux.device))
        logger.info(
            "V4.1 native PREFILL cp=%d/%d requests=%d tokens=%d CacheStore=%s",
            self.cp_rank,
            self.cp_size,
            len(lengths),
            sum(lengths),
            bridge is not None,
        )
        return (hidden, aux) if return_aux else hidden
