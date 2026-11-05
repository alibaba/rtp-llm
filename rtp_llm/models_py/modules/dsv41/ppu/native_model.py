"""Native RTP model execution over framework cache resources and scheduler IO."""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)
_SHUTDOWN_GROUP = None


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
        if dist.is_available() and dist.is_initialized():
            value = torch.tensor(
                [count, int(stopping)], dtype=torch.int64, device=self.core.device
            )
            dist.all_reduce(value, group=self.group)
            count, stopping = value.cpu().tolist()
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
        )

    def forward(self, inputs):
        primary = primary_attention_inputs(inputs.attention_inputs, self.kv_cache)
        fake = bool(inputs.is_fake_stream)
        rows = int(inputs.input_ids.numel())
        # Native DP supplies fake streams on idle ranks. Exchange work BEFORE
        # any layer collective; an entirely idle world skips ENP/EP uniformly.
        if not self._work_available(0 if fake else rows):
            return self.core.weights.embedding.new_zeros(
                (rows, self.core.geometry.hidden_size)
            )
        if fake:
            cache = PPUKVCache(self.core.config, device=self.core.device)
            self._orchestrator(cache).step({}, placeholders=max(1, rows))
            return self.core.weights.embedding.new_zeros(
                (rows, self.core.geometry.hidden_size)
            )
        if self.decode_role and not bool(inputs.warmup):
            return self._decode(inputs, primary)
        if bool(primary.is_prefill):
            return self._prefill(inputs, primary)
        return self._decode(inputs, primary)

    def _decode(self, inputs, primary):
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
        return result.hidden

    def _prefill(self, inputs, primary):
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
            cache = bridge.import_requests(prefixes)
        result = self.core.prefill(
            ids,
            positions=positions,
            cu_seqlens=cu_local,
            request_ids=requests,
            lookback_token_ids=lookback,
            cache=cache,
            parallel_context=context,
        )
        if bridge is not None:
            bridge.export_requests(
                cache,
                [a + b for a, b in zip(prefixes, lengths)],
                publish=not bool(inputs.warmup),
            )
        hidden = result.hidden
        if context is not None:
            # C++ owns the final CP output gather. Return this rank's rows,
            # including harmless padding, rather than a duplicate full batch.
            indices = torch.empty(plan.chunk_length, dtype=torch.int64)
            for row, split in enumerate(plan.requests):
                sl = slice(split.local_offset, split.local_offset + split.chunk)
                indices[sl] = cu[row] + plan.local_positions[sl] - split.prefix
            hidden = hidden.index_select(0, indices.to(hidden.device))
        logger.info(
            "V4.1 native PREFILL cp=%d/%d requests=%d tokens=%d CacheStore=%s",
            self.cp_rank,
            self.cp_size,
            len(lengths),
            sum(lengths),
            bridge is not None,
        )
        return hidden
