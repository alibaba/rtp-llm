"""Process-group and accelerator teardown for the native PPU model runtime."""

from __future__ import annotations

import logging
import weakref

import torch

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
