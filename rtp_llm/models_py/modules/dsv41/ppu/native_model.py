"""Process-group and accelerator teardown for the native PPU model runtime."""

from __future__ import annotations

import logging
import os
import time
import weakref

import torch

logger = logging.getLogger(__name__)
_SHUTDOWN_GROUP = None
_NATIVE_RUNNERS = weakref.WeakSet()
_REFERENCE_PREFILLS = []


def record_reference_prefill(state):
    if len(_REFERENCE_PREFILLS) < 4096:
        engine = getattr(getattr(state.runtime.model, "_shared_ops", None), "moe_engine", None)
        _REFERENCE_PREFILLS.append({
            "stamp_ns": time.monotonic_ns(),
            "lengths": list(state.lengths), "prefixes": list(state.query_prefixes),
            "plan": state.plan_reason, "compacted": state.compacted,
            "dense_fp4_layers": sorted(state.shared.get("reference_dense_fp4", ())),
            "sparse_fp4_layers": sorted(state.shared.get("reference_sparse_fp4", ())),
            "sg_align": bool(getattr(state, "local_prefill_sg_align", False)),
            "sg_topk_calls": list(state.shared.get("sg_topk_calls", ())),
            "sg_paged_layers": list(state.shared.get("sg_paged_layers", ())),
            "sg_rope_calls": list(state.shared.get("sg_rope_calls", ())),
            "dense_projection_calls": list(state.shared.get("dense_projection_calls", ())),
            "cp_token_topk_calls": list(state.shared.get("cp_token_topk_calls", ())),
            "cp_l20_projection": dict(state.shared.get("cp_l20_projection", {})),
            "cp_shared_global": getattr(getattr(state, "cache", None), "_cp_shared_global_stats", None) or {},
            "cp_shared_indexer": getattr(getattr(state, "cache", None), "_cp_shared_indexer_stats", None) or {},
            "cp_paged_indexer_calls": list(state.shared.get("cp_paged_indexer_calls", ())),
            "cp_indexer_layout_calls": list(state.shared.get("cp_indexer_layout_calls", ())),
            "cp_indexer_layout": getattr(getattr(state, "cache", None), "_cp_indexer_layout_stats", None) or {},
            "cp_local_swa_pack": state.shared.get("cp_local_swa_pack", {}),
            "cp_segmented_swa": getattr(getattr(state, "cache", None), "_cp_segmented_swa_stats", None) or {},
            "current_row_gathers": list(state.shared.get("current_row_gathers", ())),
            "cp_metadata": getattr(state, "_cp_metadata_evidence", {}),
            "cp_meta_stub_admissions": state.shared.get("cp_meta_stub_admissions", 0),
            "cp_meta_stub_consumers": state.shared.get("cp_meta_stub_consumers", 0),
            "cp_engram_gate_admissions": list(state.shared.get("cp_engram_gate_admissions", ())),
            "cp_fake_routes": getattr(engine, "_cp_fake_evidence", {}) if engine is not None else {},
            "sg_source_read": getattr(getattr(state, "cache", None), "_local_sg_source_stats", None) or {},
            "mega_mhc_transitions": state.normalization["mega_transitions"],
            "mhc_normalization": dict(state.normalization),
            # Host integer only: do not add a device-statistics reduction to
            # the measured forward while recording persistent buffer size.
            "mega_workspace_capacity": getattr(engine, "_mega_capacity", 0),
            "dp_prefill_capacity": dict(getattr(engine, "_dp_capacity_evidence", None) or {}),
            # Runtime replaces this host-only dict at every dispatch. Keep
            # this forward's owner so its finally block can add lease/peak
            # evidence before the records are flushed after teardown.
            "engram_rounds": getattr(state.runtime, "_engram_evidence", None) or {},
            # As above, keep the per-dispatch owner through final cleanup.
            "cross_layer_mhc": getattr(state.runtime, "_cross_mhc_evidence", None) or {},
            # Host-only per-forward counters; never synchronize a device
            # merely to observe the local old-window reader.
            "swa_native_read": getattr(getattr(state, "cache", None), "_local_swa_native_read_stats", None) or {},
            "source_native_read": getattr(getattr(state, "cache", None), "_local_source_native_read_stats", None) or {},
            "source_native_write": getattr(getattr(state, "cache", None), "_local_source_native_write_stats", None) or {},
            "csa_native_write": getattr(getattr(state, "cache", None), "_local_csa_native_write_stats", None) or {},
            "producer_slots": getattr(getattr(state, "cache", None), "_local_producer_slot_stats", None) or {},
            "attention_provider": getattr(getattr(state, "local_attention_provider", None), "stats", None) or {},
        })


def _flush_reference_prefills(directory, rank):
    import json
    from pathlib import Path

    if not _REFERENCE_PREFILLS:
        return
    with (Path(directory) / f"prefill-rank{rank}.jsonl").open("x") as output:
        for record in _REFERENCE_PREFILLS:
            output.write(json.dumps(record) + "\n")
    _REFERENCE_PREFILLS.clear()

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
        reference_dir = os.environ.get("RTP_V41_REFERENCE_PERF_DIR")
        if reference_dir:
            if os.environ.get("ROLE_TYPE") == "DECODE":
                from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import flush_reference_replays

                flush_reference_replays(reference_dir, dist.get_rank())
            else:
                _flush_reference_prefills(reference_dir, dist.get_rank())
        logger.info("V4.1 native role engine device work drained")


def _write_perf_evidence():
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
                "mega_mhc": [
                    model._shared_ops.mhc_execution_stats()
                    for runner in _NATIVE_RUNNERS
                    if (model := runner.model) is not None
                ],
                "attention_graph_capture_calls": [
                    dict(graph.attention_capture_calls)
                    for runner in _NATIVE_RUNNERS
                    if (model := runner.model) is not None
                    and (graph := model.shared_graph_runtime) is not None
                ],
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
