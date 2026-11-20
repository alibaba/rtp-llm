"""Observe worker KV acquisition and completed HBM transfers on PP TPU jobs."""

import contextvars
import functools
import inspect
import json
import logging
import os
import sys
import time

from . import ExecutionTimingProbe as _timing

_active_load = contextvars.ContextVar("navi_peer_kv_load", default=None)
_group_types = {}
_clock = time.perf_counter_ns
_logger = logging.getLogger("vllm.navi_sched.worker")
_marker = "_navi_worker_kv_load_timing_v1"


def _argument(args, kwargs, index, name):
    return kwargs[name] if name in kwargs else args[index]


def _new_load(handler, args, kwargs):
    groups = _argument(args, kwargs, 4, "group_ids")
    counts = {}
    for group in groups or ():
        name = _group_types.get(group, "unknown")
        counts[name] = counts.get(name, 0) + 1
    return dict(
        owner=id(handler),
        request_id=str(_argument(args, kwargs, 3, "request_id")),
        job_id=int(_argument(args, kwargs, 0, "job_id")),
        pid=os.getpid(),
        rank_id=int(handler._rank_id),
        key_count=len(_argument(args, kwargs, 1, "keys")),
        group_counts=counts,
        rank_page_bytes=int(handler._transfer_engine._object_nbytes),
        prepare_started_ns=0,
        prepare_finished_ns=0,
        dma_batches=[],
        source_counts={},
        published=False,
        load_failed=False,
    )


def _emit(fields):
    _timing._diagnostic_info(
        _logger,
        "[navi-worker-kv-load-timing] %s",
        json.dumps(fields, separators=(",", ":"), sort_keys=True),
    )


def _run_wrapper(original):
    @functools.wraps(original)
    async def run(handler, *args, **kwargs):
        if not handler._swap_in:
            return await original(handler, *args, **kwargs)
        try:
            state = _new_load(handler, args, kwargs)
        except Exception:
            return await original(handler, *args, **kwargs)
        state["load_started_ns"] = _clock()
        token = _active_load.set(state)
        returned = False
        try:
            result = await original(handler, *args, **kwargs)
            returned = True
            return result
        finally:
            ended = _clock()
            _active_load.reset(token)
            try:
                counts = state["group_counts"]
                state.update(
                    event="worker_kv_load_timing",
                    version=1,
                    event_time_us=time.time_ns() // 1000,
                    load_finished_ns=ended,
                    success=returned and state["published"] and not state["load_failed"],
                    requested_full_hbm_bytes=counts.get("FullAttentionSpec", 0)
                    * state["rank_page_bytes"],
                    requested_linear_hbm_bytes=counts.get("MambaSpec", 0)
                    * state["rank_page_bytes"],
                    byte_groups_complete="unknown" not in counts
                    and sum(counts.values()) == state["key_count"],
                    completed_hbm_bytes=sum(
                        b["bytes"] for b in state["dma_batches"] if b["success"]
                    ),
                )
                state.pop("owner")
                _emit(state)
            except Exception:
                pass

    return run


def _prepare_wrapper(original):
    @functools.wraps(original)
    def prepare(handler, *args, **kwargs):
        state = _active_load.get()
        if state is None or state["owner"] != id(handler):
            return original(handler, *args, **kwargs)
        state["prepare_started_ns"] = _clock()
        try:
            objects = _argument(args, kwargs, 0, "batch_objs")
            for obj in objects if isinstance(objects, list) else [objects]:
                source = getattr(getattr(obj, "_blob", None), "source", None)
                name = str(source) if type(source) in (str, int) else "unknown"
                state["source_counts"][name] = state["source_counts"].get(name, 0) + 1
        except Exception:
            state["source_observation_failed"] = True
        try:
            return original(handler, *args, **kwargs)
        finally:
            state["prepare_finished_ns"] = _clock()

    return prepare


def _dma_wrapper(original):
    @functools.wraps(original)
    async def transfer(engine, batch):
        state = _active_load.get()
        if state is None or not batch.is_h2d:
            return await original(engine, batch)
        started = _clock()
        success = False
        try:
            result = await original(engine, batch)
            success = True
            return result
        finally:
            finished = _clock()
            try:
                state["dma_batches"].append(
                    dict(
                        started_ns=started,
                        finished_ns=finished,
                        bytes=int(batch.region.nbytes) * len(batch.gpu_block_ids),
                        success=success,
                    )
                )
            except Exception:
                state["dma_observation_failed"] = True

    return transfer


def _publish_wrapper(original):
    @functools.wraps(original)
    def publish(handler, result):
        state = _active_load.get()
        if state is not None and state["owner"] == id(handler):
            try:
                state["published"] = True
                state["load_failed"] = bool(result.error or result.failed_block_ids)
                state["published_ns"] = _clock()
            except Exception:
                state["publication_observation_failed"] = True
        return original(handler, result)

    return publish


def _install(module):
    handler = getattr(module, "TorchTPUV6dSwapHandler", None)
    engine = getattr(module, "_TpuV6dTransferEngine", None)
    if handler is None or engine is None:
        return False
    if getattr(handler, _marker, False):
        return True
    methods = (
        (handler, "_run_swap", _run_wrapper, True),
        (handler, "_prepare_swap", _prepare_wrapper, False),
        (handler, "_publish_result", _publish_wrapper, False),
        (engine, "_transfer_batch", _dma_wrapper, True),
    )
    originals = []
    for cls, name, wrapper, coroutine in methods:
        method = getattr(cls, name, None)
        if not callable(method) or inspect.iscoroutinefunction(method) != coroutine:
            return False
        originals.append((cls, name, method, wrapper(method)))
    for cls, name, original, replacement in originals:
        setattr(cls, name, replacement)
    setattr(handler, _marker, True)
    return True


def install_peer_kv_load_timing(worker, vllm_config):
    parallel = getattr(vllm_config, "parallel_config", None)
    if int(getattr(parallel, "pipeline_parallel_size", 1) or 1) <= 1:
        return False
    module = sys.modules.get("kvs_connector.v6d_object_connector")
    if module is None:
        return False
    runner = getattr(worker, "model_runner", None)
    cache = getattr(runner, "kv_cache_config", None)
    groups = getattr(cache, "kv_cache_groups", None)
    if not groups or not _timing.prepare_diagnostic_logging():
        return False
    try:
        _group_types.update(
            {i: type(group.kv_cache_spec).__name__ for i, group in enumerate(groups)}
        )
        installed = _install(module)
        if installed:
            _timing._diagnostic_info(
                _logger,
                "[navi-worker-kv-load-timing-installed] %s",
                json.dumps(
                    dict(version=1, pid=os.getpid(), group_types=_group_types),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        return installed
    except Exception:
        return False
