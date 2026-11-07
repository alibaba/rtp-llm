from __future__ import annotations

import asyncio
import atexit
import json
import os
import time
from contextlib import aclosing

_registered = False


def _server_args(original, args):
    from sglang.srt.runtime_context import get_platform

    if getattr(args, "_resolution_finished", False):
        return original(args)
    if not args.kv_events_config:
        config = {"publisher": "navi_sched", "format": "dynamo"}
    else:
        config = json.loads(args.kv_events_config)
        if config.get("format", "default") != "dynamo":
            raise ValueError("SGLang NaviSched requires Dynamo KV event format")
        if config.get("publisher", "null") == "null":
            config["publisher"] = "navi_sched"
    args.kv_events_config = json.dumps(config)
    value = original(args)
    for name in ("dp_size", "attn_dp_size", "pp_size", "tokenizer_worker_num"):
        if getattr(args, name, 1) != 1:
            raise ValueError("SGLang NaviSched currently requires " + name + "=1")
    if args.disaggregation_mode != "null":
        raise ValueError("SGLang NaviSched currently requires unified prefill/decode")
    if args.speculative_algorithm not in (None, "DSPARK"):
        raise ValueError(
            "SGLang NaviSched currently supports DSPARK speculative decoding"
        )
    if args.device not in (None, "cuda") or not get_platform().is_cuda:
        raise ValueError("SGLang NaviSched currently requires a CUDA device")
    return value


def _event_publisher(**options):
    from sglang.srt.disaggregation.kv_events import NullEventPublisher

    return NullEventPublisher()


def _manager_init(original, manager, *args, **kwargs):
    original(manager, *args, **kwargs)
    from sglang.utils import TypeBasedDispatcher

    from .SGLangObserver import SGLangState
    from .SGLangRuntime import SGLangRuntime

    runtime = SGLangRuntime(manager, _original_generate[0])
    manager.navi_sched = runtime
    manager._result_dispatcher += TypeBasedDispatcher([(SGLangState, runtime.receive)])
    atexit.register(runtime.close)


_original_generate = None


async def _generate(original, manager, obj, request=None):
    from sglang.srt.managers.io_struct import GenerateReqInput

    if not isinstance(obj, GenerateReqInput):
        async for output in original(manager, obj, request):
            yield output
        return
    async with aclosing(manager.navi_sched.generate(obj, request)) as responses:
        async for output in responses:
            yield output


def _abort(original, manager, rid="", abort_all=False):
    runtime = getattr(manager, "navi_sched", None)
    if runtime is not None and runtime.transport is not None:
        if runtime.transport.abort(rid, abort_all) and not abort_all:
            return
    return original(manager, rid, abort_all)


def _create_abort_task(original, manager, obj):
    from fastapi import BackgroundTasks
    from sglang.srt.managers.io_struct import GenerateReqInput

    if not isinstance(obj, GenerateReqInput):
        return original(manager, obj)

    async def abort_request():
        await asyncio.sleep(2)
        items = [obj] if obj.is_single else [obj[i] for i in range(len(obj.rid))]
        for item in items:
            state = manager.rid_to_state.get(item.rid)
            # A disconnected HTTP request can outlive reuse of its request ID.
            if state is not None and state.obj is item:
                manager.abort_request(item.rid)

    tasks = BackgroundTasks()
    tasks.add_task(abort_request)
    return tasks


def _engine_shutdown(original, engine, *args, **kwargs):
    manager = getattr(engine, "tokenizer_manager", None)
    runtime = getattr(manager, "navi_sched", None)
    if runtime is not None:
        runtime.close()
    return original(engine, *args, **kwargs)


async def _sigterm_watchdog(original, manager, *args, **kwargs):
    try:
        return await original(manager, *args, **kwargs)
    finally:
        runtime = getattr(manager, "navi_sched", None)
        if runtime is not None:
            runtime.close()


def _scheduler_init(original, scheduler, *args, **kwargs):
    original(scheduler, *args, **kwargs)
    from .SGLangObserver import SGLangObserver

    scheduler.navi_sched = SGLangObserver(scheduler)


def _run_batch(original, scheduler, batch, *args, **kwargs):
    import torch

    observer = scheduler.navi_sched
    if not observer.enabled:
        return original(scheduler, batch, *args, **kwargs)
    for req in batch.reqs:
        if not req.output_ids:
            observer.prefill_started.setdefault(req.rid, time.monotonic_ns())
    observer.report(batch, force=True)
    observer.send({"kind": "execution_started", "request_count": len(batch.reqs)})
    # Speculative workers may change the batch mode during target verification.
    prefill_tokens = (
        int(batch.extend_num_tokens or 0)
        if batch.forward_mode.is_extend_without_speculative()
        else 0
    )
    decode_requests = (
        len(batch.reqs)
        if batch.forward_mode.is_decode() or batch.forward_mode.is_target_verify()
        else 0
    )
    stream = (
        scheduler.forward_stream
        if scheduler.enable_overlap
        else torch.cuda.current_stream()
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    result = original(scheduler, batch, *args, **kwargs)
    end.record(stream)
    result.navi_sched_timing = (
        start,
        end,
        prefill_tokens,
        decode_requests,
    )
    return result


def _process_result(original, scheduler, batch, result):
    value = original(scheduler, batch, result)
    observer = scheduler.navi_sched
    if observer.enabled:
        timing = getattr(result, "navi_sched_timing", None)
        if timing is not None:
            observer.pending_timings.append(timing)
        observer.report(force=True)
    return value


def _idle(original, scheduler, *args, **kwargs):
    value = original(scheduler, *args, **kwargs)
    scheduler.navi_sched.report()
    return value


def _release(original, scheduler, *args, **kwargs):
    try:
        return original(scheduler, *args, **kwargs)
    finally:
        scheduler.navi_sched.close()


def _register():
    global _registered, _original_generate
    if _registered or os.environ.get("SGLANG_NAVI_SCHED_ENABLED", "0") != "1":
        return
    from sglang.srt.disaggregation.kv_events import EventPublisherFactory
    from sglang.srt.managers.io_struct import hook_custom_types
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.version import __version__

    from .SGLangConfig import SGLangConfig
    from .SGLangObserver import SGLangState

    if __version__.split("+", 1)[0] != "0.5.23":
        raise RuntimeError("NaviSched SGLang adapter requires SGLang 0.5.23")
    SGLangConfig.from_env()
    EventPublisherFactory.register_publisher("navi_sched", _event_publisher)
    hook_custom_types(SGLangState)
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    # HookRegistry rewrites module-level references to patched functions. Keep
    # the native entry in a container so destination execution bypasses routing.
    _original_generate = (TokenizerManager.generate_request,)
    targets = {
        "sglang.srt.server_args.ServerArgs.resolve_once": _server_args,
        "sglang.srt.managers.tokenizer_manager.TokenizerManager.__init__": _manager_init,
        "sglang.srt.managers.tokenizer_manager.TokenizerManager.generate_request": _generate,
        "sglang.srt.managers.tokenizer_manager.TokenizerManager.abort_request": _abort,
        "sglang.srt.managers.tokenizer_manager.TokenizerManager.create_abort_task": _create_abort_task,
        "sglang.srt.managers.tokenizer_manager.TokenizerManager.sigterm_watchdog": _sigterm_watchdog,
        "sglang.srt.entrypoints.engine.Engine.shutdown": _engine_shutdown,
        "sglang.srt.managers.scheduler.Scheduler.__init__": _scheduler_init,
        "sglang.srt.managers.scheduler.Scheduler.run_batch": _run_batch,
        "sglang.srt.managers.scheduler.Scheduler.process_batch_result": _process_result,
        "sglang.srt.managers.scheduler.Scheduler.on_idle": _idle,
        "sglang.srt.managers.scheduler.Scheduler.release_host_resources": _release,
    }
    for target, callback in targets.items():
        HookRegistry.register(target, callback, HookType.AROUND)
    # The upstream loader logs and continues after hook errors. An enabled
    # adapter must stop startup if any required hook could not be installed.
    for target in targets:
        if target not in HookRegistry._patched:
            HookRegistry._apply_target(target, HookRegistry._hooks[target])
            HookRegistry._patched.add(target)
    _registered = True


def register():
    try:
        _register()
    except Exception as error:
        raise SystemExit(
            "SGLang NaviSched initialization failed: " + str(error)
        ) from error
