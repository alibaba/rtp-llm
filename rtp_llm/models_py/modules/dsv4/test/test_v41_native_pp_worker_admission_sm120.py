"""Atomic native WORLD allocation and maintained two-stage worker readiness."""

import gc
import json
import os
import socket
import sys
import tempfile
import traceback
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_pp_transport_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    resources,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    make_worker,
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_admission_sm120 import (
    caller_snapshot,
    require_caller,
    tensors_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_local_cache_sm120 import (
    release_allocations,
    root_allocations,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    ROUNDS,
    clean_worker,
    corrupt,
    global_tokens,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_transport_sm120 import (
    REQUESTS,
    cache_config,
    require_snapshot,
    snapshot,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.ops import NcclCommConfig


def reject(native, source, call, raw, free):
    before = caller_snapshot(source)
    try:
        call()
    except ops.NativeInputRejected:
        pass
    else:
        raise AssertionError("worker rejection did not reach the entire WORLD")
    assert source.batch is None and source.dispatch is None
    assert native.free_blocks == free and native.active_global == 0
    assert not native.global_failed
    require_caller(source, before)
    require_snapshot(native.cache, raw)


def marker(rank, variant, actor):
    print(
        "DSV41_NATIVE_PP_WORKER_REJECTION "
        + json.dumps(
            {
                "rank": rank,
                "variant": variant,
                "actor": actor,
                "world_rejected": True,
                "caller_unchanged": True,
                "provisional_descriptors_released": True,
                "read_only": True,
            }
        ),
        flush=True,
    )


def run_rank(rank, port, config):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "MOEDBG": "0",
        }
    )
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    new_group = dist.new_group

    def bounded_group(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=180)
        return new_group(*args, **kwargs)

    native = other = model = None
    calls = {"global": 0, "ready": 0, "stage": 0, "worker": 0}
    with patch.dict(sys.modules, {"librtp_compute_ops": ops}), patch.object(
        dist, "new_group", side_effect=bounded_group
    ):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            with torch.inference_mode(), ExitStack() as stack:
                native = ops.NativePPTransport(cache_config(), parallelism_for(rank))
                other = ops.NativePPTransport(cache_config(), parallelism_for(rank))
                cache = native.cache
                free_before = native.free_blocks
                roots = root_allocations(native, rank)
                model = make_worker(rank, config)
                assert model.initialize(resources(cache))
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.zero_()
                gather = dist.all_gather_into_tensor

                def admission_only(output, value, **kwargs):
                    if kwargs["group"] is dist.group.WORLD:
                        if value.shape == (2576,):
                            assert output.shape == (20608,)
                            calls["global"] += 1
                        else:
                            assert value.shape == (3,) and output.shape == (24,)
                            calls["ready"] += 1
                    else:
                        assert kwargs["group"] is model._dp_group
                        assert value.shape == (1, 33) and output.shape == (4, 33)
                        calls["stage"] += 1
                    return gather(output, value, **kwargs)

                stack.enter_context(
                    patch.object(
                        dist, "all_gather_into_tensor", side_effect=admission_only
                    )
                )
                for layer in model.v4.layers:
                    stack.enter_context(
                        patch.object(
                            layer,
                            "forward",
                            side_effect=AssertionError(
                                "readiness executed a maintained block"
                            ),
                        )
                    )

                def worker(request):
                    calls["worker"] += 1
                    return model.prepare_native_pp_prefill(request)

                identities = [
                    V41CacheIdentity(request, i + 1, 1)
                    for i, request in enumerate(REQUESTS)
                ]
                initial = [V41L20Span(identity, 0, 0, 1) for identity in identities]
                ids, windows = global_tokens([1, 1], [[], []], 0)
                raw = snapshot(cache)
                free = native.free_blocks
                # All three controls must reject before any STAGE callback.
                for variant, actor in (
                    ("missing-hook", 0),
                    ("mixed-mode", 0),
                    ("global-metadata", 4),
                ):
                    source = native.global_inputs(
                        tensors_for(initial, ids, windows), roots
                    )
                    if variant == "global-metadata" and rank == actor:
                        source.control("warmup")
                    hook = (
                        None if variant == "missing-hook" and rank == actor else worker
                    )
                    if variant == "mixed-mode" and rank == actor:
                        operation = lambda: native.prepare_global(source)
                    else:
                        operation = lambda: native.prepare_global_worker(source, hook)
                    reject(native, source, operation, raw, free)
                    assert calls["stage"] == calls["worker"] == 0
                    marker(rank, variant, actor)
                    del operation, source

                for variant, actor in (
                    ("missing-lease", 0),
                    ("table", 4),
                    ("common", 0),
                    ("restore", 4),
                    ("mask", 0),
                    ("cp-missing", 4),
                    ("windows", 0),
                    ("tokens", 2),
                    ("position", 4),
                    ("spans", 0),
                    ("foreign-allocator", 4),
                ):
                    source = native.global_inputs(
                        tensors_for(initial, ids, windows), roots
                    )

                    def malformed(request):
                        calls["worker"] += 1
                        if rank == actor:
                            if variant == "foreign-allocator":
                                with patch.object(model, "kv_cache", other.cache):
                                    return model.prepare_native_pp_prefill(request)
                            if variant == "position":
                                for value in request.attention_inputs.values():
                                    value.combo_position_ids = (
                                        value.combo_position_ids + 1
                                    )
                            else:
                                request = corrupt(request, variant, None)
                        return model.prepare_native_pp_prefill(request)

                    # Windows are native CPU mirrors. No CP object is needed
                    # for this control because every row, including padding,
                    # is corrupted; the maintained validator checks its mask.
                    if variant == "windows":

                        def malformed(request):
                            calls["worker"] += 1
                            if rank == actor:
                                request.engram_token_windows.zero_()
                            return model.prepare_native_pp_prefill(request)

                    reject(
                        native,
                        source,
                        lambda: native.prepare_global_worker(source, malformed),
                        raw,
                        free,
                    )
                    clean_worker(model)
                    del source
                    retry = native.global_inputs(
                        tensors_for(initial, ids, windows), roots
                    )
                    batch = native.prepare_global_worker(retry, worker)
                    assert retry.dispatch.schedule == tuple(
                        s.wire_values() for s in initial
                    )
                    assert (
                        cache.native_allocation_tables(retry.dispatch.native_inputs)
                        == batch.block_tables
                    )
                    native.abort_global(batch)
                    del retry, batch
                    gc.collect()
                    assert native.free_blocks == free and native.active_global == 0
                    clean_worker(model)
                    require_snapshot(cache, raw)
                    marker(rank, variant, actor)

                prefixes, histories = [0, 0], [[], []]
                for step, lengths in enumerate(ROUNDS):
                    spans = [
                        V41L20Span(identity, step, start, count)
                        for identity, start, count in zip(identities, prefixes, lengths)
                    ]
                    ids, windows = global_tokens(lengths, histories, step)
                    source_tensors = tensors_for(spans, ids, windows)
                    source = native.global_inputs(source_tensors, roots)
                    batch = native.prepare_global_worker(source, worker)
                    dispatch = source.dispatch
                    assert dispatch.schedule == tuple(s.wire_values() for s in spans)
                    assert (
                        cache.native_allocation_tables(dispatch.native_inputs)
                        == batch.block_tables
                    )
                    for name in (
                        "spans",
                        "ids",
                        "windows",
                        "lengths",
                        "prefixes",
                        "positions",
                    ):
                        torch.testing.assert_close(
                            source.tensors[name], source_tensors[name], rtol=0, atol=0
                        )
                        assert (
                            source.tensors[name].data_ptr()
                            != source_tensors[name].data_ptr()
                        )
                        source_tensors[name].zero_()
                    if rank < 4:
                        native.send_local(
                            {
                                "ready_ids": dispatch.input_ids,
                                "ready_rows": dispatch.rows.cuda(),
                            },
                            batch,
                        )
                    else:
                        received = native.receive_local(batch)
                    native.wait()
                    if rank >= 4:
                        torch.testing.assert_close(
                            received["ready_ids"], dispatch.input_ids, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            received["ready_rows"].cpu(), dispatch.rows, rtol=0, atol=0
                        )
                        del received
                    native.commit_global(batch)
                    native.finish()
                    clean_worker(model)
                    require_snapshot(cache, raw)
                    values = ids.tolist()
                    histories[0].extend(values[: lengths[0]])
                    histories[1].extend(values[lengths[0] :])
                    prefixes = [p + n for p, n in zip(prefixes, lengths)]
                    print(
                        "DSV41_NATIVE_PP_WORKER_ADMISSION "
                        + json.dumps(
                            {
                                "rank": rank,
                                "step": step,
                                "ends": prefixes,
                                "both_stages_ready": True,
                                "native_tables_exact": True,
                                "caller_snapshot_independent": True,
                                "read_only": True,
                                "native_named_tensor_handoff": True,
                            }
                        ),
                        flush=True,
                    )
                    del source, batch
                    if step != len(ROUNDS) - 1:
                        del dispatch
                occupied = native.free_blocks
                for identity in identities:
                    native.retire_global(list(identity.__dict__.values()))
                release_allocations(roots)
                assert native.active_global == 0 and native.free_blocks == occupied
                del dispatch
                gc.collect()
                assert native.free_blocks == free_before

                # Inject inside the maintained preflight's error agreement:
                # stage zero fails together; stage one's successful provisional
                # descriptors must drop when the second WORLD packet is fatal.
                identity = V41CacheIdentity(REQUESTS[0], 3, 1)
                fresh = [V41L20Span(identity, 0, 0, 1)]
                fresh_roots = [native.allocate(REQUESTS[0], 256)] if rank == 0 else []
                source = native.global_inputs(
                    tensors_for(fresh, ids[:1], windows[:1]), fresh_roots
                )
                before = caller_snapshot(source)
                available = native.free_blocks

                def failed_validator(*args):
                    raise RuntimeError("injected maintained worker failure")

                with ExitStack() as fault:
                    if rank == 0:
                        fault.enter_context(
                            patch(
                                "rtp_llm.models_py.modules.dsv4.native_pp_prefill.prepare_native_pp_prefill_dispatch",
                                new=failed_validator,
                            )
                        )
                    try:
                        native.prepare_global_worker(source, worker)
                    except RuntimeError:
                        assert native.global_failed
                    else:
                        raise AssertionError("fatal worker peer was admitted")
                assert model._worker_state == ("failed" if rank < 4 else "ready")
                assert source.batch is None and source.dispatch is None
                require_caller(source, before)
                require_snapshot(cache, raw)
                assert native.free_blocks == available
                del source
                release_allocations(fresh_roots)
                assert native.free_blocks == free_before
                assert calls == {"global": 29, "ready": 26, "stage": 26, "worker": 26}
                print(
                    "DSV41_NATIVE_PP_WORKER_LIFETIME "
                    + json.dumps(
                        {
                            "rank": rank,
                            "calls": calls,
                            "free_before": free_before,
                            "free_after": free_before,
                            "descriptor_holds_native_leases": True,
                            "unexpected_worker_failure_poisoned_world": True,
                        }
                    ),
                    flush=True,
                )
                model.close()
                model = None
                native.close()
                other.close()
                native = other = None
            dist.barrier()
        except BaseException:
            print(
                "DSV41_WORKER_ADMISSION_FAILURE "
                + json.dumps({"rank": rank, "calls": calls}),
                flush=True,
            )
            traceback.print_exc()
            raise
        finally:
            if model is not None:
                model.close()
            for fixture in (other, native):
                if fixture is not None:
                    fixture.wait()
                    fixture.close()
            Dsv4SharedRuntimeBufferStore._reset_for_test()
            collective.destroy_distributed_environment()


class V41NativePPWorkerAdmissionSM120Test(unittest.TestCase):
    def test_both_stages_ready_before_returning_native_admission(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41700)
            config = checkpoint(Path(root), full_schedule=True, hidden_size=5120)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
