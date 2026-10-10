"""Real PyWrappedModel preparation, CP exits and native allocation lifetime."""

import gc
import json
import os
import socket
import sys
import tempfile
import traceback
import unittest
import weakref
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_pp_transport_test as ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import restore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    resources,
)
from rtp_llm.models_py.modules.dsv4.test.test_native_l20_transfer import CONTRACT
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    make_worker,
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_admission_sm120 import (
    tensors_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_execution_sm120 import (
    expected_storage,
    full_owner,
    oracle_forward,
    raw_snapshot,
    require,
    require_raw,
    world_check,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_local_cache_sm120 import (
    release_allocations,
    root_allocations,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_preflight_sm120 import (
    ROUNDS,
    clean_worker,
    global_tokens,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_native_pp_transport_sm120 import (
    REQUESTS,
    cache_config,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.ops import NcclCommConfig


def rejected(call):
    try:
        call()
    except ops.NativeInputRejected:
        return
    raise AssertionError("inadmissible native model descriptor executed")


def run_rank(rank, port, config):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "RTP_V41_BATCHED_PRODUCER": "0",
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
    native = worker = reference = None
    with patch.dict(sys.modules, {"librtp_compute_ops": ops}):
        try:
            collective.init_distributed_environment(
                parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
            )
            with torch.inference_mode():
                native = ops.NativePPTransport(cache_config(), parallelism_for(rank))
                cache = native.cache
                free_before = native.free_blocks
                roots = root_allocations(native, rank)
                worker = make_worker(rank, config)
                assert worker.initialize(resources(cache))
                native.attach_model(
                    worker,
                    worker.v4.head_weight if rank >= 4 else torch.empty(0),
                )
                oracle, reference = full_owner(rank, config, worker)
                oracle_cache = NativeTaggedCache(
                    [layer.attn for layer in oracle.layers],
                    gen=0,
                    padding=0,
                    pages=native.capacity,
                )
                for layer in range(cache.layer_count):
                    for group in cache.get_layer_cache_groups(layer):
                        group.kv_cache_base.view(torch.uint8).fill_(37)
                identities = [
                    V41CacheIdentity(request, row + 1, 1)
                    for row, request in enumerate(REQUESTS)
                ]
                if rank != 4:
                    worker._configure_native_pp_engine(CONTRACT)
                ids, windows = global_tokens([1, 1], [[], []], 0)
                spans = [V41L20Span(identity, 0, 0, 1) for identity in identities]
                admission_tensors = tensors_for(spans, ids, windows)
                admission_tensors["lm_indexes"] = torch.tensor(
                    [0, 1], dtype=torch.int32
                )
                source = native.global_inputs(admission_tensors, roots)
                before = raw_snapshot(cache)
                occupied = native.free_blocks
                rejected(lambda: native.prepare_global_model(source, False))
                world_check(
                    lambda: (
                        require_raw(cache, before),
                        require(
                            native.free_blocks == occupied, "readiness leaked pages"
                        ),
                        require(
                            source.model_dispatch is None, "descriptor published early"
                        ),
                    ),
                    "missing contract WORLD rollback",
                )
                del source
                worker._configure_native_pp_engine(CONTRACT)
                for variant in ("output-mode", "lm-row"):
                    candidate = dict(admission_tensors)
                    if variant == "lm-row" and rank == 4:
                        candidate["lm_indexes"] = torch.tensor(
                            [0, 99], dtype=torch.int32
                        )
                    source = native.global_inputs(candidate, roots)
                    rejected(
                        lambda: native.prepare_global_model(
                            source, variant == "output-mode" and rank == 4
                        )
                    )
                    world_check(
                        lambda: (
                            require_raw(cache, before),
                            require(
                                native.free_blocks == occupied,
                                "output rejection leaked pages",
                            ),
                            require(
                                source.model_dispatch is None,
                                "output rejection published descriptor",
                            ),
                        ),
                        f"WORLD output admission {variant}",
                    )
                    del source
                for variant in ("changed-device-cu", "missing-cu", "changed-host-cu"):
                    source = native.global_inputs(admission_tensors, roots)
                    original = worker._prepare_native_pp_engine

                    def corrupt_boundaries(inputs):
                        if rank == 4:
                            tagged = inputs.attention_inputs
                            for attention in tagged.values():
                                if variant == "changed-device-cu":
                                    attention.cu_seqlens_device = (
                                        attention.cu_seqlens_device + 1
                                    )
                                elif variant == "missing-cu":
                                    # The native field is a Tensor. None fails
                                    # at pybind before STAGE agreement; an empty
                                    # tensor tests missing boundaries in preflight.
                                    attention.cu_seqlens_device = (
                                        attention.cu_seqlens_device.new_empty(0)
                                    )
                                else:
                                    attention.cu_seqlens = torch.tensor(
                                        [0, 1, 2], dtype=torch.int32
                                    )
                            inputs.attention_inputs = tagged
                        return original(inputs)

                    with patch.object(
                        worker, "_prepare_native_pp_engine", new=corrupt_boundaries
                    ):
                        rejected(lambda: native.prepare_global_model(source, True))
                    world_check(
                        lambda: (
                            require_raw(cache, before),
                            require(
                                native.free_blocks == occupied,
                                "cumulative-boundary rejection leaked pages",
                            ),
                            require(
                                source.model_dispatch is None,
                                "cumulative-boundary rejection published descriptor",
                            ),
                        ),
                        f"WORLD cumulative boundary admission {variant}",
                    )
                    del source, original, corrupt_boundaries
                try:
                    worker._configure_native_pp_engine(
                        replace(CONTRACT, checkpoint_sha256="d" * 64)
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError("active native model contract changed")
                prefixes, histories = [0, 0], [[], []]
                group = collective._get_group(collective.Group.TP)
                for step, lengths in enumerate(ROUNDS):
                    full = step != 1
                    os.environ["DSV4_PREFILL_FAST_PATH"] = "0" if step == 0 else "1"
                    ids, windows = global_tokens(lengths, histories, step)
                    spans = [
                        V41L20Span(identity, step, start, count)
                        for identity, start, count in zip(identities, prefixes, lengths)
                    ]
                    tensors = tensors_for(spans, ids, windows)
                    indexes = torch.tensor(
                        [lengths[0] - 1, sum(lengths) - 1], dtype=torch.int32
                    )
                    tensors["lm_indexes"] = indexes
                    source = native.global_inputs(tensors, roots)
                    captured = []
                    original = worker._prepare_native_pp_engine

                    def prepare(inputs):
                        descriptor = original(inputs)
                        captured.append(descriptor)
                        return descriptor

                    with patch.object(worker, "_prepare_native_pp_engine", new=prepare):
                        batch = native.prepare_global_model(source, full)
                    dispatch = captured.pop()
                    weak = weakref.ref(dispatch)
                    rejected(
                        lambda: native.execute_global_model(
                            source.model_dispatch, step, None
                        )
                    )
                    native.initialize_global(batch)
                    round_id = native.begin_global_execution(batch)
                    rejected(lambda: native.execute_global_model(None, round_id, None))
                    rejected(
                        lambda: native.execute_global_model(
                            source.model_dispatch, round_id, None, True
                        )
                    )
                    before = raw_snapshot(cache)
                    expected, expected_payload, _ = oracle_forward(
                        oracle, reference, oracle_cache, dispatch, lengths, prefixes
                    )
                    restored = restore(expected, dispatch.cp, group)
                    if rank < 4:
                        outputs = native.execute_global_model(
                            source.model_dispatch, round_id, None
                        )
                        payload = outputs["payload"]
                        native.send_local(payload, batch)
                    else:
                        payload = native.receive_local(batch)
                    native.wait()
                    if rank >= 4:
                        outputs = native.execute_global_model(
                            source.model_dispatch, round_id, payload
                        )
                    native.wait()
                    rejected(
                        lambda: native.execute_global_model(
                            source.model_dispatch,
                            round_id,
                            None if rank < 4 else payload,
                        )
                    )
                    expected_logits = None
                    if rank >= 4:
                        width = oracle.head_weight.size(0) // 4
                        head = oracle.head_weight.narrow(0, rank % 4 * width, width)
                        local = ops.native_lm_logits(restored, head.contiguous())
                        shards = [torch.empty_like(local) for _ in range(4)]
                        dist.all_gather(shards, local, group=group)
                        expected_logits = torch.cat(shards, dim=1)

                    def verify():
                        clean_worker(worker)
                        assert set(payload) == set(expected_payload)
                        for key, value in expected_payload.items():
                            torch.testing.assert_close(
                                payload[key], value, rtol=0, atol=0
                            )
                        require_raw(
                            cache,
                            expected_storage(
                                cache, oracle_cache, dispatch, rank // 4, before
                            ),
                        )
                        if rank >= 4:
                            last = restored[indexes.long().cuda()]
                            torch.testing.assert_close(
                                outputs["hidden"], last, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                outputs["all_hidden"],
                                restored if full else last,
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                outputs["logits"],
                                expected_logits[indexes.long().cuda()],
                                rtol=0,
                                atol=0,
                            )
                            if full:
                                torch.testing.assert_close(
                                    outputs["all_logits"],
                                    expected_logits,
                                    rtol=0,
                                    atol=0,
                                )
                            assert worker._native_pp_receiver._next_round == step + 1
                        for key in (
                            "ids",
                            "windows",
                            "lengths",
                            "prefixes",
                            "lm_indexes",
                        ):
                            torch.testing.assert_close(
                                source.tensors[key], tensors[key], rtol=0, atol=0
                            )
                        assert not native.global_failed

                    world_check(verify, f"native model output/cache round {step}")
                    native.complete_global_execution(batch, True)
                    native.commit_global(batch)
                    print(
                        "DSV41_NATIVE_PP_MODEL "
                        + json.dumps(
                            {
                                "rank": rank,
                                "step": step,
                                "full_outputs": full,
                                "native_cp_hidden_logits_exact": True,
                                "frame_cache_exact": True,
                            }
                        ),
                        flush=True,
                    )
                    for row, count in enumerate(lengths):
                        histories[row].extend(
                            ids[
                                sum(lengths[:row]) : sum(lengths[:row]) + count
                            ].tolist()
                        )
                        prefixes[row] += count
                    del dispatch, original, outputs, payload, batch
                    # The last Python reference is owned by C++. Its destructor
                    # must acquire the GIL when the native engine thread drops it.
                    native.release_model_descriptor(source)
                    world_check(
                        lambda: require(weak() is None, "model descriptor retained"),
                        "GIL-safe release",
                    )
                    del source, prepare
                    if step < 2:
                        native.finish()
                occupied = native.free_blocks
                for identity in identities:
                    native.retire_global_model(list(identity.__dict__.values()))
                release_allocations(roots)
                gc.collect()
                world_check(
                    lambda: require(
                        native.free_blocks == occupied, "consumer released early"
                    ),
                    "retirement",
                )
                native.finish()
                world_check(
                    lambda: require(
                        native.free_blocks == free_before, "capacity did not recover"
                    ),
                    "drain",
                )
                print(
                    "DSV41_NATIVE_PP_MODEL_LIFETIME "
                    + json.dumps({"rank": rank, "free_after": native.free_blocks}),
                    flush=True,
                )
                native.close()
                native = None
                worker.close()
                worker = None
                reference.close()
                reference = None
            dist.barrier()
        except BaseException:
            print(f"DSV41_MODEL_FAILURE rank={rank}", flush=True)
            traceback.print_exc()
            raise
        finally:
            if native is not None:
                native.wait()
                native.close()
            if worker is not None:
                worker.close()
            if reference is not None:
                reference.close()
            Dsv4SharedRuntimeBufferStore._reset_for_test()
            collective.destroy_distributed_environment()


class V41NativePPModelSM120Test(unittest.TestCase):
    def test_cpp_model_preparation_execution_and_cp_exits(self):
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
