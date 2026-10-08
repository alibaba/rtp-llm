"""Bounded 40-layer DP8 prefill through NormalEngine's live scheduler loop."""

import os
import socket
import sys
import tempfile
import threading
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test import libth_v41_native_engine_test as native_ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_dp_worker_sm120 import (
    assert_released,
    make_workers,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    inputs,
    model_config,
    tagged_cache,
)
from rtp_llm.ops.compute_ops import PyModelInitResources, PyModelOutputs


def run_rank(rank, port, config, initialize_model):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": str(Path(config["checkpoint_path"]) / "cache"),
            "DSV4_MOE_STRATEGY": "auto",
            "DSV4_MOE_CHUNK_PREFILL": "1",
            "DSV4_MOE_CHUNK_TOKENS": "4",
            "DSV4_MOE_EXTENT_COUNT_FUSE": "1",
            "MOEDBG": "0",
        }
    )
    Dsv4SharedRuntimeBufferStore._reset_for_test()
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    model = reference = engine = None
    stop_requested = threading.Event()
    stop_boundary = threading.Event()
    try:
        with torch.inference_mode():
            model, reference = make_workers(
                rank, config, initialize_model=initialize_model
            )
            assert model._materialized == initialize_model
            torch.manual_seed(41950)
            head = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
            bridge_calls = {"stage": 0, "phase": 0}
            register = native_ops.register_comm_ops

            def register_checked(broadcast, allreduce, allgather):
                def reduce(tensor, op, mode, dest):
                    assert mode == collective._CPP_PARALLEL_MODE_WORLD and op == 2
                    bridge_calls["phase"] += 1
                    return allreduce(tensor, op, mode, dest)

                def gather(recv, mode, send, inplace):
                    assert mode == collective._CPP_PARALLEL_MODE_STAGE
                    allgather(recv, mode, send, inplace)
                    assert recv[0].tolist() == list(range(8, 16))
                    bridge_calls["stage"] += 1

                register(broadcast, reduce, gather)

            # The helper has its own linked ExecOps instance. Register the
            # production callbacks into that instance using the upstream bridge.
            with patch.dict(
                sys.modules, {"librtp_compute_ops": native_ops}
            ), patch.dict(
                collective._group_map,
                {
                    collective.Group.WORLD: dist.group.WORLD,
                    collective.Group.STAGE: dist.group.WORLD,
                },
                clear=True,
            ), patch.object(
                collective, "_parallelism_config", model.parallelism_config
            ), patch.object(
                native_ops, "register_comm_ops", side_effect=register_checked
            ):
                collective._register_process_groups_to_cpp()
            forward = model.forward
            initialize = model.initialize
            initialization_calls = []
            expected_tokens = {}
            calls = {"idle": 0, "active": 0}

            def checked_initialize(resource):
                assert isinstance(resource, PyModelInitResources)
                assert resource.kv_cache is not None
                initialization_calls.append(model._materialized)
                return initialize(resource)

            def checked_forward(request, fmha_impl=None):
                try:
                    stop = torch.tensor([int(stop_requested.is_set())], device="cuda")
                    dist.all_reduce(stop, op=dist.ReduceOp.MIN)
                    if stop.item():
                        engine.request_stop()
                        stop_boundary.set()
                    lengths = next(
                        iter(request.attention_inputs.values())
                    ).input_lengths.tolist()
                    ids = request.input_ids.tolist()
                    expected = None
                    if ids:
                        reference.kv_cache = tagged_cache(reference)
                        oracle_request = inputs(reference.kv_cache, ids, lengths)
                        torch.testing.assert_close(
                            request.engram_token_windows.cpu(),
                            oracle_request.engram_token_windows.cpu(),
                            rtol=0,
                            atol=0,
                        )
                        expected = reference.forward(
                            oracle_request
                        ).hidden_states.clone()
                    actual = forward(request, fmha_impl).hidden_states.clone()
                    assert_released(model)
                    if ids:
                        torch.testing.assert_close(
                            actual, expected, rtol=0.03, atol=0.05
                        )
                        last = torch.tensor(lengths, device="cuda").cumsum(0) - 1
                        tokens = (
                            (actual[last].float() @ head.float().T).argmax(-1).tolist()
                        )
                        start = 0
                        for length, token in zip(lengths, tokens):
                            expected_tokens[tuple(ids[start : start + length])] = token
                            start += length
                        calls["active"] += 1
                    else:
                        assert actual.shape == (0, 512)
                        assert request.v41_cache_spans.shape == (0, 6)
                        calls["idle"] += 1
                    return PyModelOutputs(actual)
                except BaseException:
                    traceback.print_exc()
                    raise

            with patch.object(model, "forward", checked_forward), patch.object(
                model, "initialize", checked_initialize
            ):
                engine = native_ops.NativeEngine(
                    model,
                    model_config(model._v4_args, config),
                    model.parallelism_config,
                    head,
                )
                assert initialization_calls == [initialize_model]
                assert model._materialized and not hasattr(model, "weight")
                table_paths = [table._path for table in model._engram_tables.values()]
                assert len(table_paths) == 2
                assert all(
                    path.parent == model._engram_cache_dir for path in table_paths
                )
                identities = [
                    (path.stat().st_dev, path.stat().st_ino) for path in table_paths
                ]
                peers = [None] * 8
                dist.all_gather_object(peers, identities, group=control)
                assert all(peer == identities for peer in peers)
                for phase, lengths in enumerate(
                    (
                        (3, 2) if rank == 0 else (),
                        (rank + 1,),
                        (2, 1) if rank % 2 else (),
                        (1,),
                        (2,) if rank == 7 else (),
                    )
                ):
                    requests = [
                        [(rank * 7 + phase * 13 + row * 3 + i) % 128 for i in range(n)]
                        for row, n in enumerate(lengths)
                    ]
                    engine.enqueue(requests)
                    result = engine.outputs()
                    assert result["tokens"] == [
                        row + [expected_tokens[tuple(row)]] for row in requests
                    ]
                    assert result["steps"] == [1] * len(requests)
                    # Main-thread control never competes with the live engine's
                    # WORLD NCCL collective sequence.
                    dist.barrier(group=control)
                stop_requested.set()
                assert stop_boundary.wait(60), "engine did not reach coordinated stop"
                result = engine.stop()
                engine = None
                assert result["free_before"] == result["free_after"], result
                assert result["cache_blocks"] == 8, result
            assert calls["active"] >= 2 and calls["idle"] >= 1, calls
            assert bridge_calls["stage"] == 1
            assert bridge_calls["phase"] == calls["active"] + calls["idle"]
            reference.close()
            model.close()
            dist.barrier(group=control)
            if rank == 0:
                assert all(not path.exists() for path in table_paths)
                assert not list(model._engram_cache_dir.glob("*.lease.*"))
                print(
                    "V4.1 40-layer native DP8 NormalEngine prefill "
                    f"prematerialized={initialize_model} PASS",
                    flush=True,
                )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if engine is None:
            if reference is not None:
                reference.close()
            if model is not None:
                model.close()
            native_ops.clear_comm_ops()
            dist.destroy_process_group(control)
            dist.destroy_process_group()


class V41NativeEngineSM120Test(unittest.TestCase):
    def run_engine(self, initialize_model):
        self.assertEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41900)
            config = checkpoint(Path(root), full_schedule=True)
            (Path(root) / "cache").mkdir()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(
                run_rank, args=(port, config, initialize_model), nprocs=8, join=True
            )

    def test_eight_rank_scheduler_loop(self):
        self.run_engine(initialize_model=True)

    def test_engine_materializes_workers(self):
        self.run_engine(initialize_model=False)


if __name__ == "__main__":
    unittest.main()
