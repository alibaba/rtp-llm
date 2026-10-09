"""Bounded 40-layer DP8 generation through NormalEngine's live scheduler loop."""

import os
import socket
import sys
import tempfile
import threading
import traceback
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
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
    decode_inputs,
    inputs,
    model_config,
    tagged_cache,
)
from rtp_llm.ops.compute_ops import PyModelInitResources, PyModelOutputs
from rtp_llm.tools.dsv41_trace import capture_model_boundaries, save_model_trace


def run_rank(rank, port, config, initialize_model, new_tokens=1, context=256):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": str(Path(config["checkpoint_path"]) / "cache"),
            "DSV4_MOE_STRATEGY": "auto",
            "DSV4_MOE_CHUNK_PREFILL": "1",
            "DSV4_MOE_CHUNK_TOKENS": "32" if context > 256 else "4",
            "DSV4_MOE_EXTENT_COUNT_FUSE": "1",
            "RTP_LLM_STREAM_ASYNC": "0",
            "RTP_LLM_DEVICE_INPUT": "0",
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
    inject_armed = threading.Event()
    inject_boundary = threading.Event()
    inject_release = threading.Event()
    try:
        with torch.inference_mode():
            model, reference = make_workers(
                rank, config, initialize_model=initialize_model, max_seq_len=context
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
            diagnostic_directory = os.environ.get("DSV41_TEST_FAILURE_TRACE_DIR")
            reference_traces = []

            @contextmanager
            def failure_boundaries(worker):
                with capture_model_boundaries(
                    worker.v4
                ) as tensors, ExitStack() as stack:
                    for layer in worker.v4.layers:
                        original = layer.attn._project_output
                        label = f"layer{layer.layer_id:02d}.attn"

                        def project(
                            output, freqs, out=None, *, op=original, name=label
                        ):
                            tensors[name + ".projection_input"] = (
                                output.detach().clone()
                            )
                            tensors[name + ".freqs"] = freqs.detach().clone()
                            result = op(output, freqs, out=out)
                            tensors[name + ".projection_output"] = (
                                result.detach().clone()
                            )
                            return result

                        stack.enter_context(
                            patch.object(layer.attn, "_project_output", project)
                        )
                    yield tensors

            def reference_forward(request):
                if not diagnostic_directory:
                    return reference.forward(request).hidden_states.clone()
                with failure_boundaries(reference) as tensors:
                    result = reference.forward(request).hidden_states.clone()
                tensors["hidden"] = result.cpu()
                reference_traces.append(tensors)
                return result

            initialize = model.initialize
            initialization_calls = []
            expected_tokens = {}
            continuations = {}
            calls = {"idle": 0, "active": 0, "decode": 0, "idle_decode": 0}
            injection = {"baseline": 0, "pending": {}, "deferred": False}
            growth = {"pages": 0, "long_prefill": 0}
            reference_pages = 32
            next_reference_page = 1
            if new_tokens > 1:
                reference.kv_cache = tagged_cache(reference, pages=reference_pages)

            def continuation_reference(request):
                nonlocal next_reference_page
                attn = next(iter(request.attention_inputs.values()))
                lengths = attn.input_lengths.tolist()
                ids = request.input_ids.tolist()
                offset = 0
                keys = []
                starts = []
                histories = []
                for row, span in enumerate(request.v41_cache_spans.tolist()):
                    request_id, _, _, _, start, count = span
                    if attn.is_prefill:
                        assert start == 0 and count == lengths[row]
                        prompt = ids[offset : offset + count]
                        page_count = (context + 255) // 256
                        assert next_reference_page + page_count <= reference_pages
                        state = {
                            "prompt": tuple(prompt),
                            "tokens": prompt.copy(),
                            "pages": {},
                            "tables": {
                                tag: torch.arange(
                                    next_reference_page,
                                    next_reference_page
                                    + (
                                        1
                                        if tag in ("swa_kv", "csa_state")
                                        else page_count
                                    ),
                                    device="cuda",
                                    dtype=torch.int32,
                                )[None]
                                for tag in reference.kv_cache.group_tags
                            },
                        }
                        next_reference_page += page_count
                        assert request_id not in continuations
                        continuations[request_id] = state
                        oracle = inputs(
                            reference.kv_cache, prompt, [count], tables=state["tables"]
                        )
                        growth["long_prefill"] += int(count > 256)
                    else:
                        state = continuations[request_id]
                        sequence = state["tokens"]
                        assert start == len(sequence) - 1 and count == 1
                        assert ids[row] == sequence[-1]
                        assert lengths[row] == len(state["prompt"])
                        assert attn.sequence_lengths[row].item() == start
                        history = list(
                            reversed(sequence[max(0, len(sequence) - 4) : -1])
                        )
                        history += [-1] * (3 - len(history))
                        histories.append(history)
                        starts.append(start)
                        oracle = decode_inputs(
                            reference.kv_cache,
                            [sequence[-1]],
                            [start],
                            [history],
                            prompt_lengths=[lengths[row]],
                            tables=state["tables"],
                        )
                    if context > 256:
                        for tag, tagged in request.attention_inputs.items():
                            table = tagged.kv_cache_kernel_block_id_device[row].tolist()
                            retained = tag in ("swa_kv", "csa_state")
                            needed = 1 if retained else (start + count + 255) // 256
                            pages = table[:needed]
                            assert len(pages) == needed and min(pages) > 0
                            assert len(set(pages)) == needed
                            previous = state["pages"].get(tag, [])
                            assert pages[: len(previous)] == previous
                            growth["pages"] += int(
                                bool(previous) and len(pages) > len(previous)
                            )
                            state["pages"][tag] = pages
                            if retained:
                                assert len(table) == 1
                    torch.testing.assert_close(
                        request.engram_token_windows[offset : offset + count].cpu(),
                        oracle.engram_token_windows.cpu(),
                        rtol=0,
                        atol=0,
                    )
                    keys.append(request_id)
                    offset += count
                assert offset == len(ids)
                # Preserve the native row count: BF16 grouped wo_a can select a
                # different GEMM reduction when a batch is split into single rows.
                # Reference cache allocations and page IDs remain independent.
                tables = {
                    tag: torch.cat([continuations[key]["tables"][tag] for key in keys])
                    for tag in reference.kv_cache.group_tags
                }
                if attn.is_prefill:
                    oracle = inputs(reference.kv_cache, ids, lengths, tables=tables)
                else:
                    oracle = decode_inputs(
                        reference.kv_cache,
                        ids,
                        starts,
                        histories,
                        prompt_lengths=lengths,
                        tables=tables,
                    )
                return reference_forward(oracle), keys

            def checked_initialize(resource):
                assert isinstance(resource, PyModelInitResources)
                assert resource.kv_cache is not None
                initialization_calls.append(model._materialized)
                return initialize(resource)

            def checked_forward(request, fmha_impl=None):
                try:
                    reference_traces.clear()
                    stop = torch.tensor([int(stop_requested.is_set())], device="cuda")
                    dist.all_reduce(stop, op=dist.ReduceOp.MIN)
                    if stop.item():
                        engine.request_stop()
                        stop_boundary.set()
                    attn = next(iter(request.attention_inputs.values()))
                    lengths = attn.input_lengths.tolist()
                    ids = request.input_ids.tolist()
                    if attn.is_prefill and injection["pending"]:
                        for key, tokens in injection["pending"].items():
                            assert continuations[key]["tokens"] == tokens
                        injection["pending"].clear()
                        injection["deferred"] = True
                    expected = None
                    if ids and new_tokens > 1:
                        expected, keys = continuation_reference(request)
                    elif ids:
                        reference.kv_cache = tagged_cache(reference)
                        oracle_request = inputs(reference.kv_cache, ids, lengths)
                        torch.testing.assert_close(
                            request.engram_token_windows.cpu(),
                            oracle_request.engram_token_windows.cpu(),
                            rtol=0,
                            atol=0,
                        )
                        expected = reference_forward(oracle_request)
                    with (
                        failure_boundaries(model)
                        if diagnostic_directory and ids
                        else nullcontext({})
                    ) as actual_trace:
                        actual = forward(request, fmha_impl).hidden_states.clone()
                    assert_released(model)
                    if ids:
                        try:
                            torch.testing.assert_close(
                                actual, expected, rtol=0.03, atol=0.05
                            )
                        except AssertionError:
                            if diagnostic_directory:
                                metadata = {
                                    "input_ids": ids,
                                    "cache_spans": request.v41_cache_spans.tolist(),
                                    "is_prefill": attn.is_prefill,
                                }
                                actual_trace["hidden"] = actual.cpu()
                                expected_trace = {
                                    name: torch.cat(
                                        [trace[name] for trace in reference_traces]
                                    )
                                    for name in reference_traces[0]
                                }
                                for label, trace in (
                                    ("reference", expected_trace),
                                    ("native", actual_trace),
                                ):
                                    save_model_trace(
                                        Path(diagnostic_directory)
                                        / f"rank{rank}-{label}.pt",
                                        trace,
                                        metadata,
                                    )
                            raise
                        last = (
                            torch.tensor(lengths, device="cuda").cumsum(0) - 1
                            if attn.is_prefill
                            else torch.arange(len(lengths), device="cuda")
                        )
                        tokens = (
                            (actual[last].float() @ head.float().T).argmax(-1).tolist()
                        )
                        if new_tokens > 1:
                            for key, token in zip(keys, tokens):
                                state = continuations[key]
                                state["tokens"].append(token)
                                expected_tokens[state["prompt"]] = state[
                                    "tokens"
                                ].copy()
                        else:
                            start = 0
                            for length, token in zip(lengths, tokens):
                                expected_tokens[tuple(ids[start : start + length])] = (
                                    token
                                )
                                start += length
                        calls["active"] += 1
                        calls["decode"] += int(not attn.is_prefill)
                    else:
                        assert actual.shape == (0, 512)
                        assert request.v41_cache_spans.shape == (0, 6)
                        calls["idle"] += 1
                        calls["idle_decode"] += int(not attn.is_prefill)
                    if new_tokens > 1:
                        inject = torch.tensor(
                            [
                                int(
                                    rank == 0
                                    and inject_armed.is_set()
                                    and calls["decode"] > injection["baseline"]
                                )
                            ],
                            device="cuda",
                        )
                        dist.all_reduce(inject, op=dist.ReduceOp.MAX)
                        if inject.item():
                            inject_armed.clear()
                            injection["pending"] = {
                                key: state["tokens"].copy()
                                for key, state in continuations.items()
                                if len(state["tokens"])
                                < len(state["prompt"]) + new_tokens
                            }
                            inject_boundary.set()
                            assert inject_release.wait(
                                60
                            ), "late prompts were not enqueued"
                    return PyModelOutputs(actual)
                except BaseException:
                    traceback.print_exc()
                    sys.stderr.flush()
                    os._exit(1)

            with patch.object(model, "forward", checked_forward), patch.object(
                model, "initialize", checked_initialize
            ):
                engine = native_ops.NativeEngine(
                    model,
                    model_config(model._v4_args, config),
                    model.parallelism_config,
                    head,
                    pdfusion=new_tokens > 1,
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
                phases = (
                    (
                        (253, 255) if rank == 0 else (),
                        (253 + rank % 3,),
                        (503,) if rank % 2 else (),
                        (503,),
                        (257,) if rank == 7 else (),
                    )
                    if context > 256
                    else (
                        (3, 2) if rank == 0 else (),
                        (125 if new_tokens > 1 else rank + 1,),
                        (2, 1) if rank % 2 else (),
                        (1,),
                        (2,) if rank == 7 else (),
                    )
                )
                for phase, lengths in enumerate(phases):
                    requests = [
                        [(rank * 7 + phase * 13 + row * 3 + i) % 128 for i in range(n)]
                        for row, n in enumerate(lengths)
                    ]
                    engine.enqueue(requests, new_tokens)
                    result = engine.outputs()
                    assert result["tokens"] == [
                        (
                            expected_tokens[tuple(row)]
                            if new_tokens > 1
                            else row + [expected_tokens[tuple(row)]]
                        )
                        for row in requests
                    ]
                    assert result["steps"] == [new_tokens] * len(requests)
                    # Main-thread control never competes with the live engine's
                    # WORLD NCCL collective sequence.
                    dist.barrier(group=control)
                if new_tokens > 1:
                    injection["baseline"] = calls["decode"]
                    inject_armed.set()
                    requests = (
                        [[(31 + i) % 128 for i in range(253 if context > 256 else 3)]]
                        if rank == 0
                        else []
                    )
                    engine.enqueue(requests, new_tokens)
                    assert inject_boundary.wait(
                        60
                    ), "decode did not reach injection boundary"
                    late = []
                    if rank in (0, 7):
                        first = 43 if rank == 0 else 103
                        count = (
                            (503 if rank == 0 else 501)
                            if context > 256
                            else (2 if rank == 0 else 3)
                        )
                        late = [[(first + i) % 128 for i in range(count)]]
                    engine.enqueue(late, new_tokens)
                    dist.barrier(group=control)
                    inject_release.set()
                    result = engine.outputs()
                    assert result["tokens"] == [
                        expected_tokens[tuple(row)] for row in requests + late
                    ]
                    assert result["steps"] == [new_tokens] * len(requests + late)
                    if rank == 0:
                        assert injection[
                            "deferred"
                        ], "live decode was not deferred for late prefill"
                    dist.barrier(group=control)
                stop_requested.set()
                assert stop_boundary.wait(60), "engine did not reach coordinated stop"
                result = engine.stop()
                engine = None
                assert result["free_before"] == result["free_after"], result
                assert result["cache_blocks"] == 8, result
            assert calls["active"] >= 2 and calls["idle"] >= 1, calls
            if new_tokens > 1:
                assert calls["decode"] >= 16 and calls["idle_decode"] >= 1, calls
            if context > 256:
                assert growth["pages"] > 0 and growth["long_prefill"] > 0, growth
            assert bridge_calls["stage"] == 1
            assert bridge_calls["phase"] == calls["active"] + calls["idle"]
            reference.close()
            model.close()
            dist.barrier(group=control)
            if rank == 0:
                assert all(not path.exists() for path in table_paths)
                assert not list(model._engram_cache_dir.glob("*.lease.*"))
                print(
                    "V4.1 40-layer native DP8 NormalEngine "
                    f"new_tokens={new_tokens} context={context} prematerialized={initialize_model} PASS",
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
    def run_engine(self, initialize_model, new_tokens=1, context=256):
        self.assertEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41900)
            config = checkpoint(Path(root), full_schedule=True)
            (Path(root) / "cache").mkdir()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(
                run_rank,
                args=(port, config, initialize_model, new_tokens, context),
                nprocs=8,
                join=True,
            )

    def test_eight_rank_scheduler_loop(self):
        self.run_engine(initialize_model=True)

    def test_engine_materializes_workers(self):
        self.run_engine(initialize_model=False)

    def test_cold_pdfusion_prefill_and_eight_decode_steps(self):
        self.run_engine(initialize_model=False, new_tokens=9)

    def test_two_page_cold_pdfusion_and_late_prefill(self):
        self.run_engine(initialize_model=False, new_tokens=9, context=512)


if __name__ == "__main__":
    unittest.main()
