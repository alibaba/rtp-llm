"""Cold MLA-core traces entered through RTP's real DP/TP/DCP factory forward.

Run using the native Bazel target. MLA_BENCH_CASES is a JSON list of B/G pairs;
G includes the four current query tokens. All eight workers use the same global
request seed before the shared fixture selects DP requests/head shards/KV owners.
MLA_BENCH_MODEL_MAX_SEQ_LEN is fixed across cases (default 1M), independently
of G. The graph page-table capacity includes the Q4 speculative reserve.
"""
import json
import hashlib
import importlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import subprocess
import statistics
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import (
    init_distributed_environment, destroy_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import get_mla_impl
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import mla_dcp_comm
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.page_rr_mla_decode_test import (
    _clone_query_inputs, _fixture, _layer_collective,
)
from rtp_llm.ops import FMHAConfig, NcclCommConfig, ParallelismConfig, RoleType
from rtp_llm.test.utils.port_util import PortManager


@triton.jit
def _mla_bench_evict_l2(Buffer, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Buffer + index, index.to(tl.uint8), index < N)


def _samples(fixture, requests, head_start):
    """Small coordinate-labelled witnesses, not a claim of full tensor hashes."""
    query = fixture.q.view(len(requests), 4, fixture.q.shape[1], 192)
    sample = {}
    for local, request in enumerate(requests):
        for query_id in (0, 3):
            for head in (0, query.shape[2] - 1):
                sample[f"q/{request}/{query_id}/{head_start+head}"] = query[local, query_id, head, :8].float().cpu().tolist()
    sample["kv/0/0"] = fixture.canonical[0, 0, :8].float().cpu().tolist()
    sample["kv/last/last"] = fixture.canonical[-1, fixture.prefixes[-1]+3, :8].float().cpu().tolist()
    return sample


def _resource_snapshot():
    queries = {
        "gpu": "--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw",
        "processes": "--query-compute-apps=pid,process_name,used_memory",
    }
    result = dict(timestamp=time.time())
    for name, query in queries.items():
        value = subprocess.run(["nvidia-smi", query, "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=15)
        result[name] = dict(returncode=value.returncode, stdout=value.stdout, stderr=value.stderr)
    return result


def _tokenspeed_plan(op, device):
    # Read the same locked module used by the real Op, and call its existing
    # cached planner solely for reporting. Forward still owns all dispatch.
    api = importlib.import_module(op.__module__)._TOKENSPEED_MLA_API
    module = importlib.import_module(api.__module__)
    dtype = op._q_fp8.dtype
    capability = torch.cuda.get_device_capability(device)
    qk, pv = module.select_mla_decode_tilers(op.num_heads, 4, is_fp8=True,
                                            compute_capability=capability)
    fold = module.get_mla_decode_fold_sq_factor(op.num_heads, 4, qk[0])
    sms = module.get_num_sm(device)
    splits, workspace = module._get_split_kv_and_workspace_size(
        op._batch_size, 4//fold, op.num_heads*fold, op.kv_lora_rank,
        sms, max(op._max_context_len, op.token_per_block), dtype, qk,
    )
    return dict(mode="local", splits=splits, persistent=False, query_fold=fold,
                H_effective=op.num_heads*fold, Q_effective=4//fold,
                qk_tiler=list(qk), pv_tiler=list(pv), planner_sm_count=sms,
                planner_workspace_bytes=workspace,
                planner_source=module.__file__,
                planner_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())


def _worker(rank, world, port, topology, cases, directory, failed):
    torch.cuda.set_device(rank)
    is_dcp = topology == "DCP8"
    parallelism = ParallelismConfig()
    parallelism.world_size = parallelism.local_world_size = world
    parallelism.world_rank = parallelism.local_rank = rank
    parallelism.tp_size = 1 if topology == "DP8" else world
    parallelism.dp_size = world // parallelism.tp_size
    parallelism.tp_rank = rank % parallelism.tp_size
    parallelism.dp_rank = rank // parallelism.tp_size
    parallelism.role_type = RoleType.DECODE
    parallelism.decode_cp_kv_cache_sharded = is_dcp
    parallelism.decode_cp_q_replicated = is_dcp
    init_distributed_environment(parallelism, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=300)
    config = FMHAConfig()
    from rtp_llm.server.server_args.util import (
        str2_decode_cp_mla_backend, str2_decode_cp_mla_fusion_mode, str2_decode_cp_mla_a2a_backend,
    )
    config.decode_cp_mla_backend = str2_decode_cp_mla_backend(
        os.environ.get("DECODE_CP_MLA_BACKEND", "FIA2A")
    )
    config.decode_cp_mla_fusion_mode = str2_decode_cp_mla_fusion_mode(
        os.environ.get("DECODE_CP_MLA_FUSION_MODE", "AUTO")
    )
    config.decode_cp_mla_a2a_backend = str2_decode_cp_mla_a2a_backend(
        os.environ.get("DECODE_CP_MLA_A2A_BACKEND", "AUTO")
    )
    graph = None
    try:
        properties = torch.cuda.get_device_properties(rank)
        l2_bytes = properties.L2_cache_size
        flush_bytes = max(256 * 1024 * 1024, 2 * l2_bytes)
        assert flush_bytes > 0
        flush = torch.empty(flush_bytes, dtype=torch.uint8, device=torch.device("cuda", rank))
        evict = lambda: _mla_bench_evict_l2[(triton.cdiv(flush_bytes, 1024),)](flush, flush_bytes, 1024)
        evict()
        inner = int(os.environ.get("MLA_BENCH_INNER", "33"))
        tail = int(os.environ.get("MLA_BENCH_TAIL", "16"))
        assert 0 < tail < inner
        minimum_settle = int(os.environ.get("MLA_BENCH_PROFILE_SETTLE", "1"))
        settle_ms = float(os.environ.get("MLA_BENCH_PROFILE_SETTLE_MS", "500"))
        profile_events = os.environ.get("MLA_BENCH_PROFILE_EVENTS", "0") == "1"
        profile_enabled = os.environ.get("MLA_BENCH_PROFILE", "1") != "0"
        model_max_seq_len = int(os.environ.get("MLA_BENCH_MODEL_MAX_SEQ_LEN", str(1024 * 1024)))
        # Match CudaGraphRunner::initCaptureAttentionInputs for proposal3/Q4:
        # (ceil(model limit / physical page) + sp_steps) * kernel pages/physical.
        graph_table_columns = (triton.cdiv(model_max_seq_len, 1024) + 3) * (1024 // 128)
        assert minimum_settle >= 1 and settle_ms >= 0
        configured_fusion_mode = config.decode_cp_mla_fusion_mode
        configured_a2a_backend = config.decode_cp_mla_a2a_backend
        for case in cases:
            batch, global_kv = case["B"], case["G"]
            assert 4 <= global_kv <= model_max_seq_len
            config.decode_cp_mla_fusion_mode = (
                str2_decode_cp_mla_fusion_mode(case["fusion_mode"])
                if "fusion_mode" in case else configured_fusion_mode
            )
            config.decode_cp_mla_a2a_backend = (
                str2_decode_cp_mla_a2a_backend(case["a2a_backend"])
                if "a2a_backend" in case else configured_a2a_backend
            )
            # Paired modes have their own prepare/capture and artifact paths.
            # The fixture seeds and static split geometry are unchanged.
            arm = (f"{topology}_{config.decode_cp_mla_fusion_mode.name}"
                   if "fusion_mode" in case else topology)
            if "a2a_backend" in case:
                arm += "_A2A_" + config.decode_cp_mla_a2a_backend.name
            case_id = f"g{global_kv}_b{batch}_q4"
            output_dir = Path(directory) / case_id / arm / f"rank{rank}"
            output_dir.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                (output_dir.parent/"resource-before.json").write_text(json.dumps(_resource_snapshot(), indent=2)+"\n")
            requests = list(range(rank, batch, world)) if topology == "DP8" else list(range(batch))
            manifest = dict(case_id=case_id, B_global=batch, G=global_kv, Q=4, H_global=96,
                            topology=topology, rank=rank, world=world, request_ids=requests,
                            decode_cp_backend_selector=config.decode_cp_mla_backend.name,
                            requested_fusion_mode=config.decode_cp_mla_fusion_mode.name,
                            active=bool(requests), cache="cold", L2_bytes=l2_bytes,
                            flush_bytes=flush_bytes, inner=inner, tail=tail,
                            selected_iterations=list(range(inner-1-tail, inner-1)),
                            excluded_last_body="No following flush; followed by V projection",
                            scope="absorbed Q through MLA/local reduction/required DCP communication, before V projection",
                            seed=1701, weight_kv_seed=3301, global_reference_generation=True,
                            ownership_page=1024, kernel_page=128, dtype="FP8_E4M3",
                            q_scale=1.0, kv_scale=1.0, qrep=is_dcp,
                            model_max_seq_len=model_max_seq_len, actual_max_kv=global_kv,
                            graph_table_columns=graph_table_columns, speculative_reserve_pages=3,
                            graph_shape_policy="exact attention-input shape; scheduler/SP/capture-bucket padding is outside this operator benchmark",
                            torch=torch.__version__, capability=torch.cuda.get_device_capability(rank),
                            sms=properties.multi_processor_count,
                            bare_metric="outer CUDA events: entire graph including preparation, flushes and final projection",
                            preparation_graph_replays=2, planning_graph_replays=5, bare_graph_replays=5,
                            profiled_settling_replays=minimum_settle,
                            profiled_measurement_replays=1 if profile_enabled else 0,
                            profiling_enabled=profile_enabled,
                            requested_a2a_backend=config.decode_cp_mla_a2a_backend.name,
                            profiled_outer_events=profile_events, profile_settle_min_ms=settle_ms,
                            preparation_to_measurement="contiguous same-stream graph enqueue; no intervening host/device barrier")
            if requests:
                print(f"MLA BENCH START rank={rank} {arm} {case_id}", flush=True)
                fixture = _fixture(
                    parallelism.tp_rank, parallelism.tp_size, 4, True,
                    device_index=rank, q_replicated=is_dcp,
                    prefix_lengths=[global_kv-4]*batch, heads=96,
                    page=1024, kernel_page=128, independent_requests=True,
                    kv_size=world if is_dcp else 1, kv_rank=rank if is_dcp else 0,
                    request_ids=requests if topology == "DP8" else None,
                )
                # Capacity affects the descriptor even when few pages contain
                # live KV. Do not allocate fictitious KV pages for the padding.
                groups = []
                for table in fixture.inputs.kv_cache_kernel_block_id_device_by_group:
                    assert table.shape[1] <= graph_table_columns
                    captured = table.new_zeros((table.shape[0], graph_table_columns))
                    captured[:, :table.shape[1]].copy_(table)
                    groups.append(captured)
                fixture.inputs.kv_cache_kernel_block_id_device_by_group = groups
                if not is_dcp:
                    fixture.inputs.kv_cache_kernel_block_id_device = (
                        fixture.inputs.kv_cache_kernel_block_id_device_by_group[fixture.group_id]
                    )
                manifest["input_samples"] = _samples(fixture, requests,
                    0 if is_dcp else parallelism.tp_rank * fixture.config.head_num)
                impl = get_mla_impl(
                    fixture.config, SimpleNamespace(weights=fixture.weights,
                        get_global_weight=lambda _: fixture.cos_sin), fixture.inputs,
                    fmha_config=config, parallelism_config=parallelism,
                    is_cuda_graph=True, max_seq_len=model_max_seq_len,
                )
                expected_class = "PageRRMlaDecodeImpl" if is_dcp else "TokenSpeedMlaDecodeImpl"
                assert type(impl).__name__ == expected_class, type(impl).__name__
                backend = getattr(impl.fmha_impl, "fia2a_backend", None)

                def wait_peer_consumption():
                    # Stands in for the model's following TP collective, which
                    # orders the next peer write after this rank's reads. It is
                    # issued after the measured core and excluded by the parser.
                    # Keep the same inter-call ordering for all FIA2A transports.
                    if backend is not None:
                        _layer_collective()

                def forward():
                    # RoPE mutates its Q/K inputs; copy from immutable fixture
                    # tensors outside the measured core on every graph replay.
                    q, k_pe = _clone_query_inputs(fixture)
                    output = impl.forward(q, fixture.ckv, k_pe,
                                          fixture.cache, fixture.layer_id)
                    wait_peer_consumption()
                    return output
                result = forward()
                torch.testing.assert_close(result, fixture.expected, atol=2e-3, rtol=0.015)
                manifest.update(impl=type(impl).__module__+"."+type(impl).__name__,
                                op=type(impl.fmha_impl).__name__, local_heads=fixture.config.head_num,
                                kv_shape=list(fixture.cache.kv_cache_base.shape),
                                kv_stride=list(fixture.cache.kv_cache_base.stride()),
                                unique_kv_bytes=len(requests)*(global_kv//world if is_dcp else global_kv)*576)
                candidate = getattr(impl.fmha_impl, "fia2a_backend", None)
                if candidate is not None:
                    manifest.update(test_consumption_wait="AllReduce")
                    manifest.update(mode=candidate.mode.name.lower(), splits=candidate.splits, persistent=True,
                                    requested_mode=candidate.requested_mode.name,
                                    a2a_transport=("producer" if candidate.fused else
                                                   "pull" if candidate.a2a_buffers is not None else "nccl"),
                                    wire_dtype=str(candidate.buffers.output.dtype if candidate.fused
                                                   else candidate.output.dtype),
                                    qk_tiler=[128, 128], planner_sm_count=candidate.sm_count)
                else:
                    manifest.update(_tokenspeed_plan(impl.fmha_impl, fixture.q.device))
                # Logical valid O/LSE payload, excluding the self destination
                # and protocol/transaction overhead; not measured link traffic.
                if is_dcp:
                    fused_output = candidate is not None and candidate.fused
                    wire_splits = candidate.splits if fused_output else 1
                    wire_element_bytes = candidate.buffers.output.element_size() if fused_output else 2
                    planned_rows = batch * 4 * wire_splits
                    output_rows = planned_rows
                    if fused_output:
                        # Empty splits publish neutral LSE but leave O unwritten.
                        # Count rows from actual Page-RR bounds, outside timing.
                        bounds = candidate.bounds.detach().cpu().long()
                        tiles = (bounds[:, -1] + 127) // 128
                        tiles_per_split = (tiles + wire_splits - 1) // wire_splits
                        starts = (tiles_per_split[:, None, None] * 128
                                  * torch.arange(wire_splits)[None, None, :])
                        output_rows = int((starts < bounds[:, :, None]).sum())
                    manifest["logical_remote_bytes_per_rank"] = (
                        (output_rows * 512 * wire_element_bytes + planned_rows * 4)
                        * 96 * (world - 1) // world
                    )
                    manifest["wire_output_rows"] = output_rows
                    manifest["wire_lse_rows"] = planned_rows
                else:
                    manifest["logical_remote_bytes_per_rank"] = 0
                if os.environ.get("MLA_BENCH_SAVE_PRODUCER") == "1":
                    # Save an already-validated local producer call for isolated
                    # profiling, without NCCL/peer mappings in the replay process.
                    assert is_dcp and candidate is not None and not candidate.fused
                    if rank == 0:
                        source = Path(importlib.import_module(candidate.__module__).__file__)
                        paths = [source, source.parent / "_fia2a/mla_fused_fp8.py",
                                 source.parent / "_fia2a/mla_helpers.py"]
                        torch.save(dict(
                            query=candidate.query.cpu(), kv=fixture.cache.kv_cache_base.cpu(),
                            table=candidate.table.cpu(), bounds=candidate.bounds.cpu(),
                            capacity=candidate.buffers.capacity, splits=candidate.splits,
                            page_size=candidate.page_size, softmax_scale=candidate.softmax_scale,
                            output_scale=candidate.output_scale, sm_count=candidate.sm_count,
                            output=candidate.buffers.output[:, :candidate.rows*candidate.splits].cpu(),
                            lse=candidate.buffers.lse[:, :candidate.rows*candidate.splits].cpu(),
                            source_sha256={str(path.relative_to(source.parent)):
                                hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                            manifest=manifest,
                        ), output_dir / "producer-inputs.pt")
                    manifest.update(measurement_kind="validated_producer_snapshot")
                    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                    print(f"MLA PRODUCER SNAPSHOT PASS rank={rank} {case_id}", flush=True)
                    del impl, fixture, result, forward, candidate
                    dist.barrier()
                    continue
                del fixture.canonical
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    forward()
                    forward()
                torch.cuda.current_stream().wait_stream(stream)
                original = impl.fmha_impl._forward_mla
                def burst(*args, **kwargs):
                    if "core_q_shape" not in manifest:
                        manifest.update(core_q_shape=list(args[0].shape), core_q_stride=list(args[0].stride()))
                    for _ in range(inner):
                        evict()
                        output = original(*args, **kwargs)
                        wait_peer_consumption()
                    return output
                impl.fmha_impl._forward_mla = burst
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        captured_output = forward()
                finally:
                    del impl.fmha_impl._forward_mla
                for _ in range(2):
                    graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured_output, fixture.expected, atol=2e-3, rtol=0.015)
                allocation = torch.cuda.memory_allocated()
                events = []
                for _ in range(5):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    events.append(begin.elapsed_time(end)*1000)
                assert torch.cuda.memory_allocated() == allocation
                manifest["planning_outer_event_us"] = events
            profile_settle = minimum_settle
            if settle_ms:
                # One common replay count keeps DCP collective order identical.
                # This preparation-only reduction never enters the measured graph.
                count = math.ceil(settle_ms*1000/statistics.median(events)) if requests else 0
                common_count = torch.tensor(count, dtype=torch.int32, device=torch.device('cuda', rank))
                dist.all_reduce(common_count, op=dist.ReduceOp.MAX)
                profile_settle = max(profile_settle, int(common_count.item()))
            manifest['profiled_settling_replays'] = profile_settle if profile_enabled else 0
            manifest['bare_settling_replays'] = profile_settle
            if requests:
                for _ in range(profile_settle):
                    graph.replay()
                manifest["bare_window_start_s"] = time.time()
                bare_events = []
                for _ in range(5):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    bare_events.append(begin.elapsed_time(end)*1000)
                manifest['bare_outer_event_us'] = bare_events
                manifest['bare_window_end_s'] = time.time()
            # Empty DP owners take part only in the outer rendezvous. They do
            # not run a B0 MLA or contribute invented timing samples.
            dist.barrier()
            if profile_enabled:
                diagnostic_events = []
                if requests and profile_events:
                    diagnostic_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                                         for _ in range(profile_settle+1)]
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    dist.barrier()
                    torch.cuda.synchronize()
                    if requests:
                        with torch.profiler.record_function("MLA_TOPOLOGY_SETTLE"):
                            for index in range(profile_settle):
                                if profile_events:
                                    diagnostic_events[index][0].record()
                                graph.replay()
                                if profile_events:
                                    diagnostic_events[index][1].record()
                    if requests:
                        with torch.profiler.record_function("MLA_TOPOLOGY_MEASURE"):
                            if profile_events:
                                diagnostic_events[-1][0].record()
                            graph.replay()
                            if profile_events:
                                diagnostic_events[-1][1].record()
                        torch.cuda.synchronize()
            if requests:
                if profile_enabled and profile_events:
                    durations = [begin.elapsed_time(end)*1000 for begin, end in diagnostic_events]
                    manifest["profiled_settle_outer_event_us"] = durations[:-1]
                    manifest["profiled_measure_outer_event_us"] = durations[-1]
                if profile_enabled:
                    profiler.export_chrome_trace(str(output_dir / "trace.json"))
                else:
                    manifest["measurement_kind"] = "unprofiled whole cold graph, including preparation/flush/projection"
                torch.testing.assert_close(captured_output, fixture.expected, atol=2e-3, rtol=0.015)
                graph.reset()
                graph = None
                del impl, original, fixture, result, captured_output, forward, burst, candidate
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            if rank == 0:
                (output_dir.parent/"resource-after.json").write_text(json.dumps(_resource_snapshot(), indent=2)+"\n")
            print(f"MLA BENCH PASS rank={rank} {arm} {case_id} active={bool(requests)}", flush=True)
            dist.barrier()
    except BaseException:
        error = traceback.format_exc()
        try:
            (Path(directory)/f'failure-rank{rank}.json').write_text(json.dumps(
                dict(rank=rank, case_id=locals().get('case_id'), manifest=locals().get('manifest'),
                     error=error), indent=2)+'\n')
            print(error, flush=True)
        finally:
            # CUDA/NCCL cleanup can wait for peers after a capture failure.
            # Notify the native parent before entering that cleanup.
            failed.set()
        raise
    finally:
        if graph is not None:
            graph.reset()
        mla_dcp_comm._communicators.clear()
        destroy_distributed_environment()


class MlaTopologyBenchmark(unittest.TestCase):
    def test_cold_production_workflow(self):
        topology = os.environ["MLA_BENCH_TOPOLOGY"]
        self.assertIn(topology, ("DP8", "TP8", "DCP8"))
        cases = json.loads(os.environ["MLA_BENCH_CASES"])
        self.assertTrue(cases)
        for case in cases:
            self.assertTrue(0 < case["B"] <= 128 and case["G"] >= 8192)
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        directory = os.environ["MLA_BENCH_OUTPUT"]
        mp.set_start_method("spawn", force=True)
        failed = mp.Event()
        ports, locks = PortManager().get_consecutive_ports(1)
        workers = [mp.Process(target=_worker, args=(rank, 8, ports[0], topology, cases, directory, failed),
                              name=f"mla-bench-rank-{rank}") for rank in range(8)]
        try:
            for worker in workers:
                worker.start()
            pending = list(workers)
            deadline = time.monotonic()+4800
            while pending:
                self.assertFalse(failed.is_set(), "MLA worker failed; see failure-rank*.json")
                for worker in pending[:]:
                    if worker.exitcode is not None:
                        worker.join()
                        self.assertEqual(worker.exitcode, 0, worker.name)
                        pending.remove(worker)
                self.assertLess(time.monotonic(), deadline, "MLA benchmark workers timed out")
                if pending:
                    time.sleep(.1)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                if worker.pid is not None:
                    worker.join(timeout=10)
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=10)
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
