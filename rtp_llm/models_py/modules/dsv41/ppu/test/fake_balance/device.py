"""Eight PPU ranks: actual grouped/MegaMoE receive loads and Graph replay.

Uses small synthetic expert weights with real 384-expert/top-6 EP8 geometry,
the production PPUMoE adapter, existing native fake-balance op and real native
communication/GEMMs. This qualifies routing, not model output accuracy or TPS.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE, pad_topk_for_dispatch

WORLD, EXPERTS, LOCAL, TOP_K, HIDDEN, INTERMEDIATE = 8, 384, 48, 6, 512, 128


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_engine(rank, backend, enabled):
    os.environ["RTP_V41_PPU_MOE_BACKEND"] = backend
    os.environ["RTP_V41_PPU_MOE_PRECISION"] = "w4a4" if backend == "megamoe" else "w4a8"
    value = PPUMoE(
        num_routed_experts=EXPERTS,
        num_local_experts=LOCAL,
        ep_rank=rank,
        ep_size=WORLD,
        hidden_size=HIDDEN,
        inter_dim=INTERMEDIATE,
        group=dist.group.WORLD,
    )
    value.configure_fake_balance(enabled, top_k=TOP_K, dp_rank=rank, dp_size=WORLD)
    generator = torch.Generator().manual_seed(730 + rank)
    pairs = []
    for n, k in ((2 * INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE)):
        packed = torch.randint(
            256, (LOCAL, n, k // 2), generator=generator, dtype=torch.uint8
        ).cuda()
        scales = torch.full((LOCAL, n, k // 32), 120, dtype=torch.uint8, device="cuda")
        pairs.append((packed, scales))
    value.bind_expert_weights(0, *pairs)
    return value


def inputs(rows, live):
    x = torch.full((rows, HIDDEN), 0.125, dtype=torch.bfloat16, device="cuda")
    ids = torch.full((rows, 8), -1, dtype=torch.int64, device="cuda")
    ids[:, :TOP_K] = torch.arange(TOP_K, device="cuda")
    weights = torch.zeros((rows, 8), dtype=torch.float32, device="cuda")
    weights[:, :TOP_K] = 0.25
    ids[live:] = -1
    weights[live:] = 0
    return x, ids, weights


def receive_counts(engine, grouped_counts):
    if engine.backend == "megamoe":
        return engine._mega_recv_stats.clone().long()
    return grouped_counts.clone()


def validate_counts(engine, counts, name, enabled, live_counts):
    gathered = [torch.empty_like(counts) for _ in range(WORLD)]
    dist.all_gather(gathered, counts)
    per_expert = torch.cat(gathered).cpu()
    per_rank = per_expert.reshape(WORLD, LOCAL).sum(1)
    assignments = sum(live_counts) * TOP_K
    assert int(per_expert.sum()) == assignments, (name, per_expert.tolist())
    if enabled:
        assert int(per_rank.max() - per_rank.min()) <= 1, (name, per_rank.tolist())
        assert int(per_expert.max() - per_expert.min()) <= 1, (
            name,
            per_expert.tolist(),
        )
    else:
        # All unmodified inputs route to experts 0..5 on owner rank zero.
        assert per_rank.tolist() == [assignments] + [0] * 7, (name, per_rank.tolist())
        assert per_expert[:TOP_K].tolist() == [sum(live_counts)] * TOP_K
    return {
        "case": name,
        "backend": engine.backend,
        "precision": engine.precision,
        "fake_balance_expert": enabled,
        "live_rows_per_source_rank": live_counts,
        "received_assignments_per_rank": per_rank.tolist(),
        "received_assignments_per_expert": per_expert.tolist(),
        "total_assignments": assignments,
        "status": "PASS",
    }


@torch.no_grad()
def verify_prefill_route_pack(rank, directory):
    """Real native-pattern/Triton oracle with actual DP preparation ownership.

    No expert weights or model outputs are compared here. The existing cases
    below retain their independent real communication/Graph coverage.
    """
    os.environ['RTP_V41_PPU_MOE_BACKEND'] = 'megamoe'
    os.environ['RTP_V41_PPU_MOE_PRECISION'] = 'w4a4'
    value = PPUMoE(ep_size=WORLD, ep_rank=rank, hidden_size=5120,
                   inter_dim=2304, group=dist.group.WORLD)
    value.configure_fake_balance(True, top_k=6, dp_rank=rank, dp_size=WORLD)
    device = torch.device('cuda', rank)
    cases, timings = [], []
    receipt = None
    try:
        for capacity, row_counts in ((128, (1, 7, 127, 128)), (4096, (140, 4096, 3))):
            receipt = value.prepare_dp_prefill_forward(device, capacity, supported=True)
            if receipt is None:
                raise AssertionError('real DP preparation was not admitted')
            for rows in row_counts:
                _, ids, weights = inputs(rows, rows)
                ids[::2, 1] = -7
                ids[0, 3] = torch.iinfo(torch.int64).min
                ids[:, 6:] = 93
                weights[:, :6] = float('nan')
                snapshots = ids.clone(), weights.clone()
                expected = pad_topk_for_dispatch(*value._apply_fake_balance(ids, weights))
                actual = value._try_prefill_fake_balance(ids, weights, receipt)
                if actual is None:
                    raise AssertionError('eligible PPU route pack did not launch')
                for new, old in zip(actual, expected):
                    if not torch.equal(new.contiguous().view(torch.uint8), old.contiguous().view(torch.uint8)):
                        raise AssertionError('native legacy and cached Triton route bytes differ')
                    if torch._C._is_alias_of(new, ids) or torch._C._is_alias_of(new, weights):
                        raise AssertionError('route pack exposed an input alias')
                if not torch.equal(ids, snapshots[0]) or not torch.equal(
                    weights.view(torch.uint8), snapshots[1].view(torch.uint8),
                ):
                    raise AssertionError('route pack mutated its source')
                cases.append(dict(capacity=capacity, rows=rows, exact_route_bytes=True))
            value.finish_dp_prefill_forward(receipt)
            try:
                value._try_prefill_fake_balance(ids, weights, receipt)
            except ValueError as error:
                if 'stale or insufficient' not in str(error):
                    raise
            else:
                raise AssertionError('an expired forward receipt must be rejected')
        if value._prefill_fake_pattern_builds != 2:
            raise AssertionError('native pattern must build once per capacity generation')
        receipt = value.prepare_dp_prefill_forward(device, 4096, supported=True)
        pattern = value._prefill_fake_pattern
        if pattern.ids.is_inference() or pattern.weights.is_inference():
            raise AssertionError('cached pattern requires mutation-tracked tensors')
        for live in (0, 17, 140):
            _, ids, weights = inputs(140, live)
            expected = pad_topk_for_dispatch(*value._apply_fake_balance(ids, weights))
            actual = value._try_prefill_fake_balance(ids, weights, receipt)
            if actual is None or any(not torch.equal(new.view(torch.uint8), old.view(torch.uint8))
                                     for new, old in zip(actual, expected)):
                raise AssertionError('new-epoch validity changes differ from the native oracle')
        for rows in (140, 4096):
            _, ids, weights = inputs(rows, rows)
            legacy = lambda: value._apply_fake_balance(ids, weights)
            fused = lambda: value._try_prefill_fake_balance(ids, weights, receipt)
            for _ in range(5):
                legacy()
                if fused() is None:
                    raise AssertionError('route pack declined after warmup')
            torch.cuda.synchronize(device)
            observations = dict(rows=rows, iterations=50, legacy_wall_us=[], fused_wall_us=[])
            for repeat in range(3):
                arms = (('legacy', legacy), ('fused', fused))
                for label, operation in arms if repeat % 2 == 0 else reversed(arms):
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter_ns()
                    for _ in range(observations['iterations']):
                        operation()
                    torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter_ns() - begin) / observations['iterations'] / 1000
                    observations[label + '_wall_us'].append(elapsed)
            timings.append(observations)
        # One warmed pair supplies per-chain launch evidence, separate from
        # the clean wall loops and from the full-model benchmark boundary.
        _, ids, weights = inputs(140, 140)
        profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA])
        with profile:
            with torch.profiler.record_function('fake_routes.legacy'):
                value._apply_fake_balance(ids, weights)
            with torch.profiler.record_function('fake_routes.cached_mask_pad'):
                value._try_prefill_fake_balance(ids, weights, receipt)
            torch.cuda.synchronize(device)
        trace = directory / f'route-chain-rank-{rank}.trace.json'
        profile.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())['traceEvents']
        names = [event['name'] for event in events if event.get('cat') == 'kernel']
        if sum('_mask_fake_routes_kernel' in name for name in names) != 1:
            raise AssertionError('exactly one cached mask/pad kernel must execute')
        if sum('_simulate_balanced_routing_kernel' in name for name in names) != 1:
            raise AssertionError('compact override must execute exactly one fused kernel')
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            if value._try_prefill_fake_balance(ids, weights, receipt) is not None:
                raise AssertionError('a nondefault stream must retain the legacy path')
        torch.cuda.current_stream(device).wait_stream(side)
        pattern.ids.add_(1).sub_(1)
        if value._try_prefill_fake_balance(ids, weights, receipt) is not None:
            raise AssertionError('a mutated pattern must not be reused or reminted')
        if value._prefill_fake_pattern_builds != 2:
            raise AssertionError('same-capacity forwards must reuse the same native pattern')
        return dict(cases=cases, pattern_builds=value._prefill_fake_pattern_builds,
                    pack_calls=value._prefill_fake_pattern_calls, timings=timings,
                    trace=str(trace), kernel_names=names,
                    scope='discrete route bytes and host/device route-chain time; not model numerics or TPS')
    finally:
        value.finish_dp_prefill_forward(receipt)
        value.close()
        if value._prefill_fake_pattern is not None:
            raise AssertionError('close retained the route pattern')
        dist.barrier()


@torch.no_grad()
def verify_compact_balance_kernel(rank, directory):
    """Compare fused top6/top3 routes to the independent existing native op."""
    from rtp_llm.models_py.modules.base.cuda.select_topk import FakeBalanceExpert

    results = []
    os.environ["RTP_V41_PPU_MOE_BACKEND"] = "megamoe"
    os.environ["RTP_V41_PPU_MOE_PRECISION"] = "w4a4"
    for experts, top_k in ((384, 6), (128, 3)):
        value = PPUMoE(num_routed_experts=experts, num_local_experts=experts // WORLD,
                       ep_size=WORLD, ep_rank=rank, top_k=top_k)
        value.configure_fake_balance(True, top_k=top_k, dp_rank=rank, dp_size=WORLD)
        oracle = FakeBalanceExpert(experts, top_k, rank, WORLD, WORLD)
        for dtype in (torch.int32, torch.int64):
            for slots in (top_k, 8):
                for rows in (1, 8, 32):
                    storage = torch.zeros((rows * 2, slots * 2), dtype=dtype, device="cuda")
                    ids = storage[::2, ::2]
                    ids[::2, 0] = -7
                    ids[-1] = -1
                    ids[0, top_k - 1] = torch.iinfo(dtype).min
                    weights = torch.full(ids.shape, float("nan"), device="cuda")
                    original = ids.clone()
                    compact = ids[:, :top_k].clone(memory_format=torch.contiguous_format)
                    expected_weights = torch.empty(compact.shape, device="cuda")
                    oracle(compact, expected_weights)
                    expected_weights.mul_(1.0 / top_k)
                    valid = ids[:, :top_k] >= 0
                    expected_ids = torch.where(valid, compact, ids[:, :top_k])
                    expected_weights = torch.where(valid, expected_weights, 0)
                    actual = value._apply_fake_balance(ids, weights)
                    torch.testing.assert_close(actual[0], expected_ids, rtol=0, atol=0)
                    torch.testing.assert_close(actual[1], expected_weights, rtol=0, atol=0)
                    assert actual[0].is_contiguous() and actual[1].is_contiguous()
                    assert not torch._C._is_alias_of(actual[0], ids)
                    assert torch.equal(ids, original) and torch.isnan(weights).all()
                    results.append(dict(path="compact_fake_native_oracle", experts=experts,
                                        top_k=top_k, rows=rows, input_slots=slots,
                                        dtype=str(dtype), status="PASS"))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                value._apply_fake_balance(ids, weights)
                torch.cuda.synchronize()
            trace = directory / f"compact-top{top_k}-{dtype}-rank{rank}.trace.json"
            profile.export_chrome_trace(str(trace))
            names = [event["name"] for event in json.loads(trace.read_text())["traceEvents"]
                     if event.get("cat") == "kernel"]
            assert len(names) == 1 and "_simulate_balanced_routing_kernel" in names[0], names
    return results


@torch.no_grad()
def verify_shared_graph_balance(rank):
    """Actual fixed-workspace dispatch, target/draft geometry and live masks."""
    results = []
    hidden, intermediate = 5120, 2304
    os.environ["RTP_V41_PPU_MOE_BACKEND"] = "megamoe"
    os.environ["RTP_V41_PPU_MOE_PRECISION"] = "w4a4"
    for experts, top_k, rows in ((384, 6, 8), (384, 6, 32), (128, 3, 24)):
        local = experts // WORLD
        value = PPUMoE(num_routed_experts=experts, num_local_experts=local,
                       ep_rank=rank, ep_size=WORLD, hidden_size=hidden,
                       inter_dim=intermediate, top_k=top_k, group=dist.group.WORLD)
        value.configure_fake_balance(True, top_k=top_k, dp_rank=rank, dp_size=WORLD)
        generator = torch.Generator().manual_seed(11600 + rank)
        pairs = []
        for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
            packed = torch.randint(256, (local, n, k // 2), generator=generator,
                                   dtype=torch.uint8).cuda()
            scale = torch.full((local, n, k // 32), 120, dtype=torch.uint8, device="cuda")
            pairs.append((packed, scale))
        value.bind_expert_weights(0, *pairs)
        x = torch.full((rows, hidden), 0.125, dtype=torch.bfloat16, device="cuda")
        ids = torch.full((rows, 8), -1, dtype=torch.int64, device="cuda")
        ids[:, :top_k] = torch.arange(top_k, device="cuda")
        weights = torch.zeros((rows, 8), dtype=torch.float32, device="cuda")
        weights[:, :top_k] = 0.25
        try:
            value.prepare_shared_graph(x, (0,))
            assert value._mega_buffer.num_topk == top_k
            assert value._mega_buffer.topk_idx.shape[1] == top_k
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    value.forward_shared_graph(x, ids, weights, layer_id=0)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = value.forward_shared_graph(x, ids, weights, layer_id=0)
            for live, one_busy in ((rows, False), (rows // 2, False), (0, False), (rows, True), (rows, False)):
                actual_live = live if not one_busy or rank == 0 else 0
                ids.fill_(-1)
                ids[:actual_live, :top_k] = torch.arange(top_k, device="cuda")
                weights.zero_()
                weights[:actual_live, :top_k] = 0.5
                x.fill_(0.0625 if live != rows else 0.25)
                before = value._mega_recv_stats.clone()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.isfinite(output).all()
                assert torch.count_nonzero(output[actual_live:]) == 0
                counts = value._mega_recv_stats.long() - before.long()
                gathered = [torch.empty_like(counts) for _ in range(WORLD)]
                dist.all_gather(gathered, counts)
                per_expert = torch.cat(gathered).cpu()
                per_rank = per_expert.reshape(WORLD, local).sum(1)
                assignments = live * top_k * (1 if one_busy else WORLD)
                assert int(per_expert.sum()) == assignments
                assert int(per_rank.max() - per_rank.min()) <= 1
                assert int(per_expert.max() - per_expert.min()) <= 1
                results.append(dict(experts=experts, top_k=top_k, rows=rows,
                                    live=live, one_busy=one_busy,
                                    received_assignments_per_rank=per_rank.tolist(),
                                    total_assignments=assignments,
                                    native_top_k=value._mega_buffer.num_topk,
                                    path="forward_shared_graph", status="PASS"))
        finally:
            value.close()
            dist.barrier()
    return results


def worker(args):
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=args.rendezvous,
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    results = []
    try:
        if args.shared_graph_only:
            results = verify_compact_balance_kernel(rank, args.output)
            results.extend(verify_shared_graph_balance(rank))
            write(args.output / f"rank-{rank}.json", dict(
                rank=rank, status="PASS", cases=results,
                scope="shared_graph_fake_balance", torch=torch.__version__,
                device=torch.cuda.get_device_name(rank),
            ))
            return
        prefill_route_pack = verify_prefill_route_pack(rank, args.output)
        for backend in (() if args.route_pack_only else ("grouped", "megamoe")):
            for enabled in (False, True):
                engine = make_engine(rank, backend, enabled)
                grouped_counts = torch.zeros(LOCAL, dtype=torch.int64, device="cuda")
                local_compute = engine._local_compute

                def counted_compute(x, ids, weights, counts, layer_id):
                    # Count the actual DeepEP receive result without replacing
                    # communication or computation. Diagnostics are outside TPS.
                    grouped_counts.add_(torch.as_tensor(counts, device="cuda"))
                    return local_compute(x, ids, weights, counts, layer_id)

                engine._local_compute = counted_compute
                try:
                    for rows, one_busy in (
                        (8, False),
                        (16, False),
                        (64, False),
                        (256, False),
                        (8, True),
                    ):
                        live = rows if not one_busy or rank == 0 else 0
                        x, ids, weights = inputs(live, live)
                        before = (
                            receive_counts(engine, grouped_counts)
                            if engine._mega_recv_stats is not None
                            or backend == "grouped"
                            else torch.zeros_like(grouped_counts)
                        )
                        output = engine(x, ids, weights, layer_id=0)
                        torch.cuda.synchronize()
                        assert torch.isfinite(
                            output
                        ).all(), "non-finite native MoE output"
                        counts = receive_counts(engine, grouped_counts) - before
                        live_counts = [rows] + [0] * 7 if one_busy else [rows] * WORLD
                        results.append(
                            validate_counts(
                                engine,
                                counts,
                                f"eager-b{rows}-idle{one_busy}",
                                enabled,
                                live_counts,
                            )
                        )
                    if backend == "megamoe":
                        # A static 16-row capture must honor masks changed on
                        # replay. Warm up all native operations before capture.
                        x, ids, weights = inputs(16, 16)
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(2):
                                engine(x, ids, weights, layer_id=0)
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            graph_output = engine(x, ids, weights, layer_id=0)
                        for live in (8, 16, 0, 8):
                            new_x, new_ids, new_weights = inputs(16, live)
                            x.copy_(new_x)
                            ids.copy_(new_ids)
                            weights.copy_(new_weights)
                            before = engine._mega_recv_stats.clone()
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.isfinite(graph_output).all()
                            assert torch.count_nonzero(graph_output[live:]) == 0
                            counts = engine._mega_recv_stats.long() - before.long()
                            results.append(
                                validate_counts(
                                    engine,
                                    counts,
                                    f"graph-b16-live{live}",
                                    enabled,
                                    [live] * WORLD,
                                )
                            )
                finally:
                    engine.close()
                    dist.barrier()
        write(
            args.output / f"rank-{rank}.json",
            {
                "rank": rank,
                "status": "PASS",
                "cases": results,
                "prefill_route_pack": prefill_route_pack,
                "scope": "route_pack_only" if args.route_pack_only else "full_fake_balance_device",
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(rank),
            },
        )
    finally:
        PPUMoE.close_mega_resources()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rendezvous")
    args = parser.parse_args()
    # Separate discrete route validation from the broader legacy DeepEP and
    # Graph suite. The default continues to run every existing backend case.
    args.route_pack_only = os.environ.get("RTP_V41_TEST_FAKE_ROUTE_PACK_ONLY", "0") == "1"
    args.shared_graph_only = os.environ.get("RTP_V41_TEST_FAKE_SHARED_GRAPH_ONLY", "0") == "1"
    if args.route_pack_only and args.shared_graph_only:
        raise ValueError("choose one focused fake-balance device scope")
    if args.worker:
        worker(args)
        return
    if (
        torch.cuda.device_count() != WORLD
        or "ZW-M890P" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("this test requires eight allocated ZW-M890P devices")
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    children, logs = [], []
    started = time.time()
    try:
        for rank in range(WORLD):
            log = (args.output / f"rank-{rank}.log").open("x")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        __spec__.name,
                        "--worker",
                        "--output",
                        str(args.output),
                        "--rendezvous",
                        f"tcp://127.0.0.1:{port}",
                    ],
                    env={**os.environ, "RANK": str(rank), "WORLD_SIZE": str(WORLD)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 1500
        codes = [
            child.wait(timeout=max(1, deadline - time.monotonic()))
            for child in children
        ]
        if any(codes):
            raise RuntimeError(f"PPU rank failures: {codes}; see rank logs")
        ranks = [
            json.loads((args.output / f"rank-{rank}.json").read_text())
            for rank in range(WORLD)
        ]
        assert all(value["status"] == "PASS" for value in ranks)
        assert all(value["cases"] == ranks[0]["cases"] for value in ranks)
        write(
            args.output / "summary.json",
            {
                "status": "PASS",
                "ranks": WORLD,
                "cases": ranks[0]["cases"],
                "elapsed_seconds": time.time() - started,
                "workload": (
                    "native fake-route/Triton bitwise parity, lifecycle and route-chain time; not MoE output accuracy"
                    if args.route_pack_only else
                    "real native EP8 MoE; synthetic small weights, 384 experts/top-6; not full-model performance"
                ),
                "scope": "route_pack_only" if args.route_pack_only else "full_fake_balance_device",
            },
        )
        print(
            json.dumps(
                {"status": "PASS", "ranks": WORLD, "cases": len(ranks[0]["cases"])}
            )
        )
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
