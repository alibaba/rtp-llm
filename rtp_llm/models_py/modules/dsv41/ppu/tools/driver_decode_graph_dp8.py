#!/usr/bin/env python
"""Compare real DP8 decode Graph replay with the identical eager device body.

The checkpoint, all 40 layers, EP8 MegaMoE/W4A4, ENP8 GraphEngram and native
FlashMLA stay enabled. Two local requests are prefilled once, exported once
to native pages, and cloned into independent eager/Graph pools. Every replay
changes device tokens, Engram windows, positions, table rows and live count.
The default eight steps include 2 -> 1 -> 0 -> 2 live rows, request reordering,
ratio-2 phases, an eight-token page boundary and a real SWA ring overwrite.

This is capture fidelity evidence, with exact hidden and raw cache-byte
comparisons. It neither measures performance nor supplies an NVIDIA golden.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rtp_llm.models_py.modules.dsv41.ppu.tools import driver_decode_dp8 as dp8

_BATCH = 2
_PAGE_SPAN = 8
_LIVE_COUNTS = (2, 1, 0, 2)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=127)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=827)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.nproc != 8:
        parser.error("this fidelity driver requires DP8/EP8/ENP8")
    if args.tokens < 1 or args.steps < 8 or args.warmup < 2:
        parser.error("require --tokens >= 1, --steps >= 8 and --warmup >= 2")
    return args


def _native_pools(config, *, device, lengths, ring_span):
    import torch

    from rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache import NativePools

    class RingNativePools(NativePools):
        def get_seq_size_per_block(self, tag):
            if tag == "swa_kv":
                return ring_span
            return super().get_seq_size_per_block(tag)

    # The fixture's final block-table column contains reserved block zero.
    # Keep it beyond every reachable position, while retaining nonidentity
    # physical page ids for the full/global/indexer pools.
    columns = (config["max_position_embeddings"] + _PAGE_SPAN - 1) // _PAGE_SPAN
    native = RingNativePools(
        config, batch=_BATCH, blocks=_BATCH * (columns + 1), span=_PAGE_SPAN
    )
    for (owner, tag), handle in native.handles.items():
        pool = handle.kv_cache_base
        if tag == "swa_kv":
            # One physical ring per request, including raw positions 0 and
            # 130 in the SAME physical slot. Other pools retain eight-token
            # kernel pages; the bridge and Graph read this explicit geometry.
            pool = pool[: _BATCH + 1]
        handle.kv_cache_base = pool.to(device=device).contiguous()
    for tag, entry in native.inputs.items():
        kernel = entry.kv_cache_kernel_block_id_device.to(device=device).clone()
        physical = entry.kv_cache_block_id_device.to(device=device).clone()
        if tag == "swa_kv":
            physical.copy_(
                torch.arange(1, _BATCH + 1, device=device)[:, None].expand_as(physical)
            )
        entry.kv_cache_kernel_block_id_device = kernel
        entry.kv_cache_block_id_device = physical
        entry.is_prefill = False
        entry.input_lengths = torch.ones(_BATCH, dtype=torch.int32, device=device)
        entry.sequence_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
        entry.sequence_lengths_plus_1_device = entry.sequence_lengths + 1
    return native


def _pool_tensors(native):
    return {key: handle.kv_cache_base for key, handle in native.handles.items()}


def _table_bases(native):
    return {
        tag: (
            entry.kv_cache_kernel_block_id_device.clone(),
            entry.kv_cache_block_id_device.clone(),
        )
        for tag, entry in native.inputs.items()
    }


def _copy_pools(native, snapshot):
    for key, pool in _pool_tensors(native).items():
        pool.copy_(snapshot[key])


def _input_buffers(native, *, device, ngram):
    import torch

    return NS(
        input_ids=torch.zeros(_BATCH, dtype=torch.int32, device=device),
        engram_token_windows=torch.full(
            (_BATCH, ngram), -1, dtype=torch.int32, device=device
        ),
        attention_inputs=native.inputs,
    )


def _prepare_step(native, bases, impl, inputs, *, order, tokens, histories, live):
    """Scheduler-like updates are outside capture and preserve every address."""
    import torch

    device = inputs.input_ids.device
    row_ids = torch.tensor(order, dtype=torch.int64, device=device)
    positions = [len(histories[rid]) for rid in order]
    lengths = torch.tensor(positions, dtype=torch.int32, device=device)
    for tag, entry in native.inputs.items():
        kernel, physical = bases[tag]
        entry.kv_cache_kernel_block_id_device.copy_(kernel.index_select(0, row_ids))
        entry.kv_cache_block_id_device.copy_(physical.index_select(0, row_ids))
        entry.sequence_lengths.copy_(lengths)
        entry.sequence_lengths_plus_1_device.copy_(lengths + 1)
    windows = []
    width = inputs.engram_token_windows.shape[1]
    for row, rid in enumerate(order):
        recent = histories[rid][-(width - 1) :][::-1]
        windows.append(
            [tokens[row]] + recent + [-1] * (width - 1 - len(recent))
            if row < live
            else [-1] * width
        )
    inputs.input_ids.copy_(torch.tensor(tokens, dtype=torch.int32, device=device))
    inputs.engram_token_windows.copy_(
        torch.tensor(windows, dtype=torch.int32, device=device)
    )
    # Use the explicit persistent DEVICE scalar, as the native C++ runner does.
    impl.cuda_graph_valid_request_count_device.fill_(live)
    impl.prepare_cuda_graph(
        native.inputs,
        valid_request_count=impl.cuda_graph_valid_request_count_device,
    )
    return positions


def _inactive_snapshot(native, bases, request_ids):
    import torch

    snapshot = {}
    for key, pool in _pool_tensors(native).items():
        tag = key[1]
        # Full pools use kernel tables, rings use physical tables.
        table = bases[tag][1 if tag in ("swa_kv", "csa_state") else 0]
        rows = torch.tensor(request_ids, dtype=torch.int64, device=pool.device)
        blocks = table.index_select(0, rows).reshape(-1).unique()
        snapshot[key] = (blocks, pool.index_select(0, blocks).clone())
    return snapshot


def _byte_mismatches(left, right):
    import torch

    return int(torch.count_nonzero(left.view(torch.uint8) != right.view(torch.uint8)))


def _compare_pools(eager, captured):
    mismatches, compared = {}, 0
    for key, left in _pool_tensors(eager).items():
        right = _pool_tensors(captured)[key]
        count = _byte_mismatches(left, right)
        compared += left.numel() * left.element_size()
        if count:
            mismatches[f"{key[0]}:{key[1]}"] = count
    return {
        "bytes_compared": compared,
        "byte_mismatches": sum(mismatches.values()),
        "mismatches_by_pool": mismatches,
    }


def _inactive_mismatches(native, snapshot):
    total = 0
    pools = _pool_tensors(native)
    for key, (blocks, before) in snapshot.items():
        total += _byte_mismatches(pools[key].index_select(0, blocks), before)
    return total


def _compare_hidden(eager, captured, *, live):
    import torch

    finite = bool(torch.isfinite(eager).all() & torch.isfinite(captured).all())
    difference = (eager.float() - captured.float()).abs()
    return {
        "finite": finite,
        "exact": bool(torch.equal(eager, captured)),
        "elements_different": int(torch.count_nonzero(eager != captured)),
        "max_abs_error": float(difference.max()) if finite else None,
        "max_rel_error": (
            float((difference / eager.float().abs().clamp_min(1e-30)).max())
            if finite
            else None
        ),
        "inactive_output_zero": bool(
            torch.count_nonzero(eager[live:]) == 0
            and torch.count_nonzero(captured[live:]) == 0
        ),
    }


def _run_fidelity(args, *, rank, world, device, config, geometry, core, ops):
    import torch
    import torch.distributed as dist

    from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import PPUDecodeGraphImpl
    from rtp_llm.models_py.modules.dsv41.ppu.graph_decode import GraphDecodeRunner
    from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge

    generator = torch.Generator().manual_seed(args.seed + 1000 * rank)
    lengths = [args.tokens, args.tokens + 1]
    histories = [
        torch.randint(0, geometry.vocab_size, (length,), generator=generator).tolist()
        for length in lengths
    ]
    cu = [0, lengths[0], sum(lengths)]
    prefill = core.prefill(
        torch.tensor(histories[0] + histories[1], dtype=torch.int64, device=device),
        positions=torch.cat([torch.arange(length) for length in lengths]).to(device),
        cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
        request_ids=torch.repeat_interleave(
            torch.arange(_BATCH), torch.tensor(lengths)
        ).to(device),
        apply_lm_head=False,
    )
    prefill_finite = bool(torch.isfinite(prefill.hidden).all())
    ready = torch.tensor(int(prefill_finite), dtype=torch.int32, device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not int(ready):
        raise RuntimeError("real prefill produced non-finite hidden on a DP8 rank")

    maximum = max(lengths) + args.steps + 1
    if maximum > geometry.max_seq_len:
        raise ValueError("prefill plus replay exceeds the checkpoint position capacity")
    graph_config = {**config, "max_position_embeddings": maximum}
    ring_span = 1 << (maximum - 1).bit_length()
    eager = _native_pools(
        graph_config, device=device, lengths=lengths, ring_span=ring_span
    )
    captured = _native_pools(
        graph_config, device=device, lengths=lengths, ring_span=ring_span
    )
    # This is the ONLY logical/native export. No bridge imports or request
    # reconstruction occur in warmup, capture or the replay loop.
    NativeCacheBridge(graph_config, eager, eager.inputs, device=device).export_requests(
        core.cache, lengths
    )
    seed = {key: pool.clone() for key, pool in _pool_tensors(eager).items()}
    _copy_pools(captured, seed)
    if any(
        _pool_tensors(eager)[key].data_ptr() == pool.data_ptr()
        for key, pool in _pool_tensors(captured).items()
    ):
        raise RuntimeError("eager and Graph pools must have independent allocations")

    eager_bases, graph_bases = _table_bases(eager), _table_bases(captured)
    eager_impl = PPUDecodeGraphImpl(
        graph_config, eager, eager.inputs, device=device, batch_size=_BATCH
    )
    graph_impl = PPUDecodeGraphImpl(
        graph_config, captured, captured.inputs, device=device, batch_size=_BATCH
    )
    runner = GraphDecodeRunner(core, max_batch_size=_BATCH, dp_rank=rank, dp_size=world)
    eager_inputs = _input_buffers(
        eager, device=device, ngram=geometry.engram_max_ngram_size
    )
    graph_inputs = _input_buffers(
        captured, device=device, ngram=geometry.engram_max_ngram_size
    )
    initial_tokens = [history[-1] for history in histories]
    _prepare_step(
        captured,
        graph_bases,
        graph_impl,
        graph_inputs,
        order=[0, 1],
        tokens=initial_tokens,
        histories=histories,
        live=_BATCH,
    )
    # Match native capture: every exact-bucket backend is warmed on the SAME
    # stream that captures, including all distributed Engram/MegaMoE calls.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            runner.forward(graph_inputs, graph_impl)
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph, stream=stream):
        graph_output = runner.forward(graph_inputs, graph_impl)
    torch.cuda.synchronize(device)
    # Warmup/capture may write caches; begin BOTH sequences at the same real
    # prefill state while preserving the addresses recorded by the Graph.
    _copy_pools(eager, seed)
    _copy_pools(captured, seed)
    stream.wait_stream(torch.cuda.current_stream(device))
    seed.clear()

    coverage = {
        "ratio2_even": False,
        "ratio2_odd": False,
        "global_page_boundary": False,
        "swa_same_slot_wrap": False,
        "request_reordered": False,
        "live_counts": [],
    }
    reports = []
    swa_owner = next(
        owner
        for owner, tag in graph_impl.cache.geometry.pool_keys()
        if tag == "swa_kv"
    )
    swa_entries = graph_impl.cache.entries["swa_kv"]
    for step in range(args.steps):
        live = _LIVE_COUNTS[step % len(_LIVE_COUNTS)]
        order = [1, 0] if (step // 3) % 2 else [0, 1]
        tokens = torch.randint(
            0, geometry.vocab_size, (_BATCH,), generator=generator
        ).tolist()
        with torch.cuda.stream(stream):
            positions = _prepare_step(
                eager,
                eager_bases,
                eager_impl,
                eager_inputs,
                order=order,
                tokens=tokens,
                histories=histories,
                live=live,
            )
            _prepare_step(
                captured,
                graph_bases,
                graph_impl,
                graph_inputs,
                order=order,
                tokens=tokens,
                histories=histories,
                live=live,
            )
            inactive = order[live:]
            eager_before = _inactive_snapshot(eager, eager_bases, inactive)
            graph_before = _inactive_snapshot(captured, graph_bases, inactive)
            runner.validate()
            eager_output = runner.forward(eager_inputs, eager_impl)
            # Actual replay: Python model.forward is deliberately not called.
            graph.replay()
        torch.cuda.synchronize(device)
        hidden = _compare_hidden(eager_output, graph_output, live=live)
        cache = _compare_pools(eager, captured)
        inactive_bytes = (
            _inactive_mismatches(eager, eager_before)
            + _inactive_mismatches(captured, graph_before)
        )
        metadata_match = bool(
            torch.equal(
                graph_impl.metadata.positions,
                torch.tensor(positions, dtype=torch.int64, device=device),
            )
            and torch.equal(
                graph_impl.metadata.valid_mask,
                torch.arange(_BATCH, device=device) < live,
            )
        )
        # Verify the actual native address alias, not just raw-position modulo.
        raw = torch.tensor(positions, dtype=torch.int64, device=device)
        slots = graph_impl.cache.slots(swa_owner, "swa_kv", raw)
        previous_slots = graph_impl.cache.slots(swa_owner, "swa_kv", raw - swa_entries)
        wrap = bool(
            (
                (raw >= swa_entries)
                & (raw.remainder(swa_entries) == 0)
                & (torch.arange(_BATCH, device=device) < live)
                & (slots >= 0)
                & (slots == previous_slots)
            ).any()
        )
        coverage["swa_same_slot_wrap"] |= wrap
        coverage["request_reordered"] |= order != [0, 1]
        coverage["live_counts"].append(live)
        for position in positions[:live]:
            coverage["ratio2_even"] |= position % 2 == 0
            coverage["ratio2_odd"] |= position % 2 == 1
            coverage["global_page_boundary"] |= (
                position > 0 and position % _PAGE_SPAN == 0
            )
        ok = (
            hidden["finite"]
            and hidden["exact"]
            and hidden["inactive_output_zero"]
            and cache["byte_mismatches"] == 0
            and inactive_bytes == 0
            and metadata_match
        )
        reports.append(
            {
                "step": step,
                "request_order": order,
                "positions": positions,
                "tokens": tokens,
                "live_count": live,
                "hidden": hidden,
                "cache": cache,
                "inactive_cache_byte_mismatches": inactive_bytes,
                "metadata_matches_live_inputs": metadata_match,
                "swa_same_slot_wrap": wrap,
                "ok": bool(ok),
            }
        )
        print(
            f"[graph DP8 rank {rank}] replay {step}: live={live} positions={positions} "
            f"hidden_error={hidden['max_abs_error']} "
            f"cache_bytes={cache['byte_mismatches']} ok={ok}",
            flush=True,
        )
        for row, rid in enumerate(order[:live]):
            histories[rid].append(tokens[row])
        # The next update runs on the capture stream; order it after the
        # default-stream comparison kernels that read these persistent pools.
        stream.wait_stream(torch.cuda.current_stream(device))

    torch.cuda.synchronize(device)
    coverage_ok = all(value for key, value in coverage.items() if key != "live_counts")
    return {
        "prefill_finite": prefill_finite,
        "prefill_lengths": lengths,
        "local_request_ids": [0, 1],
        "native_exports": 1,
        "captured_batch_size": _BATCH,
        "graph_cache_max_seq_len": maximum,
        "global_kernel_page_span": _PAGE_SPAN,
        "swa_physical_span": ring_span,
        "swa_entries": swa_entries,
        "warmup_forwards": args.warmup,
        "capture_forwards": 1,
        "replay_count": len(reports),
        "eager_decode_forwards": len(reports),
        "hidden_tolerance": {"atol": 0, "rtol": 0},
        "coverage": coverage,
        "coverage_ok": coverage_ok,
        "steps": reports,
        "ok": coverage_ok
        and len(reports) == args.steps
        and all(r["ok"] for r in reports),
    }


def worker_main(args):
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    from rtp_llm.models_py.modules.dsv41.ppu.engram import DistributedEnpmGroup
    from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
    from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
    from rtp_llm.models_py.modules.dsv41.ppu.weight import V41Geometry, load_ppu_weights

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    payload = {"rank": rank, "world_size": world, "device": str(device), "ok": False}
    path = Path(args.output).resolve().parent / f"graph_dp8.rank{rank}.json"
    ops = None
    completed = False
    try:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.timeout))
        config = json.loads(Path(args.config_path).read_text())
        geometry = V41Geometry.from_config(config)
        if world != 8 or geometry.num_layers != 40 or not geometry.engram_enabled():
            raise ValueError(
                "require full 40-layer DSV4.1 with DP8/EP8/ENP8 and Engram"
            )
        enpm = DistributedEnpmGroup(dist.group.WORLD)
        ops = PPUOps(
            ep_rank=rank,
            ep_size=world,
            num_routed_experts=geometry.n_routed_experts,
            num_local_experts=geometry.num_local_experts(world),
            hidden_size=geometry.hidden_size,
            inter_dim=geometry.moe_inter_dim,
            swiglu_limit=geometry.swiglu_limit,
            mhc_config=geometry.mhc_config(),
            sm_scale=geometry.sm_scale(),
            window=geometry.window,
            enpm_group=enpm,
            deepep_group=dist.group.WORLD,
        )
        if ops.moe_engine.backend != "megamoe" or ops.moe_engine.precision != "w4a4":
            raise ValueError("fidelity driver requires existing MegaMoE/W4A4")
        bundle = load_ppu_weights(
            config,
            args.checkpoint_path,
            ep_rank=rank,
            ep_size=world,
            enp_rank=rank,
            enp_size=world,
            device=device,
        )
        bundle.bind(ops, enpm_group=enpm)
        ops = dp8._DP8Ops(ops, device)
        core = PPUModelCore(config, bundle, ops, device=device)
        payload.update(
            {
                "layers": geometry.num_layers,
                "hidden_size": geometry.hidden_size,
                "ep_size": world,
                "dp_size": world,
                "enp_size": world,
                "checkpoint": str(args.checkpoint_path),
                "device_name": torch.cuda.get_device_name(device),
                "engram_layer_ids": list(geometry.engram_layer_ids),
                "kv_source_layer_ids": list(geometry.kv_source_layer_ids),
                "moe_backend": ops.moe_engine.backend,
                "moe_precision": ops.moe_engine.precision,
                "attention_backend": "native flash_mla_sparse_fwd",
                "engram_backend": "GraphEngram device hash / ENP8 / mapped host shard",
                "mhc_backend": "auto",
            }
        )
        payload.update(
            _run_fidelity(
                args,
                rank=rank,
                world=world,
                device=device,
                config=config,
                geometry=geometry,
                core=core,
                ops=ops,
            )
        )
        payload["moe"] = ops.moe_engine.execution_stats()
        payload["memory"] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
        completed = True
    except Exception:
        payload["ok"] = False
        payload["error"] = traceback.format_exc()
        print(payload["error"], file=sys.stderr, flush=True)
    finally:
        # Only a completed, collectively symmetric loop can safely drain the
        # communication allocation. On an exception the parent terminates its
        # own remaining workers and preserves the per-rank error report.
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        if completed:
            try:
                torch.cuda.synchronize(device)
                dist.barrier()
                ops.moe_engine.close()
                dist.destroy_process_group()
                payload["shutdown"] = {
                    "mega_buffer_closed": True,
                    "process_group_destroyed": True,
                }
            except Exception:
                payload["ok"] = False
                payload["shutdown_error"] = traceback.format_exc()
            path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def _aggregate(args, processes):
    directory = Path(args.output).resolve().parent
    ranks = []
    for rank in range(args.nproc):
        path = directory / f"graph_dp8.rank{rank}.json"
        if path.is_file():
            ranks.append(json.loads(path.read_text()))
    steps = [step for rank in ranks for step in rank.get("steps", [])]
    errors = [
        step["hidden"]["max_abs_error"]
        for step in steps
        if step["hidden"]["max_abs_error"] is not None
    ]
    document = {
        "world_size": args.nproc,
        "ranks_reported": len(ranks),
        "layers": sorted({rank.get("layers", 0) for rank in ranks}),
        "moe_backends": sorted({rank.get("moe_backend", "unknown") for rank in ranks}),
        "moe_precisions": sorted(
            {rank.get("moe_precision", "unknown") for rank in ranks}
        ),
        "replay_count_by_rank": {
            str(rank["rank"]): rank.get("replay_count", 0) for rank in ranks
        },
        "max_hidden_abs_error": max(errors) if errors else None,
        "cache_byte_mismatches": sum(
            step["cache"]["byte_mismatches"] for step in steps
        ),
        "inactive_cache_byte_mismatches": sum(
            step["inactive_cache_byte_mismatches"] for step in steps
        ),
        "finite": bool(steps) and all(step["hidden"]["finite"] for step in steps),
        "worker_exit_codes": [process.returncode for process in processes],
        "per_rank": ranks,
        "all_ranks_ok": len(ranks) == args.nproc
        and all(rank.get("ok") for rank in ranks)
        and all(process.returncode == 0 for process in processes),
    }
    Path(args.output).write_text(json.dumps(document, indent=2, sort_keys=True))
    print(
        json.dumps(
            {key: value for key, value in document.items() if key != "per_rank"},
            indent=2,
        ),
        flush=True,
    )
    return document


def parent_main(args):
    directory = Path(args.output).resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    args.config_path = dp8._prepare_config(args)
    env = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(dp8._free_port()),
        "WORLD_SIZE": str(args.nproc),
        "RTP_V41_PPU_MOE_BACKEND": "megamoe",
        "RTP_V41_PPU_MOE_PRECISION": "w4a4",
    }
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    for name in (
        "config_path",
        "checkpoint_path",
        "output",
        "tokens",
        "steps",
        "warmup",
        "nproc",
        "seed",
        "timeout",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    processes = []
    try:
        for rank in range(args.nproc):
            (directory / f"graph_dp8.rank{rank}.json").unlink(missing_ok=True)
        for rank in range(args.nproc):
            rank_env = {**env, "RANK": str(rank), "LOCAL_RANK": str(rank)}
            with (directory / f"graph_dp8.rank{rank}.log").open("w") as log:
                processes.append(
                    subprocess.Popen(
                        command, env=rank_env, stdout=log, stderr=subprocess.STDOUT
                    )
                )
        deadline = time.monotonic() + args.timeout
        while any(process.poll() is None for process in processes):
            if any(process.returncode not in (None, 0) for process in processes):
                break
            if time.monotonic() >= deadline:
                print(
                    f"Graph DP8 timeout after {args.timeout}s",
                    file=sys.stderr,
                    flush=True,
                )
                break
            time.sleep(0.5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    document = _aggregate(args, processes)
    return 0 if document["all_ranks_ok"] else 1


def main(argv=None):
    args = _parse_args(argv)
    return worker_main(args) if args.worker else parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
