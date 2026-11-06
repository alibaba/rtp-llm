#!/usr/bin/env python
"""Offline 8-rank CP8 V4.1 PPU prefill driver (L2 wave 2).

Runs the zigzag context-parallel prefill over a real (or synthetic)
checkpoint on 8 PPU ranks, reusing the ``hw_probe.py`` self-spawn launcher
pattern (one process per ``cuda:RANK`` joined by
``dist.init_process_group(backend="nccl")`` — PCCL on this stack):

* CP8: every rank runs ``prefill.run_prefill`` over its zigzag chunk (see
  ``ppu/cp.py``); the compressor/SWA KV inputs are all-gathered and restored
  to original model order; the final hidden is restored on every rank.
* EP8: routed MoE through real DeepEP normal dispatch/combine
  (``PPUOps(..., deepep_group=<world>)``).
* ENP8: Engram lookup through ``DistributedEnpmGroup`` over the same world
  (each rank's pinned host shard, ~23.6 GiB for the real checkpoint).
* rank 0 writes the next-token logits summary, per-rank memory/time stats
  and the KV/state fixture (``cache.export_fixture``) for the D side.

Inside the PPU SDK container on the P machine::

    /opt/conda312/bin/python rtp_llm/models_py/modules/dsv41/ppu/tools/driver_prefill_cp8.py \
        --config-path /path/to/checkpoint/config.json \
        --checkpoint-path /path/to/checkpoint \
        --tokens 129 --device cuda \
        --token-map-cache /tmp/rtp-llm-v41-runs/token_map_cache.json \
        --output /tmp/rtp-llm-v41-runs/<run_id>/cp8_129.json \
        --fixture-dir /tmp/rtp-llm-v41-runs/<run_id>/fixture
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT = 7200  # seconds; weight loading alone takes many minutes


# ---------------------------------------------------------------------------
# Worker (one CP rank)
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config-path", help="path to config.json (real model)")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="use a tiny synthetic config/checkpoint (CPU friendly)",
    )
    parser.add_argument("--checkpoint-path", help="checkpoint directory")
    parser.add_argument(
        "--tokenizer-path", help="tokenizer.json for the engram token map"
    )
    parser.add_argument("--token-map-cache", help="JSON cache for the token map")
    parser.add_argument("--tokens", type=int, default=129, help="prefill token count")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument(
        "--ep-backend",
        choices=("deepep", "sim", "megamoe"),
        default=None,
        help="routed MoE backend: real DeepEP dispatch/combine, or the "
        "simulated EP8 over the CP group (all-gather + local experts + "
        "all-reduce; default 'sim' on CPU, 'deepep' on cuda)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="final JSON path (rank 0)")
    parser.add_argument(
        "--fixture-dir",
        help="rank 0 exports the KV/state fixture here (cache.export_fixture)",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--pd-peer", help="D host for internal fixture TCP round-trip")
    parser.add_argument("--pd-base-port", type=int, default=19400)
    parser.add_argument("--pd-run-id", help="unique identity shared with the D driver")
    parser.add_argument("--pd-owner", type=int, help="pin every request to one D owner")
    parser.add_argument("--pd-timeout", type=int, default=1800)
    parser.add_argument("--worker", action="store_true", help="internal: worker mode")
    args = parser.parse_args(argv)
    if args.pd_peer and (not args.pd_run_id or not args.fixture_dir):
        parser.error("--pd-peer requires --pd-run-id and --fixture-dir")
    if args.pd_owner is not None and not 0 <= args.pd_owner < args.nproc:
        parser.error("--pd-owner must be inside the D world")
    return args


def _load_config(args):
    if args.config_path:
        with open(args.config_path) as handle:
            raw = json.load(handle)
        text = {**raw, **raw.get("text_config", {})}
        text.pop("text_config", None)
        if "engram_token_map" not in text:
            from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill import (
                _load_token_map,
            )

            tokenizer_path = args.tokenizer_path
            if tokenizer_path is None:
                tokenizer_path = str(Path(args.config_path).parent / "tokenizer.json")
            text["engram_token_map"] = _load_token_map(
                tokenizer_path, args.token_map_cache
            )
        return text
    from rtp_llm.models_py.modules.dsv41.ppu.weight import synthetic_config

    return synthetic_config()


def _make_enpm_group(args):
    """The ENP lookup group: plain ``DistributedEnpmGroup`` on the device
    (PCCL handles bf16 all-reduce); a gloo-safe adapter that reduces in fp32
    for CPU multi-process runs."""
    import torch
    import torch.distributed as dist
    from rtp_llm.models_py.modules.dsv41.ppu.engram import DistributedEnpmGroup

    if args.device == "cuda":
        return DistributedEnpmGroup(dist.group.WORLD)

    class _GlooEnpmGroup(DistributedEnpmGroup):
        def all_reduce_sum(self, values):
            reduced = values.to(torch.float32)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.pg)
            return reduced.to(values.dtype)

    return _GlooEnpmGroup(dist.group.WORLD)


def worker_main(args: argparse.Namespace) -> int:
    import sys as _sys
    from datetime import timedelta

    # The self-spawned workers run this file directly (no ``-m``), so the
    # repository root is not on sys.path.
    # .../rtp-llm/rtp_llm/models_py/modules/dsv41/ppu/tools/<this file>
    _root = str(Path(__file__).resolve().parents[6])
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(args.device)
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl" if args.device == "cuda" else "gloo",
        init_method="env://",
        timeout=timedelta(seconds=args.timeout),
    )
    run_dir = os.path.dirname(os.path.abspath(args.output))

    from rtp_llm.models_py.modules.dsv41.ppu.cp import (
        CPPrefillContext,
        DistributedCPGroup,
        build_zigzag_plan,
    )
    from rtp_llm.models_py.modules.dsv41.ppu.engram import DistributedEnpmGroup
    from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
    from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
    from rtp_llm.models_py.modules.dsv41.ppu.weight import (
        V41Geometry,
        load_ppu_weights,
        write_synthetic_checkpoint,
    )

    config = _load_config(args)
    geometry = V41Geometry.from_config(config)
    checkpoint = args.checkpoint_path
    synthetic_dir = None
    if checkpoint is None:
        import tempfile

        synthetic_dir = Path(tempfile.mkdtemp(prefix="dsv41_cp8_synth_"))
        checkpoint = write_synthetic_checkpoint(config, synthetic_dir, seed=args.seed)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    from rtp_llm.models_py.modules.dsv41.ppu.moe_precision import moe_backend

    selected_moe = moe_backend()
    if selected_moe == "megamoe":
        if args.device != "cuda" or args.ep_backend not in (None, "megamoe"):
            raise ValueError(
                "MegaMoE requires cuda and a matching --ep-backend selection"
            )
        ep_backend = "megamoe"
    else:
        if args.ep_backend == "megamoe":
            raise ValueError(
                "--ep-backend megamoe requires RTP_V41_PPU_MOE_BACKEND=megamoe"
            )
        ep_backend = args.ep_backend or ("sim" if args.device == "cpu" else "deepep")
    load_start = time.time()
    enpm_group = _make_enpm_group(args)
    ops_cls = PPUOps
    if os.environ.get("RTP_V41_TEST_MHC_VALIDATE") == "1":
        from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_integration import (
            ValidatedPPUOps,
        )

        ops_cls = ValidatedPPUOps
    ops = ops_cls(
        ep_rank=rank,
        ep_size=world_size,
        num_routed_experts=geometry.n_routed_experts,
        num_local_experts=geometry.num_local_experts(world_size),
        hidden_size=geometry.hidden_size,
        inter_dim=geometry.moe_inter_dim,
        swiglu_limit=geometry.swiglu_limit,
        mhc_config=geometry.mhc_config(),
        sm_scale=geometry.sm_scale(),
        window=geometry.window,
        enpm_group=enpm_group,
        deepep_group=dist.group.WORLD if ep_backend in ("deepep", "megamoe") else None,
    )
    bundle = load_ppu_weights(
        config,
        checkpoint,
        ep_rank=rank,
        ep_size=world_size,
        enp_rank=rank,
        enp_size=world_size,
        device=device,
    )
    bundle.bind(ops, enpm_group=enpm_group)
    load_seconds = time.time() - load_start

    core = PPUModelCore(config, bundle, ops, device=device)
    identity = None
    if args.pd_peer:
        from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import runtime_identity

        identity = runtime_identity()

    generator = torch.Generator().manual_seed(args.seed)
    per_request = args.tokens // args.requests
    if per_request < 1:
        raise SystemExit("--tokens must be >= --requests")
    full_ids = torch.randint(
        0,
        geometry.vocab_size,
        (args.requests * per_request,),
        dtype=torch.int64,
        generator=generator,
    )
    cu = torch.arange(
        0, args.requests * per_request + 1, per_request, dtype=torch.int64
    )

    plan = build_zigzag_plan(full_ids, cu, cp_size=world_size, cp_rank=rank)
    cp_group = DistributedCPGroup(dist.group.WORLD)
    if ep_backend == "sim":
        cp_group.register_ops(ops)
    context = CPPrefillContext(
        plan=plan, group=cp_group, simulate_moe=(ep_backend == "sim")
    )

    # CP8 correctness check 1 (in-driver, the 跑通 bar): the zigzag restore
    # permutation must reproduce the ORIGINAL global order — logical
    # positions ascending per request, no duplicates, no lost tokens. Gather
    # every rank's local (clamped) positions through the SAME collective +
    # restore path the hidden uses and compare against the plan's full-order
    # positions.
    restored_positions = context.all_gather_restore(plan.local_positions.to(device))
    order_restored = bool(
        torch.equal(restored_positions, plan.full_positions.to(device))
    )

    dist.barrier()
    forward_start = time.time()
    result = core.prefill(
        plan.local_input_ids.to(device),
        positions=plan.local_positions.to(device),
        cu_seqlens=plan.cu_local.to(device),
        request_ids=plan.local_request_ids.to(device),
        apply_lm_head=True,
        parallel_context=context,
    )
    forward_seconds = time.time() - forward_start
    peak_memory_gib = None
    if args.device == "cuda":
        torch.cuda.synchronize()
        # Snapshot BEFORE the cross-rank comparison gathers below so the
        # recorded peak reflects the forward itself.
        peak_memory_gib = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    logits = result.logits

    # CP8 correctness check 2: mirror semantics — every rank's restored
    # hidden must be IDENTICAL (each token is computed by exactly one owning
    # rank and the all-gather hands every rank the same bytes). Compare the
    # full hidden and the next-token logits row across ranks through the CP
    # group; record the max abs diff so a non-bitwise result is visible.
    hidden_rows = int(result.hidden.shape[0])
    gathered_hidden = cp_group.all_gather(result.hidden.contiguous())
    hidden_diffs = [
        (
            gathered_hidden[r * hidden_rows : (r + 1) * hidden_rows].float()
            - result.hidden.float()
        ).abs()
        for r in range(world_size)
    ]
    hidden_identical = all(
        torch.equal(
            gathered_hidden[r * hidden_rows : (r + 1) * hidden_rows], result.hidden
        )
        for r in range(world_size)
    )
    hidden_max_diff = max(
        (float(diff.max()) if diff.numel() else 0.0 for diff in hidden_diffs),
        default=0.0,
    )
    last_row = logits[-1].contiguous()
    gathered_logits = cp_group.all_gather(last_row).view(world_size, -1)
    next_token_identical = bool((gathered_logits == gathered_logits[0]).all())

    # CP8 correctness check 3: no rank wrote CP padding / chunk-tail rows
    # into the real pools — every request's pool valid lengths must equal its
    # GLOBAL token count (not the rank's chunk), the restored hidden carries
    # one row per global token, and the swa pools track the global counts.
    regions = result.cache.valid_regions()
    swa_regions = {str(split.request_id): split.length for split in plan.requests}
    pool_lengths_ok = hidden_rows == plan.seq_len_full and all(
        result.cache.request_length(split.request_id) == split.length
        for split in plan.requests
    )
    swa_lengths_ok = all(
        regions.get(f"{layer}:swa_kv") == swa_regions
        for layer in (0, geometry.num_layers - 1)
    )
    checks = {
        "order_restored": order_restored,
        "cross_rank_hidden_identical": hidden_identical,
        "cross_rank_hidden_max_abs_diff": hidden_max_diff,
        "cross_rank_next_token_identical": next_token_identical,
        "pool_lengths_match_global_tokens": pool_lengths_ok,
        "swa_valid_lengths_match_global_tokens": swa_lengths_ok,
    }
    if args.pd_peer:
        # The CP implementation currently restores all producer KV and CSA
        # inputs on every rank. Prove the replicated logical state before
        # allowing rank 0 to export; never assume it owns all CP slices.
        checksum = result.cache.state_checksum()
        local_hash = torch.tensor(
            list(bytes.fromhex(checksum)), dtype=torch.uint8, device=device
        )
        hashes = [torch.empty_like(local_hash) for _ in range(world_size)]
        dist.all_gather(hashes, local_hash)
        checks["cp_cache_replicated"] = all(torch.equal(local_hash, h) for h in hashes)
    checks_pass = all(value for value in checks.values() if isinstance(value, bool))

    payload = {
        "rank": rank,
        "moe": ops.moe_engine.execution_stats(),
        "mhc": ops.mhc_execution_stats(),
        "world_size": world_size,
        "device": str(device),
        "tokens": int(full_ids.numel()),
        "requests": int(args.requests),
        "chunk_length": int(plan.chunk_length),
        "real_tokens_on_rank": int(plan.cp_rank_token_count),
        "load_seconds": round(load_seconds, 1),
        "forward_seconds": round(forward_seconds, 1),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "hidden_finite": bool(torch.isfinite(result.hidden).all()),
        "hidden_shape": list(result.hidden.shape),
        "checks": checks,
    }
    if peak_memory_gib is not None:
        payload["max_memory_allocated_gib"] = peak_memory_gib
    if rank == 0:
        last = logits[-1]
        top_values, top_ids = last.topk(min(5, last.shape[-1]))
        payload.update(
            {
                "argmax": int(last.argmax()),
                "top5_ids": [int(x) for x in top_ids],
                "top5_values": [float(x) for x in top_values],
            }
        )
    rank_path = os.path.join(run_dir, f"cp8.rank{rank}.json")
    with open(rank_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[cp8 rank {rank}] wrote {rank_path}", flush=True)

    if rank == 0 and args.fixture_dir:
        manifest = result.cache.export_fixture(
            args.fixture_dir,
            meta={
                "source_cp": int(world_size),
                "driver": "driver_prefill_cp8",
                "ep_backend": str(ep_backend),
                "tokens": int(full_ids.numel()),
                "requests": int(args.requests),
                "seed": int(args.seed),
                "checks": {
                    key: value
                    for key, value in checks.items()
                    if isinstance(value, (bool, int, float, str))
                },
            },
        )
        with open(os.path.join(run_dir, "fixture_manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        print(f"[cp8 rank 0] fixture exported to {args.fixture_dir}", flush=True)

    if rank == 0 and args.pd_peer:
        from concurrent.futures import ThreadPoolExecutor

        from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import (
            config_signature,
            export_handoffs,
            send_handoff,
        )

        if not checks_pass:
            raise RuntimeError("CP checks failed; refusing to export a PD handoff")
        histories = {
            split.request_id: full_ids[int(cu[i]) : int(cu[i + 1])].tolist()
            for i, split in enumerate(plan.requests)
        }
        seeds = {
            split.request_id: int(logits[int(cu[i + 1]) - 1].argmax())
            for i, split in enumerate(plan.requests)
        }
        owners = {
            rid: (args.pd_owner if args.pd_owner is not None else i % world_size)
            for i, rid in enumerate(histories)
        }
        paths = export_handoffs(
            result.cache,
            Path(args.fixture_dir) / "pd",
            histories=histories,
            seed_tokens=seeds,
            owner_map=owners,
            identity=identity,
            model_signature=config_signature(config, args.checkpoint_path),
            run_id=args.pd_run_id,
            world=world_size,
        )
        # All owners must receive before any rank enters decode collectives.
        # Keep the sockets independent and await each D continuation result.
        with ThreadPoolExecutor(max_workers=world_size) as pool:
            futures = [
                pool.submit(
                    send_handoff,
                    path,
                    args.pd_peer,
                    args.pd_base_port + owner,
                    timeout=args.pd_timeout,
                )
                for owner, path in enumerate(paths)
            ]
            reports = [future.result() for future in futures]
        payload["pd_roundtrip"] = {
            "identity": identity,
            "run_id": args.pd_run_id,
            "owners": owners,
            "reports": reports,
            "all_owners_ok": all(r["decode_result"]["ok"] for r in reports),
        }
        with open(rank_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print("[PD] all D owners acknowledged continuation", flush=True)

    dist.barrier()
    ops.moe_engine.close()
    dist.destroy_process_group()
    if synthetic_dir is not None:
        import shutil

        shutil.rmtree(synthetic_dir, ignore_errors=True)
    return 0 if payload["logits_finite"] and checks_pass else 1


# ---------------------------------------------------------------------------
# Parent / aggregation (the hw_probe.py self-spawn launcher pattern)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _aggregate(args: argparse.Namespace, world_size: int) -> dict:
    run_dir = os.path.dirname(os.path.abspath(args.output))
    deadline = time.time() + 120
    payloads = []
    while True:
        payloads = []
        for rank in range(world_size):
            path = os.path.join(run_dir, f"cp8.rank{rank}.json")
            if not os.path.exists(path):
                break
            with open(path, "r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        if len(payloads) == world_size or time.time() > deadline:
            break
        time.sleep(2.0)
    document = {
        "launcher": "self-spawn-subprocess",
        "pd_roundtrip": payloads[0].get("pd_roundtrip") if payloads else None,
        "world_size": world_size,
        "tokens": int(args.tokens),
        "ranks": payloads,
        "summary": {
            "ranks_reported": len(payloads),
            "moe_precisions": sorted(
                {p.get("moe", {}).get("precision", "unknown") for p in payloads}
            ),
            "moe_backends": sorted(
                {p.get("moe", {}).get("backend", "unknown") for p in payloads}
            ),
            "moe_mega_calls": sum(
                p.get("moe", {}).get("mega_moe_calls", 0) for p in payloads
            ),
            "moe_mega_received_assignments": sum(
                p.get("moe", {}).get("mega_received_assignments", 0) for p in payloads
            ),
            "moe_grouped_gemm_calls": sum(
                p.get("moe", {}).get("grouped_gemm_calls", 0) for p in payloads
            ),
            "all_finite": bool(payloads)
            and all(
                p.get("logits_finite") and p.get("hidden_finite") for p in payloads
            ),
            "checks_all_pass": bool(payloads)
            and all(
                all(
                    value
                    for value in p.get("checks", {}).values()
                    if isinstance(value, bool)
                )
                for p in payloads
            ),
            "hidden_max_abs_diff": max(
                (
                    p.get("checks", {}).get("cross_rank_hidden_max_abs_diff", 0.0)
                    for p in payloads
                ),
                default=0.0,
            ),
            "load_seconds_max": max(
                (p.get("load_seconds", 0) for p in payloads), default=0
            ),
            "forward_seconds_max": max(
                (p.get("forward_seconds", 0) for p in payloads), default=0
            ),
            "max_memory_gib": max(
                (p.get("max_memory_allocated_gib", 0) for p in payloads), default=0
            ),
            "pd_all_owners_ok": (
                not args.pd_peer
                or bool(
                    payloads
                    and payloads[0].get("pd_roundtrip", {}).get("all_owners_ok")
                )
            ),
        },
    }
    os.makedirs(run_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
    summary = document["summary"]
    print("=" * 72, flush=True)
    print(
        f"CP8 prefill summary: ranks {summary['ranks_reported']}/{world_size}, "
        f"finite={summary['all_finite']}, checks={summary['checks_all_pass']}, "
        f"hidden_max_diff={summary['hidden_max_abs_diff']}, "
        f"load {summary['load_seconds_max']}s, forward {summary['forward_seconds_max']}s, "
        f"peak mem {summary['max_memory_gib']} GiB",
        flush=True,
    )
    if payloads:
        rank0 = payloads[0]
        print(
            f"rank0 argmax={rank0.get('argmax')} top5={rank0.get('top5_ids')}",
            flush=True,
        )
    print("=" * 72, flush=True)
    return document


def parent_main(args: argparse.Namespace) -> int:
    master_port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(args.nproc),
        }
    )
    run_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(run_dir, exist_ok=True)
    processes = []
    for rank in range(args.nproc):
        rank_env = dict(env)
        rank_env["RANK"] = str(rank)
        rank_env["LOCAL_RANK"] = str(rank)
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--output",
            os.path.abspath(args.output),
        ]
        forwarded = (
            "config_path",
            "checkpoint_path",
            "tokenizer_path",
            "token_map_cache",
            "tokens",
            "requests",
            "device",
            "seed",
            "ep_backend",
            "pd_peer",
            "pd_base_port",
            "pd_run_id",
            "pd_owner",
            "pd_timeout",
            "nproc",
            "timeout",
        )
        for name in forwarded:
            value = getattr(args, name, None)
            if value is not None:
                command.extend([f"--{name.replace('_', '-')}", str(value)])
        if args.synthetic:
            command.append("--synthetic")
        if args.fixture_dir:
            command.extend(["--fixture-dir", os.path.abspath(args.fixture_dir)])
        log_path = os.path.join(run_dir, f"cp8.rank{rank}.log")
        log = open(log_path, "w", encoding="utf-8")
        processes.append(
            subprocess.Popen(
                command, env=rank_env, stdout=log, stderr=subprocess.STDOUT
            )
        )
        log.close()
    deadline = time.time() + int(args.timeout)
    try:
        while any(process.poll() is None for process in processes):
            if any(process.returncode not in (None, 0) for process in processes):
                break
            if time.time() >= deadline:
                print(
                    f"[cp8] timeout after {args.timeout}s; terminating spawned workers",
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
    document = _aggregate(args, args.nproc)
    ok = (
        document["summary"]["ranks_reported"] == args.nproc
        and document["summary"]["checks_all_pass"]
        and document["summary"]["all_finite"]
        and document["summary"]["pd_all_owners_ok"]
        and all(process.returncode == 0 for process in processes)
    )
    return 0 if ok else 1


def main() -> int:
    args = _parse_args()
    in_distributed_env = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "MASTER_ADDR" in os.environ
    )
    if args.worker or in_distributed_env:
        if not in_distributed_env:
            raise SystemExit(
                "--worker requires RANK/WORLD_SIZE/MASTER_ADDR in the environment"
            )
        return worker_main(args)
    if not args.config_path and not args.synthetic:
        args.synthetic = True
    if args.config_path and not args.checkpoint_path:
        raise SystemExit("--config-path requires --checkpoint-path")
    run_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "command.txt"), "w", encoding="utf-8") as handle:
        handle.write(" ".join(sys.argv))
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
