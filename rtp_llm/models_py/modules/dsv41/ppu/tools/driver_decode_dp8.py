#!/usr/bin/env python
"""DP8/EP8/ENP8 decode bring-up driver for the D machine (L3, wave 2).

One process per ZW-M890P (hw_probe.py launcher pattern), joined by
``init_process_group(backend="nccl")`` (PCCL). Every rank:

* loads its EP8 expert shard + ENP8 Engram host shard (real checkpoint) or
  a synthetic model (CPU-friendly smoke);
* prefills its own requests through ``PPUModelCore.prefill`` — every rank
  participates in the per-layer DeepEP dispatch/combine and the ENP8
  lookup rounds (a rank with an empty local batch still joins every
  collective);
* runs N decode steps coordinated by ``dp_coord.DPStepCoordinator``
  (step-state exchange, fixed collective order, release barrier). Requests
  are RELEASED on selected ranks before decode to build the plan section
  3.2 stress shapes: ``--scenario one-busy`` (one rank with a request,
  seven empty -> placeholder rows), ``--scenario skew`` (uneven request
  counts), ``--scenario all`` (one request per rank);
* asserts the placeholder contract in-driver: placeholder rows never enter
  sampling (hidden rows == active requests) and never enter the cache
  (per-request pool counts advance by exactly one per step);
* records per-rank step time and device memory.

Parent pre-derives the tokenizer-based ``engram_token_map`` config once and
spawns the workers; every remote run gets its own directory with per-rank
logs + JSON and an aggregate summary.

Example (D machine, inside the PPU container)::

    python rtp_llm/models_py/modules/dsv41/ppu/tools/driver_decode_dp8.py \
        --config-path <ckpt>/config.json --checkpoint-path <ckpt> \
        --tokens 32 --steps 4 --scenario one-busy \
        --output /tmp/rtp-llm-v41-runs/<run>/dp8.json
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
from typing import Any, Dict, List

# This file runs as a PLAIN SCRIPT in both the parent (launch command) and
# the self-spawned workers (no ``-m``), so the repository root is not on
# sys.path. Bootstrap it before the first rtp_llm import below.
# .../rtp-llm/rtp_llm/models_py/modules/dsv41/ppu/tools/<this file>
_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_decode import _DriverOps

PROBE_TIMEOUT_DEFAULT = 3600

SCENARIOS = ("all", "one-busy", "skew")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config-path")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint-path")
    parser.add_argument(
        "--tokens", type=int, default=16, help="prefill tokens per request"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--fixture-steps",
        type=int,
        default=0,
        help="also run the DP8 fixture check: rank 0 (an owner) exports its "
        "post-prefill cache fixture BEFORE decode, the uninterrupted loop is "
        "the reference, and a restored orchestrator loaded from the fixture "
        "(kv_io.load_fixture_state with owner_map pinning) replays the same "
        "steps; every rank keeps stepping so the collectives stay symmetric "
        "(must be <= --steps)",
    )
    parser.add_argument("--scenario", choices=SCENARIOS, default="all")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=PROBE_TIMEOUT_DEFAULT)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument(
        "--mhc-torch",
        action="store_true",
        help="force the authoritative torch mHC path (workaround: the D "
        "container's Triton rejects the mix-stats kernel's int64 arg cast)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pd-receive",
        action="store_true",
        help="receive CP8 state over internal TCP; skip D prefill",
    )
    parser.add_argument("--pd-bind-host", default="0.0.0.0")
    parser.add_argument("--pd-base-port", type=int, default=19400)
    parser.add_argument("--pd-run-id")
    parser.add_argument("--pd-timeout", type=int, default=1800)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.fixture_steps > args.steps:
        parser.error("--fixture-steps must be <= --steps (reference coverage)")
    if args.pd_receive and not args.pd_run_id:
        parser.error("--pd-receive requires --pd-run-id")
    return args


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class _DP8Ops(_DriverOps):
    """Ops wrapper keeping the ENP collective on-device.

    L2's prefill computes the Engram hash rows on CPU; the distributed ENP
    group needs them on the device for the NCCL gathers. The wrapper moves
    rows in (and leaves the result wherever the group put it). The inherited
    ``sparse_attention`` passes the per-HEAD ``attn_sink`` unsliced (see
    ``driver_decode._DriverOps`` -- fix-forward workaround handed to L2).
    """

    def __init__(self, ops, device):
        super().__init__(ops, device)

    def engram_lookup(self, rows, *, layer_id, context=None):
        rows = rows.detach().to(self._device)
        return self._ops.engram_lookup(rows, layer_id=layer_id, context=context)


def _requests_for_scenario(scenario: str, rank: int, world: int) -> int:
    """Requests kept per rank during DECODE (after the prefill phase)."""
    if scenario == "all":
        return 1
    if scenario == "one-busy":
        return 1 if rank == 0 else 0
    if scenario == "skew":
        table = (3, 0, 2, 0, 1, 0, 0, 0)
        return table[rank % world]
    raise ValueError(scenario)


def _compare_fixture_rows(reference, replay, *, atol: float = 1e-2, rtol: float = 1e-2):
    """Offline contract check (plan 7.5) over recorded step rows.

    Each entry is ``(row_request_ids, argmax tokens, hidden clone)`` from one
    step of the uninterrupted reference / the fixture-restored replay.
    """
    import torch

    if len(reference) != len(replay):
        return {
            "match": False,
            "reason": f"step count {len(replay)} != {len(reference)}",
        }
    max_diff = 0.0
    for index, (left, right) in enumerate(zip(reference, replay)):
        if left[0] != right[0]:
            return {"match": False, "reason": f"step {index}: request order diverged"}
        if left[1] != right[1]:
            return {
                "match": False,
                "reason": f"step {index}: token stream diverged "
                f"({left[1]} != {right[1]})",
            }
        if left[2] is None or right[2] is None:
            continue
        if left[2].shape != right[2].shape:
            return {"match": False, "reason": f"step {index}: hidden shape diverged"}
        diff = (
            (left[2].float() - right[2].float()).abs().max().item()
            if left[2].numel()
            else 0.0
        )
        max_diff = max(max_diff, diff)
        if not torch.allclose(left[2], right[2], atol=atol, rtol=rtol):
            return {
                "match": False,
                "reason": f"step {index}: hidden diff {diff:.3e} exceeds tolerance",
                "max_abs_diff": max_diff,
            }
    return {"match": True, "steps": len(reference), "max_abs_diff": max_diff}


def worker_main(args: argparse.Namespace) -> int:
    # The repository root is already on sys.path (the module-level bootstrap
    # above covers the plain-script worker spawn); import the heavy pieces
    # only inside the worker process.
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(
        backend="gloo" if device.type == "cpu" else "nccl",
        timeout=timedelta(seconds=args.timeout),
    )

    from rtp_llm.models_py.modules.dsv41.ppu.decode import wire_decode
    from rtp_llm.models_py.modules.dsv41.ppu.dp_coord import (
        DPStepCoordinator,
        OwnerRouter,
        RankStepState,
        TorchCollectiveBackend,
    )
    from rtp_llm.models_py.modules.dsv41.ppu.engram import DistributedEnpmGroup
    from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
    from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
    from rtp_llm.models_py.modules.dsv41.ppu.weight import (
        V41Geometry,
        load_ppu_weights,
        write_synthetic_checkpoint,
    )

    with open(args.config_path) as handle:
        config = json.load(handle)
    geometry = V41Geometry.from_config(config)
    ep_size = args.nproc
    ops = PPUOps(
        ep_rank=rank,
        ep_size=ep_size,
        num_routed_experts=geometry.n_routed_experts,
        num_local_experts=geometry.num_local_experts(ep_size),
        hidden_size=geometry.hidden_size,
        inter_dim=geometry.moe_inter_dim,
        swiglu_limit=geometry.swiglu_limit,
        mhc_config=geometry.mhc_config(),
        sm_scale=geometry.sm_scale(),
        window=geometry.window,
        enpm_group=DistributedEnpmGroup(dist.group.WORLD),
    )
    checkpoint = args.checkpoint_path
    synthetic_dir = None
    if checkpoint is None:
        synthetic_dir = Path(f"/tmp/dsv41_dp8_synth_{args.seed}")
        if rank == 0:
            write_synthetic_checkpoint(config, synthetic_dir, seed=args.seed)
        dist.barrier()
        checkpoint = str(synthetic_dir)
    bundle = load_ppu_weights(
        config,
        checkpoint,
        ep_rank=rank,
        ep_size=ep_size,
        enp_rank=rank,
        enp_size=ep_size,
        device=device,
    )
    bundle.bind(ops, enpm_group=DistributedEnpmGroup(dist.group.WORLD))
    if args.mhc_torch:
        # Workaround (fix-forward): the D container's Triton rejects the
        # mix-stats kernel's int64 scalar cast; force the authoritative torch
        # mHC path through the public bind_mhc_config seam (see driver_decode).
        import dataclasses

        ops.bind_mhc_config(
            dataclasses.replace(geometry.mhc_config(), use_triton=False)
        )
    if device.type == "cuda":
        ops = _DP8Ops(ops, device)
    else:
        # CPU synthetic smoke: deep_ep is unavailable off-device, so pin the
        # MoE to the LOCAL reference path over this rank's shard (the same
        # loudly-recorded smoke mode as the single-rank driver). The EP8
        # dispatch/combine collectives are exercised on-device only.
        # (PPUOps.moe is the contract op method; moe_engine is the PPUMoE.)
        ops.moe_engine.ep_size = 1
    core = PPUModelCore(config, bundle, ops, device=device)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    session = None
    pd_report = None
    if args.pd_receive:
        from rtp_llm.models_py.modules.dsv41.ppu.kv_io import load_fixture_state
        from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import (
            config_signature,
            receive_handoff,
            runtime_identity,
        )

        error = None
        try:
            identity = runtime_identity()
            session = receive_handoff(
                Path(out_dir) / f"received.owner{rank}",
                args.pd_bind_host,
                args.pd_base_port + rank,
                identity=identity,
                model_signature=config_signature(config, args.checkpoint_path),
                run_id=args.pd_run_id,
                owner=rank,
                world=world,
                timeout=args.pd_timeout,
            )
            pd = session.manifest["meta"]["pd"]
            histories = {int(rid): tokens for rid, tokens in pd["histories"].items()}
            seed_tokens = {int(rid): token for rid, token in pd["seed_tokens"].items()}
            owners = {int(rid): owner for rid, owner in pd["owner_map"].items()}
            orchestrator = load_fixture_state(
                session.path,
                ops=ops,
                bundle=bundle,
                request_histories=histories,
                owner_map=owners,
                dp_rank=rank,
                dp_size=world,
                device=device,
            )
            core.cache = orchestrator.cache.cache
            if core.cache.state_checksum() != pd["state_checksum"]:
                raise ValueError("D loaded cache differs from the P logical state")
            if any(
                len(history) + args.steps + 1 > 32768 for history in histories.values()
            ):
                raise ValueError("PD input + output exceeds 32K")
            pd_report = {
                "identity": identity,
                "run_id": args.pd_run_id,
                "transport": "tcp",
                "d_prefill_executed": False,
                "source_cp": session.manifest["source_cp"],
                "state_checksum": pd["state_checksum"],
                "payload_checksum": session.manifest["checksum"],
                "owner": rank,
                "seed_tokens": seed_tokens,
            }
            print(
                f"[PD owner {rank}] loaded {len(histories)} requests; no D prefill",
                flush=True,
            )
        except Exception as exc:
            error = exc
        # Every rank observes a failed load before any ENP/EP model collective.
        ready = torch.tensor([int(error is None)], dtype=torch.int32, device=device)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        if not int(ready.item()):
            failure = error or RuntimeError("another D owner failed its PD load")
            if session is not None:
                session.fail(failure)
            raise RuntimeError(f"PD load failed: {failure}") from error
        keep = len(histories)
        released = 0
        active_ids = sorted(histories)
    else:
        orchestrator, histories, seed_tokens, active_ids, released, keep = (
            _local_prefill(args, rank, world, geometry, core, device)
        )

    os.makedirs(out_dir, exist_ok=True)

    return _run_decode(
        args,
        rank,
        world,
        geometry,
        core,
        bundle,
        ops,
        device,
        orchestrator,
        histories,
        seed_tokens,
        active_ids,
        released,
        session,
        pd_report,
    )


def _local_prefill(args, rank, world, geometry, core, device):
    import torch

    from rtp_llm.models_py.modules.dsv41.ppu.decode import wire_decode

    # ---- prefill phase: every rank prefills its scenario request count (at
    # least one, so all collectives run with a non-empty local batch; the
    # empty-rank shapes are exercised during decode, after release).
    keep = _requests_for_scenario(args.scenario, rank, world)
    n_local = max(keep, 1)
    generator = torch.Generator().manual_seed(args.seed + 1000 * rank)
    histories = {}
    request_ids_map = {}
    flat_ids = []
    cu = [0]
    for index in range(n_local):
        tokens = torch.randint(
            0,
            geometry.vocab_size,
            (args.tokens,),
            dtype=torch.int64,
            generator=generator,
        ).tolist()
        request_id = rank * 1000 + index  # globally unique across ranks
        histories[request_id] = tokens
        request_ids_map[request_id] = index
        flat_ids.extend(tokens)
        cu.append(len(flat_ids))
    input_ids = torch.tensor(flat_ids, dtype=torch.int64)
    positions = torch.arange(args.tokens, dtype=torch.int64).repeat(n_local)
    per_request_ids = torch.repeat_interleave(
        torch.tensor(
            [rank * 1000 + index for index in range(n_local)], dtype=torch.int64
        ),
        torch.full((n_local,), args.tokens, dtype=torch.int64),
    )
    prefill = core.prefill(
        input_ids.to(device),
        positions=positions.to(device),
        cu_seqlens=torch.tensor(cu, dtype=torch.int64, device=device),
        request_ids=per_request_ids.to(device),
        apply_lm_head=True,
    )
    if not bool(torch.isfinite(prefill.logits).all()):
        raise SystemExit(f"rank {rank}: prefill logits non-finite")
    seed_tokens = {
        request_id: int(prefill.logits[int(cu[index + 1]) - 1].argmax())
        for request_id, index in request_ids_map.items()
    }

    orchestrator = wire_decode(core, dp_rank=rank, dp_size=world)
    for request_id, tokens in histories.items():
        orchestrator.add_request(request_id, tokens)

    # ---- release requests to build the decode-time shape.
    released = 0
    if keep == 0:
        for request_id in list(histories):
            orchestrator.release(request_id)
            released = len(histories)
    active_ids = [
        rid for rid in orchestrator.requests if not orchestrator.requests[rid].finished
    ]
    if len(active_ids) != keep:
        raise SystemExit(
            f"rank {rank}: expected {keep} active requests, has {len(active_ids)}"
        )

    return orchestrator, histories, seed_tokens, active_ids, released, keep


def _run_decode(
    args,
    rank,
    world,
    geometry,
    core,
    bundle,
    ops,
    device,
    orchestrator,
    histories,
    seed_tokens,
    active_ids,
    released,
    session,
    pd_report,
):
    import torch
    import torch.distributed as dist

    from rtp_llm.models_py.modules.dsv41.ppu.dp_coord import (
        DPStepCoordinator,
        OwnerRouter,
        RankStepState,
        TorchCollectiveBackend,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))

    # ---- optional DP8 fixture check (plan 7.5, single-owner form). Rank 0
    # (an owner) exports its post-prefill cache fixture BEFORE any decode
    # step; the uninterrupted loop below is then the reference, and a
    # restored orchestrator (kv_io.load_fixture_state with owner_map
    # pinning) replays the same steps after it. Every rank keeps stepping
    # during the replay so the ENP/EP collectives stay symmetric.
    fixture_dir = os.path.join(out_dir, "fixture") if args.fixture_steps > 0 else None
    fixture_histories = {
        rid: list(state.history)
        for rid, state in orchestrator.requests.items()
        if not state.finished
    }
    if args.pd_receive and args.fixture_steps:
        fixture_dir = str(session.path)
    elif fixture_dir is not None and rank == 0:
        orchestrator.cache.export_fixture(
            fixture_dir,
            meta={
                "source_cp": 1,
                "driver": "driver_decode_dp8",
                "scenario": args.scenario,
            },
        )
        print(
            f"[driver_decode_dp8 rank 0] fixture exported to {fixture_dir}",
            flush=True,
        )

    coordinator = DPStepCoordinator(
        TorchCollectiveBackend(),
        num_layers=geometry.num_layers,
        owner_router=OwnerRouter(world),
    )

    pending = {rid: seed_tokens[rid] for rid in active_ids}
    generated = {rid: [seed_tokens[rid]] for rid in active_ids}
    initial_lengths = {rid: len(histories[rid]) for rid in active_ids}
    steps_report = []
    reference_rows = []  # rank 0 only: the uninterrupted fixture reference
    placeholder_contract_ok = True
    for step in range(args.steps):
        started = time.perf_counter()
        local = RankStepState(valid_requests=len(active_ids), tokens=len(active_ids))
        plan, tracker = coordinator.begin_step(local)
        if not plan.run:
            coordinator.finish_step(tracker, plan)
            break
        tracker.enter("enp_query")
        # The ENP query and every per-layer EP dispatch/combine execute
        # INSIDE the single orchestrator step below, in the agreed order
        # (serial, no cross-group interleaving); the tracker records the
        # schedule around the fused call.
        result = orchestrator.step(dict(pending), placeholders=plan.placeholder_rows)
        for layer in range(geometry.num_layers):
            tracker.enter(f"ep_dispatch_L{layer}")
            tracker.enter(f"ep_combine_L{layer}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        coordinator.finish_step(tracker, plan)
        elapsed = time.perf_counter() - started

        if (
            (rank == 0 or args.pd_receive)
            and args.fixture_steps > 0
            and step < args.fixture_steps
        ):
            reference_rows.append(
                (
                    list(result.row_request_ids),
                    (
                        [
                            int(result.logits[row].argmax())
                            for row in range(len(result.row_request_ids))
                        ]
                        if result.logits is not None
                        else None
                    ),
                    None if result.hidden is None else result.hidden.detach().clone(),
                )
            )

        # ---- placeholder contract assertions (plan 3.2.3) -------------------
        if len(result.row_request_ids) != len(active_ids):
            placeholder_contract_ok = False
        if int(result.valid_mask.sum()) != len(active_ids):
            placeholder_contract_ok = False
        if plan.placeholder_rows and result.num_placeholders != plan.placeholder_rows:
            placeholder_contract_ok = False
        finite = bool(torch.isfinite(result.hidden).all()) and (
            result.logits is None or bool(torch.isfinite(result.logits).all())
        )
        for request_id_now in active_ids:
            state = orchestrator.requests[request_id_now]
            if state.seq_len != initial_lengths[request_id_now] + step + 1:
                placeholder_contract_ok = False
            # The transferred first token is consumed exactly once, and each
            # subsequent input is the last sampled token, including Engram's
            # cross-P/D n-gram window.
            if state.history != histories[request_id_now] + generated[request_id_now]:
                placeholder_contract_ok = False
        steps_report.append(
            {
                "step": step,
                "local_tokens": plan.local_tokens,
                "placeholder_rows": plan.placeholder_rows,
                "world_tokens": plan.world_tokens,
                "hidden_rows": int(result.hidden.shape[0]),
                "finite": finite,
                "placeholder_contract_ok": placeholder_contract_ok,
                "step_seconds": round(elapsed, 4),
                "argmax": (
                    {
                        int(rid): int(result.logits[row].argmax())
                        for row, rid in enumerate(result.row_request_ids)
                    }
                    if result.logits is not None
                    else None
                ),
            }
        )
        if not finite or not placeholder_contract_ok:
            break
        if result.logits is not None:
            for row, rid in enumerate(result.row_request_ids):
                pending[int(rid)] = int(result.logits[row].argmax())
                generated[int(rid)].append(pending[int(rid)])

    # ---- fixture replay: rank 0 continues decode from the LOADED fixture;
    # the other ranks keep stepping their own orchestrators so every
    # collective of the replay runs with the same shapes as the reference.
    fixture_report = None
    if args.fixture_steps > 0:
        from rtp_llm.models_py.modules.dsv41.ppu.kv_io import load_fixture_state

        if rank == 0 or args.pd_receive:
            restored = load_fixture_state(
                fixture_dir,
                ops=ops,
                bundle=bundle,
                dp_rank=rank,
                dp_size=world,
                request_histories=dict(fixture_histories),
                owner_map={int(rid): rank for rid in fixture_histories},
                device=device,
            )
            replay_pending = {
                int(rid): int(token)
                for rid, token in seed_tokens.items()
                if int(rid) in fixture_histories
            }
        replay_rows = []
        for _step in range(args.fixture_steps):
            if rank == 0 or args.pd_receive:
                result = restored.step(
                    dict(replay_pending), placeholders=plan.placeholder_rows
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                replay_rows.append(
                    (
                        list(result.row_request_ids),
                        (
                            [
                                int(result.logits[row].argmax())
                                for row in range(len(result.row_request_ids))
                            ]
                            if result.logits is not None
                            else None
                        ),
                        (
                            None
                            if result.hidden is None
                            else result.hidden.detach().clone()
                        ),
                    )
                )
                if result.logits is not None:
                    for row, rid in enumerate(result.row_request_ids):
                        replay_pending[int(rid)] = int(result.logits[row].argmax())
            else:
                # Collective symmetry: exactly one step call per replay step
                # on every rank (empty ranks rejoin with placeholder rows).
                orchestrator.step(dict(pending), placeholders=plan.placeholder_rows)
        if rank == 0 or args.pd_receive:
            fixture_report = _compare_fixture_rows(reference_rows, replay_rows)
            print(
                f"[driver_decode_dp8 rank {rank}] fixture_check={fixture_report}",
                flush=True,
            )

    memory = {}
    if device.type == "cuda":
        try:
            memory = {
                "max_allocated_mb": int(torch.cuda.max_memory_allocated() / 2**20),
                "allocated_mb": int(torch.cuda.memory_allocated() / 2**20),
            }
        except Exception as error:  # noqa: BLE001
            memory = {"error": str(error)[:200]}

    payload = {
        "rank": rank,
        "world_size": world,
        "device": str(device),
        "scenario": args.scenario,
        "tokens": args.tokens,
        "moe_mode": (
            "deepep-dispatch"
            if device.type == "cuda"
            else "local-reference-cpu-smoke (deep_ep unavailable on CPU; only "
            "this rank's expert shard computes)"
        ),
        "mhc_backend": "torch (forced)" if args.mhc_torch else "auto",
        "steps": steps_report,
        "released_before_decode": released,
        "active_requests": len(active_ids),
        "request_states": {
            rid: {"seq_len": state.seq_len, "steps_done": state.steps_done}
            for rid, state in orchestrator.requests.items()
        },
        "cache_regions": {
            key: {str(rid): count for rid, count in per_request.items()}
            for key, per_request in core.cache.valid_regions().items()
        },
        "fixture_check": fixture_report,
        "pd_handoff": pd_report,
        "generated_tokens": generated,
        "memory": memory,
        "ok": bool(steps_report)
        and all(s["finite"] and s["placeholder_contract_ok"] for s in steps_report)
        and (fixture_report is None or bool(fixture_report.get("match"))),
    }
    rank_path = os.path.join(out_dir, f"dp8.rank{rank}.json")
    with open(rank_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[driver_decode_dp8 rank {rank}] wrote {rank_path}", flush=True)
    if session is not None:
        session.finish(
            {
                "ok": payload["ok"],
                "owner": rank,
                "request_states": payload["request_states"],
                "generated_tokens": generated,
                "fixture_check": fixture_report,
                "state_checksum": pd_report["state_checksum"],
            }
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0 if payload["ok"] else 1


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------


def _prepare_config(args) -> str:
    """Merge the tokenizer-derived engram token map into a run-local config."""
    with open(args.config_path) as handle:
        raw = json.load(handle)
    text = {**raw, **raw.get("text_config", {})}
    text.pop("text_config", None)
    if "engram_token_map" in text:
        out = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "config_with_token_map.json"
        )
        Path(out).write_text(json.dumps(text))
        return out
    cache_path = Path(args.output).resolve().parent / "engram_token_map.json"
    if not cache_path.is_file():
        from transformers import AutoTokenizer

        from rtp_llm.models_py.modules.dsv41.ppu.engram import compressed_vocab_map

        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_path, trust_remote_code=True
        )
        lookup, size = compressed_vocab_map(tokenizer)
        expected = text.get("engram_compressed_vocab_size")
        if expected is not None and size != int(expected):
            raise SystemExit(f"derived compressed vocab {size} != config {expected}")
        cache_path.write_text(json.dumps(lookup))
        print(f"[driver_decode_dp8] token map cached at {cache_path}")
    text["engram_token_map"] = json.loads(cache_path.read_text())
    out = os.path.join(
        os.path.dirname(os.path.abspath(args.output)), "config_with_token_map.json"
    )
    Path(out).write_text(json.dumps(text))
    return out


def _aggregate(output: str, world: int) -> Dict[str, Any]:
    out_dir = os.path.dirname(os.path.abspath(output))
    deadline = time.time() + 60
    payloads: List[Dict[str, Any]] = []
    while True:
        payloads = []
        for rank in range(world):
            path = os.path.join(out_dir, f"dp8.rank{rank}.json")
            if not os.path.exists(path):
                break
            with open(path, "r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        if len(payloads) == world or time.time() > deadline:
            break
        time.sleep(2.0)
    step_times = [
        entry["step_seconds"]
        for payload in payloads
        for entry in payload.get("steps", [])
        if "step_seconds" in entry
    ]
    ranks_ok = [bool(p.get("ok")) for p in payloads]
    document = {
        "world_size": world,
        "ranks_reported": len(payloads),
        "scenario": payloads[0].get("scenario") if payloads else None,
        "per_rank_summary": [
            {
                "rank": p.get("rank"),
                "active_requests": p.get("active_requests"),
                "released_before_decode": p.get("released_before_decode"),
                "steps": len(p.get("steps", [])),
                "memory": p.get("memory"),
                "ok": p.get("ok"),
            }
            for p in payloads
        ],
        "step_seconds_min": min(step_times) if step_times else None,
        "step_seconds_max": max(step_times) if step_times else None,
        "fixture_check": next(
            (p.get("fixture_check") for p in payloads if p.get("fixture_check")),
            None,
        ),
        "all_ranks_ok": bool(payloads) and all(ranks_ok) and len(payloads) == world,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    print(json.dumps(document, indent=2), flush=True)
    return document


def parent_main(args: argparse.Namespace) -> int:
    # The run-local config copies land next to --output; make sure the run
    # directory exists before either branch writes into it.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    config_path = args.config_path
    if config_path and args.checkpoint_path:
        config_path = _prepare_config(args)
        args.config_path = config_path
    elif args.synthetic:
        from rtp_llm.models_py.modules.dsv41.ppu.weight import synthetic_config

        out = os.path.join(
            os.path.dirname(os.path.abspath(args.output)), "synthetic_config.json"
        )
        Path(out).write_text(json.dumps(synthetic_config()))
        args.config_path = out
    master_port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(args.nproc),
        }
    )
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    processes: List[subprocess.Popen] = []
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
            "--config-path",
            os.path.abspath(args.config_path),
            "--scenario",
            args.scenario,
            "--tokens",
            str(args.tokens),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--nproc",
            str(args.nproc),
            "--timeout",
            str(args.timeout),
        ]
        if args.mhc_torch:
            command.append("--mhc-torch")
        if args.fixture_steps:
            command += ["--fixture-steps", str(args.fixture_steps)]
        if args.pd_receive:
            command += [
                "--pd-receive",
                "--pd-bind-host",
                args.pd_bind_host,
                "--pd-base-port",
                str(args.pd_base_port),
                "--pd-run-id",
                args.pd_run_id,
                "--pd-timeout",
                str(args.pd_timeout),
            ]
        if args.checkpoint_path:
            command += ["--checkpoint-path", os.path.abspath(args.checkpoint_path)]
        log_path = os.path.join(out_dir, f"dp8.rank{rank}.log")
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
                    f"[driver_decode_dp8] timeout after {args.timeout}s; terminating",
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
    document = _aggregate(args.output, args.nproc)
    return (
        0
        if document.get("all_ranks_ok") and all(p.returncode == 0 for p in processes)
        else 1
    )


def main() -> int:
    args = _parse_args()
    if args.worker:
        return worker_main(args)
    if not args.config_path and not args.synthetic:
        args.synthetic = True
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
