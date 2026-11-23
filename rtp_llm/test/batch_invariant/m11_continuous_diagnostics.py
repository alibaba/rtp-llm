"""Verify M11 refill forwards and compare target-only gRPC logits under load."""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from m11_continuous import (
    RANKS,
    RPC_TIMEOUT,
    SHAPES,
    Load,
    Service,
    Stream,
    monitor,
    save_json,
    slots,
)

from rtp_llm.test.batch_invariant import service_test


def records(directory, offsets):
    return sorted(
        service_test.read_records(directory, offsets),
        key=lambda batch: (batch["dp_rank"], batch["forward_step"]),
    )


def rows(batch, phase=None):
    return {
        int(row["rpc_request_id"]): (position, row)
        for position, row in enumerate(batch["requests"])
        if not row.get("is_fake") and (phase is None or row["phase"] == phase)
    }


def frame(batch, request_ids):
    selected = rows(batch)
    return {
        "rank": batch["dp_rank"],
        "forward_step": batch["forward_step"],
        "exec_id": batch["exec_id"],
        "forward_us": [batch["forward_start_us"], batch["forward_end_us"]],
        "real_rows": len(selected),
        "requests": [
            {"row": selected[request_id][0], **selected[request_id][1]}
            for request_id in request_ids
        ],
    }


def refills(batches, *, older=None, newer=None):
    """Find the first joint decode of each admitted request with an older peer."""
    prefills, first_decode = {}, {}
    for batch in batches:
        rank, step = batch["dp_rank"], batch["forward_step"]
        for request_id in rows(batch, "prefill"):
            prefills[rank, request_id] = batch
        decode = rows(batch, "decode")
        for new in decode:
            if (rank, new) in first_decode or (newer is not None and newer != new):
                continue
            prefill = prefills.get((rank, new))
            if prefill is None or prefill["forward_step"] >= step:
                continue
            for old in decode if older is None else (older,):
                older_batch = first_decode.get((rank, old))
                if (
                    old in decode
                    and older_batch is not None
                    and older_batch["forward_step"] < prefill["forward_step"]
                ):
                    yield {
                        "old_rpc_id": old,
                        "new_rpc_id": new,
                        "old_decode": frame(older_batch, (old,)),
                        "new_prefill": frame(prefill, (new,)),
                        "joint_decode": frame(batch, (old, new)),
                    }
                    break
        for request_id in decode:
            first_decode.setdefault((rank, request_id), batch)


async def poll_records(log_dir, offsets, accept):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        result = accept(await asyncio.to_thread(records, log_dir, offsets))
        if result is not None:
            return result
        await asyncio.sleep(0.25)
    raise RuntimeError(f"missing required real-row forwards in {log_dir}")


async def check_refill(service, output):
    log_dir = Path(service.manifest["log_dir"])
    await service.warm(output / "warmup.jsonl")
    await service.idle()
    offsets = service_test.record_offsets(log_dir)
    old, new = Stream(0, "short"), Stream(0, "long")
    old_task = asyncio.create_task(old.run(service))
    try:
        await asyncio.wait_for(old.decode_seen.wait(), RPC_TIMEOUT)
        await new.run(service)
    finally:
        await old_task
    save_json(output / "requests.json", [old.event(), new.event()])
    if old.error or new.error:
        raise RuntimeError(f"targeted refill requests failed: {old.error}, {new.error}")

    def accept(batches):
        return next(refills(batches, older=old.id, newer=new.id), None)

    proof = await poll_records(log_dir, offsets, accept)
    save_json(output / "refill.json", proof)
    print(json.dumps(proof), flush=True)


def capacity(samples, start, end):
    active = [sample for sample in samples if start <= sample["time"] < end]
    if not active:
        raise RuntimeError("no KV/memory telemetry during target interval")
    highest = max(
        (rank for sample in active for rank in sample["kv"]["ranks"]),
        key=lambda rank: rank["active_bytes"] / rank["capacity_bytes"],
    )
    return {
        "sample_count": len(active),
        "kv_mean": float(np.mean([sample["kv"]["occupied"] for sample in active])),
        "kv_min": min(sample["kv"]["occupied"] for sample in active),
        "kv_max": max(sample["kv"]["occupied"] for sample in active),
        "mean_resident": float(
            np.mean([sample["kv"]["resident"] for sample in active])
        ),
        "mean_queued": float(np.mean([sample["kv"]["queued"] for sample in active])),
        "busiest_rank_at_peak": highest,
        "min_gpu_free_mib": [
            min(sample["gpu"][index]["free_mib"] for sample in active)
            for index in range(RANKS)
        ],
    }


async def calibrate(service, args, output):
    await service.warm(output / "warmup.jsonl")
    empty = await service.idle()
    save_json(output / "empty.json", empty)
    for concurrency in args.concurrency:
        directory = output / str(concurrency)
        directory.mkdir()
        stop, samples = asyncio.Event(), []
        sampling = asyncio.create_task(
            monitor(service, empty, samples, directory / "telemetry.jsonl", stop)
        )
        load = Load(
            service,
            slots(concurrency),
            directory,
            initial_spread_seconds=args.initial_spread_seconds,
        )
        try:
            await load.steady(10)
            begin = time.monotonic()
            await asyncio.sleep(args.seconds)
            result = capacity(samples, begin, time.monotonic())
            result["concurrency"] = concurrency
            result["initial_spread_seconds"] = args.initial_spread_seconds
            save_json(directory / "capacity.json", result)
            print(json.dumps(result), flush=True)
        finally:
            await load.stop()
            stop.set()
            await sampling
        await service.idle()


def target_forwards(batches, target, chunk):
    prefill, decode = [], []
    for batch in batches:
        match = rows(batch).get(target.id)
        if match is None:
            continue
        assert batch["dp_rank"] == target.rank, (batch, target.id)
        row = match[1]
        (prefill if row["phase"] == "prefill" else decode).append(
            frame(batch, (target.id,))
        )
    context, steps = SHAPES[target.kind]
    if (
        sum(batch["requests"][0]["q_len"] for batch in prefill) < context
        or len(decode) < steps - 1
    ):
        return None
    covered = 0
    for batch in prefill:
        row = batch["requests"][0]
        assert row["kv_len"] == covered, (covered, prefill)
        covered += row["q_len"]
    assert covered == context and len(decode) == steps - 1, (
        covered,
        context,
        len(decode),
        steps,
    )
    assert (len(prefill) > 1) == bool(chunk), prefill
    return prefill, decode


def capture_evidence(batches, target, chunk, role, injected):
    forwards = target_forwards(batches, target, chunk)
    if forwards is None:
        return None
    prefill, decode = forwards
    result = {"prefill": prefill, "output_forwards": [prefill[-1], *decode]}
    steps = {batch["forward_step"] for batch in prefill + decode}
    other_ranks = defaultdict(set)
    for batch in batches:
        if (
            batch["forward_step"] in steps
            and batch["dp_rank"] != target.rank
            and rows(batch)
        ):
            other_ranks[batch["forward_step"]].add(batch["dp_rank"])
    if role == "solo":
        assert not other_ranks and all(
            batch["real_rows"] == 1 for batch in prefill + decode
        )
        return result
    all_refills = list(refills(batches))
    target_refill = next(
        (
            refills(batches, older=target.id, newer=injected)
            if role == "old"
            else refills(batches, newer=target.id)
        ),
        None,
    )
    if target_refill is None:
        return None
    join_step = target_refill["joint_decode"]["forward_step"]
    peers = set().union(*other_ranks.values())
    if len(peers) != RANKS - 1 or not other_ranks[join_step]:
        return None
    counts = {
        rank: sum(proof["joint_decode"]["rank"] == rank for proof in all_refills)
        for rank in range(RANKS)
    }
    if counts[target.rank] < 2:
        return None
    result.update(
        target_refill=target_refill,
        refill_counts=counts,
        peer_ranks_during_target=sorted(peers),
        peer_ranks_in_joint_forward=sorted(other_ranks[join_step]),
        refill_examples=all_refills[:8],
    )
    return result


async def capture(service, args, output):
    log_dir = Path(service.manifest["log_dir"])
    await service.warm(output / "warmup.jsonl")
    empty = await service.idle()
    save_json(output / "empty.json", empty)
    plan = slots(args.concurrency) if args.role != "solo" else []
    if plan:
        save_json(output / "slots.json", plan)
    for kind in (args.kind,) if args.kind else SHAPES:
        directory = output / kind
        directory.mkdir()
        target = Stream(args.rank, kind, measure=True, fixed=args.fixed)
        offsets = service_test.record_offsets(log_dir)
        stop, samples = asyncio.Event(), []
        sampling = asyncio.create_task(
            monitor(service, empty, samples, directory / "telemetry.jsonl", stop)
        )
        load = None
        injected = None
        kv_at_submission = None
        start = time.monotonic()
        try:
            if plan:
                target.slot = plan.index((args.rank, kind))
                skip = [target.slot]
                if args.role == "old":
                    peer_slot = next(
                        slot
                        for slot, pair in enumerate(plan)
                        if pair[0] == args.rank
                        and slot != target.slot
                        and pair[1] == "short"
                    )
                    skip.append(peer_slot)
                load = Load(service, plan, directory, skip, args.initial_spread_seconds)
                await load.steady(8)
                if args.min_kv_at_submit:
                    await load.wait(
                        lambda: samples
                        and time.monotonic() - samples[-1]["time"] < 2
                        and samples[-1]["kv"]["occupied"] >= args.min_kv_at_submit,
                        RPC_TIMEOUT,
                    )
                if samples:
                    kv_at_submission = samples[-1]["kv"]["occupied"]
            if args.role == "old":
                target_task = asyncio.create_task(target.run(service))
                try:
                    await asyncio.wait_for(target.decode_seen.wait(), RPC_TIMEOUT)
                    peer = await Stream(args.rank, "short", peer_slot).run(service)
                    if peer.error:
                        raise RuntimeError(f"refill peer failed: {peer.error}")
                    injected = peer.id
                    save_json(directory / "injected.json", peer.event())
                finally:
                    await target_task
            else:
                await target.run(service)
            save_json(directory / "target.json", target.event())
            if target.error:
                raise RuntimeError(f"target failed: {target.error}")
            np.savez(directory / "output.npz", **vars(target.measured_output))
            chunk = service.manifest["prefill_chunk_size"]

            def accept(batches):
                return capture_evidence(batches, target, chunk, args.role, injected)

            result = await poll_records(log_dir, offsets, accept)
            result.update(
                rpc_request_id=target.id,
                seconds=time.monotonic() - start,
                target_seconds=target.end - target.start,
                kv_at_submission=kv_at_submission,
                initial_spread_seconds=args.initial_spread_seconds,
                capacity=capacity(samples, target.start, target.end) if plan else None,
                kind=kind,
            )
            save_json(directory / "evidence.json", result)
            print(
                json.dumps(
                    {
                        "kind": kind,
                        "role": args.role,
                        "rank": args.rank,
                        "capacity": result["capacity"],
                    }
                ),
                flush=True,
            )
        finally:
            if load:
                await load.stop()
            stop.set()
            await sampling
        await service.idle()


def load_output(directory):
    with np.load(directory / "output.npz") as data:
        return service_test.Output(**{field: data[field] for field in data.files})


def compare(args):
    failed = False
    results = []
    for kind in (args.kind,) if args.kind else SHAPES:
        ref, actual = args.reference / kind, args.actual / kind
        reference, observed = load_output(ref), load_output(actual)
        result = service_test.compare(reference, observed, kind, require_equal=False)
        first_step = (
            result["first_differences"].get("tokens", {}).get("generation_step")
        )
        if first_step is not None:
            result["same_history_through_logits_step"] = first_step
            result["natural_logits_with_shared_history"] = int(
                np.count_nonzero(
                    reference.logits[:first_step].view(np.uint32)
                    != observed.logits[:first_step].view(np.uint32)
                )
            )
        for key, folder in (("reference", ref), ("actual", actual)):
            evidence = json.loads((folder / "evidence.json").read_text())
            result[f"{key}_request_id"] = evidence["rpc_request_id"]
            result[f"{key}_forward"] = {
                field: evidence["output_forwards"][difference["generation_step"] - 1]
                for field, difference in result["first_differences"].items()
            }
        results.append(result)
        failed |= bool(result["first_differences"])
    save_json(args.output, results)
    if failed and args.require_equal:
        raise SystemExit(1)


async def run(args):
    manifest = json.loads(args.service.read_text())
    if not manifest["record"]:
        raise RuntimeError(
            "start this service with --record to preserve decode forwards"
        )
    service = Service(manifest, json.loads(args.prompts.read_text()))
    args.output.mkdir(parents=True)
    save_json(args.output / "service.json", manifest)
    try:
        await asyncio.wait_for(
            asyncio.gather(*(channel.channel_ready() for channel in service.channels)),
            60,
        )
        if args.command == "refill":
            await check_refill(service, args.output)
        elif args.command == "calibrate":
            await calibrate(service, args, args.output)
        else:
            await capture(service, args, args.output)
    finally:
        await service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for mode in ("refill", "calibrate", "capture"):
        command = commands.add_parser(mode)
        command.add_argument("--service", type=Path, required=True)
        command.add_argument("--prompts", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if mode in ("calibrate", "capture"):
            command.add_argument("--initial-spread-seconds", type=float, default=0.0)
        if mode == "calibrate":
            command.add_argument("--concurrency", type=int, nargs="+", required=True)
            command.add_argument("--seconds", type=float, default=30)
        if mode == "capture":
            command.add_argument("--concurrency", type=int)
            command.add_argument(
                "--role", choices=("solo", "old", "new"), required=True
            )
            command.add_argument("--rank", type=int, choices=range(RANKS), default=0)
            command.add_argument("--fixed", action="store_true")
            command.add_argument("--kind", choices=SHAPES)
            command.add_argument("--min-kv-at-submit", type=float, default=0.0)
    diff = commands.add_parser("compare")
    diff.add_argument("--reference", type=Path, required=True)
    diff.add_argument("--actual", type=Path, required=True)
    diff.add_argument("--output", type=Path, required=True)
    diff.add_argument("--require-equal", action="store_true")
    diff.add_argument("--kind", choices=SHAPES)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "compare":
        compare(args)
    else:
        if args.command == "capture" and args.role != "solo" and not args.concurrency:
            parser.error("high-load capture needs --concurrency")
        if args.command == "capture" and not 0.0 <= args.min_kv_at_submit <= 1.0:
            parser.error("--min-kv-at-submit must be between 0 and 1")
        if args.command in ("calibrate", "capture") and args.initial_spread_seconds < 0:
            parser.error("--initial-spread-seconds must be nonnegative")
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
