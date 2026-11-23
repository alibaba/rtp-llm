"""Client for the M11 ratio scheduler test service. See --help for the run modes."""

import argparse
import asyncio
import csv
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import grpc
import numpy as np

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.test.batch_invariant import service_test

VOCAB_SIZE = 248320
SHAPES = {"short": (257, 128), "long": (4577, 32)}
RANKS = 8
RPC_TIMEOUT = 600
REQUEST_IDS = itertools.count(time.time_ns())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_event(file, event):
    file.write(json.dumps(event, separators=(",", ":")) + "\n")


def prepare(checkpoint, path):
    assert service_test.read_vocab_size(checkpoint) == VOCAB_SIZE
    encode = service_test.load_encoder(checkpoint, m11=True)
    manifest = {}
    for kind, (length, steps) in SHAPES.items():
        prompt = service_test.make_prompt(
            encode,
            length,
            " A notebook lays open." if kind == "short" else " The lantern is blue.",
        )
        continuation = service_test.CONTINUATION * (steps // 4)
        expected = encode(continuation)
        assert len(expected) == steps, (kind, len(expected), steps)
        manifest[kind] = {
            "token_ids": prompt,
            "steps": steps,
            "continuation": continuation,
            "continuation_ids": expected,
        }
    save_json(path, manifest)


def slots(concurrency):
    if concurrency < RANKS * 4 or concurrency % 4:
        raise ValueError("concurrency must be a multiple of four and at least 32")
    long_count = concurrency // 4
    return [
        (index % RANKS, "long" if index < long_count else "short")
        for index in range(concurrency)
    ]


class Stream:
    def __init__(self, rank, kind, slot=None, *, fixed=False, measure=False):
        self.rank = rank
        self.kind = kind
        self.slot = slot
        self.fixed = fixed
        self.measure = measure
        self.id = next(REQUEST_IDS)
        self.decode_seen = asyncio.Event()
        self.arrivals = []
        self.count = 0
        self.error = None
        self.measured_output = None

    async def run(self, service):
        workload = service.prompts[self.kind]
        steps = workload["steps"]
        request = service_test.make_request(
            workload["token_ids"],
            steps,
            measure=self.measure,
            fixed=self.fixed,
            continuation=workload["continuation"],
            request_id=self.id,
            timeout_s=RPC_TIMEOUT,
        )
        request.generate_config.random_seed.value = 42
        request.generate_config.force_disable_sp_run = True
        tokens, logits, sampled = [], [], []
        self.start = time.monotonic()
        self.wall_start_ns = time.time_ns()
        finished = False
        try:
            async for response in service.stubs[self.rank].GenerateStreamCall(
                request, timeout=RPC_TIMEOUT
            ):
                now = time.monotonic()
                if response.error_info.error_code:
                    raise RuntimeError(str(response.error_info))
                output = response.flatten_output
                if output.finished:
                    finished = all(output.finished)
                if not output.output_ids.int32_data:
                    continue
                ids = service_test.read_tensor(
                    output.output_ids, pb.TensorPB.INT32, "int32_data", "<i4"
                ).ravel()
                if ids.min() < 0 or ids.max() >= VOCAB_SIZE:
                    raise RuntimeError(f"out-of-vocabulary output IDs: {ids}")
                self.count += ids.size
                self.arrivals.append((now, ids.size))
                if self.count >= 2:
                    self.decode_seen.set()
                if self.measure:
                    row, probability = service_test.read_generation_step(
                        output, ids, VOCAB_SIZE
                    )
                    tokens.append(ids.item())
                    logits.append(row)
                    sampled.append(probability)
                elif (
                    output.logits.ByteSize()
                    or output.all_probs.ByteSize()
                    or any(info.softmax_probs.ByteSize() for info in output.aux_info)
                ):
                    raise RuntimeError(
                        "unexpected logits or probabilities on load stream"
                    )
            if self.count != steps or not finished:
                raise RuntimeError(f"received {self.count}/{steps} tokens, {finished=}")
        except (grpc.RpcError, RuntimeError, AssertionError, ValueError) as error:
            self.error = str(error) or repr(error)
        self.end = time.monotonic()
        if self.measure and self.error is None:
            self.measured_output = await asyncio.to_thread(
                service_test.generation_output,
                tokens,
                logits,
                sampled,
                steps,
                VOCAB_SIZE,
            )
            if self.fixed:
                np.testing.assert_array_equal(
                    self.measured_output.tokens, workload["continuation_ids"]
                )
        return self

    def event(self):
        return {
            "rpc_request_id": self.id,
            "rank": self.rank,
            "kind": self.kind,
            "slot": self.slot,
            "sent": self.start,
            "sent_unix_ns": self.wall_start_ns,
            "ended": self.end,
            "received": self.count,
            "arrivals": self.arrivals,
            "error": self.error,
        }


class Service:
    def __init__(self, manifest, prompts):
        self.manifest = manifest
        self.prompts = prompts
        assert len(manifest["rpc_addresses"]) == RANKS
        self.channels = [
            grpc.aio.insecure_channel(
                address, options=[("grpc.max_receive_message_length", -1)]
            )
            for address in manifest["rpc_addresses"]
        ]
        self.stubs = [RpcServiceStub(channel) for channel in self.channels]

    async def worker(self, rank):
        status = await self.stubs[rank].GetWorkerStatus(
            pb.StatusVersionPB(
                include_kv_cache_pools=True, latest_finished_version=(2**63 - 1)
            ),
            timeout=5,
        )
        assert (status.dp_rank, status.tp_size, status.dp_size) == (rank, 1, RANKS)
        assert status.role == "RoleType.PDFUSION", status.role
        assert (status.max_seq_len, status.max_batch_tokens_size) == (8192, 8192)
        pools = [
            {field.name: getattr(pool, field.name) for field in pool.DESCRIPTOR.fields}
            for pool in status.kv_cache_pools
        ]
        assert pools and all(pool["block_size_bytes"] for pool in pools), status
        resident = sum(
            task.phase in (pb.TASK_PHASE_KV_ALLOCATED, pb.TASK_PHASE_RUNNING)
            for task in status.running_task_info
        )
        return {
            "rank": rank,
            "precision": status.precision,
            "resident": resident,
            "queued": len(status.running_task_info) - resident,
            "tasks": [
                [task.request_id, task.input_length, task.phase, task.iterate_count]
                for task in status.running_task_info
            ],
            "pools": pools,
        }

    async def workers(self):
        return await asyncio.gather(*(self.worker(rank) for rank in range(RANKS)))

    async def idle(self):
        async def wait_for_idle():
            while True:
                workers = await self.workers()
                if all(
                    not (worker["resident"] or worker["queued"] or worker["tasks"])
                    for worker in workers
                ):
                    return workers
                await asyncio.sleep(0.25)

        return await asyncio.wait_for(wait_for_idle(), 30)

    async def warm(self, path):
        with path.open("w", buffering=1) as file:
            for kind in SHAPES:
                results = await asyncio.gather(
                    *(Stream(rank, kind).run(self) for rank in range(RANKS))
                )
                for stream in results:
                    write_event(file, stream.event())
                if any(stream.error for stream in results):
                    raise RuntimeError(f"real prompt warmup failed: {path}")

    async def close(self):
        await asyncio.gather(*(channel.close() for channel in self.channels))


def usage(workers, empty):
    per_rank = []
    for worker, initial in zip(workers, empty, strict=True):
        assert worker["rank"] == initial["rank"]
        pools = []
        initial_pools = {pool["pool_name"]: pool for pool in initial["pools"]}
        for pool in worker["pools"]:
            idle = initial_pools[pool["pool_name"]]
            assert pool["total_blocks"] == idle["total_blocks"]
            size = pool["block_size_bytes"]
            # Empty available excludes block zero. Admission reserve is reported
            # separately because it sets a scheduler threshold on these blocks.
            active = pool["active_blocks"] - idle["active_blocks"]
            capacity = idle["available_blocks"]
            pools.append(
                {
                    "pool": pool["pool_name"],
                    "active": active,
                    "capacity": capacity,
                    "active_bytes": active * size,
                    "capacity_bytes": capacity * size,
                    "reserve_bytes": idle["reserve_blocks"] * size,
                }
            )
        per_rank.append(
            {
                "rank": worker["rank"],
                "resident": worker["resident"],
                "queued": worker["queued"],
                "pools": pools,
                "active_bytes": sum(pool["active_bytes"] for pool in pools),
                "capacity_bytes": sum(pool["capacity_bytes"] for pool in pools),
            }
        )
    return {
        "ranks": per_rank,
        "occupied": sum(rank["active_bytes"] for rank in per_rank)
        / sum(rank["capacity_bytes"] for rank in per_rank),
        "resident": sum(rank["resident"] for rank in per_rank),
        "queued": sum(rank["queued"] for rank in per_rank),
    }


async def smi():
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(stderr.decode().strip())
    rows = list(csv.reader(stdout.decode().splitlines()))
    assert len(rows) == RANKS, rows
    return [
        dict(
            zip(
                ("gpu", "total_mib", "used_mib", "free_mib"), map(int, row), strict=True
            )
        )
        for row in rows
    ]


async def monitor(service, empty, samples, path, stop):
    with path.open("w", buffering=1) as file:
        while not stop.is_set():
            event = {"time": time.monotonic(), "unix_ns": time.time_ns()}
            try:
                workers, gpu = await asyncio.gather(service.workers(), smi())
                event.update(workers=workers, gpu=gpu, kv=usage(workers, empty))
                samples.append(event)
            except (grpc.RpcError, RuntimeError, ValueError, AssertionError) as error:
                event["error"] = str(error) or repr(error)
            write_event(file, event)
            await asyncio.sleep(max(0, 1 - (time.monotonic() - event["time"])))


class Load:
    def __init__(self, service, plan, output, skip=(), initial_spread_seconds=0):
        self.service = service
        self.running = asyncio.Event()
        self.running.set()
        self.events = []
        self.streams = {}
        self.expected = {pair for slot, pair in enumerate(plan) if slot not in skip}
        self.file = (output / "requests.jsonl").open("w", buffering=1)
        self.tasks = []
        counts, ordinals = Counter(plan), Counter()
        for slot, pair in enumerate(plan):
            rank, kind = pair
            phase = (ordinals[pair] + rank / RANKS) / counts[pair]
            ordinals[pair] += 1
            if slot not in skip:
                self.tasks.append(
                    asyncio.create_task(
                        self.fill(slot, rank, kind, initial_spread_seconds * phase)
                    )
                )

    async def fill(self, slot, rank, kind, initial_delay):
        if initial_delay:
            await asyncio.sleep(initial_delay)
        while self.running.is_set():
            stream = Stream(rank, kind, slot)
            self.streams[slot] = stream
            await stream.run(self.service)
            event = stream.event()
            self.events.append(event)
            write_event(self.file, event)
            if stream.error:
                await asyncio.sleep(0.25)

    async def wait(self, predicate, seconds):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.25)

        await asyncio.wait_for(poll(), seconds)

    async def steady(self, seconds):
        started = time.monotonic()

        def ready():
            succeeded = {
                (event["rank"], event["kind"])
                for event in self.events
                if not event["error"]
            }
            return (
                len(self.streams) == len(self.tasks)
                and self.expected.issubset(succeeded)
                and time.monotonic() - started >= seconds
            )

        await self.wait(ready, RPC_TIMEOUT)

    async def stop(self):
        self.running.clear()
        try:
            await asyncio.gather(*self.tasks)
        finally:
            self.file.close()


def summarize(events, samples, begin, end):
    relevant = [event for event in events if begin <= event["ended"] < end]
    succeeded = [event for event in relevant if not event["error"]]
    tokens = sum(
        count
        for event in events
        if not event["error"]
        for arrival, count in event["arrivals"]
        if begin <= arrival < end
    )
    window = [sample for sample in samples if begin <= sample["time"] < end]
    if not window or not relevant or not succeeded:
        raise RuntimeError("timing window is missing KV samples or completed requests")
    busiest = max(
        (rank for sample in window for rank in sample["kv"]["ranks"]),
        key=lambda rank: rank["active_bytes"] / rank["capacity_bytes"],
    )
    return {
        "seconds": end - begin,
        "begin_monotonic": begin,
        "end_monotonic": end,
        "output_tokens": tokens,
        "tok_per_s": tokens / (end - begin),
        "attempts_completed": len(relevant),
        "success_rate": len(succeeded) / len(relevant),
        "successes_by_rank": Counter(event["rank"] for event in succeeded),
        "p95_ttft_ms": float(
            np.percentile(
                [
                    1000 * (event["arrivals"][0][0] - event["sent"])
                    for event in succeeded
                ],
                95,
            )
        ),
        "p95_tpot_ms": float(
            np.percentile(
                [
                    1000
                    * (event["arrivals"][-1][0] - event["arrivals"][0][0])
                    / (event["received"] - 1)
                    for event in succeeded
                ],
                95,
            )
        ),
        "kv_sample_count": len(window),
        "kv_mean": float(np.mean([sample["kv"]["occupied"] for sample in window])),
        "kv_min": min(sample["kv"]["occupied"] for sample in window),
        "kv_max": max(sample["kv"]["occupied"] for sample in window),
        "mean_resident": float(
            np.mean([sample["kv"]["resident"] for sample in window])
        ),
        "mean_queued": float(np.mean([sample["kv"]["queued"] for sample in window])),
        "busiest_rank_at_peak": busiest,
        "min_gpu_free_mib": {
            index: min(sample["gpu"][index]["free_mib"] for sample in window)
            for index in range(RANKS)
        },
    }


async def scan(service, args, output):
    if service.manifest["record"]:
        raise RuntimeError("start an unrecorded service for the throughput scan")
    await service.warm(output / "warmup.jsonl")
    empty = await service.idle()
    save_json(output / "empty.json", {"workers": empty, "gpu": await smi()})
    for concurrency in args.concurrency:
        directory = output / str(concurrency)
        directory.mkdir()
        plan = slots(concurrency)
        save_json(directory / "slots.json", plan)
        stop = asyncio.Event()
        samples = []
        sampling = asyncio.create_task(
            monitor(service, empty, samples, directory / "telemetry.jsonl", stop)
        )
        load = Load(service, plan, directory)
        windows = []
        try:
            await load.steady(args.settle_seconds)
            for _ in range(args.windows):
                begin = time.monotonic()
                await asyncio.sleep(args.window_seconds)

                def replenished():
                    counts = Counter(
                        event["rank"]
                        for event in load.events
                        if event["ended"] >= begin and not event["error"]
                    )
                    return sum(counts.values()) >= concurrency and all(
                        counts[rank] >= 2 for rank in range(RANKS)
                    )

                await load.wait(replenished, args.window_timeout)
                end = time.monotonic()
                windows.append((begin, end))
                save_json(directory / "windows.json", windows)
                print(
                    f"concurrency {concurrency}: completed window {len(windows)}",
                    flush=True,
                )
        finally:
            await load.stop()
            stop.set()
            await sampling
        result = [summarize(load.events, samples, *window) for window in windows]
        save_json(directory / "throughput.json", result)
        print(json.dumps({"concurrency": concurrency, "windows": result}), flush=True)
        await service.idle()


async def run(args):
    manifest = json.loads(args.service.read_text())
    service = Service(manifest, json.loads(args.prompts.read_text()))
    args.output.mkdir(parents=True)
    try:
        await asyncio.wait_for(
            asyncio.gather(*(channel.channel_ready() for channel in service.channels)),
            60,
        )
        await scan(service, args, args.output)
    finally:
        await service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_args = commands.add_parser("prepare")
    prepare_args.add_argument("--checkpoint", type=Path, required=True)
    prepare_args.add_argument("--prompts", type=Path, required=True)
    scan_args = commands.add_parser("scan")
    scan_args.add_argument("--service", type=Path, required=True)
    scan_args.add_argument("--prompts", type=Path, required=True)
    scan_args.add_argument("--output", type=Path, required=True)
    scan_args.add_argument(
        "--concurrency", type=int, nargs="+", default=[64, 96, 128, 144, 152, 156, 160]
    )
    scan_args.add_argument("--windows", type=int, default=3)
    scan_args.add_argument("--window-seconds", type=float, default=45)
    scan_args.add_argument("--window-timeout", type=float, default=240)
    scan_args.add_argument("--settle-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.checkpoint, args.prompts)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
