"""Run with torchrun --nproc-per-node=8 -m ...glm53_smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import platform

from .settings import BASELINE, configure


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--phase", choices=("prefill", "decode"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--write-golden", action="store_true")
    p.add_argument("--seq-len", type=int, default=131072)
    p.add_argument("--seed", type=int, default=531210)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=9)
    p.add_argument("--relative-l2-limit", type=float, default=0.0)
    p.add_argument("--max-abs-limit", type=float, default=0.0)
    p.add_argument("--max-regression", type=float, default=0.10)
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 8:
        p.error("this smoke requires exactly 8 ranks")
    if args.seq_len != 131072 or args.iterations < 3 or args.warmup < 1:
        p.error("baseline requires seq_len=131072, iterations>=3 and warmup>=1")
    configure(args.phase)
    import torch
    import torch.distributed as dist
    from .runtime import build_model, make_cache, make_inputs
    from .block import LayerBlock
    from .fixture import prepare_fixture
    from .golden import (
        compare,
        layer_logits,
        save_exclusive,
        tensor_sha256,
        weight_manifest,
        file_sha256,
    )

    rank = int(os.environ["RANK"])
    batch = 8 if args.phase == "prefill" else 48
    golden_path = args.golden / f"rank{rank}.pt"
    if args.write_golden == golden_path.exists():
        raise FileExistsError(f"golden mode/path mismatch: {golden_path}")
    if not args.write_golden and not (args.golden / "COMPLETE.json").exists():
        raise RuntimeError(
            "golden is incomplete; all eight ranks must have passed creation"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        gpt = build_model(args.checkpoint, args.phase, batch, args.seq_len)
        model = gpt.py_model
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (props.major, props.minor) != (10, 3):
            raise RuntimeError(
                f"this baseline was built for Runtime SM103, got {props.major}.{props.minor}"
            )
        cache = make_cache(model, args.phase, batch, args.seq_len)
        seed = args.seed + (rank if args.phase == "decode" else 0)
        inputs = make_inputs(args.phase, batch, args.seq_len, seed)
        snapshot, fixture_sha = prepare_fixture(
            model, cache, inputs, args.phase, seed, args.golden, rank, args.write_golden
        )
        block = LayerBlock(model, inputs)

        def restore():
            snapshot.restore()
            block.restore_inputs()

        metadata = dict(
            schema=2,
            fixture_sha256=fixture_sha,
            source_layers=[4, 5, 6, 7],
            phase=args.phase,
            rank=rank,
            world_size=8,
            local_batch=batch,
            seq_len=args.seq_len,
            seed=seed,
            mtp=False,
            history=(
                "zero initial state"
                if args.phase == "prefill"
                else "seeded synthetic BF16 KV / FP32 recurrent state"
            ),
            positions=block.positions.cpu().tolist(),
            input_ids_sha256=tensor_sha256(inputs.input_ids),
            input_hidden_sha256=tensor_sha256(block.hidden),
            weights=weight_manifest(gpt.weight),
        )
        print(f"SMOKE rank={rank}: fixture and weight manifest ready", flush=True)
        (args.output / f"input_manifest_rank{rank}.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        for _ in range(args.warmup):
            restore()
            block()
        torch.cuda.synchronize()

        restore()
        reference = layer_logits(model, block(audit=True), block.positions)
        restore()
        repeated = layer_logits(model, block(audit=True), block.positions)
        repeat_metrics = compare(repeated, reference)
        if not all(x["passed"] for x in repeat_metrics.values()):
            (args.output / f"repeat_failure_rank{rank}.json").write_text(
                json.dumps(repeat_metrics, indent=2)
            )
            raise AssertionError("fixed-state baseline is not repeatable")
        print(f"SMOKE rank={rank}: full layer repeat check passed", flush=True)
        del repeated
        if args.write_golden:
            save_exclusive(golden_path, reference, metadata)
            golden_metrics = repeat_metrics
        else:
            expected_hash = json.loads(golden_path.with_suffix(".json").read_text())[
                "sha256"
            ]
            if file_sha256(golden_path) != expected_hash:
                raise AssertionError("golden file checksum changed")
            expected = torch.load(
                golden_path, map_location="cpu", weights_only=True, mmap=True
            )
            if expected["metadata"] != metadata:
                raise AssertionError(
                    "input/weight/state contract changed; golden remains untouched"
                )
            golden_metrics = compare(
                reference,
                expected["tensors"],
                args.relative_l2_limit,
                args.max_abs_limit,
            )

        fn = block
        graph = None
        if args.phase == "decode":
            restore()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = block()
            restore()
            graph.replay()
            # Compare final mHC tensor from the actual timed graph to eager.
            torch.testing.assert_close(
                graph_output[block.positions].cpu(),
                reference["layer3.hidden"],
                rtol=0,
                atol=0,
            )
            fn = graph.replay

        # CPU golden I/O can idle the GPU for minutes. Warm up again before
        # measurement so clock ramp-up and lazy compilation are excluded.
        for _ in range(args.warmup):
            restore()
            fn()
        torch.cuda.synchronize()
        timings = []
        timing_start = time.time()
        for _ in range(args.iterations):
            restore()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            fn()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end), device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            timings.append(float(elapsed))
        timing_end = time.time()
        median = statistics.median(timings)
        target = 598.0 if args.phase == "prefill" else 1.89
        result = dict(
            phase=args.phase,
            rank=rank,
            target_ms=target,
            median_ms=median,
            timing_window_unix=[timing_start, timing_end],
            max_rank_samples_ms=timings,
            performance_pass=median <= target * (1 + args.max_regression),
            max_regression=args.max_regression,
            numeric_pass=all(x["passed"] for x in golden_metrics.values()),
            golden_metrics=golden_metrics,
            repeat_metrics=repeat_metrics,
            cuda_graph=graph is not None,
            scope="four decoder layers including final deferred mHC post; embedding, final norm/head, cache restore and metadata preparation excluded",
            gpu=dict(
                name=props.name,
                major=props.major,
                minor=props.minor,
                sms=props.multi_processor_count,
            ),
            max_allocated_bytes=torch.cuda.max_memory_allocated(),
            software=dict(
                torch=str(torch.__version__),
                cuda=str(torch.version.cuda),
                python=platform.python_version(),
                revision=os.environ.get("RTP_SMOKE_REVISION", "unrecorded"),
            ),
            configuration={
                key: os.environ.get(key)
                for key in list(BASELINE)
                + [
                    "GLM53_PREFILL_MLA_CP",
                    "GLM53_PREFILL_SEQUENCE_PARALLEL",
                    "GLM53_INDEXER_FUSED_Q_QUANT",
                ]
            },
        )
        if args.trace:
            from .trace import profile_block

            (args.output / f"performance_rank{rank}.json").write_text(
                json.dumps(result, indent=2)
            )
            profile_block(fn, restore, args.phase, args.output, median)
        (args.output / f"result_rank{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        dist.barrier()
        passed = torch.tensor(
            int(result["numeric_pass"] and result["performance_pass"]), device="cuda"
        )
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        if not int(passed):
            raise AssertionError(
                f"smoke failed: numeric={result['numeric_pass']}, {median:.4f} ms vs {target:.4f} ms"
            )
        if args.write_golden and rank == 0:
            complete = dict(
                schema=2,
                phase=args.phase,
                ranks=8,
                files={
                    p.name: json.loads(p.read_text())["sha256"]
                    for p in args.golden.glob("rank*.json")
                },
                median_ms=median,
                created_unix=time.time(),
            )
            with (args.golden / "COMPLETE.json").open("x") as stream:
                json.dump(complete, stream, indent=2)
        print(
            f"SMOKE PASS rank={rank} phase={args.phase} median_ms={median:.6f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
