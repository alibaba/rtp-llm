"""Fresh, complete sparse-attention-unit comparison (not a kernel benchmark).

Boundary: equal BF16 Q[192+64], K[512], K_rope[64], Wkc and logical metadata
through native preparation/cache write to attn_out[512]. Wvc/Wo are excluded.
Both native histories are built from the SAME generated BF16 chunks. A single
independent FP32 golden consumes saved ORIGINAL selected tokens, not either
quantized cache. This file does not import the previous benchmark or helper.

The production RTP writer consumes (mutates) Q/K_rope. Staging supplies equal
disposable copies before each invocation; it is outside timing, explicitly NOT
an immutable-input adapter benchmark. Each CUDA event interval includes the
entire forward, including first-layer FlashMLA scheduler rebuilding.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from glm53_attention_unit_paths import FlashUnit, TrtUnit

TOPK, PAGE = 2048, 64
NOPE, LATENT, ROPE = 192, 512, 64
TOLERANCE = {"atol": 0.0008, "rtol": 0.02, "relative_rms": 0.005, "cosine": 0.99999}


class CommonData:
    """Original inputs, logical history, and ONE unquantized golden for both units."""

    def __init__(self, batch, context, heads, queries=1, seed=53):
        if batch < 1 or queries < 1 or context < queries or heads not in (8, 16, 64):
            raise ValueError("Invalid GLM53 unit shape")
        self.batch, self.context, self.heads, self.queries = (
            batch,
            context,
            heads,
            queries,
        )
        self.tokens = batch * queries
        self.device = torch.device("cuda:0")
        self.seed = seed
        self.pages_per_request = math.ceil(context / PAGE)
        self.pages = batch * self.pages_per_request + 1
        generator = torch.Generator().manual_seed(seed)
        self.page_table = (
            (torch.randperm(self.pages - 1, generator=generator) + 1)
            .view(batch, self.pages_per_request)
            .to(device=self.device, dtype=torch.int32)
        )
        self.request_ids = torch.arange(
            batch, device=self.device, dtype=torch.int32
        ).repeat_interleave(queries)
        self.positions = torch.arange(
            context - queries, context, device=self.device, dtype=torch.int32
        ).repeat(batch)
        self.seq_lens = self.positions + 1
        self.slots = (
            self.page_table[
                self.request_ids.long(), self.positions.long() // PAGE
            ].long()
            * PAGE
            + self.positions % PAGE
        )
        self.topk_cpu = torch.full((self.tokens, TOPK), -1, dtype=torch.int32)
        chooser = random.Random(seed + 1)
        for row in range(self.tokens):
            position = context - queries + row % queries
            chosen = chooser.sample(range(position), min(position, TOPK - 1)) + [
                position
            ]
            chooser.shuffle(chosen)
            self.topk_cpu[row, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
        self.logical_topk = self.topk_cpu.to(self.device)
        frequency = 10000.0 ** (
            -torch.arange(0, ROPE, 2, device=self.device, dtype=torch.float32) / ROPE
        )
        angle = (
            torch.arange(context, device=self.device, dtype=torch.float32)[:, None]
            * frequency
        )
        self.cos_sin = torch.cat((angle.cos(), angle.sin()), dim=1)
        rng = torch.Generator(device=self.device).manual_seed(seed + 2)
        self.q = torch.randn(
            (self.tokens, heads, NOPE + ROPE), device=self.device, generator=rng
        ).bfloat16()
        self.k = self.raw_k(self.tokens, rng)
        self.k_rope = torch.randn(
            (self.tokens, ROPE), device=self.device, generator=rng
        ).bfloat16()
        self.wkc = (
            torch.randn((heads, NOPE, LATENT), device=self.device, generator=rng)
            / math.sqrt(NOPE)
        ).bfloat16()
        # All query rows, not sampled references. Retain exact original selected
        # tokens at generation time; never regenerate them with a guessed seed.
        self.selected_k = torch.zeros(
            (self.tokens, TOPK, LATENT), device=self.device, dtype=torch.bfloat16
        )
        self.selected_rope = torch.zeros(
            (self.tokens, TOPK, ROPE), device=self.device, dtype=torch.bfloat16
        )
        self.selected_written = torch.zeros(
            (self.tokens, TOPK), device=self.device, dtype=torch.bool
        )

    def raw_k(self, count, generator):
        value = torch.randn((count, LATENT), device=self.device, generator=generator)
        return (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()

    def retain_original(self, request, first, key, rope):
        logical = self.topk_cpu[request * self.queries : (request + 1) * self.queries]
        row, col = ((logical >= first) & (logical < first + key.shape[0])).nonzero(
            as_tuple=True
        )
        if row.numel() == 0:
            return
        offset = (logical[row, col].long() - first).to(self.device)
        dst_row = (row + request * self.queries).to(self.device)
        dst_col = col.to(self.device)
        self.selected_k[dst_row, dst_col] = key[offset]
        self.selected_rope[dst_row, dst_col] = rope[offset]
        self.selected_written[dst_row, dst_col] = True

    def populate(self, paths, chunk_tokens):
        rng = torch.Generator(device=self.device).manual_seed(self.seed + 3)
        changed = torch.zeros((), device=self.device, dtype=torch.bool)
        for request in range(self.batch):
            for first in range(0, self.context - self.queries, chunk_tokens):
                stop = min(first + chunk_tokens, self.context - self.queries)
                key = self.raw_k(stop - first, rng)
                rope = torch.randn(
                    (stop - first, ROPE), device=self.device, generator=rng
                ).bfloat16()
                positions = torch.arange(
                    first, stop, device=self.device, dtype=torch.int32
                )
                slots = (
                    self.page_table[request, positions.long() // PAGE].long() * PAGE
                    + positions % PAGE
                )
                self.retain_original(request, first, key, rope)
                # Direct CUDA writes need not increment Tensor._version.
                # Compare bytes on GPU, accumulating a flag for ALL chunks.
                original_key, original_rope = key.clone(), rope.clone()
                for path in paths:
                    path.append_history(key, rope, positions, slots)
                    changed.logical_or_(
                        torch.any(
                            key.view(torch.uint8) != original_key.view(torch.uint8)
                        )
                    )
                    changed.logical_or_(
                        torch.any(
                            rope.view(torch.uint8) != original_rope.view(torch.uint8)
                        )
                    )
            start = request * self.queries
            self.retain_original(
                request,
                self.context - self.queries,
                self.k[start : start + self.queries],
                self.k_rope[start : start + self.queries],
            )
        if bool(changed):
            raise AssertionError("Native history writer modified shared original bytes")
        torch.testing.assert_close(
            self.selected_written, self.logical_topk >= 0, atol=0, rtol=0
        )

    @staticmethod
    def rotate(value, cos_sin):
        first, second = value.float().chunk(2, dim=-1)
        cos, sin = cos_sin.float().chunk(2, dim=-1)
        return torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )

    def golden(self):
        """FP32 Wkc + FP32 RoPE + attention using original, unquantized KV."""
        outputs = []
        weight = self.wkc.float()
        for row in range(self.tokens):
            query = self.q[row].float()
            q_latent = torch.bmm(query[:, :NOPE, None].transpose(1, 2), weight).squeeze(
                1
            )
            q_rope = self.rotate(query[:, NOPE:], self.cos_sin[self.positions[row]])
            logical = self.logical_topk[row].long()
            k_rope = self.rotate(
                self.selected_rope[row], self.cos_sin[logical.clamp_min(0)]
            )
            key = self.selected_k[row].float()
            logits = (q_latent @ key.T + q_rope @ k_rope.T) * (1.0 / 16.0)
            logits.masked_fill_(logical[None] < 0, -torch.inf)
            outputs.append(logits.softmax(-1) @ key)
        return torch.stack(outputs)

    def shape(self):
        return dict(
            batch=self.batch,
            context=self.context,
            heads=self.heads,
            queries=self.queries,
        )


def difference(actual, reference):
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    finite = bool(torch.isfinite(actual).all() & torch.isfinite(reference).all())
    norm = reference.norm().item()
    nrms = delta.norm().item() / max(norm, 1e-20)
    cosine = (
        torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        ).item()
        if norm
        else float(actual.norm().item() == 0)
    )
    outside = int(
        (delta.abs() > TOLERANCE["atol"] + TOLERANCE["rtol"] * reference.abs()).sum()
    )
    return dict(
        finite=finite,
        relative_rms=nrms,
        cosine=cosine,
        max_abs=delta.abs().max().item(),
        mean_abs=delta.abs().mean().item(),
        outside_tolerance=outside,
        elements=delta.numel(),
        passed=finite
        and outside == 0
        and nrms <= TOLERANCE["relative_rms"]
        and cosine >= TOLERANCE["cosine"],
    )


class PrecisionRejected(AssertionError):
    pass


def test_precision_metrics():
    zero = torch.zeros(8)
    require_precision(difference(zero, zero))
    for actual in (torch.ones(8), torch.full((8,), float("nan"))):
        try:
            require_precision(difference(actual, zero))
        except PrecisionRejected:
            continue
        raise AssertionError("Precision gate incorrectly accepted a bad output")


def require_precision(metrics):
    if not metrics["passed"]:
        raise PrecisionRejected(
            "Complete attention unit does not meet the fixed precision gate"
        )


def check_units(common, paths):
    originals = {
        name: getattr(common, name).clone()
        for name in (
            "q",
            "k",
            "k_rope",
            "logical_topk",
            "wkc",
            "positions",
            "cos_sin",
            "page_table",
            "request_ids",
            "seq_lens",
            "slots",
        )
    }
    outputs = {}
    for name, path in paths.items():
        path.stage()
        # No branch-specific quantization is allowed before forward begins.
        path.assert_staged_equal()
        outputs[name] = path.forward().clone()
        for field, original in originals.items():
            if not torch.equal(
                getattr(common, field).contiguous().view(torch.uint8),
                original.contiguous().view(torch.uint8),
            ):
                raise AssertionError(f"Shared source bytes changed: {field}")
    golden = common.golden()
    precision = dict(
        rtp_vs_common_fp32=difference(outputs["rtp"], golden),
        trt_vs_common_fp32=difference(outputs["trt"], golden),
        trt_vs_rtp=difference(outputs["trt"], outputs["rtp"]),
    )
    if not all(item["finite"] for item in precision.values()):
        raise AssertionError("Non-finite complete-unit output or common golden")
    status = "precision_passed"
    try:
        require_precision(precision["trt_vs_rtp"])
    except PrecisionRejected:
        status = "precision_rejected"
    return precision, status


def capture(path, scrub):
    for _ in range(3):
        path.stage()
        path.forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        # Equal consumable input values at the unit boundary on every replay.
        path.stage()
        if scrub is not None:
            scrub.zero_()
        begin.record()
        output = path.forward()
        end.record()
    return graph, begin, end, output


def measure(paths, rounds, cache_mode):
    scrub = (
        torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        if cache_mode == "cold-l2"
        else None
    )
    graphs = {name: capture(path, scrub) for name, path in paths.items()}
    for _ in range(5):
        for graph, _, _, _ in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in paths}
    for iteration in range(rounds):
        for name in ("rtp", "trt") if iteration % 2 == 0 else ("trt", "rtp"):
            graph, begin, end, _ = graphs[name]
            graph.replay()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000.0)
    result = {}
    for name, values in samples.items():
        ordered = sorted(values)
        result[name] = dict(
            median_us=statistics.median(values),
            p95_us=ordered[math.ceil(0.95 * len(values)) - 1],
            samples_us=values,
        )
    return result


def graph_input_test(common, paths):
    """Replay must consume NEW Q/K/RoPE and TopK, not captured stale values."""
    graph_objects = {name: capture(path, None) for name, path in paths.items()}
    previous = {}
    for name, (graph, _, _, output) in graph_objects.items():
        graph.replay()
        previous[name] = output.clone()
    common.q.neg_()
    common.k.neg_()
    common.k_rope.neg_()
    # Reorder the same causal set, preserving the native valid-prefix layout.
    for row, length in enumerate(common.seq_lens.tolist()):
        length = min(length, TOPK)
        common.logical_topk[row, :length] = common.logical_topk[row, :length].flip(0)
    for name, (graph, _, _, output) in graph_objects.items():
        graph.replay()
        replay = output.clone()
        paths[name].stage()
        eager = paths[name].forward()
        torch.testing.assert_close(replay, eager, atol=0.002, rtol=0.01)
        if torch.equal(replay, previous[name]):
            raise AssertionError("Graph ignored modified common Q/K/RoPE")
    common.q.neg_()
    common.k.neg_()
    common.k_rope.neg_()
    common.logical_topk.copy_(common.topk_cpu)


def snapshot():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.used,memory.free,utilization.gpu,clocks.sm",
            "--format=csv",
        ],
        text=True,
        timeout=15,
    )


def save(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--contexts", nargs="+", type=int, default=[30000])
    parser.add_argument("--heads", nargs="+", type=int, default=[64, 8])
    parser.add_argument("--queries", nargs="+", type=int, default=[1])
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--cache-mode", choices=["warm", "cold-l2"], default="warm")
    args = parser.parse_args()
    test_precision_metrics()
    if min(args.rounds, args.chunk_tokens) < 1:
        parser.error("rounds and chunk-tokens must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("This native TRT unit requires Blackwell SM10x")
    report = dict(
        schema="glm53_complete_attention_unit_v1",
        args=vars(args),
        started_utc=datetime.now(timezone.utc).isoformat(),
        host=platform.node(),
        python=sys.version,
        executable=sys.executable,
        gpu_before=snapshot(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "flashinfer-python", "flash-mla", "triton")
        },
        tolerance=TOLERANCE,
        rows=[],
        boundary="equal consumable pre-RoPE BF16 Q/K + Wkc -> attn_out512; no Wvc/Wo",
        golden="same ORIGINAL BF16 selected KV, FP32 Wkc/RoPE/attention, ALL query rows",
        flash_scheduler="first-layer per-forward rebuild included",
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("glm53_attention_unit_paths.py"),
            )
        },
    )
    save(args.output, report)
    # Correctness of test mechanics is independent of the precision gate.
    try:
        check = CommonData(2, 73, 8, queries=3, seed=args.seed)
        units = {"rtp": FlashUnit(check), "trt": TrtUnit(check)}
        check.populate(units.values(), args.chunk_tokens)
        check_units(check, units)
        graph_input_test(check, units)
        report["fixture_checks"] = (
            "equal staged inputs, originals unchanged, all selected KV retained, graph consumes new Q: passed"
        )
        del check, units
    except Exception:
        report["fixture_checks"] = "failed"
        report["fixture_traceback"] = traceback.format_exc()
        save(args.output, report)
        raise
    gc.collect()
    torch.cuda.empty_cache()
    rejected = False
    combinations = list(
        itertools.product(args.batches, args.contexts, args.heads, args.queries)
    )
    for ordinal, (batch, context, heads, queries) in enumerate(combinations, 1):
        print(
            f"UNIT {ordinal}/{len(combinations)} B={batch} L={context} H={heads} Q={queries}",
            flush=True,
        )
        started = time.monotonic()
        shape = dict(batch=batch, context=context, heads=heads, queries=queries)
        try:
            torch.cuda.reset_peak_memory_stats()
            common = CommonData(batch, context, heads, queries, args.seed)
            paths = {"rtp": FlashUnit(common), "trt": TrtUnit(common)}
            common.populate(paths.values(), args.chunk_tokens)
            precision, status = check_units(common, paths)
            row = dict(shape=shape, status=status, precision=precision)
            rejected |= status == "precision_rejected"
            if args.benchmark:
                row["complete_unit_timing"] = measure(
                    paths, args.rounds, args.cache_mode
                )
                row["timing_qualification"] = (
                    "experimental only; precision gate is not waived"
                )
            row["harness_peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / 2**30
            )
            row["wall_seconds"] = time.monotonic() - started
            report["rows"].append(row)
            save(args.output, report)
            print("RESULT " + json.dumps(row), flush=True)
            del common, paths
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            report["rows"].append(
                dict(
                    shape=shape,
                    status="runtime_failed",
                    traceback=traceback.format_exc(),
                )
            )
            save(args.output, report)
            raise
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["gpu_after"] = snapshot()
    report["exit_reason"] = "precision_rejected" if rejected else "passed"
    save(args.output, report)
    raise SystemExit(2 if rejected else 0)


if __name__ == "__main__":
    main()
