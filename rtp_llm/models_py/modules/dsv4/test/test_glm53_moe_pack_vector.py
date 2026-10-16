"""Isolated layout tuning against the original MegaMoE quantization kernel."""

import json
import os
import statistics
import unittest
from pathlib import Path
from unittest.mock import patch

import test_glm53_moe_pack_dispatch as fixtures
import torch
import triton
import triton.language as tl
from rtp_llm.models_py.modules.glm5_mega_moe import input_packer_triton as ops


@triton.jit
def _bit_or(a, b):
    return a | b


@triton.jit(do_not_specialize=["M"])
def _pack_vector(
    x_ptr,
    weights_ptr,
    indices_ptr,
    out_fp8_ptr,
    out_sf_ptr,
    out_weights_ptr,
    out_indices_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    SS: tl.constexpr,
    OWS: tl.constexpr,
    OIS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    group = tl.arange(0, 4)
    cols = tile * 128 + group[:, None] * 32 + tl.arange(0, 32)[None, :]
    valid = (rows[:, None, None] < M) & (cols[None, :, :] < N)
    x = tl.load(x_ptr + rows[:, None, None] * XS + cols[None, :, :], valid, 0).to(
        tl.float32
    )
    amax = tl.maximum(tl.max(tl.abs(x), axis=2), 1.0e-4)
    raw = (amax / 448.0).to(tl.int32, bitcast=True)
    exp = tl.minimum(tl.maximum(((raw >> 23) & 0xFF) + ((raw & 0x7FFFFF) != 0), 1), 254)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    quant = tl.clamp(x / scale[:, :, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + rows[:, None, None] * OS + cols[None, :, :], quant, valid)
    packed = tl.reduce(exp << (group[None, :] * 8), axis=1, combine_fn=_bit_or)
    tl.store(out_sf_ptr + rows * SS + tile, packed, rows < M)
    if tile == 0:
        router = tl.arange(0, BK)
        mask = (rows[:, None] < M) & (router[None, :] < K)
        weights = tl.load(
            weights_ptr + rows[:, None] * WS + router[None, :], mask, 0
        ).to(tl.float32)
        indices = tl.load(
            indices_ptr + rows[:, None] * IS + router[None, :], mask, 0
        ).to(tl.int64)
        tl.store(out_weights_ptr + rows[:, None] * OWS + router[None, :], weights, mask)
        tl.store(out_indices_ptr + rows[:, None] * OIS + router[None, :], indices, mask)


def vector(args, block_m, warps=4):
    x, w, i, oq, osf, oi, ow = args
    m, n = x.shape
    if m:
        _pack_vector[(triton.cdiv(m, block_m), n // 128)](
            x,
            w,
            i,
            oq,
            osf,
            ow,
            oi,
            m,
            n,
            w.shape[1],
            x.stride(0),
            w.stride(0),
            i.stride(0),
            oq.stride(0),
            osf.stride(0),
            ow.stride(0),
            oi.stride(0),
            block_m,
            triton.next_power_of_2(w.shape[1]),
            num_warps=warps,
        )


def original_with_warps(args, warps):
    x, w, i, oq, osf, oi, ow = args
    m, n = x.shape
    if not m:
        return
    bm = int(os.environ.get("GLM5_MEGA_MOE_PACK_BLOCK_M", 8 if m >= 2048 else 2))
    if bm not in (1, 2, 4, 8):
        bm = 2
    tensors = (x, w, i, oq, osf, ow, oi)
    ops._pack_mega_moe_inputs_optimized_kernel[(triton.cdiv(m, bm), n // 128)](
        *tensors,
        m,
        n,
        w.shape[1],
        *(t.stride(0) for t in tensors),
        1.0e-4,
        448.0,
        bm,
        triton.next_power_of_2(w.shape[1]),
        num_warps=warps,
    )


def candidates(args):
    if os.environ.get("GLM53_MOE_PACK_VECTOR_TILE_SWEEP", "0") == "1":
        return {
            f"vector{bm}_w{w}": lambda bm=bm, w=w: vector(args, bm, w)
            for bm, w in ((8, 1), (16, 1), (16, 2), (16, 4), (32, 1), (32, 2), (32, 4))
        }
    sweep = os.environ.get("GLM53_MOE_PACK_VECTOR_WARP_SWEEP", "0") == "1"
    variants = {
        f"vector{bm}_w{w}": lambda bm=bm, w=w: vector(args, bm, w)
        for bm in (2, 4, 8)
        for w in ((1, 2, 4) if sweep else (4,))
    }
    if sweep:
        variants.update(
            {f"original_w{w}": lambda w=w: original_with_warps(args, w) for w in (1, 2)}
        )
    return variants


class TestMoePackVector(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.helper = fixtures.TestMoePackDispatch()
        enabled = patch.object(ops, "_COMPILED_DISPATCH", False)
        enabled.start()
        self.addCleanup(enabled.stop)
        baseline = patch.object(ops, "_VECTOR_PACK", False)
        baseline.start()
        self.addCleanup(baseline.stop)
        torch.manual_seed(1218)

    @torch.inference_mode()
    def test_exact_and_async(self):
        for tokens in (0, 1, 13, 81, 419, 2047, 2048, 5103, 9889):
            for width, topk, padded in (
                (128, 5, False),
                (4096, 8, False),
                (6144, 4, True),
            ):
                args, owners = self.helper.fixture(tokens, width, topk, padded=padded)
                inputs = tuple(t.clone() for t in args[:3])
                ops.fused_pack_mega_moe_inputs(*args)
                expected = tuple(t.clone() for t in owners)
                for fn in candidates(args).values():
                    for t in owners:
                        t.fill_(3)
                    fn()
                    self.helper.assert_bits(expected, owners)
                    self.helper.assert_bits(inputs, args[:3])
        args, owners = self.helper.fixture(13, unaligned=True)
        ops.fused_pack_mega_moe_inputs(*args)
        expected = tuple(t.clone() for t in owners)
        for fn in candidates(args).values():
            fn()
            self.helper.assert_bits(expected, owners)
        # Every finite BF16 bit pattern, including signed zero and subnormals.
        args, owners = self.helper.fixture(16, 4096)
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        values = torch.where(torch.isfinite(values), values, 0)
        args[0].copy_(values.view_as(args[0]))
        ops.fused_pack_mega_moe_inputs(*args)
        expected = tuple(t.clone() for t in owners)
        for fn in candidates(args).values():
            fn()
            self.helper.assert_bits(expected, owners)
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        for device in (0, 1):
            with torch.cuda.device(device):
                args, owners = self.helper.fixture(81)
                ops.fused_pack_mega_moe_inputs(*args)
                expected = tuple(t.clone() for t in owners)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                for fn in candidates(args).values():
                    with torch.cuda.stream(stream):
                        fn()
                        stream.synchronize()
                        torch.cuda._sleep(100_000_000)
                        pending = torch.cuda.Event()
                        pending.record()
                        fn()
                        self.assertFalse(pending.query())
                    stream.synchronize()
                    self.helper.assert_bits(expected, owners)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        fn()
                    graph.replay()
                    self.helper.assert_bits(expected, owners)

    @torch.inference_mode()
    def test_kernel_and_stream_benchmark(self):
        rows, traces = [], []
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        # Complete every clean CUDA-event measurement before starting CUPTI.
        for tokens in (1, 81, 419, 5103, 9889):
            args, owners = self.helper.fixture(tokens)
            functions = {"original": lambda: ops.fused_pack_mega_moe_inputs(*args)}
            functions.update(candidates(args))
            for fn in functions.values():
                for _ in range(30):
                    fn()
            samples = {name: [] for name in functions}
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for repeat in range(40):
                names = tuple(functions) if repeat % 2 else tuple(reversed(functions))
                for name in names:
                    torch.cuda.synchronize()
                    start.record()
                    functions[name]()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000)
            rows.append(dict(tokens=tokens, event_samples_us=samples))
            traces.append((tokens, args))
        for row, (tokens, args) in zip(rows, traces):
            functions = {"original": lambda: ops.fused_pack_mega_moe_inputs(*args)}
            functions.update(candidates(args))
            kernels = {}
            for name, fn in functions.items():
                for _ in range(10):
                    fn()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(40):
                        fn()
                    torch.cuda.synchronize()
                path = root / f"pack_{tokens}_{name}.json"
                prof.export_chrome_trace(str(path))
                events = [
                    e
                    for e in json.loads(path.read_text())["traceEvents"]
                    if e.get("cat") == "kernel"
                ]
                expected = (
                    "_pack_mega_moe_inputs_optimized_kernel"
                    if name.startswith("original")
                    else "_pack_vector"
                )
                self.assertEqual(len(events), 40)
                self.assertTrue(all(expected in e["name"] for e in events))
                kernels[name] = dict(
                    kernel_sum_us_per_iter=sum(e["dur"] for e in events) / 40,
                    launches_per_iter=1,
                    names=sorted({e["name"] for e in events}),
                    path_check="PASS",
                )
            row["kernels"] = kernels
            print(
                "MOE_PACK_VECTOR",
                tokens,
                {n: round(v["kernel_sum_us_per_iter"], 4) for n, v in kernels.items()},
                flush=True,
            )
        (root / "moe_pack_vector.json").write_text(
            json.dumps(
                dict(
                    warmup=30,
                    timing="CUDA-event stream latency first, then independent profiler actual kernel sums; 40 checked launches each; allocations and compilation excluded",
                    rows=rows,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
