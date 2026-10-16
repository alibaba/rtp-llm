"""Exact state epilogue and graph-replay performance before model integration."""

import json
import os
import statistics
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import rtp_llm.models_py.triton_kernels.kimi_kda.state_update as update
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _flat_state_update(
    state, delta, gate, output, SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = i < SIZE
    value = tl.load(state + i, mask, 0)
    increment = tl.load(delta + i, mask, 0)
    log_decay = tl.load(gate + i // 128, mask, 0).to(tl.float32)
    # Match torch.exp2(float), then independent multiply/add kernels. Do not
    # contract these operations into FMA or substitute approximate tl.exp2.
    decay = libdevice.exp2(log_decay)
    decayed = libdevice.mul_rn(value, decay)
    result = libdevice.add_rn(decayed, increment)
    tl.store(output + i, result, mask)


def flat_control(state, delta, gate):
    output = torch.empty_like(state)
    _flat_state_update[(triton.cdiv(state.numel(), 512),)](
        state,
        delta,
        gate,
        output,
        state.numel(),
        512,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return output


def reference(state, delta, gate):
    return state * torch.exp2(gate.float()).unsqueeze(-1) + delta


class StateUpdateTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.enabled = patch.object(update, "_ENABLED", True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)
        torch.manual_seed(739)

    def assert_bits(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32)),
            f"mismatches={torch.count_nonzero(actual.view(torch.int32) != expected.view(torch.int32)).item()}",
        )

    @torch.inference_mode()
    def test_a_precision_and_boundaries(self):
        for heads in (1, 16, 17, 64):
            state = torch.randn(heads, 128, 128, device="cuda")
            delta = torch.randn_like(state)
            gate = -512 * torch.rand(heads, 128, device="cuda")
            boundaries = torch.tensor(
                [
                    -512,
                    -150.5,
                    -150,
                    -149.5,
                    -149,
                    -148.5,
                    -127,
                    -126,
                    -125.99999,
                    -1,
                    -0.0,
                    0.0,
                ],
                device="cuda",
            )
            gate.flatten()[: boundaries.numel()].copy_(boundaries)
            # Finite FP32 values spanning gradual underflow through large exponents.
            bits = torch.randint(
                0, 0x7F800000, state.shape, device="cuda", dtype=torch.int32
            )
            bits[::2].bitwise_xor_(-2147483648)
            state.copy_(bits.view(torch.float32))
            for dtype in (torch.float32, torch.bfloat16):
                values = (state, delta, gate.to(dtype))
                expected = reference(*values)
                for block in (256, 512, 1024, 2048):
                    actual = update.kda_state_update(*values, _block=block)
                    self.assert_bits(actual, expected)
                # Deliberate cancellation exposes accidental fused multiply-add.
                cancellation = -(state * torch.exp2(values[2].float()).unsqueeze(-1))
                self.assert_bits(
                    update.kda_state_update(state, cancellation, values[2]),
                    reference(state, cancellation, values[2]),
                )
            # Contiguous buffers with an unaligned base must remain supported.
            unaligned = []
            for value in values:
                storage = torch.empty(
                    value.numel() + 1, device="cuda", dtype=value.dtype
                )
                view = storage[1:].view_as(value)
                view.copy_(value)
                unaligned.append(view)
            self.assert_bits(update.kda_state_update(*unaligned), reference(*unaligned))
        original = update.kda_state_update(*values)
        frozen = original.clone()
        state.add_(0.5)
        update.kda_state_update(*values)
        self.assert_bits(original, frozen)
        with patch.object(update, "_ENABLED", False):
            self.assertIsNone(update.kda_state_update(*values))
        self.assertIsNone(update.kda_state_update(state.transpose(1, 2), delta, gate))
        storage = torch.randn(2, *state.shape, device="cuda")
        source, destination = storage.unbind(0)
        frozen = source.clone()
        expected = reference(source, delta, gate)
        actual = update.kda_state_update(source, delta, gate, output=destination)
        self.assertIs(actual, destination)
        self.assert_bits(actual, expected)
        self.assert_bits(source, frozen)
        self.assertIsNone(update.kda_state_update(source, delta, gate, output=source))
        overlap = storage.flatten()[1 : source.numel() + 1].view_as(source)
        self.assertIsNone(update.kda_state_update(source, delta, gate, output=overlap))
        self.assertIsNone(update.kda_state_update(source, delta, gate, output=delta))
        self.assertIsNone(
            update.kda_state_update(source, delta, gate, output=destination.bfloat16())
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            update.kda_state_update(*values)
            stream.synchronize()
            torch.cuda._sleep(100_000_000)
            pending = torch.cuda.Event()
            pending.record()
            actual = update.kda_state_update(*values)
            self.assertFalse(pending.query())
        stream.synchronize()
        self.assert_bits(actual, reference(*values))

    @torch.inference_mode()
    def test_b_graph_benchmark(self):
        directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        rows = []
        repeats = 32
        for heads in (16, 64):
            values = (
                torch.randn(heads, 128, 128, device="cuda"),
                torch.randn(heads, 128, 128, device="cuda"),
                -64 * torch.rand(heads, 128, device="cuda"),
            )
            variants = {
                "reference": lambda: reference(*values),
                "flat_512": lambda: flat_control(*values),
            }
            for block in (256, 512, 1024, 2048):
                variants[f"block_{block}"] = (
                    lambda block=block: update.kda_state_update(*values, _block=block)
                )
            expected = reference(*values)
            for fn in variants.values():
                self.assert_bits(fn(), expected)
            graphs = {}
            for name, fn in variants.items():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(30):
                        output = fn()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(repeats):
                        output = fn()
                graphs[name] = (graph, output)
            samples = {name: [] for name in variants}
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for repeat in range(100):
                names = tuple(variants) if repeat % 2 else tuple(reversed(variants))
                for name in names:
                    torch.cuda.synchronize()
                    start.record()
                    graphs[name][0].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000 / repeats)
            kernels = {}
            for name, (graph, output) in graphs.items():
                graph.replay()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    graph.replay()
                    torch.cuda.synchronize()
                path = directory / f"state_update_h{heads}_{name}.json"
                profile.export_chrome_trace(str(path))
                events = [
                    e
                    for e in json.loads(path.read_text())["traceEvents"]
                    if e.get("cat") == "kernel"
                ]
                counts = Counter(e["name"] for e in events)
                self.assertEqual(
                    len(events), repeats * (3 if name == "reference" else 1)
                )
                if name == "reference":
                    for operation in ("exp2", "mul", "add"):
                        self.assertTrue(
                            any(operation in kernel.lower() for kernel in counts),
                            counts,
                        )
                else:
                    self.assertTrue(
                        all("_state_update" in kernel for kernel in counts), counts
                    )
                kernels[name] = {
                    "launches_per_iter": len(events) / repeats,
                    "kernel_sum_us_per_iter": sum(e["dur"] for e in events) / repeats,
                    "names_counts": dict(counts),
                    "expected_path_pass": True,
                }
            row = {
                "heads": heads,
                "shape": list(values[0].shape),
                "samples_event_us": samples,
                "kernels": kernels,
            }
            rows.append(row)
            print(
                "KDA_STATE_UPDATE",
                heads,
                {
                    name: {
                        "event_us": statistics.median(samples[name]),
                        **{
                            k: v
                            for k, v in kernels[name].items()
                            if k != "names_counts"
                        },
                    }
                    for name in variants
                },
                flush=True,
            )
        (directory / "state_update_benchmark.json").write_text(
            json.dumps(rows, indent=2)
        )


if __name__ == "__main__":
    unittest.main()
