"""K3.1 TP4 DSA projection fork with synthetic weights and gathered hidden rows."""

import json
import statistics
import unittest
from collections import Counter
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.kernels.cuda.fp8_kernel import quant_weight_ue8m0_packed
from rtp_llm.models_py.modules import LayerNorm
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.kimi_k3.dsa import split_bf16x3
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import fused_q_kv_rmsnorm
from rtp_llm.models_py.utils.stream_dag import StreamDag


HIDDEN = 7168
INPUT_WIDTH = 5248
INDEX_WIDTH = 128
Q_RANK = 1536
KV_RANK = 512
SAMPLES = 20


class ProjectionFork:
    def __init__(self, mode, input_proj, index_proj, packed_proj, head_gate):
        q_weight = torch.ones(Q_RANK, dtype=torch.bfloat16, device="cuda")
        kv_weight = torch.ones(KV_RANK, dtype=torch.bfloat16, device="cuda")
        k_norm = LayerNorm(
            torch.ones(INDEX_WIDTH, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(INDEX_WIDTH, dtype=torch.bfloat16, device="cuda"),
        )
        dag = StreamDag(torch.cuda.current_device(), side_streams=("index_k", "head_gate"))
        dependencies = ()
        if mode == "shared":
            dag.add(
                "quantize",
                lambda s: setattr(s, "encoded", input_proj.quantize_input(s.hidden)),
            )
            dependencies = ("quantize",)

        def project_input(state):
            if mode == "packed":
                state.projection, state.key = packed_proj(state.hidden).split(
                    (INPUT_WIDTH, INDEX_WIDTH), dim=-1
                )
            elif mode == "shared":
                state.projection = input_proj.forward_quantized(*state.encoded)
            else:
                state.projection = input_proj(state.hidden)

        def project_key(state):
            if mode == "shared":
                state.key = index_proj.forward_quantized(*state.encoded)
            elif mode == "separate":
                state.key = index_proj(state.hidden)
            # LayerNorm's native kernel expects compact rows.
            state.key_norm = k_norm(state.key.contiguous())

        def normalize_q_kv(state):
            state.q_norm, state.kv_norm = fused_q_kv_rmsnorm(
                state.projection[:, :Q_RANK],
                state.projection[:, Q_RANK : Q_RANK + KV_RANK],
                q_weight,
                kv_weight,
                1e-6,
            )

        dag.add("input", project_input, after=dependencies)
        dag.add(
            "index_k",
            project_key,
            stream="index_k",
            after=("input",) if mode == "packed" else dependencies,
        )
        dag.add(
            "head_gate",
            lambda s: setattr(
                s, "head_gate", torch.mm(s.hidden, head_gate, out_dtype=torch.float32)
            ),
            stream="head_gate",
        )
        dag.add("q_kv_norm", normalize_q_kv, after=("input",))
        dag.finalize()
        self.dag = dag

    def __call__(self, hidden):
        state = SimpleNamespace(hidden=hidden)
        self.dag.run(state)
        return (
            state.projection,
            state.q_norm,
            state.kv_norm,
            state.key_norm,
            state.head_gate,
        )


def capture(fork, hidden):
    for _ in range(3):
        fork(hidden)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fork(hidden)
    return graph, outputs


def kernel_counts(graph):
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    return Counter(
        "quant" if "per_token_group_quant_8bit_kernel" in event.name else "fp8_gemm"
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and (
            "per_token_group_quant_8bit_kernel" in event.name
            or "sm100_fp8_fp4_gemm_1d1d_impl" in event.name
        )
    )


def measure(graph, flush):
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    events = []
    for _ in range(SAMPLES):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        flush.zero_()
        begin.record()
        graph.replay()
        end.record()
        events.append((begin, end))
    torch.cuda.synchronize()
    return [begin.elapsed_time(end) * 1000 for begin, end in events]


class DsaProjectionBenchmark(unittest.TestCase):
    @torch.inference_mode()
    def test_projection_fork(self):
        self.assertEqual(torch.cuda.get_device_capability(), (10, 3))
        torch.manual_seed(3116)
        weights = torch.randn(
            INPUT_WIDTH + INDEX_WIDTH, HIDDEN, dtype=torch.bfloat16, device="cuda"
        ) * HIDDEN**-0.5

        def linear(matrix):
            weight, scales = quant_weight_ue8m0_packed(matrix)
            return CudaFp8DeepGEMMLinear(weight, scales)

        input_proj = linear(weights[:INPUT_WIDTH])
        index_proj = linear(weights[INPUT_WIDTH:])
        packed_proj = linear(weights)
        head_gate = split_bf16x3(torch.randn(HIDDEN, 32, device="cuda") * HIDDEN**-0.5)
        modes = ("separate", "shared", "packed")
        forks = {
            mode: ProjectionFork(mode, input_proj, index_proj, packed_proj, head_gate)
            for mode in modes
        }
        props = torch.cuda.get_device_properties(0)
        flush = torch.empty(2 * props.L2_cache_size, dtype=torch.uint8, device="cuda")
        print(
            json.dumps(
                {"gpu": props.name, "torch": torch.__version__, "cuda": torch.version.cuda}
            )
        )
        for rows in (80, 160, 320):
            with self.subTest(rows=rows):
                hidden = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
                hidden.normal_()
                captured = {mode: capture(fork, hidden) for mode, fork in forks.items()}
                max_abs = {}
                for _ in range(2):
                    hidden.normal_()
                    for graph, _ in captured.values():
                        graph.replay()
                    _, expected = captured["separate"]
                    for mode in modes[1:]:
                        _, actual = captured[mode]
                        for out, ref in zip(actual, expected):
                            torch.testing.assert_close(out, ref, rtol=0.02, atol=0.005)
                        error = max(
                            (out.float() - ref.float()).abs().max().item()
                            for out, ref in zip(actual, expected)
                        )
                        max_abs[mode] = max(max_abs.get(mode, 0.0), error)

                if rows == 160:
                    expected_counts = {
                        "separate": {"quant": 2, "fp8_gemm": 2},
                        "shared": {"quant": 1, "fp8_gemm": 2},
                        "packed": {"quant": 1, "fp8_gemm": 1},
                    }
                    for mode, (graph, _) in captured.items():
                        observed = dict(kernel_counts(graph))
                        print(json.dumps({"rows": rows, "mode": mode, "kernels": observed}))
                        self.assertEqual(observed, expected_counts[mode])

                for order in (modes, tuple(reversed(modes))):
                    for mode in order:
                        samples = measure(captured[mode][0], flush)
                        print(
                            json.dumps(
                                {
                                    "rows": rows,
                                    "order": order,
                                    "mode": mode,
                                    "p50_us": statistics.median(samples),
                                    "p90_us": sorted(samples)[17],
                                    "max_abs": max_abs.get(mode, 0.0),
                                    "samples_us": samples,
                                }
                            ),
                            flush=True,
                        )


if __name__ == "__main__":
    unittest.main()
