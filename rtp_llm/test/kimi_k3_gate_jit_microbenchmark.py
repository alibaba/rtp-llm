"""Compare the previous Inductor gate with the current K3 Triton gate."""

import argparse
import json
import os
import statistics
import tempfile
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--width", type=int, default=7168)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(*args.rows, args.width, args.iterations, args.repeats) <= 0:
        parser.error("all dimensions and repeat counts must be positive")

    with tempfile.TemporaryDirectory(prefix="k3_gate_jit_bench_") as cache:
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache, "inductor")

        import torch

        from rtp_llm.models_py.modules.kimi_k3.native_mla_ops import gate_sigmoid_mul

        if not torch.cuda.is_available():
            raise RuntimeError("K3 gate benchmark requires CUDA")

        @torch.compile(backend="inductor")
        def previous_gate(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            return out * gate.sigmoid()

        torch.manual_seed(71)
        for rows in args.rows:
            values = torch.randn(
                rows, args.width, device="cuda", dtype=torch.bfloat16
            )
            gates = torch.randn_like(values)
            first_call = {}
            outputs = {}
            operations = (
                ("previous_inductor", previous_gate),
                ("current_triton", gate_sigmoid_mul),
            )
            for name, op in operations:
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs[name] = op(values, gates)
                torch.cuda.synchronize()
                first_call[name] = time.perf_counter() - start
            reference = values * gates.sigmoid()
            torch.testing.assert_close(
                outputs["current_triton"], reference, rtol=0, atol=0
            )
            torch.testing.assert_close(
                outputs["previous_inductor"], reference, rtol=0.02, atol=0.02
            )
            previous_gap = (
                outputs["previous_inductor"].float() - reference.float()
            ).abs()
            previous_mismatch = (
                outputs["previous_inductor"] != reference
            ).float().mean().item()

            steady = {}
            for name, op in operations:
                for _ in range(20):
                    op(values, gates)
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.repeats):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        op(values, gates)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / args.iterations)
                steady[name] = statistics.median(samples)
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "width": args.width,
                        "dtype": "bfloat16",
                        "first_call_s": first_call,
                        "median_event_interval_us": steady,
                        "previous_vs_eager_max_abs": previous_gap.max().item(),
                        "previous_vs_eager_mismatch_fraction": previous_mismatch,
                        "iterations_per_repeat": args.iterations,
                        "repeats": args.repeats,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
