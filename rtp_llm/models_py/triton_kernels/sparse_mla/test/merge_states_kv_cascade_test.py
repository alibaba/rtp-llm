"""Accuracy and benchmark tests for ``merge_states_kv_cascade`` (Triton vs PyTorch reference).

Run under Bazel (GPU):
  bazel test //rtp_llm/models_py/triton_kernels/sparse_mla/test:merge_states_kv_cascade_test

Run directly:
  python -m rtp_llm.models_py.triton_kernels.sparse_mla.test.merge_states_kv_cascade_test
  python .../merge_states_kv_cascade_test.py --seq-len 3489 --head-dim 512
  python .../merge_states_kv_cascade_test.py --skip-bench
"""

from __future__ import annotations

import argparse
import sys
import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.merge_states_kv_cascade import (
    _TRITON_AVAILABLE,
    MAX_STATES,
    merge_states_kv_cascade_torch_reference,
    triton_merge_states_kv_cascade,
)
from rtp_llm.test.utils.bench_util import bench
from rtp_llm.test.utils.numeric_util import calc_diff


def _make_inputs(
    seq_len: int,
    num_states: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    v = torch.randn(
        seq_len,
        num_states,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=g,
    )
    s = torch.randn(
        seq_len,
        num_states,
        num_heads,
        device=device,
        dtype=torch.float32,
        generator=g,
    )
    return v.contiguous(), s.contiguous()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


class MergeStatesKvCascadeTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        self.device = torch.device("cuda", torch.cuda.current_device())
        if not _TRITON_AVAILABLE:
            raise unittest.SkipTest("Triton is not installed")

    def _assert_close_to_ref(
        self,
        v: torch.Tensor,
        s: torch.Tensor,
        *,
        block_d: int = 128,
        max_atol: float = 0.02,
    ) -> None:
        ref = merge_states_kv_cascade_torch_reference(v, s)
        out = triton_merge_states_kv_cascade(v, s, block_d=block_d)
        mad = _max_abs_diff(out, ref)
        self.assertLessEqual(
            mad,
            max_atol,
            msg=f"max_abs_diff={mad}, shape={tuple(v.shape)}, dtype={v.dtype}, block_d={block_d}",
        )
        if v.dtype == torch.float32:
            sim_err = calc_diff(out.reshape(-1), ref.reshape(-1))
            self.assertLess(sim_err, 1e-6, msg=f"calc_diff={sim_err}, mad={mad}")

    def test_accuracy_bf16_mla_like(self) -> None:
        """Dimensions from real Sparse MLA CP prefill (e.g. total_q=3489, kv_lora_rank=512)."""
        v, s = _make_inputs(3489, 4, 64, 512, torch.bfloat16, self.device)
        self._assert_close_to_ref(v, s, block_d=128)
        self._assert_close_to_ref(v, s, block_d=64)

    def test_accuracy_fp16_fp32_states(self) -> None:
        for dtype, atol in ((torch.float16, 2e-2), (torch.float32, 5e-6)):
            v, s = _make_inputs(256, 8, 32, 128, dtype, self.device, seed=1)
            self._assert_close_to_ref(v, s, max_atol=atol)

    def test_accuracy_num_states_edge(self) -> None:
        """num_states=2 and num_states at small/large seq."""
        v, s = _make_inputs(1, 2, 8, 512, torch.bfloat16, self.device, seed=2)
        self._assert_close_to_ref(v, s)
        v, s = _make_inputs(4096, 2, 64, 256, torch.bfloat16, self.device, seed=3)
        self._assert_close_to_ref(v, s)

    def test_head_dim_not_multiple_of_block_d(self) -> None:
        """Tail block must be masked correctly."""
        v, s = _make_inputs(32, 4, 8, 300, torch.bfloat16, self.device, seed=4)
        self._assert_close_to_ref(v, s, block_d=128)

    def test_benchmark_torch_vs_triton(self) -> None:
        """Sanity timing: Triton should run without error; prints ms (not a strict perf gate)."""
        seq_len, num_states, num_heads, head_dim = 2048, 4, 64, 512
        v, s = _make_inputs(
            seq_len, num_states, num_heads, head_dim, torch.bfloat16, self.device
        )

        def run_torch() -> None:
            merge_states_kv_cascade_torch_reference(v, s)

        def run_triton() -> None:
            triton_merge_states_kv_cascade(v, s)

        mean_t, min_t, max_t = bench(run_torch, num_warmups=10, num_tests=20)
        mean_tr, min_tr, max_tr = bench(run_triton, num_warmups=10, num_tests=20)
        # bench returns (mean, min, max) in seconds — see bench_util.bench
        print(
            f"\n[merge_states_kv_cascade] seq={seq_len} states={num_states} "
            f"heads={num_heads} dim={head_dim} bf16\n"
            f"  torch  mean/min/max ms: {mean_t*1e3:.4f} / {min_t*1e3:.4f} / {max_t*1e3:.4f}\n"
            f"  triton mean/min/max ms: {mean_tr*1e3:.4f} / {min_tr*1e3:.4f} / {max_tr*1e3:.4f}\n",
            file=sys.stderr,
        )
        self.assertGreater(mean_t, 0)
        self.assertGreater(mean_tr, 0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="merge_states_kv_cascade bench / quick check"
    )
    p.add_argument(
        "--skip-bench",
        action="store_true",
        help="Only compare Triton vs torch accuracy, skip timing",
    )
    p.add_argument("--seq-len", type=int, default=3489)
    p.add_argument("--num-states", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--block-d", type=int, default=128)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=50)
    return p.parse_args(argv)


def _cli_main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)
    if not _TRITON_AVAILABLE:
        print("Triton required", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    v, s = _make_inputs(
        args.seq_len,
        args.num_states,
        args.num_heads,
        args.head_dim,
        dtype,
        device,
        seed=0,
    )
    if args.num_states > MAX_STATES:
        print(f"num_states must be <= {MAX_STATES}", file=sys.stderr)
        sys.exit(1)

    ref = merge_states_kv_cascade_torch_reference(v, s)
    out = triton_merge_states_kv_cascade(v, s, block_d=args.block_d)
    mad = _max_abs_diff(out, ref)
    sim_err = calc_diff(out.reshape(-1), ref.reshape(-1))
    print(
        f"max_abs_diff={mad:.6g}  calc_diff={sim_err:.6g}  "
        f"shape=({args.seq_len},{args.num_states},{args.num_heads},{args.head_dim})"
    )
    if mad > 0.02:
        print(
            "WARNING: max_abs_diff exceeds 0.02 (bf16 typical bound)", file=sys.stderr
        )
        sys.exit(1)

    if args.skip_bench:
        return

    def run_torch() -> None:
        merge_states_kv_cascade_torch_reference(v, s)

    def run_triton() -> None:
        triton_merge_states_kv_cascade(v, s, block_d=args.block_d)

    t_mean, t_min, t_max = bench(
        run_torch, num_warmups=args.warmup, num_tests=args.repeat
    )
    tr_mean, tr_min, tr_max = bench(
        run_triton, num_warmups=args.warmup, num_tests=args.repeat
    )
    print(
        f"torch  mean/min/max ms: {t_mean*1e3:.4f} / {t_min*1e3:.4f} / {t_max*1e3:.4f}"
    )
    print(
        f"triton mean/min/max ms: {tr_mean*1e3:.4f} / {tr_min*1e3:.4f} / {tr_max*1e3:.4f}"
    )


if __name__ == "__main__":
    _cli_flags = (
        "--skip-bench",
        "--seq-len",
        "--num-states",
        "--num-heads",
        "--head-dim",
        "--block-d",
        "--warmup",
        "--repeat",
        "-h",
        "--help",
    )
    if any(a in sys.argv for a in _cli_flags):
        _cli_main()
    else:
        unittest.main()
