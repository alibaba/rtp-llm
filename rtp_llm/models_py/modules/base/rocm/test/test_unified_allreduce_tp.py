# SPDX-License-Identifier: Apache-2.0
"""End-to-end multi-GPU tests for the MoE unified-allreduce optimization.

The optimization in ``GenericMoeLayer`` (rtp_llm/models_py/model_desc/generic_moe.py)
skips the per-path allreduce inside ``FusedMoe.finalize`` and ``DenseMLP.forward``
when both a routed MoE path and a shared-expert path exist under TP, then issues
a single ``all_reduce`` on the summed output. This relies on two properties under
real NCCL semantics:

1. Linearity:  allreduce(A) + allreduce(B) == allreduce(A + B)
2. Gate consistency: when the gate input ``x`` is identical across ranks
   (which is the real case, since it is post-RMSNorm), gate values are bit-exact
   identical, so ``gate * allreduce(B)`` is rank-consistent.

These properties hold mathematically, but the reviewer's P1 concern was bf16
precision / in-place ``SigmoidGateScaleAdd`` side-effects under real NCCL. These
tests exercise the real collective on ≥2 GPUs.

Pattern mirrors ``rtp_llm/models_py/modules/base/rocm/test/trt_allreduce_test.py``.
Skipped when fewer than 2 GPUs are available.
"""

import os
import socket
import sys
import unittest
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.filterwarnings(
    "ignore", message="barrier.*using the device under current context"
)
warnings.filterwarnings("ignore", message="Guessing device ID based on global rank")


def _setup_distributed(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )


def _cleanup_distributed():
    try:
        dist.barrier()
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _launch_workers(worker_fn, world_size: int, timeout: int = 120, **kwargs):
    port = _get_free_port()
    processes = []
    for rank in range(world_size):
        proc = mp.Process(
            target=worker_fn,
            args=(rank, world_size, port),
            kwargs=kwargs,
            name=f"rank-{rank}",
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join(timeout=timeout)
        if proc.exitcode != 0:
            raise RuntimeError(f"Process {proc.name} exited with code {proc.exitcode}")


def _worker_linearity(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    """Property 1: allreduce(A) + allreduce(B) == allreduce(A + B).

    Each rank holds its own local A_rank / B_rank. After allreduce both paths
    should agree bit-exact in fp32 and within bf16 round-off otherwise.
    """
    try:
        _setup_distributed(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")

        torch.manual_seed(1000 + rank)
        A = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        B = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

        # Separated path: two independent allreduces then sum
        ar_A = A.clone()
        ar_B = B.clone()
        dist.all_reduce(ar_A)
        dist.all_reduce(ar_B)
        separated = ar_A + ar_B

        # Unified path: sum first, single allreduce
        ar_sum = (A + B).clone()
        dist.all_reduce(ar_sum)
        unified = ar_sum

        diff = (separated - unified).abs().max().item()
        ref_max = separated.abs().max().item()
        rel = diff / ref_max if ref_max > 0 else 0.0

        # bf16 accumulates across 2 ranks; one extra fp32-rounded add on the
        # separated path is well within bf16 mantissa precision.
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-6
        assert (
            rel < rtol or diff < atol
        ), f"[Rank {rank}] linearity mismatch: rel={rel:.3e} abs={diff:.3e}"
        if rank == 0:
            print(f"  [linearity] hidden={hidden_size} dtype={dtype} rel={rel:.2e} ✓")
    except Exception as exc:
        print(f"[Rank {rank}] linearity FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


def _worker_gate_consistency(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    """Property 2 (foundation): when input is identical across ranks, gate values
    produced by a Linear+sigmoid must be bit-exact identical across ranks.

    This is the precondition for ``gate * allreduce(B)`` being safe to factor
    out of the shared allreduce. Violated only if non-deterministic kernels are
    used, which PyTorch Linear in bf16 is not.
    """
    try:
        _setup_distributed(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")

        # Identical seed across ranks → identical x. In real code, x is
        # post-RMSNorm over an identical input, guaranteeing this.
        torch.manual_seed(42)
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        W_gate = torch.randn(hidden_size, 1, dtype=dtype, device=device)

        gate_local = torch.sigmoid(x @ W_gate)

        # Gather rank-0's gate to every rank for bit-exact comparison.
        gate_rank0 = gate_local.clone()
        dist.broadcast(gate_rank0, src=0)

        diff = (gate_local - gate_rank0).abs().max().item()
        assert diff == 0.0, (
            f"[Rank {rank}] gate NOT bit-exact across ranks: max_abs_diff={diff:.3e}. "
            f"The unified-allreduce optimization relies on this invariant."
        )
        if rank == 0:
            print(f"  [gate_consistency] hidden={hidden_size} bit-exact ✓")
    except Exception as exc:
        print(f"[Rank {rank}] gate_consistency FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


def _worker_gated_linearity(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    """Property 1 extended with a shared-expert sigmoid gate:

        separated = allreduce(A) + gate * allreduce(B)
        unified   = allreduce(A + gate * B)

    Uses the real ``SigmoidGateScaleAdd`` op (in-place on *experts*) to match
    the code path exercised in ``GenericMoeLayer.forward``. This covers the
    reviewer's concern about in-place side-effects under real NCCL.
    """
    try:
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.modules import SigmoidGateScaleAdd

        device = torch.device(f"cuda:{rank}")

        # Identical input across ranks → identical gate (verified in the
        # separate gate_consistency test).
        torch.manual_seed(2026)
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        W_gate = torch.randn(hidden_size, 1, dtype=dtype, device=device)
        gate_input = x @ W_gate  # pre-sigmoid; op applies sigmoid internally

        # Per-rank random A / B (routed experts / shared expert outputs).
        torch.manual_seed(7000 + rank)
        A_local = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        B_local = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

        gate_scale = SigmoidGateScaleAdd()

        # ----- Separated path -----
        A_sep = A_local.clone()
        B_sep = B_local.clone()
        dist.all_reduce(A_sep)
        dist.all_reduce(B_sep)
        # experts += sigmoid(gate) * shared  (in-place on experts=A_sep)
        separated = gate_scale(gate_input, B_sep, A_sep)

        # ----- Unified path -----
        A_uni = A_local.clone()
        B_uni = B_local.clone()
        # Fuse the gated-add locally first, then single allreduce.
        unified = gate_scale(gate_input, B_uni, A_uni)
        dist.all_reduce(unified)

        diff = (separated - unified).abs().max().item()
        ref_max = separated.abs().max().item()
        rel = diff / ref_max if ref_max > 0 else 0.0

        # Looser tolerance than pure linearity: two bf16 rounding steps
        # (sigmoid*shared, add) happen on different data per path.
        rtol = 2e-2 if dtype == torch.bfloat16 else 1e-5
        atol = 2e-3 if dtype == torch.bfloat16 else 1e-6
        assert rel < rtol or diff < atol, (
            f"[Rank {rank}] gated linearity mismatch: " f"rel={rel:.3e} abs={diff:.3e}"
        )
        if rank == 0:
            print(
                f"  [gated_linearity] hidden={hidden_size} dtype={dtype} "
                f"rel={rel:.2e} ✓"
            )
    except Exception as exc:
        print(f"[Rank {rank}] gated_linearity FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
class TestUnifiedAllreduceTP(unittest.TestCase):
    """Multi-GPU end-to-end tests for the MoE unified-allreduce optimization.

    Run via ``python -m unittest rtp_llm.models_py.modules.hybrid.test.
    test_unified_allreduce_tp`` on a host with ≥2 GPUs; auto-skipped otherwise.
    """

    def setUp(self):
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need >=2 GPUs, found {gpu_count}")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # --- Property 1: linearity -------------------------------------------------
    def test_linearity_bf16_hidden4096_ws2(self):
        _launch_workers(
            _worker_linearity,
            world_size=2,
            batch_size=8,
            hidden_size=4096,
            dtype=torch.bfloat16,
        )

    def test_linearity_fp32_hidden4096_ws2(self):
        _launch_workers(
            _worker_linearity,
            world_size=2,
            batch_size=8,
            hidden_size=4096,
            dtype=torch.float32,
        )

    def test_linearity_bf16_large_batch_ws2(self):
        _launch_workers(
            _worker_linearity,
            world_size=2,
            batch_size=128,
            hidden_size=4096,
            dtype=torch.bfloat16,
        )

    # --- Property 2: gate consistency -----------------------------------------
    def test_gate_consistency_bf16_ws2(self):
        _launch_workers(
            _worker_gate_consistency,
            world_size=2,
            batch_size=8,
            hidden_size=4096,
            dtype=torch.bfloat16,
        )

    # --- Combined: gated linearity with real SigmoidGateScaleAdd --------------
    def test_gated_linearity_bf16_hidden4096_ws2(self):
        _launch_workers(
            _worker_gated_linearity,
            world_size=2,
            batch_size=8,
            hidden_size=4096,
            dtype=torch.bfloat16,
        )

    def test_gated_linearity_bf16_large_batch_ws2(self):
        _launch_workers(
            _worker_gated_linearity,
            world_size=2,
            batch_size=64,
            hidden_size=4096,
            dtype=torch.bfloat16,
        )


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    unittest.main()
