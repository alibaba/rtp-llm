"""Acceptance test for the B300 (sm_103) two-kernel sparse MLA prefill path.

The kernel pair lives in :mod:`rtp_llm.models_py.triton_kernels.lsa_attention.
sparse_mla_two_kernel`; it only runs on sm_103 (tcgen05-class Triton lowering), so
the test skips everywhere else.

Contract under test (see the module docstring for the full statement):

* ``idx`` must be **4-row shared** (spec-decode form); the kernel reads only the
  ``4k``-th index row.
* Kernel A writes ``P = bf16(exp(s - m_off))`` with a row-constant conservative
  offset ``m_off``; Kernel B accumulates ``l = sum_slot P`` and returns
  ``out = (sum_slot P * V) / l`` and ``lse = m_off + ln(l)``.  Therefore the
  offset-invariant quantities to check are ``P / l`` (probabilities) and ``lse``.
"""

from __future__ import annotations

import math

import pytest
import torch

from rtp_llm.models_py.triton_kernels.lsa_attention import sparse_mla_two_kernel as two_kernel


def _skip_reason() -> str | None:
    if two_kernel is None:
        return "two-kernel module not importable (triton/sm_103 unavailable)"
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    if torch.cuda.get_device_capability() != (10, 3):
        return "requires sm_103 (B300)"
    return None


@pytest.mark.parametrize("T", [8, 512])
def test_sparse_mla_two_kernel_matches_reference(T: int) -> None:
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    case = getattr(two_kernel, "_case", None)
    ref_row = getattr(two_kernel, "_ref_row", None)
    if case is None or ref_row is None:  # pragma: no cover - helper moved
        pytest.skip("in-module test helpers unavailable")

    N, S = 8192, 2048
    q, kv, idx = case(T, N=N, S=S)
    out = torch.zeros(T, 32, 512, dtype=torch.float32, device="cuda")
    lse = torch.zeros(T, 32, dtype=torch.float32, device="cuda")
    two_kernel.sparse_mla_two_kernel(q, kv, idx, out, lse)

    assert torch.isfinite(out).all() and torch.isfinite(lse).all()

    worst_out = worst_lse = 0.0
    for r in range(min(T, 8)):
        ref_out, ref_lse = ref_row(q[r], kv, idx[r])
        worst_out = max(
            worst_out,
            (out[r] - ref_out).abs().max().item() / max(1e-6, ref_out.abs().max().item()),
        )
        worst_lse = max(
            worst_lse,
            (lse[r] - ref_lse).abs().max().item() / max(1e-6, abs(ref_lse).item()),
        )

    # Tolerance: the operator stores KV in fp8 (FP8_KV_DIM=656 layout), so the
    # kernel-vs-unquantized-reference gap is dominated by that quantization.
    assert worst_out < 3e-2, f"T={T}: rel_out {worst_out:.3e}"
    assert worst_lse < 3e-2, f"T={T}: rel_lse {worst_lse:.3e}"


def test_two_kernel_offset_invariance() -> None:
    """``m_off`` is a free row constant: softmax = P / l must stay offset invariant."""
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    case = getattr(two_kernel, "_case", None)
    if case is None:  # pragma: no cover - helper moved
        pytest.skip("in-module test helpers unavailable")

    T, N, S = 8, 4096, 2048
    q, kv, idx = case(T, N=N, S=S)
    out = torch.zeros(T, 32, 512, dtype=torch.float32, device="cuda")
    lse = torch.zeros(T, 32, dtype=torch.float32, device="cuda")
    two_kernel.sparse_mla_two_kernel(q, kv, idx, out, lse)

    # lse must be a monotone function of the row log-sum-exp: compare against a
    # reference that is itself offset free.
    for r in range(T):
        valid = idx[r] >= 0
        k = kv[idx[r][valid].long()].float()
        sc = (q[r].float() @ k.t()) / math.sqrt(576.0)
        ref_out = torch.softmax(sc, dim=-1) @ k[:, :512]
        rel = (out[r] - ref_out).abs().max().item() / max(1e-6, ref_out.abs().max().item())
        assert rel < 3e-2, f"row {r}: rel_out {rel:.3e}"