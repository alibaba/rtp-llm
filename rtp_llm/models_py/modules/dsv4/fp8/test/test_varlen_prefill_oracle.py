"""Phase-0 oracle for the DSV4 prefill varlen migration.

Compares two ways of running prefill on the same (input_ids, prefix) batch:

  * **Per-request**: call the prefill arm B times, once per request, the
    way the legacy ``max_context_batch_size=1`` FIFO scheduler does today.
  * **Batched**: pack the same B requests into one flat ``[T_total]`` call
    (cu_seqlens / position_ids derived in ``forward_layers``), the way the
    framework will drive prefill once Phase 2/3 lands.

The two outputs MUST be bit-equal (BF16 paths) or within FP8 atol/rtol
(FP8 paths) for every (compress_ratio, sp_b mix, layer) combination
covered by the fixtures below.

Fixtures use the small random-weight V4 config from
``transformer_smoke_test.py`` because:
  * the prefill code path is dtype-agnostic — slot_mapping math, freqs_cis
    gather, sparse_attn indices all run on the same shapes regardless of
    weight magnitude;
  * full-model FP8 weights are 600+ GB and not available in the test
    sandbox.

The production path now always uses varlen metadata; this oracle keeps the
pack/unpack comparison helpers for batched prefill fixtures.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

# NOTE: actual fixture imports + model construction are stubbed below
# pending a small-FP8 fixture (Phase 0 follow-up). The harness itself —
# pack/unpack helpers, comparison thresholds, xfail markers per phase —
# is what's load-bearing for parallel Engineer A/B work.


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _pack_batch(
    per_req_input_ids: List[torch.Tensor],
    per_req_prefix_lengths: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the flat ``[T_total]`` inputs the framework hands to
    ``forward_layers``: ``input_ids`` / ``positions`` / ``cu_seqlens`` /
    ``input_lengths`` / ``prefix_lengths``."""
    device = per_req_input_ids[0].device
    input_ids = torch.cat(per_req_input_ids, dim=0)
    input_lengths = torch.tensor(
        [int(t.numel()) for t in per_req_input_ids], dtype=torch.int32, device=device
    )
    prefix_lengths = torch.tensor(
        per_req_prefix_lengths, dtype=torch.int32, device=device
    )
    cu_seqlens = torch.zeros(
        input_lengths.numel() + 1, dtype=torch.int32, device=device
    )
    cu_seqlens[1:] = torch.cumsum(input_lengths, dim=0).to(torch.int32)
    positions = torch.cat(
        [
            torch.arange(int(p), int(p) + int(L), dtype=torch.int64, device=device)
            for L, p in zip(input_lengths.tolist(), per_req_prefix_lengths)
        ],
        dim=0,
    )
    return input_ids, positions, cu_seqlens, input_lengths, prefix_lengths


def _unpack_batch(
    flat_hidden: torch.Tensor, cu_seqlens: torch.Tensor
) -> List[torch.Tensor]:
    """Split a packed ``[T_total, dim]`` output back into per-request
    ``[S_b, dim]`` tensors so the comparison harness can match against
    the per-request loop output token-by-token."""
    cu = cu_seqlens.tolist()
    return [flat_hidden[cu[b] : cu[b + 1]] for b in range(len(cu) - 1)]


def _assert_close(
    batched: List[torch.Tensor],
    per_req: List[torch.Tensor],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    label: str = "",
) -> None:
    assert len(batched) == len(per_req), f"{label}: B mismatch"
    for b, (lhs, rhs) in enumerate(zip(batched, per_req)):
        torch.testing.assert_close(
            lhs,
            rhs,
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"{label} req={b}: {m}",
        )


# ---------------------------------------------------------------------------
# Fixture builder placeholder. Real implementation lands with Phase 0
# follow-up (needs a small FP8-KV V4 model + bound paged pools).
# ---------------------------------------------------------------------------
def _build_v4_fp8_test_model():
    pytest.skip(
        "Phase-0 follow-up: small FP8-KV V4 fixture not yet built. "
        "Engineer A/B will land this alongside their first phase patch."
    )


# ---------------------------------------------------------------------------
# Per-phase oracle tests. Each starts as xfail until its phase lands.
# Flip the marker to a plain pass once the corresponding builder body
# has been switched from B==1 scalar plumbing to varlen.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="Phase 2 (SWA) not yet landed — _build_swa_prefill_meta still B==1",
    strict=False,
)
def test_swa_only_prefill_b2_mixed_sp() -> None:
    """compress_ratio == 0 layer, B=2: one cold (sp=0), one continuation (sp>0)."""
    model = _build_v4_fp8_test_model()
    # TODO(engineer-A): drive layer with mixed-sp B=2 fixture and compare
    # batched-call output to per-request loop output.
    del model


@pytest.mark.xfail(
    reason=(
        "Phase 3a (compressor/indexer flat) not yet landed — "
        "_build_compressor_meta + IndexerFP8.prepare still B==1"
    ),
    strict=False,
)
def test_csa_layer_prefill_b2_mixed_sp() -> None:
    """compress_ratio == 4 (CSA): indexer + compressor + workspace path."""
    model = _build_v4_fp8_test_model()
    del model


@pytest.mark.xfail(
    reason=(
        "Phase 3b workspace builder has landed (_build_workspace_meta + "
        "_attn_via_workspace are varlen-capable; field-level coverage in "
        "test_workspace_prefill_varlen). End-to-end oracle still pending the "
        "small FP8-KV V4 fixture for ``_build_v4_fp8_test_model``."
    ),
    strict=False,
)
def test_hca_layer_prefill_b2_mixed_sp() -> None:
    """compress_ratio == 128 (HCA): compressor + dense workspace path."""
    model = _build_v4_fp8_test_model()
    del model


# ---------------------------------------------------------------------------
# End-to-end (all three layer types interleaved) — the gate that closes
# the migration. Stays xfail until Phase 4 cleanup.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="End-to-end gate — flips when Phase 2 + Phase 3 both land",
    strict=False,
)
def test_full_v4_prefill_b4_mixed_sp() -> None:
    """Full-stack: 4 requests, mix of cold and continuation, all 3 ratios."""
    model = _build_v4_fp8_test_model()
    del model
