"""Batched multi-request FP4 pool reads: not ported to SM120.

The full implementation lives on the prefill-overlap branches (commit 4b72b0bad9,
"perf(dsv41): optimize batched high-reuse prefill") and is gated twice over for hardware
this branch does not target:

  * its entry gate returns None unless `torch.cuda.get_device_capability(device)[0] == 10`
    (SM100), while the SM120 parts this branch runs report 12.x;
  * it additionally requires `rtp_llm.models_py.modules.dsv4.fp8._v41_joint_pool`, which
    is not present in this branch.

AttentionV41FP8 imports this module unconditionally on the batched producer path
(`fp8/attention_v41.py`, the `fused_indexer and len(ends) > 1` branch), so without this
module a two-request prefill chunk raises ImportError (the P6 blocker). Providing the
interface with an unconditional None routes every batched chunk to the existing
per-request fallback loop in the caller, which is byte-identical in result and slower
only in host-side gather time.

If this branch is ever expected to serve batched prefill on SM100, replace this file with
the real implementation (and its `_v41_joint_pool` dependency) rather than extending the
stub.
"""

from __future__ import annotations


def try_gather_prefill_pools(attn, main_pool, index_pool, ends, seq_ends):
    """Decline every batch; the caller's per-request gather path handles it."""
    return None
