"""V4.1 sync varlen CP gather per-forward scratch (forward_scratch=True).

Covers ``cp_all_gather_full_varlen``'s CPContext-cached NCCL destination and
non-prefix restore destination:

* outputs are bit-identical to the default fresh-allocation path (same
  ``torch.distributed.all_gather_into_tensor`` semantics, same
  ``index_select`` restore);
* the scratch buffers are allocated once per (shape, dtype) and REUSED by
  subsequent calls of the same forward (no fresh ``torch.empty`` per layer);
* the prefix fast path ignores the restore scratch and returns a view;
* the scratch path is skipped when ``torch.distributed`` is not initialized
  (CPU/reference execution keeps the original path).

The NCCL collective is faked by patching ``torch.distributed`` so the test
runs on CPU; the fake writes each rank's block into the destination exactly
like ``all_gather_into_tensor`` would.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rtp_llm.models_py.modules.dsv4.cp import (  # noqa: E402
    CPContext,
    cp_all_gather_full_varlen,
)


def _make_ctx(*, prefix: bool, chunk: int = 4, seq_full: int = 6) -> CPContext:
    cp_size = 2
    if prefix:
        unpad_restore = torch.arange(seq_full, dtype=torch.long)
    else:
        # Non-prefix permutation: every gather pays the index_select restore.
        unpad_restore = torch.arange(cp_size * chunk, dtype=torch.long)
        unpad_restore = torch.roll(unpad_restore, shifts=1)[:seq_full].contiguous()
    return CPContext(
        cp_size=cp_size,
        cp_rank=0,
        chunk_length=chunk,
        padded_seq_len=cp_size * chunk,
        seq_len_full=seq_full,
        relative_positions=torch.arange(chunk, dtype=torch.long),
        prefix_length=0,
        global_positions=torch.arange(chunk, dtype=torch.long),
        local_is_real=torch.ones(chunk, dtype=torch.bool),
        unpad_restore=unpad_restore,
        seq_len_total=seq_full,
        cp_info=object(),
        unpad_restore_is_prefix=prefix,
    )


class _FakeDist:
    """Simulate torch.distributed for a 2-rank TP group on CPU."""

    def __init__(self):
        self.gather_calls = 0

    def is_initialized(self):
        return True

    def get_world_size(self, group=None):
        return 2

    def all_gather_into_tensor(self, out, tensor, group=None, async_op=False):
        # rank 0 contributes ``tensor``; rank 1 contributes a deterministic
        # shifted copy so the restore permutation is observable.
        self.gather_calls += 1
        rows = tensor.shape[0]
        out[:rows].copy_(tensor)
        out[rows:].copy_(tensor + 100.0)
        return None


def _fake_group(group):
    return object()


def _run(local, ctx, scratch):
    fake = _FakeDist()
    with patch("torch.distributed.is_initialized", fake.is_initialized), patch(
        "torch.distributed.get_world_size", fake.get_world_size
    ), patch(
        "torch.distributed.all_gather_into_tensor", fake.all_gather_into_tensor
    ), patch(
        "rtp_llm.models_py.distributed.collective_torch._get_group", _fake_group
    ):
        return (
            cp_all_gather_full_varlen(local, ctx, forward_scratch=scratch),
            fake,
        )


class CPSyncVarlenScratchTest(unittest.TestCase):
    def test_bit_identical_to_default_path_non_prefix(self):
        ctx = _make_ctx(prefix=False)
        torch.manual_seed(0)
        local = torch.randn(ctx.chunk_length, 5)
        reference, _ = _run(local, ctx, scratch=False)
        ctx2 = _make_ctx(prefix=False)
        assert ctx2._sync_varlen_scratch is None
        scratch_out, _ = _run(local, ctx2, scratch=True)
        self.assertTrue(torch.equal(reference, scratch_out))
        # Both scratch roles were allocated exactly once.
        self.assertEqual(
            sorted(k[0] for k in ctx2._sync_varlen_scratch.keys()),
            ["gathered", "restored"],
        )

    def test_scratch_reused_across_calls_without_new_allocations(self):
        ctx = _make_ctx(prefix=False)
        local = torch.randn(ctx.chunk_length, 5)
        first, _ = _run(local, ctx, scratch=True)
        scratch_ptrs = {
            role: buf.data_ptr() for role, buf in ctx._sync_varlen_scratch.items()
        }
        local2 = torch.randn(ctx.chunk_length, 5)
        second, _ = _run(local2, ctx, scratch=True)
        # Same buffers, no growth of the cache.
        self.assertEqual(len(ctx._sync_varlen_scratch), 2)
        self.assertEqual(
            {role: buf.data_ptr() for role, buf in ctx._sync_varlen_scratch.items()},
            scratch_ptrs,
        )
        # Second result is the restored permutation of the second gather.
        expected = torch.cat([local2, local2 + 100.0], 0).index_select(
            0, ctx.unpad_restore
        )
        self.assertTrue(torch.equal(second, expected))
        # The restore buffer is REUSED, so the first call's returned view now
        # aliases the second result (same contract as the async impl's
        # workspace role buffers: consumers must finish before the next
        # layer's gather of the same role).
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(torch.equal(first, second))

    def test_prefix_fast_path_uses_view_and_skips_restore_scratch(self):
        ctx = _make_ctx(prefix=True, chunk=3, seq_full=6)
        local = torch.randn(ctx.chunk_length, 5)
        out, _ = _run(local, ctx, scratch=True)
        # Only the gathered role was allocated; restore is a view of gathered.
        self.assertEqual([k[0] for k in ctx._sync_varlen_scratch.keys()], ["gathered"])
        self.assertEqual(out.shape[0], ctx.seq_len_full)
        expected = torch.cat([local, local + 100.0], 0)[: ctx.seq_len_full]
        self.assertTrue(torch.equal(out, expected))

    def test_scratch_ignored_without_distributed(self):
        ctx = _make_ctx(prefix=False)
        local = torch.randn(ctx.chunk_length, 5)
        gathered = torch.arange(40, dtype=torch.float32).reshape(8, 5)
        # The generic collective fallback returns the pre-made gathered tensor.
        import rtp_llm.models_py.modules.dsv4.cp as cp_mod

        with patch("torch.distributed.is_initialized", lambda: False), patch.object(
            cp_mod, "all_gather", lambda t, group=None: gathered
        ):
            out = cp_all_gather_full_varlen(local, ctx, forward_scratch=True)
        expected = gathered.index_select(0, ctx.unpad_restore)
        self.assertTrue(torch.equal(out, expected))
        # No scratch was created on the fallback path.
        self.assertIsNone(ctx._sync_varlen_scratch)

    def test_trailing_dims_view_matches(self):
        ctx = _make_ctx(prefix=False)
        local = torch.randn(ctx.chunk_length, 2, 5)
        out, _ = _run(local, ctx, scratch=True)
        self.assertEqual(tuple(out.shape), (ctx.seq_len_full, 2, 5))
        expected = (
            torch.cat([local, local + 100.0], 0)
            .index_select(0, ctx.unpad_restore)
            .view(ctx.seq_len_full, 2, 5)
        )
        self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()
