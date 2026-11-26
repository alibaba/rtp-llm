"""CPU tests for synchronous prefill scalar derivation.

Loads the L11-E7 synchronous-metadata fixture (``test_prefill_async_head_cpu``)
and drives the ACTUAL indexer/attention builders it extracts. Expected
scalars — indexer ``T``/``end_pos``/``ks``/``ke``/``cu_kv_seqlens`` and
workspace ``N``/``M``/gather/fresh slots — come from the sibling test's
independent Python coordinate oracle, never from a second production
builder.

The second test drops ONLY the retained framework host mirrors
(``input_lengths_global_host`` / ``prefix_lengths_host``) so the same
builders must reach the same integers through their tensor fallback paths.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).parent

_FIXTURE_NAME = "_dsv4_sync_fixture"
_FIXTURE_PATH = HERE / "test_prefill_async_head_cpu.py"
_MISSING = object()


def _load_fixture():
    """Source-load the sibling synchronous-metadata fixture module."""
    spec = importlib.util.spec_from_file_location(_FIXTURE_NAME, _FIXTURE_PATH)
    mod = importlib.util.module_from_spec(spec)
    prev = sys.modules.get(_FIXTURE_NAME, _MISSING)
    sys.modules[_FIXTURE_NAME] = mod

    def _restore():
        if prev is _MISSING:
            sys.modules.pop(_FIXTURE_NAME, None)
        else:
            sys.modules[_FIXTURE_NAME] = prev

    unittest.addModuleCleanup(_restore)
    spec.loader.exec_module(mod)
    return mod


class SynchronousScalarTest(unittest.TestCase):
    """Actual builders vs the sibling's independent integer oracle."""

    def _assert_oracle(self, rank, prefix, ctx, metas):
        fixture = sys.modules[_FIXTURE_NAME]
        gp = torch.tensor(
            [prefix + i for i in fixture._cp_zigzag_slices(4096, 4, rank)],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(ctx.global_positions, gp), "ctx.global_positions")

        idx = metas[4].csa_meta.indexer_meta
        T = (prefix + 4096) // 4
        self.assertEqual(idx.T, T)
        self.assertEqual(idx.end_pos, prefix + 4096)
        self.assertTrue(
            torch.equal(idx.ks, torch.zeros(1024, dtype=torch.int32)),
            "indexer ks",
        )
        want_ke = torch.clamp((gp.to(torch.int64) + 1) // 4, max=T).to(torch.int32)
        self.assertTrue(torch.equal(idx.ke, want_ke), "indexer ke")
        self.assertEqual(idx.cu_kv_seqlens.tolist(), [0, T], "indexer cu_kv_seqlens")

        for ratio in (4, 128):
            meta = metas[ratio]
            wm = (
                meta.csa_meta.workspace_meta
                if ratio == 4
                else meta.hca_meta.workspace_meta
            )
            N = (prefix + 4096) // ratio
            want_M = N + 4096 + min(prefix, 63)
            self.assertEqual(wm.N, N, f"ratio={ratio} N")
            self.assertEqual(wm.M, want_M, f"ratio={ratio} M")
            self.assertEqual(
                wm.cmp_seq_lens.tolist(), [N], f"ratio={ratio} cmp_seq_lens"
            )
            self.assertEqual(
                wm.swa_gather_lens.tolist(),
                [4096 + min(prefix, 63)],
                f"ratio={ratio} swa_gather_lens",
            )
            self.assertEqual(
                wm.new_k_slot_in_flat.tolist(),
                [N + min(prefix, 63) + i for i in range(4096)],
                f"ratio={ratio} new_k_slot_in_flat",
            )
        for ratio in (0, 4, 128):
            self.assertEqual(
                metas[ratio].any_cont, prefix > 0, f"ratio={ratio} any_cont"
            )
            self.assertEqual(
                metas[ratio].row_seqlens_full.tolist(),
                [4096],
                f"ratio={ratio} row_seqlens_full",
            )

    def test_cp4_indexer_and_workspace_bounds(self):
        fixture = sys.modules[_FIXTURE_NAME]
        for rank in (0, 1, 2, 3):
            for prefix in (0, 4096, 28672):
                with self.subTest(rank=rank, prefix=prefix):
                    ctx, metas, _attns = fixture._legacy_build(
                        rank, prefix, fixture._make_tables(128)
                    )
                    self._assert_oracle(rank, prefix, ctx, metas)

    def test_framework_host_and_tensor_fallback_use_same_integer_oracle(self):
        fixture = sys.modules[_FIXTURE_NAME]
        for rank in (0, 1, 2, 3):
            for prefix in (0, 4096, 28672):
                with self.subTest(rank=rank, prefix=prefix):
                    ctx = fixture._build_ctx(rank, prefix)
                    # Drop ONLY the retained framework host mirrors; the
                    # builders must reach the same integers through their
                    # tensor fallback paths (legitimate eager .item()).
                    ctx.input_lengths_global_host = None
                    ctx.prefix_lengths_host = None
                    with patch.object(fixture, "_build_ctx", return_value=ctx):
                        built_ctx, metas, _attns = fixture._legacy_build(
                            rank, prefix, fixture._make_tables(128)
                        )
                    self.assertIs(built_ctx, ctx)
                    self._assert_oracle(rank, prefix, built_ctx, metas)


def setUpModule():
    _fixture = _load_fixture()
    # The fixture's own setUpModule isolates runtime modules, loads the real
    # dsv4 leaf modules, extracts the production builders and binds them onto
    # its stub classes. Its registered cleanup restores the package stubs.
    _fixture.setUpModule()


if __name__ == "__main__":
    unittest.main(verbosity=2)
