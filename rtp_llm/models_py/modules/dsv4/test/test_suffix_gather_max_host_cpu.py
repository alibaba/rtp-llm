"""CPU tests for deriving suffix gather extents from host prefix mirrors.

Compare host-derived extents with the legacy reductions, check domain rejection,
and verify that the production builders retain their device-readback fallback."""

from __future__ import annotations

import ast
import pathlib
import unittest
from runpy import run_path
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
ATTN_PATH = DSV4 / "fp8" / "attention.py"
ATTN_SRC = ATTN_PATH.read_text()


def _extract(names):
    tree = ast.parse(ATTN_SRC)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), f"helper missing: {names}"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    import os as _os
    import typing

    env = {
        "torch": torch,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "os": _os,
    }
    exec(compile(mod, str(ATTN_PATH), "exec"), env)
    return env


ENV = _extract(
    {
        "_suffix_gather_lens_max_host",
        "_build_suffix_pool_slot_mapping",
        "_build_suffix_cp_sliced_slot_mapping",
    }
)
suffix_gather_lens_max_host = ENV["_suffix_gather_lens_max_host"]
build_suffix_pool_slot_mapping = ENV["_build_suffix_pool_slot_mapping"]
build_suffix_cp_sliced_slot_mapping = ENV["_build_suffix_cp_sliced_slot_mapping"]


class _BanHostSync:
    def __enter__(self):
        def banned(*a, **k):
            raise AssertionError("host synchronizing call on the hot path")

        self._patchers = [
            patch.object(torch.Tensor, "item", banned),
            patch.object(torch.Tensor, "cpu", banned),
            patch.object(torch.Tensor, "tolist", banned),
        ]
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._patchers:
            p.stop()
        return False


# (prefixes, count) matrix: B=1..3, zero/large prefixes, window clamp edges.
CASES = [
    ([0], 1),
    ([4096], 1),
    ([28672], 1),
    ([2048], 1),  # == win-1 boundary is 2047; prefix > win
    ([2047], 1),  # exactly win-1
    ([0, 512], 2),
    ([100, 4096, 0], 3),
    ([100, 4096, 0], 2),  # sliced count < B
]
WINDOW = 2048


def _legacy_swa_gather_lens(prefixes, count, win):
    """The legacy device value: clamp_max(prefix, win-1)[:count]."""
    pl = torch.tensor(prefixes, dtype=torch.int32)
    return torch.clamp_max(pl, win - 1)[:count].to(torch.int64)


class SuffixGatherLensMaxHost(unittest.TestCase):
    def test_parity_over_matrix(self):
        for prefixes, count in CASES:
            with self.subTest(prefixes=prefixes, count=count):
                prefix_t = torch.tensor(prefixes, dtype=torch.int32)
                want = int(
                    _legacy_swa_gather_lens(prefixes, count, WINDOW).max().item()
                )
                with _BanHostSync():
                    got = suffix_gather_lens_max_host(
                        prefix_t, tuple(prefixes), count, WINDOW
                    )
                self.assertIsNotNone(got)
                self.assertEqual(got, want)

    def test_declines_on_domain_mismatch(self):
        prefix_t = torch.tensor([0, 4096], dtype=torch.int32)
        # Mirror shorter than the device arg.
        self.assertIsNone(suffix_gather_lens_max_host(prefix_t, (0,), 2, WINDOW))
        # No mirror / no device arg.
        self.assertIsNone(suffix_gather_lens_max_host(prefix_t, None, 2, WINDOW))
        self.assertIsNone(suffix_gather_lens_max_host(None, (0, 4096), 2, WINDOW))
        # count out of range.
        self.assertIsNone(suffix_gather_lens_max_host(prefix_t, (0, 4096), 0, WINDOW))
        self.assertIsNone(suffix_gather_lens_max_host(prefix_t, (0, 4096), 3, WINDOW))

    def test_ban_bites_legacy_path(self):
        with self.assertRaises(AssertionError):
            with _BanHostSync():
                _legacy_swa_gather_lens([100, 4096], 2, WINDOW).max().item()


class SuffixBuildersWithOverride(unittest.TestCase):
    """Drive the REAL suffix builders with the host override under the ban."""

    def _fixtures(self, prefixes, lengths, win):
        # seq_lens = prefix + input (total), gather = clamp(prefix, win-1).
        B = len(prefixes)
        seq_lens = torch.tensor(
            [p + l for p, l in zip(prefixes, lengths)], dtype=torch.int32
        )
        gather_lens = torch.clamp_max(
            seq_lens - seq_lens.new_tensor(lengths), 0
        )  # = prefix
        gather_lens = torch.clamp_max(
            torch.tensor(prefixes, dtype=torch.int32), win - 1
        )
        nblocks = 8
        block_table = torch.arange(1, B * nblocks + 1, dtype=torch.int32).view(
            B, nblocks
        )
        return seq_lens, gather_lens, block_table

    def test_pool_builder_override_parity(self):
        for prefixes, lengths, count in (
            ([0], [4096], 1),
            ([4096], [4096], 1),
            ([2048], [4096], 1),
            ([0, 512], [2048, 2048], 2),
            ([100, 4096, 0], [1000, 4096, 5], 3),
        ):
            with self.subTest(prefixes=prefixes):
                seq_lens, gather_lens, bt = self._fixtures(prefixes, lengths, WINDOW)
                prefix_t = torch.tensor(prefixes, dtype=torch.int32)
                override = suffix_gather_lens_max_host(
                    prefix_t, tuple(prefixes), len(prefixes), WINDOW
                )
                legacy = build_suffix_pool_slot_mapping(
                    block_table=bt,
                    seq_lens=seq_lens,
                    gather_lens=gather_lens,
                    entries_per_block=64,
                    tokens_per_block_for_block_table=256,
                    ring_entries=64,
                )
                with _BanHostSync():
                    got = build_suffix_pool_slot_mapping(
                        block_table=bt,
                        seq_lens=seq_lens,
                        gather_lens=gather_lens,
                        entries_per_block=64,
                        tokens_per_block_for_block_table=256,
                        ring_entries=64,
                        max_gather_host=override,
                    )
                self.assertTrue(torch.equal(got, legacy))

    def test_cp_sliced_builder_override_parity(self):
        for prefixes, lengths in (
            ([0], [4096]),
            ([4096], [4096]),
            ([0, 512], [2048, 2048]),
        ):
            with self.subTest(prefixes=prefixes):
                seq_lens, gather_lens, bt = self._fixtures(prefixes, lengths, WINDOW)
                prefix_t = torch.tensor(prefixes, dtype=torch.int32)
                override = suffix_gather_lens_max_host(
                    prefix_t, tuple(prefixes), len(prefixes), WINDOW
                )
                kwargs = dict(
                    block_table=bt,
                    seq_lens=seq_lens,
                    gather_lens=gather_lens,
                    local_entries_per_block=16,
                    tokens_per_block_for_block_table=256,
                    cp_rank=1,
                    cp_size=4,
                )
                legacy = build_suffix_cp_sliced_slot_mapping(**kwargs)
                with _BanHostSync():
                    got = build_suffix_cp_sliced_slot_mapping(
                        max_gather_host=override, **kwargs
                    )
                self.assertTrue(torch.equal(got, legacy))

    def test_ban_bites_legacy_builder(self):
        # Anti-vacuity: without the override the legacy CPU-source branch calls
        # .item() — the ban must catch it.
        seq_lens, gather_lens, bt = self._fixtures([4096], [4096], WINDOW)
        with self.assertRaises(AssertionError):
            with _BanHostSync():
                build_suffix_pool_slot_mapping(
                    block_table=bt,
                    seq_lens=seq_lens,
                    gather_lens=gather_lens,
                    entries_per_block=64,
                    tokens_per_block_for_block_table=256,
                    ring_entries=64,
                )

    def test_wrong_override_mutant_changes_output(self):
        # Rejecting mutant discipline: a corrupted override must observably
        # change the builder output (wider trailing all -1 columns).
        seq_lens, gather_lens, bt = self._fixtures([4096], [4096], WINDOW)
        good = build_suffix_pool_slot_mapping(
            block_table=bt,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            entries_per_block=64,
            tokens_per_block_for_block_table=256,
            ring_entries=64,
        )
        bad = build_suffix_pool_slot_mapping(
            block_table=bt,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            entries_per_block=64,
            tokens_per_block_for_block_table=256,
            ring_entries=64,
            max_gather_host=int(gather_lens.max().item()) + 1,
        )
        self.assertNotEqual(tuple(good.shape), tuple(bad.shape))

    def test_guard_stripped_helper_mutant_misfires(self):
        # Mutant: drop the numel domain guard.  With a short mirror the real
        # helper declines (None) while the mutant silently computes a wrong max.
        fn_src = ast.get_source_segment(
            ATTN_SRC,
            next(
                n
                for n in ast.parse(ATTN_SRC).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "_suffix_gather_lens_max_host"
            ),
        )
        mutant_src = fn_src.replace(
            "    if int(prefix_lengths.numel()) != len(prefix_lengths_full_host):\n"
            "        return None\n",
            "",
        )
        assert mutant_src != fn_src, "mutant edit did not apply"
        env = {
            "torch": torch,
            "Optional": __import__("typing").Optional,
            "Tuple": __import__("typing").Tuple,
        }
        exec(mutant_src, env)
        mutant = env["_suffix_gather_lens_max_host"]
        prefix_t = torch.tensor([0, 4096, 8192], dtype=torch.int32)
        self.assertIsNone(suffix_gather_lens_max_host(prefix_t, (0,), 3, WINDOW))
        legacy = int(_legacy_swa_gather_lens([0, 4096, 8192], 3, WINDOW).max().item())
        self.assertNotEqual(mutant(prefix_t, (0,), 3, WINDOW), legacy)


class ProductionCallSites(unittest.TestCase):
    def test_three_call_sites_pass_override(self):
        # All three production call sites pass max_gather_host.
        self.assertEqual(ATTN_SRC.count("max_gather_host="), 3)
        self.assertIn("_suffix_gather_lens_max_host(", ATTN_SRC)

    def test_legacy_readback_survives_as_fallback(self):
        # The builders keep both legacy branches after the override.
        self.assertIn(
            "if max_gather_host is not None:\n"
            "        max_gather = int(max_gather_host)\n"
            '    elif gather_lens.device.type == "cpu"',
            ATTN_SRC,
        )
        self.assertIn(
            "max_gather = int(gather_lens_l.max().item()) if gather_lens_l.numel() else 0",
            ATTN_SRC,
        )

    def test_cmp_pool_uses_host_maxes_n_max(self):
        self.assertIn(
            "max_gather_host=N_max if host_maxes is not None else None", ATTN_SRC
        )


def setUpModule():
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()


if __name__ == "__main__":
    unittest.main(verbosity=2)
