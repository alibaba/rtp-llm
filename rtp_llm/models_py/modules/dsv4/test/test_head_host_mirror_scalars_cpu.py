"""CPU tests for prefill scalar derivation from host mirrors.

Compare the production helpers with tensor reductions while forbidding blocking
host readbacks. Invalid mirror geometry must use the legacy fallback."""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types
import unittest
from runpy import run_path
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
INDEXER_PATH = DSV4 / "fp8" / "indexer.py"
INDEXER_SRC = INDEXER_PATH.read_text()
ATTN_PATH = DSV4 / "fp8" / "attention.py"
ATTN_SRC = ATTN_PATH.read_text()
CP_PATH = DSV4 / "cp.py"
CP_SRC = CP_PATH.read_text()
CP_NAME = "rtp_llm.models_py.modules.dsv4.cp"


def _extract(path, src, names):
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), f"helper missing from {path}"
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    mod = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    import typing

    env = {
        "torch": torch,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
    }
    exec(compile(mod, str(path), "exec"), env)
    return env


IDX = _extract(INDEXER_PATH, INDEXER_SRC, {"_compressed_k_scalars_host"})
compressed_k_scalars_host = IDX["_compressed_k_scalars_host"]
ATT = _extract(ATTN_PATH, ATTN_SRC, {"_gather_len_max_host", "_prefill_maxes_host"})
gather_len_max_host = ATT["_gather_len_max_host"]
prefill_maxes_host = ATT["_prefill_maxes_host"]


def _package(name, path=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path] if path else []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_package(parent), child, mod)
    return sys.modules[name]


def _load_cp():
    if CP_NAME in sys.modules:
        return sys.modules[CP_NAME]
    _package("rtp_llm")
    _package("rtp_llm.models_py")
    _package("rtp_llm.models_py.modules")
    _package("rtp_llm.models_py.modules.dsv4", str(DSV4))
    dist = _package("rtp_llm.models_py.distributed")
    collective = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")

    class _Group:
        TP = "TP"

    collective.Group = _Group
    collective.all_gather = lambda *a, **k: None
    collective._get_group = lambda g: g
    sys.modules[collective.__name__] = collective
    dist.collective_torch = collective

    spec = importlib.util.spec_from_file_location(CP_NAME, CP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[CP_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _ctx(prefixes, lengths):
    return types.SimpleNamespace(
        prefix_lengths_full_host=tuple(prefixes),
        input_lengths_full_host=tuple(lengths),
    )


# Matrix: (prefixes, lengths) covering B=1..3, zero/large prefixes, partial
# last chunk lengths, and the production full chunk (4096).
CASES = [
    ([0], [4096]),
    ([4096], [4096]),
    ([28672], [4096]),
    ([7], [3000]),
    ([0, 512], [2048, 2048]),
    ([100, 4096, 0], [1000, 4096, 5]),
]
RATIOS = (4, 128)
WINDOW = 2048


class CompressedKScalars(unittest.TestCase):
    """indexer.prepare's (T, end_pos)."""

    def _legacy(self, prefixes, lengths, ratio):
        prefix_t = torch.tensor(prefixes, dtype=torch.int32)
        len_t = torch.tensor(lengths, dtype=torch.int32)
        seq_total = prefix_t.to(torch.int64) + len_t.to(torch.int64)
        t_per_req = (seq_total // ratio).to(torch.int32)
        cu = torch.zeros(len(prefixes) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(t_per_req.to(torch.int64), dim=0).to(torch.int32)
        return int(cu[-1].item()), int(seq_total[0].item())

    def test_parity_over_matrix(self):
        for prefixes, lengths in CASES:
            for ratio in RATIOS:
                ctx = _ctx(prefixes, lengths)
                prefix_t = torch.tensor(prefixes, dtype=torch.int32)
                len_t = torch.tensor(lengths, dtype=torch.int32)
                want = self._legacy(prefixes, lengths, ratio)
                with _BanHostSync():
                    got = compressed_k_scalars_host(ctx, prefix_t, len_t, ratio)
                self.assertIsNotNone(got)
                self.assertEqual(tuple(got), want)

    def test_ban_bites_legacy_path(self):
        # Anti-vacuity: the legacy reduction really does trip the ban.
        prefix_t = torch.tensor([4096], dtype=torch.int32)
        len_t = torch.tensor([4096], dtype=torch.int32)
        with self.assertRaises(AssertionError):
            with _BanHostSync():
                self._legacy([4096], [4096], 4)
        del prefix_t, len_t

    def test_domain_mismatch_returns_none(self):
        prefix_t = torch.tensor([0, 1, 2], dtype=torch.int32)
        len_t = torch.tensor([10, 20, 30], dtype=torch.int32)
        # Mirror shorter than the device arg -> decline.
        self.assertIsNone(
            compressed_k_scalars_host(_ctx([0], [10]), prefix_t, len_t, 4)
        )
        # No context / no mirrors -> decline.
        self.assertIsNone(compressed_k_scalars_host(None, prefix_t, len_t, 4))
        self.assertIsNone(
            compressed_k_scalars_host(types.SimpleNamespace(), prefix_t, len_t, 4)
        )
        # Empty mirrors -> decline.
        self.assertIsNone(
            compressed_k_scalars_host(_ctx([], []), prefix_t[:0], len_t[:0], 4)
        )

    def test_guard_stripped_mutant_misfires(self):
        # Mutant: drop the numel domain guards.  With a short mirror the real
        # helper declines (None) while the mutant silently computes a wrong T.
        fn_src = ast.get_source_segment(
            INDEXER_SRC,
            next(
                n
                for n in ast.parse(INDEXER_SRC).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "_compressed_k_scalars_host"
            ),
        )
        mutant_src = fn_src.replace(
            "    if int(prefix_lengths.numel()) != len(prefix_host):\n        return None\n"
            "    if int(eff_input_lengths.numel()) != len(lengths_host):\n        return None\n",
            "",
        )
        assert mutant_src != fn_src, "mutant edit did not apply"
        env = {"torch": torch}
        exec(mutant_src, env)
        mutant = env["_compressed_k_scalars_host"]
        prefix_t = torch.tensor([0, 4096, 8192], dtype=torch.int32)
        len_t = torch.tensor([4096, 4096, 4096], dtype=torch.int32)
        short = _ctx([0], [4096])
        self.assertIsNone(compressed_k_scalars_host(short, prefix_t, len_t, 4))
        legacy_t, _ = self._legacy([0, 4096, 8192], [4096, 4096, 4096], 4)
        mutant_t, _ = mutant(short, prefix_t, len_t, 4)
        self.assertNotEqual(mutant_t, legacy_t)


class GatherLenMax(unittest.TestCase):
    """attention._build_swa_prefill_meta_varlen's combined_gather_len_max."""

    def _legacy(self, prefixes, lengths, window):
        # The fused kernel: gather_len[b] = q_len[b] + min(prefix_len[b], win-1)
        # with q_len == lengths[b] and prefix_len == prefixes[b] under the CP
        # write trio.
        lens = torch.tensor(
            [l + min(p, window - 1) for p, l in zip(prefixes, lengths)],
            dtype=torch.int32,
        )
        return int(lens.max().item())

    def _call(self, prefixes, lengths):
        ctx = _ctx(prefixes, lengths)
        ctx.input_lengths_global = torch.tensor(lengths, dtype=torch.int32)
        prefix_t = torch.tensor(prefixes, dtype=torch.int32)
        return gather_len_max_host(ctx, prefix_t, len(prefixes), WINDOW)

    def test_parity_over_matrix(self):
        for prefixes, lengths in CASES:
            with self.subTest(prefixes=prefixes, lengths=lengths):
                want = self._legacy(prefixes, lengths, WINDOW)
                with _BanHostSync():
                    got = self._call(prefixes, lengths)
                self.assertEqual(got, want)

    def test_declines_without_matching_domains(self):
        ctx = _ctx([0], [4096])
        ctx.input_lengths_global = torch.tensor([4096, 5], dtype=torch.int32)
        prefix_t = torch.tensor([0], dtype=torch.int32)
        self.assertIsNone(gather_len_max_host(ctx, prefix_t, 1, WINDOW))
        self.assertIsNone(gather_len_max_host(None, prefix_t, 1, WINDOW))
        ctx2 = _ctx([0], [4096])
        ctx2.input_lengths_global = None
        self.assertIsNone(gather_len_max_host(ctx2, prefix_t, 1, WINDOW))

    def test_ban_bites_legacy_path(self):
        with self.assertRaises(AssertionError):
            with _BanHostSync():
                self._legacy([1], [4096], WINDOW)


class PrefillMaxes(unittest.TestCase):
    """attention._build_workspace_meta's (N_max, gather_len_max)."""

    def _legacy(self, prefixes, lengths, ratio, window):
        sp = torch.tensor(prefixes, dtype=torch.int32)
        s = torch.tensor(lengths, dtype=torch.int32)
        n_per_req = (sp + s) // ratio
        gather_per_req = s + torch.clamp_max(sp, window - 1)
        maxes = torch.stack([n_per_req.max(), gather_per_req.max()])
        n_max, g_max = (int(v) for v in maxes.tolist())
        return n_max, g_max

    def _call(self, prefixes, lengths, ratio):
        prefix_t = torch.tensor(prefixes, dtype=torch.int32)
        len_t = torch.tensor(lengths, dtype=torch.int32)
        return prefill_maxes_host(
            tuple(prefixes),
            tuple(lengths),
            ratio,
            WINDOW,
            prefix_arg=prefix_t,
            lengths_arg=len_t,
        )

    def test_parity_over_matrix(self):
        for prefixes, lengths in CASES:
            for ratio in RATIOS:
                with self.subTest(prefixes=prefixes, lengths=lengths, ratio=ratio):
                    want = self._legacy(prefixes, lengths, ratio, WINDOW)
                    with _BanHostSync():
                        got = self._call(prefixes, lengths, ratio)
                    self.assertIsNotNone(got)
                    self.assertEqual(tuple(got), want)

    def test_declines_on_domain_mismatch(self):
        prefix_t = torch.tensor([0, 1], dtype=torch.int32)
        len_t = torch.tensor([5, 6], dtype=torch.int32)
        self.assertIsNone(
            prefill_maxes_host(
                (0,), (5,), 4, WINDOW, prefix_arg=prefix_t, lengths_arg=len_t
            )
        )
        self.assertIsNone(
            prefill_maxes_host(
                None, (5, 6), 4, WINDOW, prefix_arg=prefix_t, lengths_arg=len_t
            )
        )
        self.assertIsNone(
            prefill_maxes_host(
                (0, 1), (5, 6), 4, WINDOW, prefix_arg=None, lengths_arg=len_t
            )
        )

    def test_ban_bites_legacy_path(self):
        with self.assertRaises(AssertionError):
            with _BanHostSync():
                self._legacy([1], [4096], 4, WINDOW)


class CpFullPrefillPositions(unittest.TestCase):
    """cp.build_cp_full_prefill_positions under the mirror path."""

    def _make_ctx(self, prefixes, lengths, with_mirrors=True):
        return types.SimpleNamespace(
            input_lengths_global=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.tensor(prefixes, dtype=torch.long),
            prefix_length=prefixes[0] if prefixes else 0,
            input_lengths_full_host=tuple(lengths) if with_mirrors else None,
            prefix_lengths_full_host=tuple(prefixes) if with_mirrors else None,
        )

    def _oracle(self, prefixes, lengths, device):
        """The uncached item-driven loop, verbatim."""
        positions = []
        b_idx = []
        for req_id in range(len(lengths)):
            length = int(lengths[req_id])
            start = int(prefixes[req_id])
            if length <= 0:
                continue
            positions.append(
                torch.arange(start, start + length, dtype=torch.long, device=device)
            )
            b_idx.append(torch.full((length,), req_id, dtype=torch.long, device=device))
        pos = (
            torch.cat(positions).contiguous()
            if positions
            else torch.empty(0, dtype=torch.long)
        )
        req = (
            torch.cat(b_idx).contiguous() if b_idx else torch.empty(0, dtype=torch.long)
        )
        zero = torch.zeros(1, dtype=torch.long, device=device)
        cu = torch.cat(
            [zero, torch.cumsum(torch.tensor(lengths).to(torch.long), dim=0)]
        ).contiguous()
        return pos, req, torch.tensor(prefixes, dtype=torch.long), cu

    def test_parity_over_matrix(self):
        for prefixes, lengths in CASES:
            with self.subTest(prefixes=prefixes, lengths=lengths):
                ctx = self._make_ctx(prefixes, lengths)
                with _BanHostSync():
                    got = cp.build_cp_full_prefill_positions(ctx, torch.device("cpu"))
                want = self._oracle(prefixes, lengths, torch.device("cpu"))
                for g, w in zip(got, want):
                    self.assertTrue(torch.equal(g, w))

    def test_mirror_path_is_ban_clean_and_legacy_falls_back(self):
        # Mirrors present: no item() under the ban (parity test above).
        # Mirrors absent: the legacy .item() loop still works (and would trip
        # the ban — proven by the ban-bites check on the oracle below).
        prefixes, lengths = [4096], [4096]
        ctx = self._make_ctx(prefixes, lengths, with_mirrors=False)
        got = cp.build_cp_full_prefill_positions(ctx, torch.device("cpu"))
        want = self._oracle(prefixes, lengths, torch.device("cpu"))
        for g, w in zip(got, want):
            self.assertTrue(torch.equal(g, w))

    def test_mirror_wrong_value_changes_output(self):
        # Rejecting mutant discipline: a corrupted mirror must change the
        # output (the mirror genuinely drives the arange bounds).
        ctx_good = self._make_ctx([100], [1000])
        ctx_bad = self._make_ctx([100], [1000])
        ctx_bad.input_lengths_full_host = (999,)
        good = cp.build_cp_full_prefill_positions(ctx_good, torch.device("cpu"))
        bad = cp.build_cp_full_prefill_positions(ctx_bad, torch.device("cpu"))
        self.assertFalse(torch.equal(good[0], bad[0]))


class ProductionCallSites(unittest.TestCase):
    def test_indexer_prepare_uses_host_scalars(self):
        self.assertIn("_compressed_k_scalars_host(", INDEXER_SRC)
        self.assertIn("host_scalars is not None", INDEXER_SRC)

    def test_attention_sites_use_host_maxes(self):
        self.assertIn("_gather_len_max_host(", ATTN_SRC)
        self.assertIn("_prefill_maxes_host(", ATTN_SRC)
        # The legacy syncs survive only on the mirror-miss fallback branches.
        self.assertIn(
            "        if combined_gather_len_max is None:\n"
            "            combined_gather_len_max = int(combined_gather_lens.max().item())",
            ATTN_SRC,
        )
        self.assertIn(
            "            else:\n"
            "                # Single stacked .tolist() sync when mirrors are unavailable.\n"
            "                maxes = torch.stack([N_per_req.max(), gather_len_per_req.max()])",
            ATTN_SRC,
        )

    def test_row_seqlens_full_is_device_filled(self):
        self.assertIn("row_seqlens_full = torch.full(", ATTN_SRC)
        self.assertNotIn("row_seqlens_full = torch.tensor([seqlen_full]", ATTN_SRC)

    def test_torch_full_matches_tensor_constructor_values(self):
        # The production swap: torch.full (device fill kernel, no pageable
        # HtoD + sync) must produce the identical tensor to the legacy
        # torch.tensor([v], device=...) constructor.
        for v in (0, 1, 4096, 32768):
            legacy = torch.tensor([v], dtype=torch.long)
            got = torch.full((1,), v, dtype=torch.long)
            self.assertEqual(got.dtype, legacy.dtype)
            self.assertTrue(torch.equal(got, legacy))

    def test_cp_context_carries_input_lengths_mirror(self):
        self.assertIn("input_lengths_full_host", CP_SRC)
        self.assertIn(
            "input_lengths_full_host=(\n            tuple(input_lengths_host)", CP_SRC
        )


def setUpModule():
    global cp
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    cp = _load_cp()


if __name__ == "__main__":
    unittest.main(verbosity=2)
