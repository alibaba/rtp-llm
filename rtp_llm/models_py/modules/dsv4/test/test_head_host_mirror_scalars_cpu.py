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
ATT = _extract(ATTN_PATH, ATTN_SRC, {"_gather_len_max_host"})
gather_len_max_host = ATT["_gather_len_max_host"]


def _package(name, path=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path] if path else []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_package(parent), child, mod)
    return sys.modules[name]


_GENERIC_PROFILER = "rtp_llm.models_py.utils.profiler"


def _load_generic_profiler():
    """Load main's torch-only generic profiler that ``dsv4._profiler`` wraps."""
    if _GENERIC_PROFILER in sys.modules:
        return
    _package("rtp_llm.models_py.utils")
    path = DSV4.parents[1] / "utils" / "profiler.py"
    spec = importlib.util.spec_from_file_location(_GENERIC_PROFILER, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_GENERIC_PROFILER] = mod
    spec.loader.exec_module(mod)


def _load_cp():
    if CP_NAME in sys.modules:
        return sys.modules[CP_NAME]
    _package("rtp_llm")
    _package("rtp_llm.models_py")
    _package("rtp_llm.models_py.modules")
    _load_generic_profiler()
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

    def test_independently_mismatched_mirrors_decline_without_readback(self):
        prefix_t = torch.tensor([0, 4096, 8192], dtype=torch.int32)
        len_t = torch.tensor([4096, 4096, 4096], dtype=torch.int32)
        for prefixes, lengths in (
            ([0], [4096, 4096, 4096]),
            ([0, 4096, 8192], [4096]),
            ([0], [4096]),
        ):
            with self.subTest(prefixes=prefixes, lengths=lengths), _BanHostSync():
                self.assertIsNone(
                    compressed_k_scalars_host(
                        _ctx(prefixes, lengths), prefix_t, len_t, 4
                    )
                )


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


class CPHostMirrors(unittest.TestCase):
    def test_attachment_preserves_domains_and_clamps_first_position(self):
        for rank, length in ((0, 4096), (3, 4096), (3, 5)):
            with self.subTest(rank=rank, length=length):
                ctx = types.SimpleNamespace(
                    prefix_lengths_host=(4096,),
                    input_lengths_global_host=(length,),
                    cp_rank=rank,
                    seq_len_full=length,
                )
                with _BanHostSync():
                    cp._attach_cp_host_mirrors(ctx, (1024,), (4096,))
                self.assertEqual(ctx.prefix_lengths_full_host, (4096,))
                self.assertEqual(ctx.input_lengths_full_host, (length,))
                self.assertEqual(
                    ctx.first_global_position, 4096 + min(rank * 512, length - 1)
                )


def setUpModule():
    global cp
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    cp = _load_cp()


if __name__ == "__main__":
    unittest.main(verbosity=2)
