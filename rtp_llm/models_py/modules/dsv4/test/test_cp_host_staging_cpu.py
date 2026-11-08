"""CPU tests for CP context geometry and retained host mirrors.

Loads the REAL ``cp.py`` via importlib with package stubs (no built rtp_llm
package needed). The first-position assertions read
``ctx.global_positions[0]`` through ordinary tensor access and compare it
against an independent zigzag coordinate — no host-side first-position
helper is involved.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from runpy import run_path

import torch

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
CP_PATH = DSV4 / "cp.py"
CP_NAME = "rtp_llm.models_py.modules.dsv4.cp"


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


def _fake_cp_info(real_lengths, chunk_lengths, cp_size, *, full_mask=False):
    chunk_length = sum(chunk_lengths)
    padded = cp_size * chunk_length
    if full_mask:
        mask = torch.ones(padded, dtype=torch.int32)
    else:
        mask = torch.zeros(padded, dtype=torch.int32)
        mask[: int(sum(real_lengths))] = 1
    restore = torch.randperm(padded, dtype=torch.int64)
    return types.SimpleNamespace(
        prefill_qkv_padding_mask=mask,
        prefill_qkv_restore_indice=restore,
        prefill_actual_input_lengths_cpu=torch.tensor(real_lengths, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor(chunk_lengths, dtype=torch.int64),
    )


class HostMirrorDerivation(unittest.TestCase):
    """``int(ctx.global_positions[0])`` is the authoritative first position."""

    def _check(self, cp_rank, prefix, real_lengths, chunk_lengths):
        cp_size = 4
        cp_info = _fake_cp_info(real_lengths, chunk_lengths, cp_size)
        device = torch.device("cpu")
        prefix_t = (
            torch.tensor([prefix] * len(real_lengths), dtype=torch.int64)
            if prefix
            else None
        )
        ctx = cp.build_cp_context_for_forward(
            cp_info,
            cp_size,
            cp_rank,
            sum(chunk_lengths),
            device,
            prefix_lengths=prefix_t,
        )
        # The authoritative read: ordinary tensor access on the derived
        # global positions (no first-position helper involved).
        first = int(ctx.global_positions[0])
        # Closed-form zigzag coordinate for request 0's first rank-local
        # token: front slice offset rank*(chunk//2), clamped to the real
        # length, shifted by the request prefix.
        pair0 = chunk_lengths[0] // 2
        expect_local = min(cp_rank * pair0, max(int(real_lengths[0]) - 1, 0))
        self.assertEqual(first, prefix + expect_local)
        # Retained framework host mirrors: the per-request real lengths from
        # ``prefill_actual_input_lengths_cpu`` and the prefix lengths derived
        # from the position-offset source.
        self.assertEqual(
            tuple(ctx.input_lengths_global_host),
            tuple(int(x) for x in real_lengths),
        )
        if prefix:
            self.assertEqual(
                tuple(ctx.prefix_lengths_host),
                tuple([prefix] * len(real_lengths)),
            )
        else:
            self.assertEqual(
                tuple(ctx.prefix_lengths_host),
                tuple([0] * len(real_lengths)),
            )
        return ctx

    def test_matrix_all_ranks_prefix_and_padding(self):
        for cp_rank in range(4):
            for prefix in (0, 100, 4096):
                self._check(cp_rank, prefix, [4096], [1024])  # full chunk
                self._check(cp_rank, prefix, [3000], [1024])  # padded chunk
                self._check(cp_rank, prefix, [97], [1024])  # clamp binds

    def test_two_request_chunks(self):
        for cp_rank in range(4):
            self._check(cp_rank, 0, [2000, 1500], [512, 512])
            self._check(cp_rank, 64, [2000, 1500], [512, 512])

    def test_missing_actual_lengths_uses_padding_mask_bound(self):
        for rank in range(4):
            for length in (1, 97, 3000):
                with self.subTest(rank=rank, length=length):
                    cp_info = _fake_cp_info([length], [1024], 4)
                    cp_info.prefill_actual_input_lengths_cpu = None
                    ctx = cp.build_cp_context(
                        cp_info, 4, rank, 1024, torch.device("cpu"), position_offset=100
                    )
                    self.assertEqual(
                        int(ctx.global_positions[0]), 100 + min(rank * 512, length - 1)
                    )
                    # Without actual lengths the mirrors decline and the
                    # real-token bound falls back to the padding-mask count.
                    self.assertIsNone(ctx.input_lengths_global_host)
                    self.assertEqual(
                        int(ctx.seq_len_full),
                        int((cp_info.prefill_qkv_padding_mask == 1).sum().item()),
                    )

    def test_first_position_nonempty_fallback_byte_identical(self):
        # The ordinary positions read ``int(pos.reshape(-1)[0])`` must return
        # exactly the legacy expression's value for every shape/content, and
        # the context-built positions must deliver the absolute coordinate.
        for data in ([0], [7], [10, 11, 12], [4095, 0, 3, 3, 3]):
            pos = torch.tensor(data, dtype=torch.int64)
            legacy = int(pos.reshape(-1)[0].item())
            self.assertEqual(int(pos.reshape(-1)[0]), legacy)
        # 2-D input collapses like the legacy reshape(-1) did.
        pos2d = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)
        self.assertEqual(int(pos2d.reshape(-1)[0]), 5)
        # A real CP context: the same ordinary read yields the zigzag
        # coordinate — prefix + min(rank*pair, len-1) — identically.
        for rank in range(4):
            with self.subTest(rank=rank):
                cp_info = _fake_cp_info([4096], [1024], 4)
                ctx = cp.build_cp_context_for_forward(
                    cp_info,
                    4,
                    rank,
                    1024,
                    torch.device("cpu"),
                    prefix_lengths=torch.tensor([64], dtype=torch.int64),
                )
                self.assertEqual(
                    int(ctx.global_positions.reshape(-1)[0].item()),
                    64 + min(rank * 512, 4095),
                )


def setUpModule():
    global cp
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    cp = _load_cp()


if __name__ == "__main__":
    unittest.main(verbosity=2)
