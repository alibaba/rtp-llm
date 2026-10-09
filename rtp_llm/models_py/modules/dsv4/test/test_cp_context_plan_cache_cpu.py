"""CPU tests for the context-parallel geometry plan cache.

Compare cache hits with uncached construction across ranks, chunk layouts and
prefixes. Cover read-only consumers, eviction, content changes and the kill
switch; the cache must never reuse content-dependent tensors."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import unittest
from runpy import run_path
from unittest.mock import patch

import torch

HERE = pathlib.Path(__file__).resolve().parent
DSV4 = HERE.parent
CP_PATH = DSV4 / "cp.py"
CP_SRC = CP_PATH.read_text()
CP_NAME = "rtp_llm.models_py.modules.dsv4.cp"
FLAG = "DSV4_CP_CONTEXT_PLAN_CACHE"


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


class _CpInfo:
    """Minimal stand-in for ``PyContextParallelParams`` (mirrors
    test_cp_context_build._CpInfo)."""

    def __init__(
        self,
        padding_mask,
        restore_indice,
        prefill_actual_input_lengths_cpu=None,
        prefill_cp_chunk_lengths=None,
    ):
        self.prefill_qkv_padding_mask = padding_mask
        self.prefill_qkv_restore_indice = restore_indice
        if prefill_actual_input_lengths_cpu is None:
            prefill_actual_input_lengths_cpu = torch.empty(0, dtype=torch.int32)
        self.prefill_actual_input_lengths_cpu = prefill_actual_input_lengths_cpu
        if prefill_cp_chunk_lengths is not None:
            self.prefill_cp_chunk_lengths = prefill_cp_chunk_lengths


def _zigzag_restore_multi(chunk_lengths, cp_size):
    """Reproduce ZigZagProcessor's restore mapping (verbatim oracle from
    test_cp_context_build)."""
    total_chunk = sum(chunk_lengths)
    restore = torch.empty(cp_size * total_chunk, dtype=torch.int32)
    chunk_offset = 0
    seq_offset = 0
    for chunk in chunk_lengths:
        pair = chunk // 2
        padded = chunk * cp_size
        for rank in range(cp_size):
            dst_base = rank * total_chunk + chunk_offset
            even = torch.arange(
                seq_offset + rank * pair,
                seq_offset + rank * pair + pair,
                dtype=torch.long,
            )
            odd = torch.arange(
                seq_offset + padded - (rank + 1) * pair,
                seq_offset + padded - rank * pair,
                dtype=torch.long,
            )
            restore[even] = torch.arange(dst_base, dst_base + pair, dtype=torch.int32)
            restore[odd] = torch.arange(
                dst_base + pair, dst_base + 2 * pair, dtype=torch.int32
            )
        chunk_offset += chunk
        seq_offset += padded
    return restore


def _padding_mask_multi(chunk_lengths, actual_lengths, cp_size):
    parts = []
    for chunk, actual in zip(chunk_lengths, actual_lengths):
        padded = chunk * cp_size
        part = torch.zeros(padded, dtype=torch.int32)
        part[:actual] = 1
        parts.append(part)
    return torch.cat(parts)


def _make_cp_info(chunk_lengths, actual_lengths, cp_size):
    return _CpInfo(
        _padding_mask_multi(chunk_lengths, actual_lengths, cp_size),
        _zigzag_restore_multi(chunk_lengths, cp_size),
        prefill_actual_input_lengths_cpu=torch.tensor(
            actual_lengths, dtype=torch.int32
        ),
        prefill_cp_chunk_lengths=torch.tensor(chunk_lengths, dtype=torch.int32),
    )


def _clear_cache():
    cp._CONTEXT_PLAN_CACHE.clear()


def _fields(ctx):
    """Every CPContext field, as a comparable structure."""
    out = {}
    for k, v in vars(ctx).items():
        if k == "cp_info":
            continue
        if isinstance(v, torch.Tensor):
            out[k] = (tuple(v.shape), v.dtype, v.clone())
        else:
            out[k] = v
    return out


def _assert_ctx_equal(tc, got, want):
    g, w = _fields(got), _fields(want)
    tc.assertEqual(set(g), set(w))
    for k in g:
        gv, wv = g[k], w[k]
        if isinstance(gv, tuple) and len(gv) == 3 and isinstance(gv[2], torch.Tensor):
            tc.assertEqual(gv[0], wv[0], k)
            tc.assertEqual(gv[1], wv[1], k)
            tc.assertTrue(torch.equal(gv[2], wv[2]), k)
        else:
            tc.assertEqual(gv, wv, k)


# Production matrix: (cp_rank, chunk_lengths, real_lengths).  chunk 1024/rank
# (4096 global / CP4); real lengths cover full chunks and partial tails.
GEOMETRIES = [
    (0, (1024,), (4096,)),
    (1, (1024,), (4096,)),
    (2, (1024,), (4096,)),
    (3, (1024,), (4096,)),
    (0, (1024,), (3000,)),  # partial tail
    (2, (1024,), (7,)),  # tiny partial
    (1, (768, 256), (3072, 1024)),  # B=2
    (3, (512, 384, 128), (2048, 100, 5)),  # B=3 with ragged partial
]
PREFIXES = (0, 4096, 2 * 4096, 7 * 4096)


class ContextPlanCache(unittest.TestCase):
    def setUp(self):
        _clear_cache()
        self._env = patch.dict(os.environ, {FLAG: "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        _clear_cache()

    def test_byte_identity_across_matrix_and_chunk_starts(self):
        for cp_rank, chunk_lengths, real_lengths in GEOMETRIES:
            for prefix in PREFIXES:
                with self.subTest(
                    rank=cp_rank, cl=chunk_lengths, rl=real_lengths, sp=prefix
                ):
                    info = _make_cp_info(chunk_lengths, real_lengths, 4)
                    chunk_length = sum(chunk_lengths)
                    with patch.dict(os.environ, {FLAG: "1"}):
                        got = cp.build_cp_context(
                            info,
                            4,
                            cp_rank,
                            chunk_length,
                            torch.device("cpu"),
                            position_offset=prefix,
                        )
                    with patch.dict(os.environ, {FLAG: "0"}):
                        want = cp.build_cp_context(
                            info,
                            4,
                            cp_rank,
                            chunk_length,
                            torch.device("cpu"),
                            position_offset=prefix,
                        )
                    _assert_ctx_equal(self, got, want)

    def test_warm_hit_shares_cached_objects_and_skips_ops(self):
        cp_rank, chunk_lengths, real_lengths = 1, (1024,), (4096,)
        chunk_length = 1024
        info = _make_cp_info(chunk_lengths, real_lengths, 4)
        ctx0 = cp.build_cp_context(
            info, 4, cp_rank, chunk_length, torch.device("cpu"), position_offset=0
        )
        # Warm hit: patch the op counters around the SECOND build.
        counts = {"arange": 0, "cat": 0, "zeros": 0, "full": 0, "stage": 0}
        real = {
            "arange": torch.arange,
            "cat": torch.cat,
            "zeros": torch.zeros,
            "full": torch.full,
        }
        stage_real = cp.stage_host_to_device

        def _count(name, fn):
            def wrapped(*a, **k):
                counts[name] += 1
                return fn(*a, **k)

            return wrapped

        with patch.object(torch, "arange", _count("arange", real["arange"])):
            with patch.object(torch, "cat", _count("cat", real["cat"])):
                with patch.object(torch, "zeros", _count("zeros", real["zeros"])):
                    with patch.object(torch, "full", _count("full", real["full"])):
                        with patch.object(
                            cp, "stage_host_to_device", _count("stage", stage_real)
                        ):
                            ctx1 = cp.build_cp_context(
                                info,
                                4,
                                cp_rank,
                                chunk_length,
                                torch.device("cpu"),
                                position_offset=4096,
                            )
        # Warm hit: no zigzag arange, no cat (zigzag + cu chain), no cu zeros,
        # no input_lengths staging.  The ONE surviving arange is the
        # content-dependent ``unpad_restore_is_prefix`` host check (CPU source
        # tensors here; per-chunk by design), and the ONE torch.full is the
        # scalar prefix fill (prefix varies per chunk).
        self.assertEqual(counts["arange"], 1)
        self.assertEqual(counts["cat"], 0)
        self.assertEqual(counts["zeros"], 0)
        self.assertEqual(counts["stage"], 0)
        self.assertEqual(counts["full"], 1)  # the scalar prefix fill (per chunk)
        # Cached tensors are the same OBJECT across warm chunks.
        self.assertIs(ctx1.relative_positions, ctx0.relative_positions)
        self.assertIs(ctx1.req_id_per_token, ctx0.req_id_per_token)
        self.assertIs(ctx1.input_lengths_global, ctx0.input_lengths_global)
        self.assertIs(ctx1.cu_seqlens_global, ctx0.cu_seqlens_global)
        # The prefix-dependent global_positions differ (chunk 1 prefix).
        self.assertFalse(torch.equal(ctx0.global_positions, ctx1.global_positions))
        self.assertTrue(
            torch.equal(ctx1.global_positions, ctx0.global_positions + 4096)
        )

    def test_cold_build_op_budget_positive(self):
        # Anti-vacuity for the warm-hit budget: a cold build DOES issue the
        # zigzag/cu ops the hit skips.
        info = _make_cp_info((1024,), (4096,), 4)
        counts = {"arange": 0, "cat": 0}
        real_arange, real_cat = torch.arange, torch.cat

        def _count(name, fn):
            def wrapped(*a, **k):
                counts[name] += 1
                return fn(*a, **k)

            return wrapped

        with patch.object(torch, "arange", _count("arange", real_arange)):
            with patch.object(torch, "cat", _count("cat", real_cat)):
                cp.build_cp_context(
                    info, 4, 0, 1024, torch.device("cpu"), position_offset=0
                )
        self.assertGreater(counts["arange"], 0)
        self.assertGreater(counts["cat"], 0)

    def test_partial_tail_gets_own_key(self):
        # Same rank/chunk_lengths but a partial real length: different key,
        # fresh build, correct (clamped) values.
        info_full = _make_cp_info((1024,), (4096,), 4)
        info_part = _make_cp_info((1024,), (3000,), 4)
        ctx_full = cp.build_cp_context(
            info_full, 4, 2, 1024, torch.device("cpu"), position_offset=0
        )
        ctx_part = cp.build_cp_context(
            info_part, 4, 2, 1024, torch.device("cpu"), position_offset=0
        )
        self.assertIsNot(ctx_full.relative_positions, ctx_part.relative_positions)
        self.assertEqual(ctx_part.seq_len_full, 3000)
        # And the partial build equals the flag-off build.
        with patch.dict(os.environ, {FLAG: "0"}):
            want = cp.build_cp_context(
                info_part, 4, 2, 1024, torch.device("cpu"), position_offset=0
            )
        _assert_ctx_equal(self, ctx_part, want)
        # Two distinct cache entries now exist.
        self.assertEqual(len(cp._CONTEXT_PLAN_CACHE), 2)

    def test_kill_switch_flag_off(self):
        with patch.dict(os.environ, {FLAG: "0"}):
            info = _make_cp_info((1024,), (4096,), 4)
            a = cp.build_cp_context(
                info, 4, 0, 1024, torch.device("cpu"), position_offset=0
            )
            b = cp.build_cp_context(
                info, 4, 0, 1024, torch.device("cpu"), position_offset=0
            )
            self.assertIsNot(a.relative_positions, b.relative_positions)
            self.assertTrue(torch.equal(a.relative_positions, b.relative_positions))
            self.assertEqual(len(cp._CONTEXT_PLAN_CACHE), 0)

    def test_flag_parse_fail_closed(self):
        info = _make_cp_info((1024,), (4096,), 4)
        with patch.dict(os.environ, {FLAG: "yes"}):
            with self.assertRaises(ValueError):
                cp.build_cp_context(
                    info, 4, 0, 1024, torch.device("cpu"), position_offset=0
                )

    def test_lru_bound(self):
        # Fill beyond the cap with distinct geometries; the oldest evict.
        cap = cp._CONTEXT_PLAN_CACHE_MAX_ENTRIES
        keys = []
        for i in range(cap + 4):
            info = _make_cp_info((1024,), (4096 + i,), 4)  # distinct real lengths
            cp.build_cp_context(
                info, 4, 0, 1024, torch.device("cpu"), position_offset=0
            )
            keys.append((4096 + i,))
        self.assertEqual(len(cp._CONTEXT_PLAN_CACHE), cap)
        cached_lengths = {k[5] for k in cp._CONTEXT_PLAN_CACHE}
        self.assertNotIn((4096,), cached_lengths)  # oldest evicted
        self.assertIn((4096 + cap + 3,), cached_lengths)  # newest present

    def test_mutation_guard_consumers_read_only(self):
        # Drive every consumer op of the cached tensors, then re-check the
        # cached bytes are untouched.
        info = _make_cp_info((1024,), (4096,), 4)
        ctx = cp.build_cp_context(
            info, 4, 1, 1024, torch.device("cpu"), position_offset=0
        )
        (key,) = list(cp._CONTEXT_PLAN_CACHE.keys())
        entry = cp._CONTEXT_PLAN_CACHE[key]
        snapshots = [t.clone() for t in entry]
        # Every downstream consumer op (all read-only):
        _ = info.prefill_qkv_padding_mask[entry[3]] == 1  # local_is_real gather
        _ = torch.zeros(2, dtype=torch.long).gather(0, entry[6].clamp_max(1)[:2])
        _ = entry[5].to(torch.long)  # req_id cast
        _ = entry[0].to(torch.int32)  # input_lengths cast
        _ = torch.cumsum(entry[0].to(torch.int64), dim=0)  # cu recompute
        _ = entry[1].reshape(-1)
        _ = entry[2].to(torch.long)
        _ = torch.arange(4).index_select(0, entry[3].clamp_max(3)[:4])
        for t, snap in zip(entry, snapshots):
            self.assertTrue(torch.equal(t, snap))

    def test_poisoned_cache_changes_output(self):
        # Anti-vacuity: the cache genuinely drives downstream values — a
        # corrupted entry must change the next chunk's context.
        info = _make_cp_info((1024,), (4096,), 4)
        good = cp.build_cp_context(
            info, 4, 1, 1024, torch.device("cpu"), position_offset=0
        )
        (key,) = list(cp._CONTEXT_PLAN_CACHE.keys())
        entry = list(cp._CONTEXT_PLAN_CACHE[key])
        poisoned = entry[3].clone()
        poisoned[0] = (int(poisoned[0]) + 1) % 4096
        entry[3] = poisoned
        cp._CONTEXT_PLAN_CACHE[key] = tuple(entry)
        bad = cp.build_cp_context(
            info, 4, 1, 1024, torch.device("cpu"), position_offset=4096
        )
        self.assertFalse(torch.equal(good.relative_positions, bad.relative_positions))

    def test_mask_content_not_in_key_but_content_rebuilt(self):
        # Same geometry key, DIFFERENT mask content: the plan cache still hits
        # (mask content is deliberately not in the key) but the
        # content-derived ``local_is_real`` is rebuilt from the CURRENT chunk's
        # mask — prove both.
        info_a = _make_cp_info((1024,), (4096,), 4)  # all-ones mask
        mask_b = torch.ones(4096, dtype=torch.int32)
        # Rank 1 owns padded positions [512,1024) + [3072,3584); zero a range
        # inside its ownership so local_is_real actually flips.
        mask_b[522:542] = 0  # inconsistent with real lengths — synthetic
        info_b = _CpInfo(
            mask_b,
            _zigzag_restore_multi((1024,), 4),
            prefill_actual_input_lengths_cpu=torch.tensor([4096], dtype=torch.int32),
            prefill_cp_chunk_lengths=torch.tensor([1024], dtype=torch.int32),
        )
        ctx_a = cp.build_cp_context(
            info_a, 4, 1, 1024, torch.device("cpu"), position_offset=0
        )
        ctx_b = cp.build_cp_context(
            info_b, 4, 1, 1024, torch.device("cpu"), position_offset=0
        )
        # Plan cache hit (same geometry): shared relative_positions object.
        self.assertIs(ctx_a.relative_positions, ctx_b.relative_positions)
        # But local_is_real differs (rebuilt from the current mask).
        self.assertTrue(ctx_a.local_is_real.all())
        self.assertFalse(ctx_b.local_is_real.all())
        # And it matches a flag-off build on the same mask.
        with patch.dict(os.environ, {FLAG: "0"}):
            want_b = cp.build_cp_context(
                info_b, 4, 1, 1024, torch.device("cpu"), position_offset=0
            )
        self.assertTrue(torch.equal(ctx_b.local_is_real, want_b.local_is_real))
        self.assertTrue(torch.equal(ctx_b.unpad_restore, want_b.unpad_restore))

    def test_no_actual_lengths_never_caches(self):
        # Without prefill_actual_input_lengths_cpu the cache is not consulted
        # (real lengths would need a device readback to key on).
        info = _CpInfo(
            torch.ones(4096, dtype=torch.int32),
            _zigzag_restore_multi((1024,), 4),
        )
        cp.build_cp_context(info, 4, 0, 1024, torch.device("cpu"), position_offset=0)
        cp.build_cp_context(info, 4, 0, 1024, torch.device("cpu"), position_offset=4096)
        self.assertEqual(len(cp._CONTEXT_PLAN_CACHE), 0)


class SourceInvariants(unittest.TestCase):
    def test_flag_and_cache_structure(self):
        self.assertIn('_CONTEXT_PLAN_CACHE_FLAG = "DSV4_CP_CONTEXT_PLAN_CACHE"', CP_SRC)
        self.assertIn('os.environ.get(_CONTEXT_PLAN_CACHE_FLAG, "1")', CP_SRC)
        self.assertIn('raise ValueError(f"{_CONTEXT_PLAN_CACHE_FLAG}', CP_SRC)
        # The zigzag loop only runs on the miss path; the store is gated on
        # (flag on + lengths present) and miss.
        self.assertIn("    if plan is not None:\n", CP_SRC)
        self.assertIn("    if plan_key is not None:\n", CP_SRC)
        self.assertIn(
            "        if plan is None:\n            _context_plan_store(", CP_SRC
        )


def setUpModule():
    global cp
    run_path(str(HERE / "cpu_test_utils.py"))["isolate_cpu_test_module"]()

    cp = _load_cp()


if __name__ == "__main__":
    unittest.main(verbosity=2)
