"""CPU tests for #19 CED (DSV41_CED): flag parsing, chunked zigzag geometry
(owner/relative/restore/tail-source maps must be exact bijections), the halo
arithmetic behind the 659-exact-row guarantee, the tail CPContext builder,
the full-geometry indexer-projection trick, and the gate predicates. The
GPU/collective path (compact/restore all_gathers, meta rebuild) is proven by
the node parity run's [CEDPLAN]/[CEDTAIL] announces + the sha gate."""

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _load_ced():
    """Load _v41_ced.py with hermetic stubs for every package import it
    performs (top-level cp/collective_torch + the deferred bounded_replay /
    forward_ep_plan gates), so the suite never touches the real rtp_llm
    package import chain."""
    import sys
    import typing
    from dataclasses import dataclass, field

    @dataclass
    class CPContext:
        cp_size: int
        cp_rank: int
        chunk_length: int
        padded_seq_len: int
        seq_len_full: int
        relative_positions: torch.Tensor
        prefix_length: int
        global_positions: torch.Tensor
        local_is_real: torch.Tensor
        unpad_restore: torch.Tensor
        seq_len_total: int
        cp_info: object
        req_id_per_token: typing.Optional[torch.Tensor] = None
        prefix_lengths: typing.Optional[torch.Tensor] = None
        input_lengths_global: typing.Optional[torch.Tensor] = None
        cu_seqlens_global: typing.Optional[torch.Tensor] = None
        unpad_restore_is_prefix: bool = False
        chunk_lengths_per_req: typing.Optional[typing.Tuple[int, ...]] = None
        kv_cache_sharded: bool = False
        input_lengths_global_host: typing.Optional[typing.Tuple[int, ...]] = None
        prefix_lengths_host: typing.Optional[typing.Tuple[int, ...]] = None
        compact_geometry_verified: bool = False
        _sync_varlen_scratch: typing.Optional[dict] = field(
            default=None, init=False, repr=False, compare=False
        )

    names = [
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.distributed",
        "rtp_llm.models_py.distributed.collective_torch",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
        "rtp_llm.models_py.modules.dsv4.cp",
        "rtp_llm.models_py.modules.dsv4.bounded_replay",
        "rtp_llm.models_py.modules.dsv4.moe",
        "rtp_llm.models_py.modules.dsv4.moe.forward_ep_plan",
    ]
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules["rtp_llm.models_py.modules.dsv4.cp"] = SimpleNamespace(
        CPContext=CPContext
    )
    sys.modules["rtp_llm.models_py.distributed.collective_torch"] = SimpleNamespace(
        Group=SimpleNamespace(TP="tp"), _get_group=lambda name: None
    )
    sys.modules["rtp_llm.models_py.distributed"] = SimpleNamespace(
        collective_torch=sys.modules["rtp_llm.models_py.distributed.collective_torch"]
    )
    sys.modules["rtp_llm.models_py.modules.dsv4.bounded_replay"] = SimpleNamespace(
        enabled=lambda: False, bounded_replay_enabled=lambda: False
    )
    sys.modules["rtp_llm.models_py.modules.dsv4.moe.forward_ep_plan"] = SimpleNamespace(
        enabled=lambda: False
    )
    for name in names:
        sys.modules.setdefault(name, SimpleNamespace())
    try:
        spec = importlib.util.spec_from_file_location(
            "_r19_ced", ROOT / "prefill" / "_v41_ced.py"
        )
        module = importlib.util.module_from_spec(spec)
        # dataclasses._is_type resolves bare annotations through
        # sys.modules[cls.__module__]; the dynamic module must be registered.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_r19_ced", None)
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return module, CPContext


ced, CPContext = _load_ced()

ROWS = 1024  # bench chunk geometry: 4096-token chunks over CP4
N = 4096
HALF = ROWS // 2
TAIL = ced.TAIL_TOKENS  # 3072
CP = 4


def make_ctx(rank, prefix=28672, total=32768, rows=ROWS, n=N, restore=None):
    relative = torch.tensor(ced.zigzag_relative(rank, rows), dtype=torch.int64)
    if restore is None:
        restore = torch.tensor(ced.zigzag_unpad_restore(rows, n), dtype=torch.int64)
    return CPContext(
        cp_size=CP,
        cp_rank=rank,
        chunk_length=rows,
        padded_seq_len=n,
        seq_len_full=n,
        relative_positions=relative,
        prefix_length=prefix,
        global_positions=(relative + prefix).contiguous(),
        local_is_real=torch.ones(rows, dtype=torch.bool),
        unpad_restore=restore,
        seq_len_total=total,
        cp_info=SimpleNamespace(),
        req_id_per_token=torch.zeros(rows, dtype=torch.int32),
        prefix_lengths=torch.tensor([prefix], dtype=torch.int64),
        input_lengths_global=torch.tensor([n], dtype=torch.int32),
        cu_seqlens_global=torch.tensor([0, n], dtype=torch.int32),
        unpad_restore_is_prefix=False,
        chunk_lengths_per_req=(rows,),
        kv_cache_sharded=True,
        input_lengths_global_host=(n,),
        prefix_lengths_host=(prefix,),
    )


class FlagTest(unittest.TestCase):
    def test_mode_parsing(self):
        for value, expect in (("0", 0), ("1", 1)):
            with mock.patch.dict(os.environ, {"DSV41_CED": value}):
                self.assertEqual(ced.ced_mode(), expect)
        clean = dict(os.environ)
        clean.pop("DSV41_CED", None)
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertEqual(ced.ced_mode(), 0)
        for bad in ("2", "x", ""):
            with mock.patch.dict(os.environ, {"DSV41_CED": bad}):
                with self.assertRaises(ValueError):
                    ced.ced_mode()

    def test_halo_arithmetic(self):
        self.assertEqual(ced.EXACT_TAIL_ROWS, 659)
        self.assertEqual(ced.EXACT_TAIL_ROWS, TAIL - 19 * 127)
        self.assertGreaterEqual(ced.EXACT_TAIL_ROWS, 128)
        self.assertEqual(ced._LOCAL_ROWS, 768)


class ZigzagTest(unittest.TestCase):
    def test_owner_local_bijection(self):
        seen = set()
        for p in range(N):
            owner, local = ced.zigzag_owner_local(p, HALF)
            self.assertIn(owner, range(CP))
            self.assertIn(local, range(ROWS))
            self.assertNotIn((owner, local), seen)
            seen.add((owner, local))
        self.assertEqual(len(seen), N)

    def test_relative_matches_owner_local(self):
        for rank in range(CP):
            relative = ced.zigzag_relative(rank, ROWS)
            self.assertEqual(len(relative), ROWS)
            for local, p in enumerate(relative):
                self.assertEqual(ced.zigzag_owner_local(p, HALF), (rank, local))

    def test_unpad_restore_is_gather_inverse(self):
        restore = ced.zigzag_unpad_restore(ROWS, N)
        self.assertEqual(sorted(restore), list(range(N)))
        for rank in range(CP):
            for local, p in enumerate(ced.zigzag_relative(rank, ROWS)):
                self.assertEqual(restore[p], rank * ROWS + local)

    def test_known_owners(self):
        # chunk pairs 2..7 (the tail) -> ranks 2,3,3,2,1,0 (design doc map)
        expected = {2: 2, 3: 3, 4: 3, 5: 2, 6: 1, 7: 0}
        for pair, rank in expected.items():
            owner, _ = ced.zigzag_owner_local(pair * HALF, HALF)
            self.assertEqual(owner, rank)

    def test_tail_source_map(self):
        start = N - TAIL
        all_src, per_rank_dst = [], {}
        for rank in range(CP):
            dst, src = ced.tail_source_map(rank, ROWS, N, TAIL)
            self.assertEqual(len(dst), len(src))
            per_rank_dst[rank] = dst
            all_src.extend(src)
            for i, s in zip(dst, src):
                pair = rank if i < HALF else 2 * CP - 1 - rank
                p = pair * HALF + i % HALF
                self.assertEqual(p, start + s)
        self.assertEqual(sorted(all_src), list(range(TAIL)))  # exact cover
        # rank0: only its back half (pair 7) is in the tail
        self.assertEqual(len(per_rank_dst[0]), HALF)
        self.assertTrue(all(i >= HALF for i in per_rank_dst[0]))
        # ranks 2 and 3 own tail pairs entirely: full 1024 rows each
        self.assertEqual(len(per_rank_dst[2]), ROWS)
        self.assertEqual(len(per_rank_dst[3]), ROWS)
        self.assertEqual(len(per_rank_dst[1]), HALF)


class TailContextTest(unittest.TestCase):
    def test_tail_context_fields(self):
        original = make_ctx(1, prefix=28672)
        start = 28672 + N - TAIL  # 29696
        ctx = ced._tail_context(original, start)
        self.assertEqual(ctx.chunk_length, 768)
        self.assertEqual(ctx.padded_seq_len, TAIL)
        self.assertEqual(ctx.seq_len_full, TAIL)
        self.assertEqual(ctx.prefix_length, start)
        self.assertEqual(ctx.seq_len_total, 32768)
        self.assertEqual(ctx.input_lengths_global_host, (TAIL,))
        self.assertEqual(ctx.prefix_lengths_host, (start,))
        self.assertEqual(ctx.chunk_lengths_per_req, (768,))
        self.assertTrue(ctx.kv_cache_sharded)
        self.assertFalse(ctx.compact_geometry_verified)
        self.assertFalse(ctx.unpad_restore_is_prefix)
        self.assertTrue(
            torch.equal(ctx.global_positions, ctx.relative_positions + start)
        )
        self.assertTrue(ctx.local_is_real.all())
        # tail restore map is a bijection into the 4x768 gather flat
        self.assertEqual(sorted(ctx.unpad_restore.tolist()), list(range(TAIL)))
        # relative positions follow the tail zigzag (front 384 + back 384)
        rel = ctx.relative_positions.tolist()
        self.assertEqual(rel[:384], list(range(1 * 384, 2 * 384)))
        self.assertEqual(rel[384:], list(range(TAIL - 2 * 384, TAIL - 1 * 384)))
        info = ctx.cp_info
        self.assertEqual(info.prefill_cp_chunk_lengths.tolist(), [768])
        self.assertEqual(info.prefill_actual_input_lengths_cpu.tolist(), [TAIL])
        self.assertEqual(info.prefill_prefix_lengths_cpu.tolist(), [start])
        self.assertTrue((info.prefill_qkv_padding_mask == 1).all())
        self.assertEqual(info.prefill_qkv_restore_indice.dtype, torch.int32)
        # replace() reset the init=False scratch
        self.assertIsNone(ctx._sync_varlen_scratch)

    def test_geometry_verified_accepts_zigzag(self):
        for rank in range(CP):
            self.assertTrue(ced._geometry_verified(make_ctx(rank)))

    def test_geometry_verified_rejects_foreign_layout(self):
        bogus = torch.arange(N, dtype=torch.int64)  # identity != zigzag
        self.assertFalse(ced._geometry_verified(make_ctx(0, restore=bogus)))


class IndexerProjectionTest(unittest.TestCase):
    def test_row_independent_full_geometry(self):
        torch.manual_seed(7)
        plan = ced.CEDChunkPlan(
            mode=1,
            original_context=make_ctx(0),
            context=make_ctx(0),
            cu_seqlens=torch.tensor([0, 768], dtype=torch.int32),
        )
        weights = torch.randn(16, 32, dtype=torch.float64)
        full = torch.randn(ROWS, 32, dtype=torch.float64)
        compact = full[:768]
        direct = torch.nn.functional.linear(full, weights)[:768]
        trick = plan.indexer_projection(compact, weights)
        self.assertTrue(torch.equal(direct, trick))
        # padding rows never leak into the selected outputs
        self.assertEqual(trick.shape, (768, 16))


class GateTest(unittest.TestCase):
    def _v4(self, **overrides):
        layers = []
        for local in range(20):
            index = 20 + local
            attn = SimpleNamespace(
                layer_id=index,
                kv_source_layer_id=20,
                index_source_layer_id=20 + (index - 20) // 4 * 4,
                compress_ratio=1,
                window_size=128,
                is_kv_source=(index == 20),
                is_index_source=(index % 4 == 0),
            )
            layers.append(SimpleNamespace(attn=attn, engram=None))
        base = dict(
            args=SimpleNamespace(v41_config={}),
            layers=tuple(layers),
            pp_global_layer_ids=tuple(range(20, 40)),
            capture_aux_hidden_layer_ids=(),
            fp8_kv_cache=True,
            norm=object(),
            _mtp_hidden_buffer=None,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_supported_model(self):
        self.assertTrue(ced._supported_model(self._v4()))
        self.assertFalse(ced._supported_model(self._v4(norm=None)))
        self.assertFalse(ced._supported_model(self._v4(_mtp_hidden_buffer=object())))
        self.assertFalse(
            ced._supported_model(self._v4(capture_aux_hidden_layer_ids=(37,)))
        )
        self.assertFalse(ced._supported_model(self._v4(fp8_kv_cache=False)))
        bad = self._v4()
        bad.layers[5].attn.compress_ratio = 2
        self.assertFalse(ced._supported_model(bad))
        bad2 = self._v4()
        bad2.layers[1].engram = object()
        self.assertFalse(ced._supported_model(bad2))
        bad3 = self._v4(pp_global_layer_ids=tuple(range(0, 20)))
        self.assertFalse(ced._supported_model(bad3))

    def test_env_gates(self):
        clean = {
            "REUSE_CACHE": "0",
            "ENABLE_MEMORY_CACHE": "0",
            "ENABLE_GPU_PREFIX_TREE": "0",
        }
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertTrue(ced._env_gates_ok())
        for key in ("ENABLE_MEMORY_CACHE", "ENABLE_GPU_PREFIX_TREE"):
            with mock.patch.dict(os.environ, {**clean, key: "1"}, clear=True):
                self.assertFalse(ced._env_gates_ok())
        with mock.patch.dict(os.environ, {**clean, "REUSE_CACHE": "1"}, clear=True):
            self.assertFalse(ced._env_gates_ok())

    def test_plan_requires_initialized_dist(self):
        # torch.distributed is not initialized in CPU tests -> rank-symmetric
        # precondition returns None before any collective.
        with mock.patch.dict(os.environ, {"DSV41_CED": "1"}):
            attn_inputs = SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                cache_store_inputs=None,
            )
            plan = ced.plan_ced_chunk(
                self._v4(),
                make_ctx(0),
                attn_inputs,
                capture_aux=False,
                cache_store_active=False,
            )
        self.assertIsNone(plan)


if __name__ == "__main__":
    unittest.main(verbosity=2)
