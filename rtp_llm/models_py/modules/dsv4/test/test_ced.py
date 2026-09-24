"""CPU/Gloo contracts for checkpoint-aware CED and its dependency cone.

The toy decoder below is an independent numerical oracle: global KV is frozen
encoder state, while only local SWA propagates dependencies between tokens.
It also demonstrates why replaying just one SWA window is not exact.
"""

import ast
import importlib
import os
import sys
import tempfile
import time
import types
import unittest
import weakref
from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


def _load_helpers():
    """Import complete helpers, replacing only the native collective boundary."""
    root = Path(__file__).resolve().parents[5]
    packages = {}
    for name in (
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.distributed",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
        "rtp_llm.models_py.modules.dsv4.prefill",
        "rtp_llm.models_py.modules.dsv4.fp8",
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        packages[name] = module
    collective = types.ModuleType("rtp_llm.models_py.distributed.collective_torch")
    collective.Group = types.SimpleNamespace(TP="TP")
    collective._get_group = mock.Mock(return_value=object())
    collective.broadcast = mock.Mock()
    collective.all_gather = mock.Mock(side_effect=AssertionError("unexpected gather"))
    packages[collective.__name__] = collective
    # No native ops, CUDA packages, or prefill.forward import is needed.
    with mock.patch.dict(sys.modules, packages):
        cp = importlib.import_module("rtp_llm.models_py.modules.dsv4.cp")
        ced = importlib.import_module("rtp_llm.models_py.modules.dsv4.prefill.ced")
        bounded = importlib.import_module(
            "rtp_llm.models_py.modules.dsv4.bounded_replay"
        )
        index_plan = importlib.import_module(
            "rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_index_plan"
        )
        kv_workspace = importlib.import_module(
            "rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_kv_workspace"
        )
        prefill_meta = importlib.import_module(
            "rtp_llm.models_py.modules.dsv4.fp8.prefill_meta"
        )
    return cp, ced, collective, bounded, index_plan, kv_workspace, prefill_meta


(
    _CP,
    _CED,
    _COLLECTIVE,
    _BOUNDED,
    _INDEX_PLAN,
    _KV_WORKSPACE,
    _PREFILL_META,
) = _load_helpers()


def _swa_owner(bounded):
    """Run complete production methods, isolating unrelated GPU boundaries.

    Importing attention_v41 needs native/CUDA packages unavailable to this CPU
    suite. Compile its unchanged methods, not a copied SWA planner algorithm.
    """
    path = Path(_CED.__file__).parent.parent / "fp8/attention_v41.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AttentionV41FP8"
    )
    names = {
        "_host_prefill_lengths",
        "_host_prefill_prefixes",
        "_swa_prefill_workspace",
        "_prefill_chunk_meta",
        "_forward_prefill",
        "_prefill_produce",
        "_prefill_query",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in methods} == names
    namespace = {
        "torch": torch,
        "F": F,
        "_use_small_cp_x_gather": lambda ctx: False,
        "prefill_index_plan": _INDEX_PLAN,
        "prefill_kv_workspace": _KV_WORKSPACE,
    }
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    owner = types.SimpleNamespace(
        swa_bounded_replay=bounded,
        window_size=128,
        compress_ratio=1,
        layer_id=21,
        kv_source_layer_id=20,
        is_kv_source=False,
        _kv_cache=object(),
        _shared_attention={},
        _begin_forward=mock.Mock(),
        _source_pool=mock.Mock(return_value=object()),
        _global_region=lambda: 2,
        _swa_prefill_concat=mock.Mock(
            side_effect=AssertionError("unexpected cached prefix read")
        ),
        _prefill_write_swa_fp8_paged=mock.Mock(),
        _prefill_sparse_attention=mock.Mock(),
    )
    for name in names:
        setattr(owner, name, types.MethodType(namespace[name], owner))
    return owner


def _cp4_layout(length):
    """Independent input oracle: assign paired front/back eighths to ranks."""
    padded = ((length + 7) // 8) * 8
    eighths = torch.arange(padded).chunk(8)
    ranks = [torch.cat((eighths[r], eighths[7 - r])) for r in range(4)]
    restore = torch.argsort(torch.cat(ranks))
    info = types.SimpleNamespace(
        prefill_qkv_padding_mask=(torch.arange(padded) < length).to(torch.int32),
        prefill_qkv_restore_indice=restore.to(torch.int32),
        prefill_actual_input_lengths_cpu=torch.tensor([length], dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor([padded // 4], dtype=torch.int32),
    )
    return ranks, info


def _model():
    layers = [
        types.SimpleNamespace(
            attn=types.SimpleNamespace(
                layer_id=i,
                kv_source_layer_id=20,
                index_source_layer_id=20 + ((i - 20) // 4) * 4,
                compress_ratio=1,
                window_size=128,
                is_kv_source=i == 20,
                is_index_source=i % 4 == 0,
            ),
            engram=None,
            ffn_hc=types.SimpleNamespace(pre_mix_out=None),
        )
        for i in range(40)
    ]
    return types.SimpleNamespace(
        args=types.SimpleNamespace(
            v41_config=object(),
            n_layers=40,
            ep_size=4,
            window_size=128,
            dim=5120,
            hc_mult=4,
            n_hash_layers=0,
        ),
        layers=layers,
        fp8_kv_cache=True,
        capture_aux_hidden_layer_ids=(37, 38, 39),
        _mtp_hidden_buffer=None,
        _note_aux_hidden_rows=mock.Mock(),
    )


def _context(length, rank, prefix=0, *, device="cpu", permuted=False):
    positions, info = _cp4_layout(length)
    if permuted:
        # Still a complete engine inverse map, but not the usual rank pairing.
        positions = [positions[(r + 1) % 4].flip(0) for r in range(4)]
        info.prefill_qkv_restore_indice = torch.argsort(torch.cat(positions)).int()
    ctx = _CP.build_cp_context(
        info,
        cp_size=4,
        cp_rank=rank,
        chunk_length=len(positions[rank]),
        device=torch.device("cpu"),
        position_offset=prefix,
        kv_cache_sharded=True,
    )
    if permuted:
        ctx = replace(
            ctx,
            relative_positions=positions[rank],
            global_positions=positions[rank] + prefix,
            local_is_real=positions[rank] < length,
        )
    # Move only tensor storage. Avoid importing unrelated CUDA metadata kernels
    # in the standalone real-NCCL test, which uses this same actual CPContext.
    if torch.device(device).type != "cpu":
        ctx = replace(
            ctx,
            **{
                f.name: getattr(ctx, f.name).to(device)
                for f in fields(ctx)
                if f.init and isinstance(getattr(ctx, f.name), torch.Tensor)
            },
        )
    return ctx, positions


def _plan(
    length,
    rank,
    selected,
    prefix=0,
    *,
    group=None,
    device="cpu",
    permuted=False,
    candidate_publication=False,
):
    original, positions = _context(
        length, rank, prefix, device=device, permuted=permuted
    )
    context, exchange, groups = _CED._query_layout(
        original, selected, group, keep_candidate_rows=candidate_publication
    )
    if candidate_publication:
        context = replace(context, swa_replay_start=length - 128)
    plan = _CED.CEDPlan(
        original,
        context,
        torch.tensor([0, context.chunk_length], dtype=torch.int32, device=device),
        exchange,
        groups,
        (37, 38, 39),
    )
    return plan, positions


def _cache_fixture(length, prefix=0, *, ring=136, dtype=torch.uint8, region=None):
    region = _CED.SWA_KV if region is None else region
    columns = (prefix + length + 511) // 512 + 4
    ids = torch.zeros((2, 1, columns), dtype=torch.int32)
    # Group 0 is deliberately dense but belongs to another pool (encoder SWA
    # for bounded mode). Only the requested region's host IDs may be used.
    ids[0] = 99
    for column in range(31, columns, 32):
        ids[1, 0, column] = 1000 + column
    ids[1, 0, -1] = 919  # allocation reserve, outside the request
    element_size = torch.empty((), dtype=dtype).element_size()
    # alignDsv41Fp8KvBlockBytes pads the full native block to 512 bytes
    # before CP4 byte slicing; padding is not another ring entry.
    full_stride_bytes = ((ring * 528 + 511) // 512) * 512
    base = torch.empty((2, full_stride_bytes // (4 * element_size)), dtype=dtype)
    cache = types.SimpleNamespace(
        group_region_names=(
            int(_CED.SWA_KV) + 17 if region == _CED.SWA_KV else _CED.SWA_KV,
            region,
        ),
        group_seq_size_per_block=(512, 512),
        get_layer_cache=mock.Mock(
            return_value=types.SimpleNamespace(kv_cache_base=base)
        ),
    )
    attn = types.SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        is_cuda_graph=False,
        kv_cache_block_id_host=ids,
    )
    return attn, cache


def _compact_oracle(full, selected, rank):
    """Select by global row identity, then independently pair canonical eighths."""
    positions, _ = _cp4_layout(selected.numel())
    padded = full.new_zeros((sum(len(p) for p in positions), *full.shape[1:]))
    padded[: len(selected)] = full.index_select(0, selected.to(full.device))
    return padded.index_select(0, positions[rank].to(full.device))


class _SingleThreadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)


class CedCheckpointTest(_SingleThreadTest):
    def assert_positions(self, prefix, length, ids, span=512, ring=136):
        actual = _CED.checkpoint_positions(prefix, length, ids, span, ring)
        # Scalar set oracle; deliberately no interval-merging implementation.
        required = set()
        endpoints = {prefix + length}
        for column, block in enumerate(ids):
            if (
                block > 0
                and column * span < prefix + length
                and (column + 1) * span > prefix
            ):
                endpoints.add(min((column + 1) * span, prefix + length))
        for stop in endpoints:
            for absolute in range(max(prefix, stop - max(3072, 19 * 127 + ring)), stop):
                required.add(absolute - prefix)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(actual.tolist(), sorted(required))
        return actual

    def test_nonperiodic_ids_holes_reserve_and_cached_prefix(self):
        for prefix, length in ((0, 32769), (16387, 32769), (32767, 32768)):
            ids = [0] * 160
            for column in (0, 1, 11, 29, 31, 32, 44, 62, 80, 120, 159):
                ids[column] = 137 + column * 5
            ids[17] = -1
            with self.subTest(prefix=prefix, length=length):
                actual = self.assert_positions(prefix, length, ids)
                changed_ids = [99999 if i > 0 else i for i in ids]
                self.assertTrue(
                    torch.equal(
                        actual, self.assert_positions(prefix, length, changed_ids)
                    )
                )

    def test_overlap_adjacent_windows_partial_end_and_no_allocated_blocks(self):
        ids = [0] * 100
        for column in (15, 16, 22, 63, 64, 99):
            ids[column] = column + 1
        self.assert_positions(13, 32769, ids)
        tail = self.assert_positions(16384, 32769, [0] * 150)
        self.assertEqual(tail.tolist(), list(range(32769 - 3072, 32769)))
        self.assertEqual(self.assert_positions(8190, 3, [1] * 40).tolist(), [0, 1, 2])

    def test_actual_ring_stride_can_require_more_than_3072(self):
        rows = self.assert_positions(123, 32768, [], ring=800)
        self.assertEqual(len(rows), 19 * 127 + 800)
        for bad in (
            (-1, 4, [], 512, 136),
            (0, 0, [], 512, 136),
            (0, 4, [], 0, 136),
            (0, 4, [], 512, 0),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _CED.checkpoint_positions(*bad)

    def test_every_allocated_cache_write_has_all_decoder_ancestors(self):
        prefix, length, span, ring = 16387, 32769, 512, 136
        ids = [0] * 104
        for i in (31, 32, 51, 63, 64, 90, 95, 96, 103):
            ids[i] = 300 + i
        retained = set(self.assert_positions(prefix, length, ids).tolist())
        for column, block in enumerate(ids):
            stop = min((column + 1) * span, prefix + length)
            if block <= 0 or column * span >= stop or stop <= prefix:
                continue
            # Required physical ring rows, including speculative lookahead,
            # depend on fresh ancestors at every depth, not just the last logit.
            for row in range(max(prefix, column * span, stop - ring), stop):
                for depth in range(1, 20):
                    self.assertTrue(
                        set(
                            range(
                                max(prefix, row - depth * 127) - prefix,
                                row - prefix + 1,
                            )
                        ).issubset(retained)
                    )


class CedCreationTest(_SingleThreadTest):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.rank = 0
        for patch in (
            mock.patch.object(torch.Tensor, "is_cuda", new=property(lambda t: True)),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
            mock.patch.object(
                torch.cuda, "_lazy_init", side_effect=AssertionError("GPU call")
            ),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(torch.distributed, "get_world_size", return_value=4),
            mock.patch.object(
                torch.distributed, "get_rank", side_effect=lambda g: self.rank
            ),
            mock.patch.object(
                torch.distributed,
                "all_to_all_single",
                side_effect=AssertionError("create must not transport"),
            ),
        ):
            self.stack.enter_context(patch)

    def fixture(self, length=32769, prefix=0, **cache_kw):
        ctx, _ = _context(length, self.rank, prefix)
        attn, cache = _cache_fixture(length, prefix, **cache_kw)
        return _model(), ctx, attn, cache

    def test_full_output_gate_requires_explicit_false_for_both_flags(self):
        for logits in (False, True, None, 0):
            for hidden in (False, True, None, 0):
                with self.subTest(logits=logits, hidden=hidden):
                    inputs = types.SimpleNamespace(
                        need_all_logits=logits, need_all_hidden_states=hidden
                    )
                    self.assertEqual(
                        _CED.permits_ced(inputs), logits is False and hidden is False
                    )
        self.assertFalse(_CED.permits_ced(types.SimpleNamespace()))
        self.assertFalse(_CED.permits_ced(types.SimpleNamespace(need_all_logits=False)))
        self.assertFalse(
            _CED.permits_ced(types.SimpleNamespace(need_all_hidden_states=False))
        )

    def test_sparse_cache_plan_preserves_true_prefix_and_fresh_geometry(self):
        for length, prefix in ((32768, 0), (32769, 16387), (131072, 0)):
            for rank in range(4):
                self.rank = rank
                with self.subTest(length=length, prefix=prefix, rank=rank):
                    model, original, attn, cache = self.fixture(length, prefix)
                    original._full_prefill_positions_cache = (object(),) * 4
                    attn.cache_store_inputs = object()  # formal cache/PD is allowed
                    plan = _CED.CEDPlan.create(model, original, attn, cache)
                    self.assertIsNotNone(plan)
                    ctx = plan.context
                    self.assertIs(plan.original_context, original)
                    self.assertIs(ctx.cp_info, original.cp_info)
                    self.assertIsNone(ctx._full_prefill_positions_cache)
                    self.assertIsNotNone(original._full_prefill_positions_cache)
                    self.assertEqual(ctx.seq_len_full, length)
                    self.assertEqual(ctx.seq_len_total, prefix + length)
                    self.assertEqual(ctx.prefix_length, prefix)
                    self.assertEqual(ctx.prefix_lengths_host, (prefix,))
                    self.assertEqual(ctx.input_lengths_global_host, (length,))
                    self.assertEqual(ctx.cu_seqlens_global.tolist(), [0, length])
                    self.assertEqual(ctx.prefix_lengths.tolist(), [prefix])
                    self.assertEqual(ctx.chunk_lengths_per_req, (ctx.chunk_length,))
                    self.assertEqual(plan.cu_seqlens.tolist(), [0, ctx.chunk_length])
                    self.assertTrue(
                        torch.equal(
                            ctx.global_positions, ctx.relative_positions + prefix
                        )
                    )
                    selected = _CED.checkpoint_positions(
                        prefix,
                        length,
                        attn.kv_cache_block_id_host[1, 0].tolist(),
                        512,
                        136,
                    )
                    self.assertTrue(torch.equal(ctx.gather_restore_positions, selected))
                    cache.get_layer_cache.assert_called_once_with(21, _CED.SWA_KV)

    def test_native_tma_aligned_stride_preserves_136_ring_entries(self):
        for dtype in (torch.uint8, torch.bfloat16, torch.int32):
            with self.subTest(dtype=dtype):
                model, ctx, attn, cache = self.fixture(prefix=16387, dtype=dtype)
                base = cache.get_layer_cache.return_value.kv_cache_base
                self.assertEqual(136 * 528, 71808)
                self.assertEqual(base.shape[1] * base.element_size(), 18048)
                self.assertEqual(18048 * ctx.cp_size, 72192)
                self.assertEqual(72192 % 528, 384)
                with mock.patch.object(
                    _CED, "checkpoint_positions", wraps=_CED.checkpoint_positions
                ) as checkpoints:
                    plan = _CED.CEDPlan.create(model, ctx, attn, cache)
                self.assertIsNotNone(plan)
                # The 3072 minimum halo masks small ring-count errors in the
                # resulting positions, so inspect the actual planner argument.
                checkpoints.assert_called_once_with(
                    ctx.prefix_length,
                    ctx.seq_len_full,
                    attn.kv_cache_block_id_host[1, 0].tolist(),
                    512,
                    136,
                )

    def test_bounded_policy_default_off_preserves_exact_plan(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIs(_BOUNDED.enabled(), False)
        for value in ("0", "false", "off", "no", "1", "true", "yes", "on"):
            with self.subTest(value=value), mock.patch.dict(
                "os.environ", {"DSV41_SWA_BOUNDED_REPLAY": value}, clear=True
            ):
                self.assertEqual(
                    _BOUNDED.enabled(), value in ("1", "true", "yes", "on")
                )
        for rank in range(4):
            self.rank = rank
            args = self.fixture(prefix=16387)
            default = _CED.CEDPlan.create(*args)
            explicit = _CED.CEDPlan.create(*args, bounded_replay=False)
            self.assertIsNone(default.context.swa_replay_start)
            self.assertIsNone(explicit.context.swa_replay_start)
            self.assertTrue(
                torch.equal(
                    default.context.gather_restore_positions,
                    explicit.context.gather_restore_positions,
                )
            )
            self.assertGreater(default.context.gather_restore_positions.numel(), 128)
            for field in ("send_indices", "receive_positions"):
                self.assertTrue(
                    torch.equal(
                        getattr(default.exchange, field),
                        getattr(explicit.exchange, field),
                    )
                )
            self.assertEqual(default.exchange.send_sizes, explicit.exchange.send_sizes)
            self.assertEqual(
                default.exchange.receive_sizes, explicit.exchange.receive_sizes
            )

    def test_bounded_final128_uses_fresh_offset_and_true_absolute_positions(self):
        for length, prefix in (
            (129, 16387),
            (257, 16387),
            (32768, 0),
            (32769, 16387),
            (131072, 32768),
            (131073, 0),
        ):
            for rank in range(4):
                self.rank = rank
                with self.subTest(length=length, prefix=prefix, rank=rank):
                    model, original, attn, cache = self.fixture(
                        length, prefix, region=8
                    )
                    # Bounded mode deliberately does not preserve checkpoints;
                    # native policy must make decoder/draft pools non-reusable.
                    attn.kv_cache_block_id_host[1].fill_(9)
                    with mock.patch.object(
                        _CED,
                        "checkpoint_positions",
                        side_effect=AssertionError(
                            "checkpoint plan used in bounded mode"
                        ),
                    ):
                        plan = _CED.CEDPlan.create(
                            model, original, attn, cache, bounded_replay=True
                        )
                    self.assertIsNotNone(plan)
                    cache.get_layer_cache.assert_called_once_with(21, 8)
                    self.assertEqual(plan.context.swa_replay_start, length - 128)
                    self.assertEqual(
                        plan.exchange.candidate_rows_host,
                        tuple(
                            torch.where(
                                original.local_is_real
                                & (original.global_positions >= prefix + length - 128)
                            )[0].tolist()
                        ),
                    )
                    self.assertIsNone(original.swa_replay_start)
                    self.assertEqual(plan.context.chunk_length, 32)
                    self.assertEqual(plan.context.seq_len_full, length)
                    self.assertEqual(plan.context.prefix_length, prefix)
                    self.assertTrue(plan.context.local_is_real.all())
                    selected = torch.arange(length - 128, length)
                    self.assertTrue(
                        torch.equal(plan.context.gather_restore_positions, selected)
                    )
                    expected_rows, _ = _cp4_layout(128)
                    self.assertTrue(
                        torch.equal(
                            plan.context.global_positions,
                            expected_rows[rank] + prefix + length - 128,
                        )
                    )

    def test_bounded_minimum_does_not_inherit_exact_half_cost_gate(self):
        # A None plan means no CP compaction, not permission to read decoder
        # prefix cache. The separately tested bounded SWA consumer handles it.
        for length in (127, 128):
            with self.subTest(length=length):
                args = self.fixture(length=length, prefix=16387, region=8)
                self.assertIsNone(_CED.CEDPlan.create(*args, bounded_replay=True))
                self.assertIsNone(args[1].swa_replay_start)
        for length in (129, 255, 256, 257):
            with self.subTest(length=length):
                plan = _CED.CEDPlan.create(
                    *self.fixture(length=length, prefix=16387, region=8),
                    bounded_replay=True,
                )
                self.assertIsNotNone(plan)
                self.assertEqual(plan.context.chunk_length, 32)
                self.assertEqual(plan.context.swa_replay_start, length - 128)

    def test_bounded_requires_exactly_one_native_decoder_region(self):
        self.assertEqual(_CED.DECODER_SWA_KV, 8)
        for regions in ((24, 7), (8, 8)):
            with self.subTest(regions=regions):
                args = self.fixture()
                args[3].group_region_names = regions
                with self.assertRaisesRegex(ValueError, "native decoder SWA pool"):
                    _CED.CEDPlan.create(*args, bounded_replay=True)
                args[3].get_layer_cache.assert_not_called()

    def test_aligned_byte_stride_uses_element_size_and_excludes_padding(self):
        # Above the 3072 halo floor, one extra/missing ring entry changes
        # selected rows. int32 also detects ignoring base.element_size().
        model, ctx, attn, cache = self.fixture(ring=801, dtype=torch.int32)
        base = cache.get_layer_cache.return_value.kv_cache_base
        self.assertEqual(base.shape[1] * base.element_size() * ctx.cp_size, 423424)
        self.assertEqual(423424 % 528, 496)
        plan = _CED.CEDPlan.create(model, ctx, attn, cache)
        self.assertIsNotNone(plan)
        expected = _CED.checkpoint_positions(
            0, ctx.seq_len_full, attn.kv_cache_block_id_host[1, 0].tolist(), 512, 801
        )
        self.assertTrue(torch.equal(plan.context.gather_restore_positions, expected))

    def test_dense_short_and_unsupported_requests_fall_back(self):
        changes = (
            ("dense", lambda m, c, a, k: a.kv_cache_block_id_host[1].fill_(1)),
            ("unsharded", lambda m, c, a, k: setattr(c, "kv_cache_sharded", False)),
            (
                "multiple_requests",
                lambda m, c, a, k: setattr(
                    c, "input_lengths_global_host", (1, c.seq_len_full - 1)
                ),
            ),
            (
                "wrong_prefix_host",
                lambda m, c, a, k: setattr(c, "prefix_lengths_host", (99,)),
            ),
            ("decode", lambda m, c, a, k: setattr(a, "is_prefill", False)),
            ("verify", lambda m, c, a, k: setattr(a, "is_target_verify", True)),
            ("graph", lambda m, c, a, k: setattr(a, "is_cuda_graph", True)),
            (
                "missing_host_ids",
                lambda m, c, a, k: delattr(a, "kv_cache_block_id_host"),
            ),
            (
                "wrong_host_shape",
                lambda m, c, a, k: setattr(
                    a, "kv_cache_block_id_host", a.kv_cache_block_id_host[:, 0]
                ),
            ),
            (
                "two_swa_groups",
                lambda m, c, a, k: setattr(
                    k, "group_region_names", (_CED.SWA_KV, _CED.SWA_KV)
                ),
            ),
            (
                "zero_span",
                lambda m, c, a, k: setattr(k, "group_seq_size_per_block", (512, 0)),
            ),
            (
                "undersized_byte_stride",
                lambda m, c, a, k: setattr(
                    k.get_layer_cache.return_value,
                    "kv_cache_base",
                    # Full stride 524 is just below one 528-byte entry.
                    torch.empty(2, 131, dtype=torch.uint8),
                ),
            ),
            (
                "empty_cache",
                lambda m, c, a, k: setattr(
                    k.get_layer_cache.return_value, "kv_cache_base", torch.empty(0, 528)
                ),
            ),
        )
        for name, change in changes:
            with self.subTest(name=name):
                args = self.fixture()
                change(*args)
                self.assertIsNone(_CED.CEDPlan.create(*args))
        self.assertIsNone(_CED.CEDPlan.create(*self.fixture(length=32767)))
        self.assertIsNone(
            _CED.CEDPlan.create(*self.fixture(), prepare_hidden_fn=lambda: None)
        )
        model, ctx, attn, cache = self.fixture()
        self.assertIsNone(_CED.CEDPlan.create(model, ctx, attn, None))
        with mock.patch.object(torch.distributed, "get_rank", return_value=3):
            with self.assertRaisesRegex(ValueError, "process group"):
                _CED.CEDPlan.create(model, ctx, attn, cache)

    def test_model_and_dspark_contracts_are_not_weakened(self):
        self.assertTrue(_CED._supported_model(_model()))
        for captures in ((), (37,), (38, 39), (37, 38, 39)):
            model = _model()
            model.capture_aux_hidden_layer_ids = captures
            self.assertTrue(_CED._supported_model(model))
        changes = (
            lambda m: setattr(m, "fp8_kv_cache", False),
            lambda m: setattr(m.args, "window_size", 256),
            lambda m: setattr(m.args, "dim", 4096),
            lambda m: setattr(m.args, "n_hash_layers", 1),
            lambda m: setattr(m, "capture_aux_hidden_layer_ids", (37, 37)),
            lambda m: setattr(m, "capture_aux_hidden_layer_ids", (36, 38, 39)),
            lambda m: setattr(m.layers[24].attn, "index_source_layer_id", 20),
            lambda m: setattr(m.layers[21].attn, "kv_source_layer_id", 21),
            lambda m: setattr(m.layers[24].attn, "is_index_source", False),
            lambda m: setattr(m.layers[39], "engram", object()),
        )
        for change in changes:
            model = _model()
            change(model)
            self.assertFalse(_CED._supported_model(model))
        model = _model()
        model.capture_aux_hidden_layer_ids = ()
        model._mtp_hidden_buffer = torch.empty(1)
        self.assertFalse(_CED._supported_model(model))


class CedLayoutTest(_SingleThreadTest):
    def test_identity_receives_reuse_owned_storage_and_ragged_or_reordered_scatter(
        self,
    ):
        seen = set()
        for length, selected, permuted in (
            (16384, torch.arange(16256, 16384), False),
            (65, torch.arange(64), False),
            (65, torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 64]), False),
            (65, torch.arange(64), True),
            (65, torch.tensor([0]), False),
        ):
            for rank in range(4):
                plan, positions = _plan(length, rank, selected, permuted=permuted)
                exchange = plan.exchange
                order = exchange.receive_positions.tolist()
                identity = order == list(range(plan.context.chunk_length))
                self.assertEqual(exchange.receive_is_identity, identity)
                seen.add("identity" if identity else "scatter")
                if len(order) < plan.context.chunk_length:
                    seen.add("ragged")
                elif not identity:
                    seen.add("reordered")
                if not order:
                    seen.add("empty")
                for dtype, trailing in ((torch.int64, ()), (torch.bfloat16, (2, 3))):
                    padded = sum(len(rows) for rows in positions)
                    full = (
                        torch.arange(padded)
                        .to(dtype)
                        .reshape(padded, *([1] * len(trailing)))
                        .expand(padded, *trailing)
                    )
                    value = full[positions[rank]]
                    expected = _compact_oracle(full, selected, rank)
                    destinations = []

                    def receive(destination, send, **kwargs):
                        destinations.append(destination)
                        destination.copy_(
                            expected.index_select(0, exchange.receive_positions)
                        )

                    with mock.patch.object(
                        torch.distributed, "all_to_all_single", side_effect=receive
                    ):
                        actual = exchange.compact(value)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(actual.is_contiguous())
                    self.assertEqual(actual is destinations[0], identity)
                    self.assertIsNone(actual._base)
                    self.assertNotEqual(actual.data_ptr(), value.data_ptr())
                    if identity:
                        # Older/manual descriptors default to the original copy path.
                        conservative = replace(exchange, receive_is_identity=False)
                        with mock.patch.object(
                            torch.distributed, "all_to_all_single", side_effect=receive
                        ):
                            copied = conservative.compact(value)
                        self.assertIsNot(copied, destinations[-1])
                        torch.testing.assert_close(copied, actual, rtol=0, atol=0)
        self.assertEqual(seen, {"identity", "scatter", "ragged", "reordered", "empty"})

    def test_inverse_map_split_sizes_and_original_owner_row_collisions(self):
        selected = torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 64])
        for permuted in (False, True):
            plans = [
                _plan(65, rank, selected, 16387, permuted=permuted)[0]
                for rank in range(4)
            ]
            for rank, plan in enumerate(plans):
                ctx, ex = plan.context, plan.exchange
                self.assertEqual(ctx.chunk_length, 4)
                self.assertEqual(int(ctx.local_is_real.sum()), sum(ex.receive_sizes))
                self.assertEqual(sum(ex.send_sizes), len(ex.send_indices))
                self.assertEqual(sum(ex.receive_sizes), len(ex.receive_positions))
                for peer in range(4):
                    self.assertEqual(
                        ex.send_sizes[peer], plans[peer].exchange.receive_sizes[rank]
                    )
                self.assertTrue(
                    torch.equal(ctx.global_positions, ctx.relative_positions + 16387)
                )
                self.assertEqual(ctx.seq_len_full, 65)
                self.assertTrue(
                    (ctx.relative_positions[~ctx.local_is_real] == 64).all()
                )
            # Rank0's first two compact rows come from different original owners,
            # both at local row 0. A single scatter-to-original-M would collide.
            if not permuted:
                groups = plans[0].indexer_groups
                original_rows = torch.cat([g[1] for g in groups])
                self.assertLess(len(original_rows.unique()), len(original_rows))

    def test_indexer_projection_matches_each_original_owner_gemm(self):
        selected = torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 64])
        x = torch.arange(65 * 5, dtype=torch.float64).reshape(65, 5) / 64
        weight = torch.arange(15, dtype=torch.float64).reshape(3, 5) / 16
        expected = F.linear(x, weight)
        for rank in range(4):
            plan, _ = _plan(65, rank, selected)
            compact = _compact_oracle(x, selected, rank)
            with mock.patch.object(_CED.F, "linear", wraps=F.linear) as linear:
                actual = plan.project_indexer_weights(compact, weight)
            torch.testing.assert_close(
                actual, _compact_oracle(expected, selected, rank), rtol=0, atol=0
            )
            self.assertEqual(linear.call_count, len(plan.indexer_groups))
            for call in linear.call_args_list:
                self.assertEqual(
                    call.args[0].shape, (plan.original_context.chunk_length, 5)
                )
            destinations = torch.cat([g[0] for g in plan.indexer_groups])
            self.assertEqual(len(destinations.unique()), len(destinations))
            self.assertTrue((actual[~plan.context.local_is_real] == 0).all())

    def test_gather_scatter_restores_fresh_coordinates_and_zeros_gaps(self):
        selected = torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 64])
        full = torch.arange(65 * 3).reshape(65, 3)
        payloads = [_compact_oracle(full, selected, r) for r in range(4)]
        plan, _ = _plan(65, 0, selected, 16387)
        expected = torch.zeros_like(full)
        expected[selected] = full[selected]
        for provided in (False, True):
            output = torch.full_like(full, -919) if provided else None
            actual = _CP._cp_restore_gathered_full_2d(
                torch.cat(payloads), plan.context, out=output
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if provided:
                self.assertIs(actual, output)
        for output in (
            torch.empty(64, 3, dtype=full.dtype),
            torch.empty(65, 3, dtype=torch.float32),
        ):
            with self.assertRaises(ValueError):
                _CP._cp_restore_gathered_full_2d(
                    torch.cat(payloads), plan.context, out=output
                )
        with self.assertRaises(ValueError):
            _CP._cp_restore_gathered_full_2d(torch.cat(payloads)[:-1], plan.context)

    def test_invalid_inverse_map_and_lifecycle_rejected_before_payload(self):
        selected = torch.tensor([1, 3])
        original, _ = _context(65, 0)
        original.cp_info.prefill_qkv_padding_mask[0] = 0
        with self.assertRaisesRegex(ValueError, "inverse map"):
            _CED._query_layout(original, selected, None)
        plan, _ = _plan(65, 0, selected)
        with mock.patch.object(
            torch.distributed,
            "all_to_all_single",
            side_effect=AssertionError("unexpected transport"),
        ):
            with self.assertRaisesRegex(ValueError, "before compact"):
                plan.restore_rows(torch.empty(plan.context.chunk_length, 2))
            with self.assertRaisesRegex(ValueError, "out of order"):
                plan.restore_aux(_model())
            with self.assertRaisesRegex(ValueError, "source row count"):
                plan.exchange.compact(
                    torch.empty(plan.original_context.chunk_length - 1)
                )
            with self.assertRaisesRegex(ValueError, "compact row count"):
                plan.exchange.restore(torch.empty(plan.context.chunk_length + 1))
            plan._compacted = True
            model = _model()
            model.capture_aux_hidden_layer_ids = (39, 38, 37)
            with self.assertRaisesRegex(ValueError, "capture contract"):
                plan.restore_aux(model)


class V41KVWorkspaceTest(_SingleThreadTest):
    def test_same_global_reuses_buffer_and_only_updates_swa(self):
        shared = {}
        global_kv = torch.arange(15).reshape(5, 3).bfloat16()
        original = global_kv.clone()
        swa = torch.full((4, 3), -1.0, dtype=global_kv.dtype)
        output = _KV_WORKSPACE.combine_kv(shared, [(global_kv, None)], [swa])
        pointer = output.data_ptr()
        self.assertNotEqual(pointer, global_kv.data_ptr())
        for value in (7.0, -3.0, 11.0):
            swa.fill_(value)
            updated = _KV_WORKSPACE.combine_kv(shared, [(global_kv, None)], [swa])
            self.assertIs(updated, output)
            self.assertEqual(updated.data_ptr(), pointer)
            self.assertTrue(torch.equal(updated[:5], original))
            self.assertTrue(torch.equal(updated[5:], swa))
            self.assertTrue(torch.equal(global_kv, original))

    def test_equal_shape_new_global_rebuilds_and_releases_old_source(self):
        shared = {}
        global_kv = torch.full((5, 3), 2.0, dtype=torch.bfloat16)
        swa = torch.full((4, 3), -1.0, dtype=global_kv.dtype)
        output = _KV_WORKSPACE.combine_kv(shared, [(global_kv, None)], [swa])
        source_ref, output_ref = weakref.ref(global_kv), weakref.ref(output)
        del global_kv, output
        self.assertIsNotNone(source_ref())
        self.assertIsNotNone(output_ref())

        new_global = torch.full((5, 3), 19.0, dtype=torch.bfloat16)
        swa.fill_(7.0)
        output = _KV_WORKSPACE.combine_kv(shared, [(new_global, None)], [swa])
        self.assertIsNone(source_ref())
        self.assertIsNone(output_ref())
        self.assertTrue(torch.equal(output, torch.cat((new_global, swa))))
        self.assertTrue(torch.equal(new_global, torch.full_like(new_global, 19.0)))

    def test_swa_shape_change_rebuilds_and_releases_old_buffer(self):
        shared = {}
        global_kv = torch.arange(15).reshape(5, 3).bfloat16()
        original = global_kv.clone()
        output = _KV_WORKSPACE.combine_kv(
            shared, [(global_kv, None)], [torch.zeros(4, 3, dtype=global_kv.dtype)]
        )
        for rows in (2, 7):
            with self.subTest(rows=rows):
                output_ref = weakref.ref(output)
                del output
                self.assertIsNotNone(output_ref())
                swa = torch.full((rows, 3), -rows, dtype=global_kv.dtype)
                output = _KV_WORKSPACE.combine_kv(shared, [(global_kv, None)], [swa])
                self.assertIsNone(output_ref())
                self.assertEqual(output.shape, (5 + rows, 3))
                self.assertTrue(torch.equal(output, torch.cat((original, swa))))
                self.assertTrue(torch.equal(global_kv, original))

    def test_release_drops_workspace_at_last_consumer_and_forward_exit(self):
        for last_layer in (2, None):
            with self.subTest(last_layer=last_layer):
                layers = {
                    i: types.SimpleNamespace(kv_source_layer_id=1) for i in (1, 2)
                }
                global_kv = torch.ones(5, 3, dtype=torch.bfloat16)
                shared = {"layers": layers, "global": {1: [(global_kv, None)]}}
                output = _KV_WORKSPACE.combine_kv(
                    shared,
                    shared["global"][1],
                    [torch.zeros(4, 3, dtype=global_kv.dtype)],
                )
                source_ref, output_ref = weakref.ref(global_kv), weakref.ref(output)
                del global_kv, output
                _PREFILL_META.release_v41_prefill_shared(shared, 1)
                self.assertIsNotNone(source_ref())
                self.assertIsNotNone(output_ref())

                _PREFILL_META.release_v41_prefill_shared(shared, last_layer)
                self.assertIsNone(source_ref())
                self.assertIsNone(output_ref())
                self.assertNotIn("prefill_kv_workspace", shared)
                self.assertFalse(shared.get("global"))
                self.assertIs(shared["layers"], layers)


class CedCandidatePublicationTest(_SingleThreadTest):
    @staticmethod
    def selection_namespace():
        path = Path(_CED.__file__).parent.parent / "fp8/attention_v41.py"
        tree = ast.parse(path.read_text())
        names = {
            "select_candidate_blocks",
            "mask_candidate_logits",
            "_apply_prefill_candidates",
        }
        nodes = [
            n
            for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.ImportFrom) and n.module == "bisect")
        ]
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AttentionV41FP8"
        )
        nodes.append(
            next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "_select_indices"
            )
        )
        namespace = {"torch": torch, "F": F}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace

    def test_host_rows_follow_actual_inverse_map_prefix_and_padding(self):
        for length, prefix in (
            (129, 0),
            (257, 16387),
            (16385, 0),
            (32768, 0),
            (131073, 32768),
        ):
            for permuted in (False, True):
                selected = torch.arange(length - 128, length)
                total = 0
                for rank in range(4):
                    plan, positions = _plan(
                        length,
                        rank,
                        selected,
                        prefix,
                        permuted=permuted,
                        candidate_publication=True,
                    )
                    expected = tuple(
                        i
                        for i, p in enumerate(positions[rank].tolist())
                        if length - 128 <= p < length
                    )
                    self.assertEqual(plan.exchange.candidate_rows_host, expected)
                    self.assertTrue(all(type(i) is int for i in expected))
                    total += len(expected)
                self.assertEqual(total, 128)
        exact, _ = _plan(32768, 0, torch.arange(32768 - 3072, 32768))
        self.assertIsNone(exact.exchange.candidate_rows_host)
        with exact.candidate_publication(shared := {}):
            self.assertNotIn("ced_candidate_rows", shared)

    def test_publish_keeps_whole_chunk_shape_stride_and_source_logits(self):
        ns = self.selection_namespace()
        seen = []

        def select(logits, visible, block, count, **kwargs):
            seen.append((logits.shape, logits.stride(), logits.data_ptr()))
            kwargs["out"].copy_(
                ns["select_candidate_blocks"](logits, visible, block, count)
            )
            return kwargs["out"], None

        ns["prefill_candidates"] = types.SimpleNamespace(select_candidates=select)
        backing = torch.zeros(12, 35)
        logits = backing[:, 1:34]
        logits[:, ::7] = 2.0  # Exact ties; retained chunks keep their full shape.
        visible = torch.arange(21, 33)
        logits.masked_fill_(torch.arange(33)[None] >= visible[:, None], -torch.inf)
        before = logits.clone()
        baseline = {"candidates": torch.full((12, 3), -1, dtype=torch.int32)}
        retained = {
            "candidates": baseline["candidates"].clone(),
            "ced_candidate_rows": (4, 11),
        }
        apply = ns["_apply_prefill_candidates"]
        for shared in (baseline, retained):
            for start in range(0, 12, 4):
                rows = slice(start, start + 4)
                apply(shared, logits[rows], visible[rows], rows, 4, 3, True)
        self.assertEqual(len(seen), 5)
        self.assertEqual(seen[:3][1:], seen[3:])
        self.assertTrue(
            torch.equal(retained["candidates"][4:], baseline["candidates"][4:])
        )
        self.assertTrue((retained["candidates"][:4] == -1).all())
        self.assertTrue(torch.equal(logits, before))
        # Empty original-owner ranks do no work, even if their new CP rank receives rows.
        with mock.patch.dict(ns, prefill_candidates=mock.Mock()) as _:
            shared = {"ced_candidate_rows": ()}
            apply(shared, logits, visible, slice(0, 12), 4, 3, True)
            ns["prefill_candidates"].select_candidates.assert_not_called()

    def test_full_selector_keeps_all_l20_topk_and_retained_candidates(self):
        ns = self.selection_namespace()
        calls = []

        def candidate_fallback(logits, *args, **kwargs):
            calls.append((logits.shape, logits.stride()))
            return None

        ns.update(
            prefill_candidates=types.SimpleNamespace(
                select_candidates=candidate_fallback
            ),
            prefill_deepselect=types.SimpleNamespace(is_available=lambda device: False),
            prefill_indexer=types.SimpleNamespace(
                PrefillIndexerKeys=type("Packed", (), {})
            ),
            prefill_topk=types.SimpleNamespace(try_select_tokens=lambda *a, **k: None),
            rope_only=lambda q, freqs, dim: q,
            fp8_roundtrip=lambda q: q,
        )
        torch.manual_seed(23)
        x, qr, keys = torch.randn(193, 2), torch.randn(193, 2), torch.randn(257, 2)
        positions = torch.arange(193) + 64
        reqs = torch.zeros(193, dtype=torch.long)
        outputs = []
        for rows in (None, (130, 170), ()):
            shared = {"global": {20: [(torch.empty(257, 2), keys)]}}
            if rows is not None:
                shared["ced_candidate_rows"] = rows
            owner = types.SimpleNamespace(
                _shared_attention=shared,
                is_index_source=True,
                layer_id=20,
                kv_source_layer_id=20,
                index_topk=4,
                index_n_heads=1,
                index_head_dim=2,
                compress_ratio=1,
                rope_head_dim=0,
                index_wq=None,
                index_weights=torch.tensor([[0.5, -0.25]]),
                freqs_cis=torch.empty(257, 0),
                _lin=lambda linear, value: value,
                v41_config={
                    "candidate_source_layer_id": 20,
                    "candidate_topk_blocks": 3,
                    "candidate_block_size": 4,
                },
            )
            calls.clear()
            topk = ns["_select_indices"](
                owner, x, qr, positions, reqs, request_row_slices=(slice(0, 193),)
            )
            outputs.append((topk, shared["candidates"], list(calls)))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        self.assertTrue(torch.equal(outputs[0][0], outputs[2][0]))
        self.assertEqual(len(outputs[0][2]), 4)
        self.assertEqual(outputs[1][2], outputs[0][2][2:3])
        self.assertEqual(outputs[2][2], [])
        self.assertTrue(torch.equal(outputs[1][1][128:192], outputs[0][1][128:192]))
        self.assertTrue((outputs[1][1][:128] == -1).all())
        self.assertTrue((outputs[2][1] == -1).all())

    def test_consumers_and_unknown_row_mapping_keep_original_path(self):
        ns = self.selection_namespace()
        ns["prefill_candidates"] = types.SimpleNamespace(
            build_flags=lambda *a: None,
            select_candidates=mock.Mock(return_value=None),
        )
        logits = torch.arange(48).float().view(6, 8)
        shared = {
            "candidates": torch.tensor([[0, 2]]).int().expand(6, -1).clone(),
            "ced_candidate_rows": (),
        }
        expected = ns["mask_candidate_logits"](logits.clone(), shared["candidates"], 2)
        ns["_apply_prefill_candidates"](
            shared, logits, torch.full((6,), 8), slice(0, 6), 2, 2, False
        )
        self.assertTrue(torch.equal(logits, expected))
        for mapping in (slice(None), torch.tensor([1, 3])):
            subset = logits[mapping]
            ns["_apply_prefill_candidates"](
                shared, subset, torch.full((subset.shape[0],), 8), mapping, 2, 2, True
            )
        self.assertEqual(ns["prefill_candidates"].select_candidates.call_count, 2)

    def test_publication_scope_cleans_up_on_failure_and_forward_uses_it_only_at_l20(
        self,
    ):
        plan, _ = _plan(
            32768, 0, torch.arange(32640, 32768), candidate_publication=True
        )
        path = Path(_CED.__file__).parent / "forward.py"
        tree = ast.parse(path.read_text())
        loop = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple)
            and [e.id for e in n.target.elts] == ["layer_idx", "layer_call"]
        )
        # Execute the original layer-call loop with only model/kernel boundaries mocked.
        code = compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec")
        for exact, absent, fail in (
            (False, False, False),
            (False, False, True),
            (True, False, False),
            (False, True, False),
        ):
            shared, seen = {}, []
            plan.context = replace(
                plan.context, swa_replay_start=None if exact else 32640
            )

            def compact(v4, h, ids, state):
                self.assertNotIn("ced_candidate_rows", state)
                return h, ids

            def layer(i):
                def call(h, *args, **kwargs):
                    active = i == 20 and not exact and not absent
                    self.assertEqual("ced_candidate_rows" in shared, active)
                    if active:
                        self.assertEqual(
                            shared["ced_candidate_rows"],
                            plan.exchange.candidate_rows_host,
                        )
                    seen.append(i)
                    if i == 20 and fail:
                        raise RuntimeError("layer failure")
                    return h

                return call

            state = dict(
                layer_calls=[layer(i) for i in range(40)],
                ced_in_l20=False,
                ced_tail=None if absent else plan,
                shared_prefill=shared,
                layer_forward_range=lambda i: nullcontext(),
                nullcontext=nullcontext,
                h=torch.empty(1),
                input_ids=torch.empty(1),
                positions=None,
                cu_seqlens=None,
                v4=types.SimpleNamespace(_propagate_cp_ctx=lambda cp: None),
                kv_cache=None,
                block_tables_by_type=None,
                ws=None,
                capture_ids=(),
                _rt_on=False,
                write_cache_store_impl=None,
                release_v41_prefill_shared=lambda *a: None,
                clear_prefill_meta_shared_fp8=lambda *a: None,
                build_and_propagate_prefill_meta_fp8=lambda *a, **k: None,
                _profiler=types.SimpleNamespace(
                    record_function_range=lambda *a: nullcontext()
                ),
            )
            with mock.patch.object(plan, "compact", side_effect=compact):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "layer failure"):
                        exec(code, state)
                else:
                    exec(code, state)
                    self.assertEqual(seen, list(range(40)))
            self.assertNotIn("ced_candidate_rows", shared)


def _l20_common(ctx, freqs):
    values = dict(
        seqlen=ctx.chunk_length,
        seqlen_full=ctx.seq_len_full,
        cp_ctx=ctx,
        cp_on=True,
        freqs_cis=freqs,
        cu_seqlens=torch.tensor([0, ctx.chunk_length]),
        input_lengths=torch.tensor([ctx.chunk_length]),
        prefix_lengths=ctx.prefix_lengths,
        position_ids=ctx.global_positions,
        req_id_per_token=ctx.req_id_per_token,
        max_seqlen_q=ctx.chunk_length,
        request_row_slices=(slice(0, ctx.chunk_length),),
        swa_meta=object(),
        workspace=object(),
        batch_size=1,
    )
    return namedtuple("L20Common", values)(**values)


class CedRouterProjectionTest(_SingleThreadTest):
    @staticmethod
    def _ffn(gate, rows_per_chunk=16384, enabled=True):
        return types.SimpleNamespace(
            gate=gate,
            max_tokens_per_rank=rows_per_chunk,
            _should_chunk=lambda rows: enabled and rows > rows_per_chunk,
        )

    @staticmethod
    def _gate():
        # Execute the production gate without importing native model bindings.
        path = Path(_CED.__file__).parent.parent / "moe/gate.py"
        tree = ast.parse(path.read_text())
        nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in ("Gate", "_select_routes_with_nonfinite_fallback")
        ]
        ns = dict(
            torch=torch, nn=torch.nn, F=F, os=os, _use_fused_gate=lambda *a: False
        )
        prefix = ast.parse("from __future__ import annotations").body
        exec(
            compile(
                ast.Module(body=prefix + nodes, type_ignores=[]), str(path), "exec"
            ),
            ns,
        )
        gate = ns["Gate"].__new__(ns["Gate"])
        torch.nn.Module.__init__(gate)
        gate.weight = torch.arange(15).reshape(3, 5).float() / 16
        gate.bias = torch.zeros(3)
        gate.bias_vl = None
        gate.hash = False
        gate.topk = 2
        gate.score_func = "softmax"
        gate.route_scale = 1.0
        gate._dbg_prefix = None
        return gate

    def test_router_scope_restores_absent_none_and_nested_callbacks_on_exception(self):
        outer, _ = _plan(
            65,
            0,
            torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 64]),
            candidate_publication=True,
        )
        inner, _ = _plan(129, 0, torch.arange(1, 129), candidate_publication=True)
        outer._compacted = inner._compacted = True
        gate = self._gate()
        ffn = self._ffn(gate)
        for existing in ("absent", None, F.linear):
            if existing != "absent":
                gate._ced_row_projection = existing
            with self.assertRaisesRegex(RuntimeError, "inner failure"):
                with outer.router_projection(ffn):
                    self.assertEqual(
                        gate._ced_row_projection, outer.project_indexer_weights
                    )
                    with inner.router_projection(ffn):
                        self.assertEqual(
                            gate._ced_row_projection, inner.project_indexer_weights
                        )
                    self.assertEqual(
                        gate._ced_row_projection, outer.project_indexer_weights
                    )
                    with outer.router_projection(ffn):
                        raise RuntimeError("inner failure")
            if existing == "absent":
                self.assertFalse(hasattr(gate, "_ced_row_projection"))
            else:
                self.assertIs(gate._ced_row_projection, existing)
                del gate._ced_row_projection

    def test_router_scope_rejects_invalid_plan_before_mutating_gate(self):
        plan, _ = _plan(129, 0, torch.arange(1, 129), candidate_publication=True)
        gate = self._gate()
        ffn = self._ffn(gate)
        with self.assertRaises(ValueError), plan.router_projection(ffn):
            self.fail("uncompacted plan admitted")
        self.assertFalse(hasattr(gate, "_ced_row_projection"))
        plan._compacted = True
        with self.assertRaises(ValueError), plan.router_projection(object()):
            self.fail("missing projection interface admitted")
        with self.assertRaises(ValueError), plan.router_projection(self._ffn(gate, 16)):
            self.fail("chunked compact FFN admitted without call-local offsets")
        plan.context = replace(plan.context, swa_replay_start=None)
        with self.assertRaises(ValueError), plan.router_projection(ffn):
            self.fail("exact checkpoint plan admitted")

    def test_router_projection_reuses_original_owner_rows_and_dtype(self):
        selected = torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 64])
        gate = self._gate()
        for dtype in (torch.bfloat16, torch.float32):
            full = (torch.arange(65 * 5).reshape(65, 5) / 64).to(dtype)
            weight = gate.weight.to(dtype)
            expected = F.linear(full, weight)
            for rank in range(4):
                plan, _ = _plan(65, rank, selected, candidate_publication=True)
                plan._compacted = True
                x = _compact_oracle(full, selected, rank)
                torch.testing.assert_close(
                    gate._project_scores(x, weight), F.linear(x, weight), rtol=0, atol=0
                )
                with plan.router_projection(self._ffn(gate)):
                    with mock.patch.object(_CED.F, "linear", wraps=F.linear) as linear:
                        result = gate._project_scores(x, weight)
                    self.assertEqual(linear.call_count, len(plan.indexer_groups))
                    for call in linear.call_args_list:
                        self.assertEqual(
                            call.args[0].shape[0], plan.original_context.chunk_length
                        )
                        self.assertEqual(call.args[0].dtype, dtype)
                torch.testing.assert_close(
                    result, _compact_oracle(expected, selected, rank), rtol=0, atol=0
                )

    def test_router_keeps_original_chunk_m_offsets_partial_chunk_and_owner(self):
        linear = F.linear

        def m_sensitive_linear(x, weight):
            # CPU GEMM need not change its arithmetic with M. Make the recipe
            # observable to catch a numerically silent full-M/local-row mistake.
            return linear(x, weight) + x.shape[0] / 16

        for length in (131072, 131073):
            owners, _ = _cp4_layout(length)
            original_m = len(owners[0])
            local_rows = sorted(
                {0, 1, 16382, 16383, 16384, 16385, original_m - 2, original_m - 1}
            )
            selected = torch.cat([rows[local_rows] for rows in owners])
            selected = selected[selected < length].sort().values
            full = (torch.arange(length * 5).reshape(length, 5) % 127).double() / 128
            weight = self._gate().weight.double()
            for chunking in (False, True):
                chunk_rows = 16384 if chunking else original_m
                expected = torch.empty(length, weight.shape[0], dtype=torch.float64)
                for owner_rows in owners:
                    real = owner_rows < length
                    owner_x = torch.zeros(original_m, 5, dtype=torch.float64)
                    owner_x[real] = full[owner_rows[real]]
                    for start in range(0, original_m, chunk_rows):
                        stop = min(start + chunk_rows, original_m)
                        values = m_sensitive_linear(owner_x[start:stop], weight)
                        valid = real[start:stop]
                        expected[owner_rows[start:stop][valid]] = values[valid]
                observed_m = set()
                for rank in range(4):
                    plan, _ = _plan(length, rank, selected, candidate_publication=True)
                    plan._compacted = True
                    gate = self._gate()
                    ffn = self._ffn(gate, 16384, enabled=chunking)
                    plan._prepare_router_projection(ffn)
                    x = _compact_oracle(full, selected, rank)
                    with mock.patch.object(
                        torch.Tensor,
                        "cpu",
                        side_effect=AssertionError("D2H in router scope"),
                    ):
                        with plan.router_projection(ffn):
                            with mock.patch.object(
                                _CED.F, "linear", side_effect=m_sensitive_linear
                            ) as project:
                                result = gate._project_scores(x, weight)
                        observed_m.update(
                            call.args[0].shape[0] for call in project.call_args_list
                        )
                    torch.testing.assert_close(
                        result,
                        _compact_oracle(expected, selected, rank),
                        rtol=0,
                        atol=0,
                    )
                    self.assertFalse(hasattr(gate, "_ced_row_projection"))
                expected_m = {
                    min(chunk_rows, original_m - start)
                    for start in range(0, original_m, chunk_rows)
                }
                self.assertEqual(observed_m, expected_m)

    def test_router_scope_follows_actual_moe_chunk_policy(self):
        path = Path(_CED.__file__).parent.parent / "moe/moe_layer.py"
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "MoE"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_should_chunk"
        )
        enabled = mock.Mock(return_value=True)
        ns = dict(torch=torch, chunked_moe_enabled=enabled)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        plan, _ = _plan(
            131072, 0, torch.arange(130944, 131072), candidate_publication=True
        )
        plan._compacted = True
        gate = self._gate()
        ffn = types.SimpleNamespace(
            gate=gate, max_tokens_per_rank=16384, _is_decode_role=False
        )
        ffn._should_chunk = types.MethodType(ns["_should_chunk"], ffn)
        for policy, decode, capture, expect_chunk in (
            (True, False, False, True),
            (False, False, False, False),
            (True, True, False, False),
            (True, False, True, False),
        ):
            enabled.return_value = policy
            ffn._is_decode_role = decode
            with mock.patch.object(
                torch.cuda, "is_available", return_value=True
            ), mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capture
            ), mock.patch.object(
                plan, "_router_chunk_groups", wraps=plan._router_chunk_groups
            ) as groups:
                plan._prepare_router_projection(ffn)
                prepared_calls = groups.call_count
                with plan.router_projection(ffn):
                    self.assertEqual(groups.call_count, int(expect_chunk))
                    self.assertEqual(groups.call_count, prepared_calls)
                    if expect_chunk:
                        groups.assert_called_once_with(16384)
                    else:
                        self.assertEqual(
                            gate._ced_row_projection, plan.project_indexer_weights
                        )

    def test_router_maps_prepare_once_only_after_split_admission(self):
        plan, _ = _plan(
            131073, 0, torch.arange(130945, 131073), candidate_publication=True
        )
        gate = self._gate()
        ffn = self._ffn(gate)
        block = types.SimpleNamespace(
            forward_prefill_ced_l20=mock.Mock(),
            attn=types.SimpleNamespace(
                _prefill_produce=mock.Mock(), _prefill_query=mock.Mock()
            ),
            attn_hc=types.SimpleNamespace(pre_mix_out=None),
            ffn=ffn,
        )
        v4 = types.SimpleNamespace(layers=[None] * 20 + [block])
        supported = mock.Mock(return_value=False)
        module = types.SimpleNamespace(can_preserve_compact_mhc=supported)
        with mock.patch.dict(
            sys.modules, {"rtp_llm.models_py.modules.dsv4.hc.v41_mega_mhc": module}
        ), mock.patch.object(
            plan, "_router_chunk_groups", wraps=plan._router_chunk_groups
        ) as build:
            self.assertFalse(plan.can_split_l20(v4))
            self.assertIsNone(plan._router_groups)
            build.assert_not_called()
            supported.return_value = True
            self.assertTrue(plan.can_split_l20(v4))
            prepared = plan._router_groups
            self.assertTrue(plan.can_split_l20(v4))
            self.assertIs(plan._router_groups, prepared)
            build.assert_called_once_with(16384)
        plan._compacted = True
        x = torch.ones(plan.context.chunk_length, 5)
        with mock.patch.object(
            torch, "tensor", side_effect=AssertionError("mapping tensor created in L20")
        ), mock.patch.object(
            plan,
            "_router_chunk_groups",
            side_effect=AssertionError("mapping rebuilt in L20"),
        ):
            with plan.router_projection(ffn):
                projected = gate._project_scores(x, gate.weight)
            torch.testing.assert_close(
                projected, F.linear(x, gate.weight), rtol=0, atol=0
            )
            ffn.max_tokens_per_rank = 8192
            with self.assertRaisesRegex(
                ValueError, "not prepared"
            ), plan.router_projection(ffn):
                self.fail("changed chunk geometry admitted")
            ffn._should_chunk = lambda rows: False
            with self.assertRaisesRegex(
                ValueError, "policy changed"
            ), plan.router_projection(ffn):
                self.fail("changed chunk policy admitted")
        self.assertFalse(hasattr(gate, "_ced_row_projection"))

    def test_gate_forward_routes_through_projection_in_both_precision_modes(self):
        gate = self._gate()
        x = torch.ones(4, 5, dtype=torch.bfloat16)
        projected = torch.tensor([[1.0, 3.0, -1.0]]).expand(4, -1)
        parent = types.ModuleType("rtp_llm.models_py.modules.dsv4")
        parent._record_tensor = types.SimpleNamespace(ENABLED=False)
        for fp32 in ("0", "1"):
            callback = mock.Mock(side_effect=lambda x, w: projected.to(x.dtype))
            gate._ced_row_projection = callback
            with mock.patch.dict(os.environ, DSV4_GATE_FP32=fp32), mock.patch.dict(
                sys.modules, {parent.__name__: parent}
            ):
                weights, indices = gate(x)
            callback.assert_called_once()
            dtype = torch.float32 if fp32 == "1" else torch.bfloat16
            self.assertEqual(
                tuple(t.dtype for t in callback.call_args.args), (dtype, dtype)
            )
            expected_weights, expected_indices = projected.softmax(-1).topk(2, dim=-1)
            torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
            self.assertTrue(torch.equal(indices, expected_indices))

    def test_mega_and_mega_se_gate_pack_consume_gate_projection(self):
        root = Path(_CED.__file__).parent.parent / "moe/strategies"
        for filename, classname, getter in (
            ("mega.py", "MegaMoEStrategy", "_get_gate_pack_kernels"),
            ("mega_se.py", "MegaMoEStrategySE", "_get_mega_se_gate_pack_kernels"),
        ):
            path = root / filename
            cls = next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == classname
            )
            method = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "forward_with_gate_pack"
            )
            pack = mock.Mock()
            ns = dict(
                torch=torch,
                record_function_range=lambda *a: nullcontext(),
                sync_cuda_graph_warmup_ranks=lambda *a: None,
                FP4_BLOCK=32,
            )
            ns[getter] = lambda: (pack, pack, None)
            exec(
                compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
                ns,
            )
            buf = types.SimpleNamespace(
                num_max_tokens_per_rank=4,
                x=torch.empty(4, 5),
                x_sf=torch.empty(4, 1),
                topk_idx=torch.empty(4, 2),
                topk_weights=torch.empty(4, 2),
                shared_l1_acts_sf=torch.empty(4, 1),
            )
            owner = types.SimpleNamespace(
                _mega_buf=buf,
                _mega_y=torch.zeros(4, 5),
                _mega_l1_w=None,
                _mega_l1_sf=None,
                _mega_l2_w=None,
                _mega_l2_sf=None,
                cfg=types.SimpleNamespace(layer_id=20, swiglu_limit=0),
                _maybe_pre_kernel_barrier=lambda *a: None,
                _validate_capacity=lambda *a: None,
                _block_m=lambda *a: 128,
                _launch=mock.Mock(),
            )
            gate = self._gate()
            scores = torch.full((4, 3), 7, dtype=torch.bfloat16)
            gate._ced_row_projection = mock.Mock(return_value=scores)
            dg = types.ModuleType("deep_gemm")
            dg.fp8_fp4_mega_moe = mock.Mock()
            for hash_gate in (False, True):
                gate.hash = hash_gate
                gate.tid2eid = torch.zeros(4, 2, dtype=torch.long)
                gate._ced_row_projection.reset_mock()
                pack.reset_mock()
                with mock.patch.dict(sys.modules, deep_gemm=dg):
                    ns["forward_with_gate_pack"](
                        owner,
                        torch.ones(4, 5, dtype=torch.bfloat16),
                        gate,
                        torch.arange(4),
                    )
                gate._ced_row_projection.assert_called_once()
                pack.assert_called_once()
                self.assertIs(pack.call_args.args[1], scores)


class CedL20SplitTest(_SingleThreadTest):
    def test_split_minimum_counts_fresh_rows_before_model_checks_or_uploads(self):
        for fresh in (129, 16384, 32768, 65535):
            for prefix in (0, 131072):
                plan, _ = _plan(
                    fresh,
                    0,
                    torch.arange(fresh - 128, fresh),
                    prefix,
                    candidate_publication=True,
                )
                with mock.patch.object(
                    plan,
                    "_prepare_router_projection",
                    side_effect=AssertionError("short prefill prepared maps"),
                ):
                    # No model interface or mHC capability may be consulted.
                    self.assertFalse(plan.can_split_l20(object()))
                self.assertIsNone(plan._router_groups)
                with plan.candidate_publication(shared := {}):
                    self.assertIn("ced_candidate_rows", shared)

    def test_unavailable_split_interface_retains_stage0_plan(self):
        supports_mhc = mock.Mock(return_value=True)
        modules = mock.patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv4.hc.v41_mega_mhc": types.SimpleNamespace(
                    can_preserve_compact_mhc=supports_mhc
                )
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        plan, _ = _plan(
            65536, 0, torch.arange(65408, 65536), candidate_publication=True
        )
        block = types.SimpleNamespace(
            forward_prefill_ced_l20=mock.Mock(),
            attn=types.SimpleNamespace(
                _prefill_produce=mock.Mock(), _prefill_query=mock.Mock()
            ),
            attn_hc=types.SimpleNamespace(pre_mix_out=None),
            ffn_hc=object(),
            ffn_norm=object(),
            ffn=types.SimpleNamespace(
                gate=types.SimpleNamespace(_project_scores=F.linear),
                _should_chunk=mock.Mock(return_value=False),
            ),
        )
        v4 = types.SimpleNamespace(layers=[None] * 20 + [block])
        self.assertTrue(plan.can_split_l20(v4))
        supports_mhc.assert_called_once_with(
            block.attn_hc,
            block.ffn_hc,
            block.ffn_norm,
            original_tokens=16384,
            compact_tokens=32,
        )
        supports_mhc.return_value = False
        self.assertFalse(plan.can_split_l20(v4))
        supports_mhc.return_value = True
        supports_mhc.reset_mock()
        block.ffn._should_chunk.return_value = True
        self.assertFalse(plan.can_split_l20(v4))
        block.ffn._should_chunk.return_value = False
        for owner, field in (
            (block, "forward_prefill_ced_l20"),
            (block.attn, "_prefill_produce"),
            (block.attn, "_prefill_query"),
            (block.ffn.gate, "_project_scores"),
            (block.ffn, "_should_chunk"),
        ):
            with mock.patch.object(owner, field, None):
                self.assertFalse(plan.can_split_l20(v4))
                with plan.candidate_publication(shared := {}):
                    self.assertEqual(
                        shared["ced_candidate_rows"], plan.exchange.candidate_rows_host
                    )
        block.attn_hc = types.SimpleNamespace()
        self.assertFalse(plan.can_split_l20(v4))
        block.attn_hc.pre_mix_out = None
        with mock.patch.object(block, "ffn", None):
            self.assertFalse(plan.can_split_l20(v4))
        with mock.patch.object(block.ffn, "gate", None):
            self.assertFalse(plan.can_split_l20(v4))
        plan.context = replace(plan.context, swa_replay_start=None)
        self.assertFalse(plan.can_split_l20(v4))
        self.assertFalse(plan._compacted)
        supports_mhc.assert_not_called()

    def test_unsplit_swa_only_keeps_cold_and_prefix_order_and_outputs(self):
        for prefix in (0, 127):
            owner = _swa_owner(False)
            owner.compress_ratio = 0
            owner.head_dim = 2
            common = types.SimpleNamespace(
                cp_on=False,
                cp_ctx=None,
                any_cont=bool(prefix),
                topk_idxs=torch.tensor([[[0, -1], [1, 0]]]),
                swa_meta=types.SimpleNamespace(
                    combined_indices=torch.tensor([[127, 126], [128, 127]]),
                    combined_lens=torch.tensor([2, 2]),
                    topk_length_kv_full=torch.tensor([1, 2]),
                ),
            )
            qkv = types.SimpleNamespace(
                kv_full=torch.arange(4).reshape(2, 2).bfloat16()
            )
            source = qkv.kv_full.clone()
            concat = torch.empty(prefix + 2, 2)
            events = []
            owner._prefill_common_setup = lambda *a: common
            owner._prefill_compute_qkv = lambda *a, **kw: qkv
            owner._swa_prefill_concat = lambda *a: (
                events.append("read-prefix"),
                concat,
            )[1]
            owner._prefill_write_swa_fp8_paged = lambda meta, kv: events.append("write")
            owner._produce_global = mock.Mock(
                side_effect=AssertionError("SWA has no global source")
            )
            marker = object()
            owner._prefill_sparse_attention.side_effect = lambda *a, **kw: (
                events.append("attention"),
                marker,
            )[1]
            ns = owner._prefill_query.__func__.__globals__
            with mock.patch.dict(ns, dispose_tensor=lambda t: events.append("dispose")):
                result = owner._forward_prefill(torch.empty(2, 2), torch.arange(2))
            self.assertIs(result, marker)
            self.assertEqual(
                events,
                (
                    ["read-prefix", "write", "dispose", "attention"]
                    if prefix
                    else ["write", "attention", "dispose"]
                ),
            )
            launch = owner._prefill_sparse_attention.call_args.kwargs
            expected_kv = concat if prefix else qkv.kv_full
            self.assertEqual(launch["kv"].data_ptr(), expected_kv.data_ptr())
            expected_indices = (
                common.swa_meta.combined_indices
                if prefix
                else common.topk_idxs.squeeze(0)
            )
            self.assertTrue(torch.equal(launch["indices"].squeeze(1), expected_indices))
            self.assertTrue(torch.equal(qkv.kv_full, source))

    def test_forward_compacts_inside_l20_once_and_keeps_cache_and_aux_callbacks(self):
        path = Path(_CED.__file__).parent / "forward.py"
        loop = next(
            n
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple)
            and [e.id for e in n.target.elts] == ["layer_idx", "layer_call"]
        )
        plan, _ = _plan(
            32768, 0, torch.arange(32640, 32768), candidate_publication=True
        )
        shared, seen, writes, captures = {}, [], [], []
        original = plan.original_context

        def layer(i):
            def call(h, ids, positions, cu, **kwargs):
                self.assertNotIn("ced_candidate_rows", shared)
                self.assertNotEqual(i, 20)
                expected_ctx = original if i < 20 else plan.context
                self.assertEqual(h.shape[0], expected_ctx.chunk_length)
                self.assertIs(positions, expected_ctx.global_positions)
                seen.append(i)
                return h

            return call

        def split(h, ids, positions, supplied_plan, **kwargs):
            self.assertIs(supplied_plan, plan)
            self.assertIs(positions, original.global_positions)
            self.assertNotIn("ced_candidate_rows", shared)
            seen.append(20)
            return h[:32], ids[:32]

        layers = [layer(i) for i in range(40)]
        layers[20] = types.SimpleNamespace(forward_prefill_ced_l20=split)
        v4 = types.SimpleNamespace(
            layers=layers,
            _propagate_cp_ctx=mock.Mock(),
            capture_aux_hidden=lambda i, h: captures.append((i, h.shape[0])),
        )
        rebuild = mock.Mock()
        state = dict(
            layer_calls=[layer(i) for i in range(40)],
            ced_in_l20=True,
            ced_tail=plan,
            shared_prefill=shared,
            layer_forward_range=lambda i: nullcontext(),
            nullcontext=nullcontext,
            h=torch.empty(original.chunk_length, 4, 3),
            input_ids=torch.arange(original.chunk_length),
            positions=original.global_positions,
            cu_seqlens=torch.tensor([0, original.chunk_length]),
            cp_ctx=original,
            v4=v4,
            kv_cache=types.SimpleNamespace(get_layer_caches=lambda i: i),
            block_tables_by_type=object(),
            ws=object(),
            capture_ids=(37, 38, 39),
            _rt_on=False,
            write_cache_store_impl=writes.append,
            release_v41_prefill_shared=lambda *a: None,
            clear_prefill_meta_shared_fp8=mock.Mock(),
            build_and_propagate_prefill_meta_fp8=rebuild,
            _profiler=types.SimpleNamespace(
                record_function_range=lambda *a: nullcontext()
            ),
        )
        with mock.patch.object(
            plan, "compact", side_effect=AssertionError("double compact")
        ):
            exec(
                compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"),
                state,
            )
        self.assertEqual(seen, list(range(40)))
        self.assertEqual(writes, list(range(40)))
        self.assertEqual(captures, [(37, 32), (38, 32), (39, 32)])
        v4._propagate_cp_ctx.assert_called_once_with(plan.context)
        rebuild.assert_called_once()
        self.assertIs(
            rebuild.call_args.kwargs["position_ids"], plan.context.global_positions
        )
        self.assertIs(
            rebuild.call_args.kwargs["prefix_lengths"], original.prefix_lengths
        )
        self.assertEqual(rebuild.call_args.kwargs["input_lengths"].tolist(), [32])

    def test_source_phase_finishes_full_writes_without_selecting_queries(self):
        owner = _swa_owner(False)
        events = []
        x = torch.arange(258).reshape(129, 2).float()
        common = types.SimpleNamespace(
            cp_on=False,
            cp_ctx=None,
            batch_size=1,
            input_lengths=torch.tensor([129]),
            prefix_lengths=torch.tensor([16384]),
            position_ids=torch.arange(16384, 16513),
            req_id_per_token=torch.zeros(129, dtype=torch.int32),
            any_cont=True,
        )
        qkv = types.SimpleNamespace(qr=x, kv_full=x.clone())
        owner.is_kv_source = True
        owner._prefill_common_setup = mock.Mock(return_value=common)
        owner._prefill_compute_qkv = mock.Mock(return_value=qkv)
        old_prefix = torch.full((127, 2), 7.0)

        def snapshot(qkv, common):
            events.append("read-prefix")
            return [torch.cat((old_prefix, qkv.kv_full))], [16384 - 127]

        def write(common, values):
            events.append("write-swa")
            self.assertEqual(values.shape[0], 129)
            old_prefix.fill_(-999)

        def global_write(values, positions, reqs, starts, lengths, **kwargs):
            events.append("write-global")
            self.assertIs(values, x)
            self.assertEqual(positions.tolist(), list(range(16384, 16513)))
            self.assertEqual(lengths.tolist(), [129])

        owner._swa_prefill_workspace = snapshot
        owner._prefill_write_swa_fp8_paged = write
        owner._produce_global = global_write
        owner._select_indices = mock.Mock(
            side_effect=AssertionError("query before transition")
        )
        prepared = owner._prefill_produce(x, common.position_ids)
        self.assertEqual(events, ["read-prefix", "write-swa", "write-global"])
        self.assertIs(prepared[1], qkv)
        self.assertTrue((prepared[2][0][:127] == 7).all())
        owner._select_indices.assert_not_called()

    def test_l20_queries_keep_prefix_halo_and_full_global_visibility(self):
        length, prefix = 129, 16384
        plan, _ = _plan(
            length, 0, torch.arange(1, length), prefix, candidate_publication=True
        )
        query_ctx = replace(plan.context, swa_replay_start=None)
        common = _l20_common(query_ctx, torch.empty(32, 1))
        owner = _swa_owner(False)
        end, swstart = prefix + length, prefix - 127
        global_kv = torch.full((end, 2), 11.0)
        swa = torch.arange(swstart, end).float()[:, None].expand(-1, 2).clone()
        owner._shared_attention["global"] = {20: [(global_kv, None)]}
        owner._select_indices = mock.Mock(
            return_value=torch.tensor([[0, -1]], dtype=torch.int32).expand(32, -1)
        )
        qkv = types.SimpleNamespace(qr=torch.empty(32, 2), kv_full=object())
        owner._prefill_query(torch.empty(32, 2), common, qkv, [swa], [swstart])
        launch = owner._prefill_sparse_attention.call_args.kwargs
        self.assertEqual(launch["kv"].shape, (end + 256, 1, 2))
        for row, pos in enumerate(query_ctx.global_positions.tolist()):
            indices = launch["indices"][row, 0]
            expected = [0] + [end + p - swstart for p in range(pos - 127, pos + 1)]
            self.assertEqual(indices[:129].tolist(), expected)
        self.assertEqual(query_ctx.global_positions[0].item(), end - 128)
        first_swa = launch["indices"][0, 0, 1].item()
        self.assertEqual(launch["kv"][first_swa, 0, 0].item(), end - 255)
        owner._prefill_write_swa_fp8_paged.assert_not_called()
        self.assertIsNone(common.cp_ctx.swa_replay_start)
        self.assertEqual(plan.context.swa_replay_start, 1)

    def test_block_exchanges_delayed_state_and_keeps_source_and_query_domains_separate(
        self,
    ):
        root = Path(_CED.__file__).parent.parent
        ns = {"torch": torch, "F": F, "Tuple": tuple}
        fallback = ast.parse((root / "hc/fallback_impl.py").read_text())
        nodes = [
            next(
                n
                for n in fallback.body
                if isinstance(n, ast.FunctionDef) and n.name == "_hc_split_sinkhorn"
            )
        ]
        delayed = ast.parse((root / "hc/delayed.py").read_text())
        nodes.append(
            next(
                n
                for n in delayed.body
                if isinstance(n, ast.FunctionDef) and n.name == "collapse_delayed"
            )
        )
        cls = next(
            n
            for n in delayed.body
            if isinstance(n, ast.ClassDef) and n.name == "DelayedHCUnit"
        )
        nodes += [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_pre_impl", "_post_impl")
        ]
        block_ast = ast.parse((root / "block.py").read_text())
        cls = next(
            n
            for n in block_ast.body
            if isinstance(n, ast.ClassDef) and n.name == "Block"
        )
        nodes.append(
            next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "forward_prefill_ced_l20"
            )
        )
        ns["_prefill_fast_norm"] = lambda norm, x: norm(x)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root), "exec"), ns)

        def hc(previous):
            unit = types.SimpleNamespace(
                hc_mult=4,
                dim=3,
                fn=torch.zeros(24, 12),
                base=torch.linspace(-1, 1, 24),
                scale=torch.ones(3),
                norm_eps=1e-6,
                hc_eps=1e-6,
                hc_sinkhorn_iters=4,
                _previous_ref=lambda: previous,
                pre_mix_out=None,
            )
            unit.pre = types.MethodType(ns["_pre_impl"], unit)
            unit.post = types.MethodType(ns["_post_impl"], unit)
            return unit

        class Norm:
            weight = torch.ones(3)

            def __call__(self, x):
                return (
                    x.float()
                    * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                ).to(x.dtype)

        for fail in (False, True):
            plan, _ = _plan(
                1024, 0, torch.arange(896, 1024), candidate_publication=True
            )
            rows = torch.empty(32, dtype=torch.long)
            for compact_rows, source_rows in plan.indexer_groups:
                rows[compact_rows] = source_rows
            plan.exchange.compact = lambda value: value.index_select(0, rows)
            hidden = (
                torch.arange(256 * 12).reshape(256, 4, 3).float() / 123
            ).bfloat16()
            ids = torch.arange(256) + 10000  # Preserve vision/sentinel IDs verbatim.
            previous = types.SimpleNamespace(
                pre_mix_out=torch.tensor([0.1, 0.2, 0.3, 0.4]).expand(256, 4)
            )
            a, f = hc(previous), hc(None)
            f._previous_ref = lambda: a
            expected_pre, expected_post, expected_comb = a.pre(hidden)
            original_mix = a.pre_mix_out.clone()
            norm = Norm()
            expected_norm = norm(expected_pre)
            expected_attn = expected_norm[rows] * 0.5
            expected_residual = a.post(
                expected_attn, hidden[rows], expected_post[rows], expected_comb[rows]
            )
            expected_ffn_readout = ns["collapse_delayed"](
                expected_residual, original_mix[rows]
            )
            common = _l20_common(plan.original_context, torch.arange(256)[:, None])
            fullkv = torch.arange(1024 * 3).reshape(1024, 3).bfloat16()
            QKV = namedtuple("QKV", "qr q kv_full")
            events = []
            global_state = object()
            attn = types.SimpleNamespace(
                _shared_attention={"global": {20: global_state}},
                _cp_ctx=plan.original_context,
                index_weights=torch.ones(1, 3, dtype=torch.bfloat16),
                can_fuse_prefill_attn_norm_input_quant=lambda *args: False,
                _set_compressor_pool_context=lambda: events.append("bind"),
                _clear_compressor_pool_context=lambda: events.append("clear"),
            )

            def produce(x, positions, quant):
                events.append("producer")
                self.assertEqual(x.shape[0], 256)
                self.assertIs(attn._cp_ctx, plan.original_context)
                self.assertTrue(torch.equal(x, expected_norm))
                return common, QKV(x.clone(), None, fullkv), [fullkv], [0], None

            def query(x, meta, qkv, swa, starts):
                events.append("query")
                self.assertEqual(x.shape[0], 32)
                self.assertEqual(meta.request_row_slices, (slice(0, 32),))
                self.assertTrue(torch.equal(meta.freqs_cis, common.freqs_cis[rows]))
                self.assertIs(meta.workspace, common.workspace)
                self.assertIs(meta.prefix_lengths, common.prefix_lengths)
                self.assertIsNone(meta.cp_ctx.swa_replay_start)
                self.assertIs(attn._cp_ctx, meta.cp_ctx)
                self.assertTrue(torch.equal(a.pre_mix_out, original_mix[rows]))
                self.assertIsNone(f.pre_mix_out)
                self.assertIs(qkv.kv_full, fullkv)
                self.assertIs(swa[0], fullkv)
                self.assertIs(attn._shared_attention["global"][20], global_state)
                projection = attn._shared_attention["ced_indexer_projection"]
                with mock.patch.object(_CED.F, "linear", wraps=F.linear) as linear:
                    raw_index_weights = projection(x, attn.index_weights)
                    self.assertTrue(
                        all(
                            call.args[0].shape[0] == 256
                            for call in linear.call_args_list
                        )
                    )
                self.assertTrue(
                    torch.equal(
                        raw_index_weights,
                        F.linear(expected_norm, attn.index_weights)[rows],
                    )
                )
                if fail:
                    raise RuntimeError("query failure")
                return x * 0.5

            @contextmanager
            def bind(attn, *args, **kwargs):
                previous_ctx = attn._cp_ctx
                attn._cp_ctx = kwargs.get("cp_ctx", previous_ctx)
                try:
                    yield
                finally:
                    attn._cp_ctx = previous_ctx

            attn._prefill_produce, attn._prefill_query = produce, query
            gate = CedRouterProjectionTest._gate()

            def ffn(x, ids):
                self.assertEqual(gate._ced_row_projection, plan.project_indexer_weights)
                return x * 0.25

            block = types.SimpleNamespace(
                layer_id=20,
                engram=None,
                attn=attn,
                attn_hc=a,
                ffn_hc=f,
                attn_norm=norm,
                ffn_norm=norm,
                _prefill_fast_hc_impls=lambda: (a.pre, f.pre, a.post, f.post),
                _try_mega_mhc=mock.Mock(return_value=None),
                _sync_after_first_cp_prefill_attention=lambda: None,
                ffn=mock.Mock(
                    side_effect=ffn,
                    gate=gate,
                    _should_chunk=mock.Mock(return_value=False),
                ),
            )
            with mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.modules.dsv4.fp8.attention": types.SimpleNamespace(
                        bind_attn_cache=bind
                    )
                },
            ):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "query failure"):
                        ns["forward_prefill_ced_l20"](
                            block, hidden, ids, common.position_ids, plan
                        )
                    block.ffn.assert_not_called()
                else:
                    result, result_ids = ns["forward_prefill_ced_l20"](
                        block, hidden, ids, common.position_ids, plan
                    )
                    self.assertEqual(result.shape, (32, 4, 3))
                    self.assertTrue(torch.equal(result_ids, ids[rows]))
                    self.assertTrue(
                        torch.equal(
                            block.ffn.call_args.args[0], norm(expected_ffn_readout)
                        )
                    )
                    self.assertEqual(f.pre_mix_out.shape, (32, 4))
                    self.assertEqual(
                        block._try_mega_mhc.call_args.kwargs,
                        {"original_tokens": plan.original_context.chunk_length},
                    )
            self.assertFalse(hasattr(gate, "_ced_row_projection"))
            self.assertEqual(events, ["bind", "producer", "query", "clear"])
            self.assertIs(attn._cp_ctx, plan.original_context)
            self.assertTrue(
                torch.equal(
                    hidden,
                    (
                        torch.arange(256 * 12).reshape(256, 4, 3).float() / 123
                    ).bfloat16(),
                )
            )


class BoundedSwaConsumerTest(_SingleThreadTest):
    def test_writer_keeps_slot_alignment_and_only_writes_replay_rows(self):
        path = Path(_CED.__file__).parent.parent / "fp8/attention_v41.py"
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "AttentionV41FP8"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_prefill_write_swa_fp8_paged"
        )
        codec = mock.Mock()
        namespace = {"torch": torch, "swa_codec": codec}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        from collections import namedtuple

        Compaction = namedtuple("Compaction", "unique_blocks compact_slots")
        owner = types.SimpleNamespace(
            head_dim=2,
            window_size=128,
            _swa_cp_byte_sliced=lambda: True,
            _pool_raw_u8=lambda region: object(),
            _swa_entries_per_block=lambda: 136,
        )
        namespace["SWA_KV"] = 7
        for length in (129, 512, 513, 32768):
            slots = torch.arange(length, dtype=torch.int64) + 1000
            compact_slots = torch.arange(length, dtype=torch.int64) + 2000
            blocks = torch.tensor([17, 31])
            meta = types.SimpleNamespace(
                slot_mapping=slots, slot_compaction=Compaction(blocks, compact_slots)
            )
            cp = types.SimpleNamespace(
                swa_replay_start=length - 128, cp_rank=2, cp_size=4
            )
            common = types.SimpleNamespace(cp_ctx=cp, swa_meta=meta)
            kv = torch.arange(256).reshape(128, 2).bfloat16()
            namespace[method.name](owner, common, kv)
            call = codec.quantize_and_insert_k_cache_cp_byte_sliced.call_args
            self.assertEqual(call.args[0].data_ptr(), kv.data_ptr())
            torch.testing.assert_close(call.args[0], kv, rtol=0, atol=0)
            self.assertEqual(
                call.args[2].tolist(), list(range(1000 + length - 128, 1000 + length))
            )
            actual = call.kwargs["compaction"]
            self.assertIs(actual.unique_blocks, blocks)
            self.assertEqual(
                actual.compact_slots.tolist(),
                list(range(2000 + length - 128, 2000 + length)),
            )
            self.assertIs(meta.slot_mapping, slots)
            self.assertIs(meta.slot_compaction.compact_slots, compact_slots)
            with self.assertRaisesRegex(ValueError, "exactly 128"):
                namespace[method.name](owner, common, kv[:-1])

    def fixture(self, *, length=385, prefix=16387, compact=True, bounded=True):
        selected = torch.arange(length - 128, length)
        if compact:
            plan, _ = _plan(length, 0, selected, prefix)
            ctx = replace(plan.context, swa_replay_start=length - 128)
        else:
            ctx, _ = _context(length, 0, prefix)
        common = types.SimpleNamespace(
            cp_ctx=ctx,
            cp_on=True,
            any_cont=prefix > 0,
            batch_size=1,
            prefix_lengths=ctx.prefix_lengths,
            req_id_per_token=ctx.req_id_per_token,
        )
        full = torch.full((length, 2), torch.nan, dtype=torch.float64)
        real_rows = 128 if compact else length
        full[-real_rows:] = (
            torch.arange(real_rows * 2, dtype=torch.float64).reshape(real_rows, 2) / 32
            + 1
        )
        owner = _swa_owner(bounded)
        qkv = types.SimpleNamespace(
            kv_full=full[-128:].clone() if compact else full,
            qr=torch.zeros(ctx.chunk_length, 2),
        )
        return owner, common, qkv

    def test_actual_workspace_keeps_compact_storage_and_absolute_origin(self):
        owner, common, qkv = self.fixture()
        buffers, starts = owner._swa_prefill_workspace(qkv, common)
        self.assertEqual(starts, [16387 + 385 - 128])
        self.assertEqual(buffers[0].shape, (128, 2))
        self.assertIs(buffers[0], qkv.kv_full)
        self.assertTrue(torch.isfinite(buffers[0]).all())
        owner._swa_prefill_concat.assert_not_called()

    def test_actual_sparse_indices_exclude_missing_history_and_keep_global_kv(self):
        owner, common, qkv = self.fixture()
        ctx = common.cp_ctx
        end, origin = ctx.seq_len_total, ctx.prefix_length + ctx.swa_replay_start
        global_kv = torch.full((end, 2), 2.0, dtype=torch.float64)
        selected = (
            torch.tensor([0, end // 2, -1], dtype=torch.int32)
            .expand(ctx.chunk_length, -1)
            .clone()
        )
        owner._shared_attention["global"] = {20: [(global_kv, None)]}
        # L20's cached plan is stale once the SWA domain changes at L21.
        owner._shared_attention["prefill_chunk_meta"] = object()
        owner._shared_attention["prefill_index_plan"] = object()
        owner._prefill_common_setup = mock.Mock(return_value=common)
        owner._prefill_compute_qkv = mock.Mock(return_value=qkv)
        owner._select_indices = mock.Mock(return_value=selected)
        marker = object()
        owner._prefill_sparse_attention.return_value = marker
        result = owner._forward_prefill(
            torch.zeros(ctx.chunk_length, 2), ctx.global_positions
        )
        self.assertIs(result, marker)
        launch = owner._prefill_sparse_attention.call_args.kwargs
        kv, indices, lengths = (
            launch["kv"].squeeze(1),
            launch["indices"].squeeze(1),
            launch["topk_length"],
        )
        self.assertEqual(kv.shape, (end + 128, 2))
        self.assertTrue(torch.equal(kv[:end], global_kv))
        self.assertEqual(indices.shape[1] % 64, 0)
        for row, position in enumerate(ctx.global_positions.tolist()):
            expected_swa = list(range(max(origin, position - 127), position + 1))
            expected = [0, end // 2] + [end + p - origin for p in expected_swa]
            self.assertEqual(lengths[row].item(), len(expected))
            self.assertEqual(indices[row, : len(expected)].tolist(), expected)
            self.assertTrue((indices[row, len(expected) :] == -1).all())
            self.assertTrue(
                torch.isfinite(kv[indices[row, : len(expected)].long()]).all()
            )
        # Writer receives compact KV and slices both original/compacted slots.
        self.assertIs(owner._prefill_write_swa_fp8_paged.call_args.args[1], qkv.kv_full)

    def test_noncompacted_bounded_path_never_reads_decoder_prefix_cache(self):
        for length in (128, 256):
            owner, common, qkv = self.fixture(length=length, compact=False)
            buffers, starts = owner._swa_prefill_workspace(qkv, common)
            self.assertEqual(starts, [common.cp_ctx.prefix_length])
            self.assertEqual(buffers[0].data_ptr(), qkv.kv_full.data_ptr())
            self.assertEqual(buffers[0].shape[0], length)
            owner._swa_prefill_concat.assert_not_called()
        owner, common, qkv = self.fixture(length=127, compact=False)
        with self.assertRaisesRegex(ValueError, "128 fresh"):
            owner._swa_prefill_workspace(qkv, common)
        for length in (127, 128):
            cold, common, qkv = self.fixture(length=length, prefix=0, compact=False)
            buffers, starts = cold._swa_prefill_workspace(qkv, common)
            self.assertEqual(starts, [0])
            self.assertEqual(len(buffers[0]), length)
            cold._swa_prefill_concat.assert_not_called()

    def test_disabled_workspace_keeps_cached_prefix_and_cold_paths(self):
        owner, common, qkv = self.fixture(compact=False, bounded=False)
        prefix = common.cp_ctx.prefix_length
        merged = torch.ones(1, 385 + 127, 2)
        owner._swa_prefill_concat = mock.Mock(return_value=merged)
        buffers, starts = owner._swa_prefill_workspace(qkv, common)
        self.assertEqual(starts, [prefix - 127])
        self.assertEqual(buffers[0].shape[0], 385 + 127)
        owner._swa_prefill_concat.assert_called_once_with(qkv, common)
        cold, common, qkv = self.fixture(prefix=0, compact=False, bounded=False)
        buffers, starts = cold._swa_prefill_workspace(qkv, common)
        self.assertEqual(starts, [0])
        self.assertEqual(buffers[0].data_ptr(), qkv.kv_full.data_ptr())
        cold._swa_prefill_concat.assert_not_called()

    def test_noncompacted_batched_chunk_offsets_use_fresh_lengths_only(self):
        owner = _swa_owner(True)
        common = types.SimpleNamespace(
            cp_ctx=None,
            cp_on=False,
            any_cont=True,
            batch_size=2,
            input_lengths=torch.tensor([128, 256]),
            prefix_lengths=torch.tensor([511, 1024]),
        )
        full = torch.arange(384 * 2).reshape(384, 2)
        swa, starts = owner._swa_prefill_workspace(
            types.SimpleNamespace(kv_full=full), common
        )
        self.assertEqual([len(rows) for rows in swa], [128, 256])
        self.assertEqual(starts, [511, 1024])
        globals_by_req = [(torch.zeros(639, 2), None), (torch.zeros(1280, 2), None)]
        requests = torch.tensor([1, 0, 1, 0])
        offsets, sizes, origins = owner._prefill_chunk_meta(
            globals_by_req, swa, starts, requests, full.device, common=common
        )
        self.assertEqual(offsets.flatten().tolist(), [767, 0, 767, 0])
        self.assertEqual(sizes.flatten().tolist(), [1280, 639, 1280, 639])
        self.assertEqual(origins.flatten().tolist(), [1024, 511, 1024, 511])
        owner._swa_prefill_concat.assert_not_called()


def _transport_cases():
    # A single selected row leaves three ranks with zero receives; [0..4]
    # leaves three ranks with zero sends. All still join every collective.
    yield 65, 0, torch.tensor([0]), False
    yield 65, 16387, torch.arange(5), False
    yield 65, 16387, torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 64]), False
    yield 65, 16387, torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 64]), True
    for length, prefix in ((32768, 0), (32769, 16387), (131072, 0)):
        attn, _ = _cache_fixture(length, prefix)
        selected = _CED.checkpoint_positions(
            prefix, length, attn.kv_cache_block_id_host[1, 0].tolist(), 512, 136
        )
        yield length, prefix, selected, False
    # Final128 transport geometry; create/marker semantics are tested above.
    for length, prefix, permuted in (
        (129, 0, False),
        (512, 0, False),
        (16384, 0, False),
        (16385, 0, False),
        (129, 16387, False),
        (257, 16387, False),
        (32769, 16387, False),
        (32769, 0, True),
        (131073, 0, False),
    ):
        yield length, prefix, torch.arange(length - 128, length), permuted


def _exercise_transport(rank, device, group):
    """Actual compact, reverse all-to-all, aux aliasing and KV scatter.

    Shared by the CPU/Gloo unittest and the standalone real-NCCL artifact.
    Only descriptors/model metadata are fake; payload collectives are real.
    """
    results = []
    for length, prefix, selected, permuted in _transport_cases():
        bounded = len(selected) == 128 and torch.equal(
            selected, torch.arange(length - 128, length)
        )
        plan, original_positions = _plan(
            length,
            rank,
            selected,
            prefix,
            group=group,
            device=device,
            permuted=permuted,
            candidate_publication=bounded,
        )
        model = _model()
        rows = original_positions[rank].to(device)
        padded = len(torch.cat(original_positions))
        positions = torch.arange(padded, device=device)
        # Mixed dtypes, arbitrary trailing dimensions, large IDs and sentinels.
        hidden = (positions % 127).bfloat16()[:, None, None] + torch.arange(
            12, device=device
        ).bfloat16().reshape(1, 3, 4)
        ids = positions + (1 << 33)
        pre_mix = torch.stack((positions.float(), -positions.float()), dim=1)
        topk = torch.stack(
            (positions.int(), -torch.ones_like(positions, dtype=torch.int32)), dim=1
        )
        # The real candidate producer skips pruning when all global blocks
        # fit its top-8 selection. None is a valid cross-layer shared value.
        candidates = (
            None
            if (prefix + length + 2047) // 2048 <= 8
            else (positions // 8).int()[:, None]
        )
        payloads = (hidden, ids, pre_mix, topk, candidates)
        model.layers[20].ffn_hc.pre_mix_out = pre_mix[rows]
        global_kv = object()
        shared = {
            "global": {20: global_kv},
            "topk": {20: topk[rows]},
            "candidates": candidates[rows] if candidates is not None else None,
        }
        if bounded and candidates is not None:
            ns = CedCandidatePublicationTest.selection_namespace()

            def publish(logits, values, *args, out, **kwargs):
                out.copy_(values[:, None].int())
                return out, None

            ns["prefill_candidates"] = types.SimpleNamespace(select_candidates=publish)
            shared["candidates"].fill_(-1)
            with plan.candidate_publication(shared):
                for start in range(0, rows.numel(), 512):
                    stop = min(start + 512, rows.numel())
                    ns["_apply_prefill_candidates"](
                        shared,
                        torch.empty((stop - start, 8), device=device),
                        candidates[rows[start:stop], 0],
                        slice(start, stop),
                        8,
                        1,
                        True,
                    )
            assert "ced_candidate_rows" not in shared
            expected_owner_rows = plan.exchange.candidate_rows_host
            if not expected_owner_rows:
                assert (shared["candidates"] == -1).all()
        for key in _CED._DERIVED_KEYS:
            shared[key] = object()
        actual_h, actual_ids = plan.compact(model, hidden[rows], ids[rows], shared)
        actual = (
            actual_h,
            actual_ids,
            model.layers[20].ffn_hc.pre_mix_out,
            shared["topk"][20],
            shared["candidates"],
        )
        for value, full in zip(actual, payloads):
            if full is None:
                assert value is None
                continue
            torch.testing.assert_close(
                value, _compact_oracle(full, selected, rank), rtol=0, atol=0
            )
        assert shared["global"][20] is global_kv
        assert not any(key in shared for key in _CED._DERIVED_KEYS)
        assert shared["ced_indexer_projection"].__self__ is plan
        try:
            plan.compact(model, hidden[rows], ids[rows], shared)
            raise AssertionError("plan reuse accepted")
        except ValueError:
            pass
        selected_device = selected.to(device)
        expected_h = torch.zeros_like(hidden)
        expected_h[selected_device] = hidden[selected_device]
        restored_h = plan.restore_rows(actual_h)
        torch.testing.assert_close(restored_h, expected_h[rows], rtol=0, atol=0)

        # All three target captures occupy distinct columns, with aliased
        # compact input/full output and spare capacity after the output view.
        aux_full = torch.stack(
            tuple((positions % 41 + 50 * i).bfloat16() for i in range(3)), dim=1
        )
        compact_aux = _compact_oracle(aux_full, selected, rank)
        model._mtp_hidden_buffer = torch.full(
            (plan.original_context.chunk_length + 3, 3),
            -7,
            dtype=torch.bfloat16,
            device=device,
        )
        model._mtp_hidden_buffer[: plan.context.chunk_length].copy_(compact_aux)
        address = model._mtp_hidden_buffer.data_ptr()
        plan.restore_aux(model)
        expected_aux = torch.zeros_like(aux_full)
        expected_aux[selected_device] = aux_full[selected_device]
        torch.testing.assert_close(
            model._mtp_hidden_buffer[: len(rows)], expected_aux[rows], rtol=0, atol=0
        )
        assert model._mtp_hidden_buffer.data_ptr() == address
        assert (model._mtp_hidden_buffer[len(rows) :] == -7).all()
        model._note_aux_hidden_rows.assert_called_once_with(
            len(rows), is_cuda_graph=False
        )
        try:
            plan.restore_aux(model)
            raise AssertionError("aux double restore accepted")
        except ValueError:
            pass

        compact_kv = actual_h.flatten(1).contiguous()
        gathered = compact_kv.new_empty((4 * compact_kv.shape[0], compact_kv.shape[1]))
        torch.distributed.all_gather_into_tensor(gathered, compact_kv, group=group)
        out = hidden.new_full((length, 12), -999)
        restored_kv = _CP._cp_restore_gathered_full_2d(gathered, plan.context, out=out)
        assert restored_kv is out
        torch.testing.assert_close(
            restored_kv, expected_h[:length].flatten(1), rtol=0, atol=0
        )
        if len(selected) == 128 and torch.equal(
            selected, torch.arange(length - 128, length)
        ):
            replay_ctx = replace(plan.context, swa_replay_start=length - 128)
            with mock.patch.object(_COLLECTIVE, "_get_group", return_value=group):
                compact_result = _CP.cp_all_gather_full_varlen(
                    compact_kv, replay_ctx, replay_only=True
                )
            torch.testing.assert_close(
                compact_result, hidden[selected_device].flatten(1), rtol=0, atol=0
            )
            assert compact_result.shape == (128, 12)
            assert (
                compact_result.untyped_storage().nbytes()
                == compact_result.numel() * compact_result.element_size()
            )
        # Detect a skipped zero-count participant or mismatched collective order.
        heartbeat = torch.tensor([rank + 1], device=device)
        torch.distributed.all_reduce(heartbeat, group=group)
        assert heartbeat.item() == 10
        results.append(
            dict(
                length=length,
                prefix=prefix,
                selected=len(selected),
                permuted=permuted,
                send_sizes=plan.exchange.send_sizes,
                receive_sizes=plan.exchange.receive_sizes,
            )
        )
    return results


def _exercise_l20_transport(rank, device, group):
    """Exercise the actual in-Block transition, including complex RoPE rows."""
    for length, prefix, permuted in (
        (129, 0, False),
        (129, 16387, False),
        (32769, 16387, True),
        (131073, 0, False),
    ):
        selected = torch.arange(length - 128, length)
        plan, original_positions = _plan(
            length,
            rank,
            selected,
            prefix,
            group=group,
            device=device,
            permuted=permuted,
            candidate_publication=True,
        )
        positions = torch.arange(sum(map(len, original_positions)), device=device)
        rows = original_positions[rank].to(device)
        hidden = (positions % 47).bfloat16()[:, None, None].expand(-1, 4, 3).clone()
        post = torch.stack((positions.float(), -positions.float()), -1)
        comb = post[:, :, None].expand(-1, -1, 2).clone()
        mix = post + 7
        ids = positions + (1 << 33)
        freqs = torch.complex(positions.float(), -positions.float())[:, None]
        global_kv, global_keys = object(), object()
        shared = {
            "global": {20: [(global_kv, global_keys)]},
            "topk": {},
            "candidates": None,
        }
        for key in _CED._DERIVED_KEYS:
            shared[key] = object()
        block = types.SimpleNamespace(
            layer_id=20,
            attn=types.SimpleNamespace(_shared_attention=shared),
            attn_hc=types.SimpleNamespace(pre_mix_out=mix[rows]),
            ffn_hc=types.SimpleNamespace(pre_mix_out=None),
        )
        actual = plan.compact_l20(
            block, hidden[rows], post[rows], comb[rows], ids[rows]
        )
        for compact, full in zip(actual, (hidden, post, comb, ids)):
            torch.testing.assert_close(
                compact, _compact_oracle(full, selected, rank), rtol=0, atol=0
            )
        torch.testing.assert_close(
            block.attn_hc.pre_mix_out,
            _compact_oracle(mix, selected, rank),
            rtol=0,
            atol=0,
        )
        assert block.ffn_hc.pre_mix_out is None
        assert shared["global"][20][0] == (global_kv, global_keys)
        assert not any(key in shared for key in _CED._DERIVED_KEYS)
        common = _l20_common(plan.original_context, freqs[rows])
        query = plan.l20_query_metadata(common)
        torch.testing.assert_close(
            query.freqs_cis, _compact_oracle(freqs, selected, rank), rtol=0, atol=0
        )
        assert query.cp_ctx.swa_replay_start is None
        assert plan.context.swa_replay_start == length - 128
        assert query.prefix_lengths is common.prefix_lengths
        assert query.seqlen_full == length
        assert query.cp_ctx.input_lengths_global_host == (length,)
        assert query.workspace is common.workspace
        assert query.swa_meta is None
        assert query.request_row_slices == (slice(0, 32),)
        assert (
            query.cp_ctx.global_positions.tolist()
            == _compact_oracle(positions + prefix, selected, rank).tolist()
        )
        restored = plan.restore_rows(actual[0])
        expected = torch.zeros_like(hidden)
        expected[selected.to(device)] = hidden[selected.to(device)]
        torch.testing.assert_close(restored, expected[rows], rtol=0, atol=0)
        try:
            plan.compact_l20(block, *actual)
            raise AssertionError("double L20 transition accepted")
        except ValueError:
            pass
        heartbeat = torch.tensor([rank + 1], device=device)
        torch.distributed.all_reduce(heartbeat, group=group)
        assert heartbeat.item() == 10


def _gloo_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        _exercise_transport(rank, torch.device("cpu"), torch.distributed.group.WORLD)
        _exercise_l20_transport(
            rank, torch.device("cpu"), torch.distributed.group.WORLD
        )
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_gloo_available(),
    "Gloo unavailable",
)
class CedGlooTransportTest(unittest.TestCase):
    def test_four_process_roundtrip_including_zero_participants(self):
        with tempfile.TemporaryDirectory(prefix="ced-gloo-") as directory:
            ctx = torch.multiprocessing.spawn(
                _gloo_worker,
                args=("file://" + directory + "/store",),
                nprocs=4,
                join=False,
            )
            deadline = time.monotonic() + 90
            try:
                while not ctx.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("CED Gloo transport exceeded 90 seconds")
            finally:
                # Only children owned by this test; never touch external jobs.
                for process in ctx.processes:
                    if process.is_alive():
                        process.terminate()
                for process in ctx.processes:
                    process.join(timeout=5)


def _toy_decoder(encoder, window, layers, start=0):
    """Uniform causal SWA plus query-dependent attention to full encoder KV."""
    hidden = encoder[start:].clone()
    positions = torch.arange(start, encoder.shape[0])
    visible = torch.arange(encoder.shape[0])[None, :] <= positions[:, None]
    counts = torch.arange(1, hidden.shape[0] + 1).clamp(max=window)[:, None]
    outputs = []
    for _ in range(layers):
        # Each query still sees the entire causal encoder KV, even in tail mode.
        scores = hidden @ encoder.T / encoder.shape[1] ** 0.5
        global_out = scores.masked_fill(~visible, -torch.inf).softmax(-1) @ encoder
        local_out = (
            F.pad(hidden, (0, 0, window - 1, 0)).unfold(0, window, 1).sum(-1) / counts
        )
        hidden = 0.6 * hidden + 0.3 * local_out + 0.1 * torch.tanh(global_out)
        outputs.append(hidden)
    return outputs


def _explicit_truncated_swa_decoder(
    encoder, window, layers, start, *, include_zero_holes=False
):
    """Independent scalar-row oracle: enumerate only materialized SWA keys.

    The adversarial option models the incorrect zero-filled full-KV domain.
    Global encoder keys remain fully visible up to each absolute query row.
    """
    state = {row: encoder[row].clone() for row in range(start, len(encoder))}
    zero = torch.zeros_like(encoder[0])
    outputs = []
    for _ in range(layers):
        following = {}
        for row, query in state.items():
            global_keys = encoder[: row + 1]
            scores = global_keys @ query / encoder.shape[1] ** 0.5
            global_out = scores.softmax(0) @ global_keys
            left = max(0 if include_zero_holes else start, row - window + 1)
            local_keys = torch.stack(
                [state.get(key, zero) for key in range(left, row + 1)]
            )
            following[row] = (
                0.6 * query + 0.3 * local_keys.mean(0) + 0.1 * global_out.tanh()
            )
        state = following
        outputs.append(torch.stack(list(state.values())))
    return outputs


class BoundedReplayApproximationTest(_SingleThreadTest):
    def test_actual_dspark_indices_read_only_live128_and_current_queries(self):
        path = Path(_CP.__file__).parents[2] / "model_desc/deepseek_v4_dspark_model.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DeepSeekV4DSparkModel"
        )
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_global_pool_slots", "_build_noncausal_indices")
        ]
        namespace = {"torch": torch, "Tuple": tuple}
        exec(
            compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        owner = types.SimpleNamespace(
            _gen_num_per_cycle=7,
            _v4_args=types.SimpleNamespace(window_size=128),
            _global_pool_slots=namespace["_global_pool_slots"].__func__,
        )
        build = types.MethodType(namespace["_build_noncausal_indices"], owner)
        for end in (131072, 131073, 131199, 131200):
            with self.subTest(end=end):
                table = torch.zeros(1, 260, dtype=torch.int32)
                table[0, 255:258] = torch.tensor([13, 37, 5])
                indices, lengths = build(
                    torch.tensor([end]), torch.tensor([True]), table, 136, 512
                )
                expected = [
                    int(table[0, p // 512]) * 136 + p % 136
                    for p in range(end - 128, end + 7)
                ]
                self.assertEqual(lengths.tolist(), [135])
                self.assertEqual(indices.shape, (7, 256))
                expected_tensor = torch.tensor(expected, dtype=torch.int32)
                self.assertTrue(
                    torch.equal(indices[:, :135], expected_tensor.expand(7, -1))
                )
                self.assertTrue((indices[:, 135:] == -1).all())
                # All physical cells except actual context/query writes remain
                # poisoned. Do not assert equality of unused ring slack.
                pool = torch.full((38 * 136,), torch.nan)
                pool[expected_tensor.long()] = torch.arange(135).float()
                self.assertTrue(torch.isfinite(pool[indices[:, :135].long()]).all())

    def test_final128_matches_explicit_truncated_oracle_not_full_decoder(self):
        encoder = torch.zeros(384, 2, dtype=torch.float64)
        encoder[:256, 0] = 1
        encoder[256:, 1] = 0.25
        full = _toy_decoder(encoder, 128, 19)
        replay = _toy_decoder(encoder, 128, 19, start=256)
        explicit = _explicit_truncated_swa_decoder(encoder, 128, 19, 256)
        for got, expected in zip(replay, explicit):
            torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
        self.assertGreater((replay[-1][-1] - full[-1][-1]).abs().max().item(), 1e-3)
        # All three target captures are approximate; layout equality does not
        # imply equality with the full model or with exact CED.
        for got, expected in zip(replay[-3:], full[-3:]):
            self.assertGreater((got - expected[-128:]).abs().max().item(), 1e-3)

    def test_zero_filled_omitted_history_is_not_a_masked_key(self):
        encoder = torch.ones(384, 2, dtype=torch.float64)
        replay = _explicit_truncated_swa_decoder(encoder, 128, 19, 256)
        contaminated = _explicit_truncated_swa_decoder(
            encoder, 128, 19, 256, include_zero_holes=True
        )
        # Even zero-valued omitted keys change the attention denominator.
        self.assertGreater((replay[0][0] - contaminated[0][0]).abs().max().item(), 0.1)
        self.assertGreater(
            (replay[-1][-1] - contaminated[-1][-1]).abs().max().item(), 1e-3
        )

    def test_live128_survives_two_block_crossings_and_speculative_rollback(self):
        window, span, ring, gamma = 128, 512, 136, 7

        def write(cache, positions):
            for position in positions:
                cache[(position // span, position % ring)] = position

        def check(cache, positions):
            for position in positions:
                self.assertEqual(
                    cache.get((position // span, position % ring)), position
                )

        for residue in (0, 1, 7, 127, 128, 135, 136, 255, 505, 511):
            end = 131072 + residue
            initial = {}
            write(initial, range(end - window, end))
            draft, target = initial.copy(), initial.copy()
            write(draft, range(end, end + gamma))
            check(draft, range(end - window, end + gamma))
            write(target, range(end, end + gamma + 1))
            for query in range(end, end + gamma + 1):
                check(target, range(query - window + 1, query + 1))
            write(draft, range(end, end + gamma + 1))
            for advance in range(1, gamma + 2):
                with self.subTest(residue=residue, accepted_advance=advance):
                    committed = end + advance
                    next_draft = draft.copy()
                    write(next_draft, range(committed, committed + gamma))
                    check(next_draft, range(committed - window, committed + gamma))
        # Replay length 128 does NOT permit shrinking physical ring capacity.
        end, bad_ring = 4096 + 256, 128
        slots = {p % bad_ring: p for p in range(end - window, end)}
        slots.update({p % bad_ring: p for p in range(end, end + gamma)})
        self.assertNotEqual(slots[(end - window) % bad_ring], end - window)


class CedDependencyConeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_full_and_tail_match_inside_cone_including_aux_windows(self):
        generator = torch.Generator().manual_seed(41)
        for window, layers in ((3, 4), (5, 7)):
            with self.subTest(window=window, layers=layers):
                keep = window + layers * (window - 1)
                encoder = torch.randn(keep + 37, 3, generator=generator).double()
                start = encoder.shape[0] - keep
                full = _toy_decoder(encoder, window, layers)
                tail = _toy_decoder(encoder, window, layers, start)
                for depth, (expected, actual) in enumerate(zip(full, tail), 1):
                    halo = depth * (window - 1)
                    torch.testing.assert_close(
                        actual[halo:], expected[start + halo :], rtol=1e-12, atol=1e-12
                    )
                # Three target captures feed parallel draft KV projections;
                # all three must describe the same absolute last-W positions.
                expected_aux = torch.cat([h[-window:] for h in full[-3:]], dim=1)
                actual_aux = torch.cat([h[-window:] for h in tail[-3:]], dim=1)
                torch.testing.assert_close(
                    actual_aux, expected_aux, rtol=1e-12, atol=1e-12
                )
                projection = torch.randn(9, 4, generator=generator).double()
                torch.testing.assert_close(
                    actual_aux @ projection,
                    expected_aux @ projection,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_one_less_halo_row_corrupts_required_aux_boundary(self):
        window, layers = 5, 4
        keep = window + layers * (window - 1)
        # Place an impulse at the earliest required ancestor of the first aux
        # row. A shortened tail loses it; a sufficient tail must preserve it.
        encoder = torch.zeros(64, 2, dtype=torch.float64)
        encoder[-keep, 0] = 1
        full = _toy_decoder(encoder, window, layers)[-1]
        enough = _toy_decoder(encoder, window, layers, 64 - keep)[-1]
        short = _toy_decoder(encoder, window, layers, 64 - keep + 1)[-1]
        torch.testing.assert_close(enough[-window:], full[-window:], rtol=0, atol=1e-14)
        self.assertGreater((short[-window] - full[-window]).abs().max().item(), 1e-6)

    def test_128_rows_only_is_not_exact_for_19_decoder_layers(self):
        encoder = torch.zeros(384, 2, dtype=torch.float64)
        encoder[:256, 0] = 1
        encoder[256:, 1] = 0.25
        full = _toy_decoder(encoder, window=128, layers=19)
        replay = _toy_decoder(encoder, window=128, layers=19, start=256)
        # Keeping full global KV does not recover decoder-side SWA history.
        self.assertGreater((full[-1][-1] - replay[-1][-1]).abs().max().item(), 1e-3)
        for expected, actual in zip(full[-3:], replay[-3:]):
            self.assertGreater((expected[-128:] - actual).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
