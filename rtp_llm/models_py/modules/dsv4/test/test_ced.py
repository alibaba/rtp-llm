"""CPU/Gloo contracts for checkpoint-aware CED and its dependency cone.

The toy decoder below is an independent numerical oracle: global KV is frozen
encoder state, while only local SWA propagates dependencies between tokens.
It also demonstrates why replaying just one SWA window is not exact.
"""

import importlib
import sys
import tempfile
import time
import types
import unittest
from contextlib import ExitStack
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
    return cp, ced, collective


_CP, _CED, _COLLECTIVE = _load_helpers()


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
            }
        )
    return ctx, positions


def _plan(
    length, rank, selected, prefix=0, *, group=None, device="cpu", permuted=False
):
    original, positions = _context(
        length, rank, prefix, device=device, permuted=permuted
    )
    context, exchange, groups = _CED._query_layout(original, selected, group)
    plan = _CED.CEDPlan(
        original,
        context,
        torch.tensor([0, context.chunk_length], dtype=torch.int32, device=device),
        exchange,
        groups,
        (37, 38, 39),
    )
    return plan, positions


def _cache_fixture(length, prefix=0, *, ring=136, dtype=torch.uint8):
    columns = (prefix + length + 511) // 512 + 4
    ids = torch.zeros((2, 1, columns), dtype=torch.int32)
    # Group 0 is deliberately dense and is NOT SWA. Only actual SWA IDs count.
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
        group_region_names=(int(_CED.SWA_KV) + 17, _CED.SWA_KV),
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


def _exercise_transport(rank, device, group):
    """Actual compact, reverse all-to-all, aux aliasing and KV scatter.

    Shared by the CPU/Gloo unittest and the standalone real-NCCL artifact.
    Only descriptors/model metadata are fake; payload collectives are real.
    """
    results = []
    for length, prefix, selected, permuted in _transport_cases():
        plan, original_positions = _plan(
            length,
            rank,
            selected,
            prefix,
            group=group,
            device=device,
            permuted=permuted,
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
        candidates = (positions // 8).int()[:, None]
        payloads = (hidden, ids, pre_mix, topk, candidates)
        model.layers[20].ffn_hc.pre_mix_out = pre_mix[rows]
        global_kv = object()
        shared = {
            "global": {20: global_kv},
            "topk": {20: topk[rows]},
            "candidates": candidates[rows],
        }
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
