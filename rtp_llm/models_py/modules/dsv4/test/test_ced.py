"""CPU contracts for CED tail execution and its finite dependency cone.

The toy decoder below is an independent numerical oracle: global KV is frozen
encoder state, while only local SWA propagates dependencies between tokens.
It also demonstrates why replaying just one SWA window is not exact.
"""

import importlib
import sys
import types
import unittest
from contextlib import ExitStack
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


class CedTransportTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.rank = 0
        self.peer_rejects = False
        self.broadcasts = []
        self.broadcast_index = 0
        self.gather_payloads = None
        # Keep real CPU tensors and real metadata validation. Only simulate
        # CUDA availability and transport; a real GPU call is an error.
        patches = (
            mock.patch.object(torch.Tensor, "is_cuda", new=property(lambda t: True)),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(torch.distributed, "get_world_size", return_value=4),
            mock.patch.object(
                torch.distributed, "get_rank", side_effect=lambda g: self.rank
            ),
            mock.patch.object(torch.distributed, "get_global_rank", return_value=8),
            mock.patch.object(
                torch.distributed, "all_gather_into_tensor", side_effect=self._gather
            ),
            mock.patch.object(_COLLECTIVE, "broadcast", side_effect=self._broadcast),
            mock.patch.object(
                torch.cuda, "_lazy_init", side_effect=AssertionError("GPU call")
            ),
        )
        for patch in patches:
            self.stack.enter_context(patch)

    def _gather(self, output, local, group):
        if local.shape[0] == 1:
            output.copy_(local.expand_as(output))
            if self.peer_rejects:
                output[-1, 0] = 0
        else:
            self.assertIsNotNone(self.gather_payloads)
            torch.testing.assert_close(local, self.gather_payloads[self.rank])
            output.copy_(torch.cat(self.gather_payloads))

    def _broadcast(self, tensor, source, group):
        self.assertEqual(source, 8, "broadcast must use WORLD rank of TP rank 0")
        if self.rank == 0:
            self.broadcasts.append(tensor)
        else:
            tensor.copy_(self.broadcasts[self.broadcast_index])
            self.broadcast_index += 1

    def fixture(self, length=32768, rank=0):
        self.rank = rank
        positions, info = _cp4_layout(length)
        context = _CP.build_cp_context(
            info,
            cp_size=4,
            cp_rank=rank,
            chunk_length=len(positions[rank]),
            device=torch.device("cpu"),
            kv_cache_sharded=True,
        )
        attn = types.SimpleNamespace(
            input_lengths=torch.tensor([context.chunk_length], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
            is_prefill=True,
        )
        return _model(), context, attn

    def test_context_cp4_absolute_positions_and_padding(self):
        for length in (32768, 32769, 131072):
            expected_positions, expected_info = _cp4_layout(3072)
            actual_positions = []
            for rank in range(4):
                with self.subTest(length=length, rank=rank):
                    model, original, attn = self.fixture(length, rank)
                    original._full_prefill_positions_cache = (object(),) * 4
                    plan = _CED.CEDTail.create(model, original, attn)
                    self.assertIsNotNone(plan)
                    ctx = plan.context
                    self.assertIs(plan.original_context, original)
                    self.assertIsNot(ctx.cp_info, original.cp_info)
                    self.assertIsNone(ctx._full_prefill_positions_cache)
                    self.assertIsNotNone(original._full_prefill_positions_cache)
                    self.assertEqual(ctx.chunk_length, 768)
                    self.assertEqual(ctx.seq_len_full, 3072)
                    self.assertEqual(ctx.padded_seq_len, 3072)
                    self.assertEqual(ctx.seq_len_total, length)
                    self.assertEqual(ctx.prefix_length, length - 3072)
                    self.assertEqual(ctx.input_lengths_global_host, (3072,))
                    self.assertEqual(ctx.prefix_lengths_host, (length - 3072,))
                    self.assertEqual(ctx.chunk_lengths_per_req, (768,))
                    self.assertTrue(ctx.local_is_real.all())
                    self.assertFalse(ctx.req_id_per_token.any())
                    self.assertEqual(ctx.input_lengths_global.tolist(), [3072])
                    self.assertEqual(ctx.cu_seqlens_global.tolist(), [0, 3072])
                    self.assertEqual(ctx.prefix_lengths.tolist(), [length - 3072])
                    self.assertEqual(
                        ctx.cp_info.prefill_actual_input_lengths_cpu.tolist(), [3072]
                    )
                    self.assertEqual(
                        ctx.cp_info.prefill_prefix_lengths_cpu.tolist(), [length - 3072]
                    )
                    self.assertEqual(
                        ctx.cp_info.prefill_cp_padding_lengths.tolist(), [0]
                    )
                    torch.testing.assert_close(
                        ctx.relative_positions, expected_positions[rank]
                    )
                    torch.testing.assert_close(
                        ctx.global_positions, expected_positions[rank] + length - 3072
                    )
                    torch.testing.assert_close(
                        ctx.unpad_restore,
                        expected_info.prefill_qkv_restore_indice.long(),
                    )
                    torch.testing.assert_close(
                        plan.cu_seqlens, torch.tensor([0, 768], dtype=torch.int32)
                    )
                    self.assertEqual(original.prefix_length, 0)
                    actual_positions.append(ctx.global_positions)
            self.assertEqual(
                torch.cat(actual_positions).sort().values.tolist(),
                list(range(length - 3072, length)),
            )

    def test_rejects_unsafe_metadata_and_shapes_before_transport(self):
        changes = (
            ("short", lambda m, c, a: setattr(c, "seq_len_full", 16384)),
            ("prefix_host", lambda m, c, a: setattr(c, "prefix_lengths_host", (1,))),
            ("prefix_device", lambda m, c, a: a.prefix_lengths.fill_(1)),
            (
                "multiple_requests",
                lambda m, c, a: setattr(c, "input_lengths_global_host", (16384, 16384)),
            ),
            ("decode", lambda m, c, a: setattr(a, "is_prefill", False)),
            ("verify", lambda m, c, a: setattr(a, "is_target_verify", True)),
            ("graph", lambda m, c, a: setattr(a, "is_cuda_graph", True)),
            ("cache_store", lambda m, c, a: setattr(a, "cache_store_inputs", object())),
            ("unsharded", lambda m, c, a: setattr(c, "kv_cache_sharded", False)),
            (
                "position_shape",
                lambda m, c, a: setattr(
                    c, "relative_positions", c.relative_positions[:-1]
                ),
            ),
            ("wrong_tail_owner", lambda m, c, a: c.unpad_restore.__setitem__(-1, 9000)),
            (
                "wrong_absolute_position",
                lambda m, c, a: c.global_positions.__setitem__(-1, 0),
            ),
            ("window", lambda m, c, a: setattr(m.args, "window_size", 256)),
            (
                "wrong_kv_source",
                lambda m, c, a: setattr(m.layers[21].attn, "kv_source_layer_id", 21),
            ),
        )
        for name, change in changes:
            with self.subTest(name=name):
                model, context, attn = self.fixture()
                change(model, context, attn)
                self.assertIsNone(_CED.CEDTail.create(model, context, attn))
        model, context, attn = self.fixture()
        self.assertIsNone(
            _CED.CEDTail.create(model, context, attn, prepare_hidden_fn=lambda: None)
        )
        self.assertIsNone(
            _CED.CEDTail.create(model, context, attn, cache_store_active=True)
        )
        self.peer_rejects = True
        self.assertIsNone(_CED.CEDTail.create(model, context, attn))
        self.assertFalse(self.broadcasts)

    def test_local_request_may_keep_native_cache_store_metadata(self):
        model, context, attn = self.fixture()
        store_inputs = object()
        attn.cache_store_inputs = store_inputs
        self.assertIsNone(
            _CED.CEDTail.create(model, context, attn, cache_store_active=True)
        )
        self.assertIsNotNone(
            _CED.CEDTail.create(
                model,
                context,
                attn,
                cache_store_active=True,
                allow_local_cache_store=True,
            )
        )
        self.assertIs(attn.cache_store_inputs, store_inputs)

    def compact_inputs(self, model, context):
        positions = context.relative_positions
        # Expanded input views avoid allocating full-prompt hidden/selector
        # matrices, while exercising the production dimensions and dtypes.
        hidden = (positions % 251).bfloat16()[:, None, None].expand(-1, 4, 5120)
        model.layers[20].ffn_hc.pre_mix_out = positions.float()[:, None].expand(-1, 4)
        shared = {
            "global": {
                20: [
                    (
                        torch.zeros(1, 512).expand(context.seq_len_full, -1),
                        range(context.seq_len_full),
                    )
                ]
            },
            "topk": {20: positions.int()[:, None].expand(-1, 512)},
            "candidates": (positions // 8).int()[:, None].expand(-1, 2048),
            "prefill_sparse_plans": object(),
            "prefill_chunk_meta": object(),
            "candidate_mask": object(),
            "prefill_meta_common": object(),
        }
        model.layers[20].attn._shared_attention = shared
        return hidden, positions.clone(), shared

    def test_compact_cp4_preserves_global_selection_ids_and_delayed_mix(self):
        length = 32769
        expected_ranks, _ = _cp4_layout(3072)
        for rank in range(4):
            with self.subTest(rank=rank):
                model, original, attn = self.fixture(length, rank)
                plan = _CED.CEDTail.create(model, original, attn)
                hidden, ids, shared = self.compact_inputs(model, original)
                full_global = shared["global"][20]
                self.broadcast_index = 0
                actual_hidden, actual_ids = plan.compact(model, hidden, ids, shared)
                expected = expected_ranks[rank] + length - 3072
                torch.testing.assert_close(actual_ids, expected)
                torch.testing.assert_close(
                    actual_hidden,
                    (expected % 251).bfloat16()[:, None, None].expand(-1, 4, 5120),
                )
                torch.testing.assert_close(
                    model.layers[20].ffn_hc.pre_mix_out,
                    expected.float()[:, None].expand(-1, 4),
                )
                torch.testing.assert_close(
                    shared["topk"][20], expected.int()[:, None].expand(-1, 512)
                )
                torch.testing.assert_close(
                    shared["candidates"],
                    (expected // 8).int()[:, None].expand(-1, 2048),
                )
                self.assertIs(shared["global"][20], full_global)
                for key in (
                    "prefill_sparse_plans",
                    "prefill_chunk_meta",
                    "candidate_mask",
                    "prefill_meta_common",
                ):
                    self.assertNotIn(key, shared)
                with self.assertRaisesRegex(ValueError, "geometry"):
                    plan.compact(model, hidden, ids, shared)

    def test_bad_compact_shape_rejected_without_mutation(self):
        model, context, attn = self.fixture()
        plan = _CED.CEDTail.create(model, context, attn)
        hidden, ids, shared = self.compact_inputs(model, context)
        old_mix = model.layers[20].ffn_hc.pre_mix_out
        old_topk = shared["topk"]
        with self.assertRaisesRegex(ValueError, "geometry"):
            plan.compact(model, hidden[:-1], ids, shared)
        self.assertIs(model.layers[20].ffn_hc.pre_mix_out, old_mix)
        self.assertIs(shared["topk"], old_topk)
        self.assertFalse(self.broadcasts)

    def test_restore_rows_returns_original_cp_layout_including_padding(self):
        length = 32769
        tail_ranks, _ = _cp4_layout(3072)
        original_ranks, _ = _cp4_layout(length)
        values = torch.stack((torch.arange(length), -torch.arange(length)), dim=1)
        self.gather_payloads = [values[rows + length - 3072] for rows in tail_ranks]
        full = torch.zeros((len(torch.cat(original_ranks)), 2), dtype=torch.int64)
        full[length - 3072 : length] = values[-3072:]
        for rank in range(4):
            with self.subTest(rank=rank):
                model, context, attn = self.fixture(length, rank)
                plan = _CED.CEDTail.create(model, context, attn)
                with self.assertRaises(ValueError):
                    plan.restore_rows(self.gather_payloads[rank])
                plan._compacted = True
                actual = plan.restore_rows(self.gather_payloads[rank])
                torch.testing.assert_close(actual, full[original_ranks[rank]])
                with self.assertRaises(ValueError):
                    plan.restore_rows(self.gather_payloads[rank][:-1])

    def test_aux_restores_common_capture_rows_and_keeps_buffer_address(self):
        length = 32769
        model, context, attn = self.fixture(length)
        plan = _CED.CEDTail.create(model, context, attn)
        plan._compacted = True
        tail_ranks, _ = _cp4_layout(3072)
        captures = []
        for rows in tail_ranks:
            pos = rows + length - 3072
            captures.append(
                torch.cat(
                    [
                        (pos % 41 + layer * 50).bfloat16()[:, None].expand(-1, 5120)
                        for layer in range(3)
                    ],
                    dim=1,
                )
            )
        self.gather_payloads = captures
        model._mtp_hidden_buffer = torch.full(
            (context.chunk_length + 1, 15360), -7, dtype=torch.bfloat16
        )
        model._mtp_hidden_buffer[:768] = captures[0]
        address = model._mtp_hidden_buffer.data_ptr()
        plan.restore_aux(model)
        self.assertEqual(model._mtp_hidden_buffer.data_ptr(), address)
        model._note_aux_hidden_rows.assert_called_once_with(
            context.chunk_length, is_cuda_graph=False
        )
        original_positions, _ = _cp4_layout(length)
        final_rows = torch.nonzero(
            (original_positions[0] >= length - 128) & (original_positions[0] < length)
        ).flatten()
        expected_positions = original_positions[0][final_rows]
        for layer in range(3):
            torch.testing.assert_close(
                model._mtp_hidden_buffer[final_rows, layer * 5120],
                (expected_positions % 41 + layer * 50).bfloat16(),
            )
        self.assertTrue((model._mtp_hidden_buffer[context.chunk_length] == -7).all())
        self.assertFalse(
            model._mtp_hidden_buffer[
                context.chunk_length - 7 : context.chunk_length
            ].any()
        )
        with self.assertRaises(ValueError):
            plan.restore_aux(model)


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
