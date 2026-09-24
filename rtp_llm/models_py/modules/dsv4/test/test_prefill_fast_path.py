import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.prefill import forward as prefill_forward
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


class _FakeAttention(AttentionFP8):
    pass


class _FakeLayer(Block):
    def __init__(self, layer_id, calls):
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.calls = calls
        object.__setattr__(self, "attn", _FakeAttention.__new__(_FakeAttention))

    def prefill_fast_callable(self):
        return self.forward_prefill_fast

    def forward_prefill_fast(
        self,
        h,
        input_ids,
        positions,
        cu_seqlens,
        kv_cache=None,
        block_tables_by_type=None,
    ):
        self.calls.append(
            (
                "fast",
                self.layer_id,
                h.clone(),
                input_ids,
                positions,
                cu_seqlens,
                kv_cache,
                block_tables_by_type,
            )
        )
        return h + (self.layer_id + 1)

    def forward(
        self,
        h,
        input_ids,
        positions,
        cu_seqlens,
        kv_cache=None,
        block_tables_by_type=None,
    ):
        self.calls.append(
            (
                "normal",
                self.layer_id,
                h.clone(),
                input_ids,
                positions,
                cu_seqlens,
                kv_cache,
                block_tables_by_type,
            )
        )
        return h + (self.layer_id + 10)


class _FakeV4:
    def __init__(self):
        self.calls = []
        self.fp8_kv_cache = True
        self.hc_mult = 1
        self.layers = [_FakeLayer(0, self.calls), _FakeLayer(1, self.calls)]
        self._cp_info = None
        self._cp_size = 1
        self._cp_rank = 0
        self._kv_cache_sharded = False
        self._prefill_ws_q_rows = 0
        self._prefill_ws_q_dim = 0
        self._prefill_ws_full_rows = 0
        self._prefill_ws_main_w = 0
        self._prefill_ws_idx_w = 0
        self._mtp_hidden_buffer = None
        self._mtp_last_hidden_buffer = None
        self.capture_aux_hidden_layer_ids = ()
        self.norm = lambda h: h + 100

    def _propagate_cp_ctx(self, cp_ctx):
        self.cp_ctx = cp_ctx

    def embed(self, input_ids):
        return torch.stack((input_ids.float(), input_ids.float() + 0.5), dim=-1)

    def _hc_head_reduce(self, h):
        self.calls.append(("head_reduce", h.clone()))
        return h.squeeze(-2)


class PrefillFastPathTest(unittest.TestCase):
    def test_workspace_capacity_is_dynamic_only_for_v41_without_tensor_reads(self):
        from rtp_llm.models_py.modules.dsv4 import chunk_env

        class AllocationCaptured(Exception):
            pass

        chunk_rows = 4096
        cases = (
            # V4 keeps its bound capacities, independent of the live batch.
            (4, True, [107074], [0], 20, False),
            (4, True, [100930], [6144], 20, False),
            (4, True, [162350], [0], 20, False),
            (4, True, [536, 3202, 5400, 106050], [0, 0, 0, 1024], 20, False),
            (4, True, [100930, 100930], [6144, 6144], 20, False),
            (4, True, [1048575], [0], 20, False),
            (4, True, [1, 7, 8, 9], [100, 200, 0, 6144], 20, False),
            (2, True, [7, 9], [0, 100], 20, False),
            (8, True, [7, 9], [0, 100], 20, False),
            (4, True, [0], [0], 20, False),
            (1, False, [536, 3202, 5400, 106050], [0, 0, 0, 1024], 16, False),
            (4, False, [17], [99], 16, False),
            # V4.1 caps padded local Q rows to one chunk and rounds the
            # resulting bytes to 64-MiB allocation buckets.
            (4, True, [107074], [0], 1, True),
            (4, True, [100930], [6144], 1, True),
            (4, True, [162350], [0], 1, True),
            (4, True, [536, 3202, 5400, 106050], [0, 0, 0, 1024], 1, True),
            (4, True, [100930, 100930], [6144, 6144], 1, True),
            (4, True, [1048575], [0], 1, True),
            (1, False, [17], [99], 1, True),
            (1, False, [536, 3202, 5400, 106050], [0, 0, 0, 1024], 1, True),
            (1, False, [1048576], [0], 1, True),
            (2, True, [7, 9], [0, 100], 1, True),
            (8, True, [7, 9], [0, 100], 1, True),
            (4, True, [0], [0], 0, True),
        )
        workspace_type = prefill_forward.PrefillWorkspace
        for cp_size, cp_active, lengths, prefixes, expected_gib, v41 in cases:
            with self.subTest(
                cp_size=cp_size, cp_active=cp_active, lengths=lengths, v41=v41
            ):
                rows = (
                    sum(((n + 2 * cp_size - 1) // (2 * cp_size)) * 2 for n in lengths)
                    if cp_active
                    else sum(lengths)
                )
                v4 = _FakeV4()
                v4.args = SimpleNamespace(v41_config={} if v41 else None)
                v4._cp_size = cp_size
                v4._cp_info = object() if cp_active else None
                v4._prefill_ws_q_rows = 262144
                v4._prefill_ws_q_dim = 64 * 512
                v4._prefill_ws_full_rows = 1048576
                v4._prefill_ws_main_w = 2048
                v4._prefill_ws_idx_w = 512
                inputs = torch.empty(rows, dtype=torch.int32, device="meta")
                attn = SimpleNamespace(
                    prefix_lengths=torch.tensor(prefixes, device="meta")
                )
                captured = []

                def capture_workspace(*args, **kwargs):
                    captured.append(workspace_type(*args, **kwargs))
                    raise AllocationCaptured

                with patch.object(
                    chunk_env, "FLASH_MLA_SPARSE_Q_CHUNK", chunk_rows
                ), patch.dict(
                    prefill_forward.os.environ, {"DSV41_PREFILL_Q_CHUNKED": "0"}
                ), patch.object(
                    prefill_forward, "PrefillWorkspace", side_effect=capture_workspace
                ), patch.object(
                    prefill_forward, "build_cp_context_for_forward"
                ) as build_cp, patch.object(
                    v4, "embed"
                ) as embed, patch.object(
                    torch.Tensor,
                    "item",
                    side_effect=AssertionError("unexpected tensor read"),
                ), patch.object(
                    torch.Tensor,
                    "cpu",
                    side_effect=AssertionError("unexpected device copy"),
                ), patch.object(
                    torch.Tensor,
                    "tolist",
                    side_effect=AssertionError("unexpected tensor read"),
                ), patch.object(
                    torch.cuda,
                    "synchronize",
                    side_effect=AssertionError("unexpected synchronization"),
                ):
                    with self.assertRaises(AllocationCaptured):
                        prefill_forward.forward_layers(
                            v4, None, inputs, None, None, None, attn_inputs=attn
                        )
                    build_cp.assert_not_called()
                    embed.assert_not_called()
                ws = captured[0]
                expected_rows = min(rows, chunk_rows) if v41 else v4._prefill_ws_q_rows
                self.assertEqual(ws._q_rows, expected_rows)
                expected_bytes = expected_gib * (1 << 30)
                if v41:
                    bucket = 64 << 20
                    q_bytes = expected_rows * 64 * 512 * 2
                    expected_bytes = ((q_bytes + bucket - 1) // bucket) * bucket
                self.assertEqual(ws._union.numel(), expected_bytes)
                self.assertEqual(
                    ws._main_bytes,
                    v4._prefill_ws_full_rows * 2048 * 4 if cp_active and not v41 else 0,
                )
                self.assertEqual(
                    ws._idx_bytes,
                    v4._prefill_ws_full_rows * 512 * 4 if cp_active and not v41 else 0,
                )
                live_rows = min(rows, expected_rows)
                self.assertEqual(ws.prefill_q(live_rows).shape, (live_rows, 64 * 512))

    def test_v41_workspace_is_bounded_by_attention_chunk(self):
        from rtp_llm.models_py.modules.dsv4.chunk_env import (
            FLASH_MLA_SPARSE_Q_CHUNK as _FLASH_MLA_SPARSE_Q_CHUNK,
        )

        class AllocationCaptured(Exception):
            pass

        for rows in (0, 2, _FLASH_MLA_SPARSE_Q_CHUNK + 2, 262214, 1048576):
            with self.subTest(rows=rows):
                v4 = _FakeV4()
                v4.args = SimpleNamespace(v41_config={})
                v4._prefill_ws_q_dim = 64 * 512
                captured = []

                def allocate(*args, **kwargs):
                    captured.append(kwargs)
                    raise AllocationCaptured

                with patch.dict(
                    prefill_forward.os.environ, {"DSV41_PREFILL_Q_CHUNKED": "0"}
                ), patch.object(
                    prefill_forward, "PrefillWorkspace", side_effect=allocate
                ):
                    with self.assertRaises(AllocationCaptured):
                        prefill_forward.forward_layers(
                            v4, None, torch.empty(rows, device="meta"), None, None, None
                        )
                self.assertEqual(
                    captured[0]["q_rows"], min(rows, _FLASH_MLA_SPARSE_Q_CHUNK)
                )
                self.assertFalse(captured[0]["reserve_cp"])

    def test_v41_shared_scratch_is_always_released_on_success_and_failure(self):
        for enabled in (False, True):
            for fail in (False, True):
                with self.subTest(enabled=enabled, fail=fail):
                    v4 = _FakeV4()
                    registry = {i: layer.attn for i, layer in enumerate(v4.layers)}
                    shared = {"layers": registry, "global": {0: torch.ones(2)}}
                    for attn in registry.values():
                        object.__setattr__(attn, "_shared_attention", shared)

                    def layer_call(hidden, *_args, **_kwargs):
                        shared["global"] = {0: torch.ones(2)}
                        shared["prefill_index_plan"] = torch.ones(3)
                        if fail:
                            raise RuntimeError("layer failed")
                        return hidden

                    with patch.dict(
                        prefill_forward.os.environ,
                        {"DSV41_PREFILL_RELEASE_SHARED": str(int(enabled))},
                    ), patch.object(
                        prefill_forward,
                        "_prefill_fast_path_layer_calls",
                        return_value=(layer_call, layer_call),
                    ), patch.object(
                        prefill_forward, "_prefill_fast_path_enabled", return_value=True
                    ), patch.object(
                        prefill_forward, "build_and_propagate_prefill_meta_fp8"
                    ), patch.object(
                        prefill_forward, "clear_prefill_meta_shared_fp8"
                    ):

                        def run():
                            return prefill_forward.forward_layers(
                                v4,
                                None,
                                torch.tensor([3, 4]),
                                torch.tensor([0, 1]),
                                torch.tensor([0, 2]),
                                None,
                            )

                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "layer failed"):
                                run()
                        else:
                            run()
                    self.assertIs(shared["layers"], registry)
                    # A stale A/B env must never retain per-forward scratch.
                    self.assertEqual(set(shared), {"layers"})

    def test_workspace_is_allocated_before_cp_setup_and_embedding(self):
        v4 = _FakeV4()
        events = []
        original_embed = v4.embed

        def make_workspace(*args, **kwargs):
            events.append("workspace")
            return object()

        def propagate_cp(cp_ctx):
            events.append("propagate_cp")

        def embed(input_ids):
            events.append("embed")
            return original_embed(input_ids)

        with patch.object(
            prefill_forward, "PrefillWorkspace", side_effect=make_workspace
        ), patch.object(
            v4, "_propagate_cp_ctx", side_effect=propagate_cp
        ), patch.object(
            v4, "embed", side_effect=embed
        ), patch.object(
            prefill_forward, "build_and_propagate_prefill_meta_fp8"
        ), patch.object(
            prefill_forward, "clear_prefill_meta_shared_fp8"
        ):
            prefill_forward.forward_layers(
                v4,
                kv_cache=None,
                input_ids=torch.tensor([3, 4], dtype=torch.long),
                positions=torch.tensor([7, 8], dtype=torch.long),
                cu_seqlens=torch.tensor([0, 2], dtype=torch.long),
                block_tables_by_type=None,
            )

        self.assertEqual(events, ["workspace", "propagate_cp", "embed"])

    def test_cp_prepare_skips_dead_framework_positions(self):
        v4 = SimpleNamespace(_cp_info=None, _cp_size=1)
        attn = SimpleNamespace(
            position_ids=None,
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            input_lengths=torch.tensor([2], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
        )
        inputs = SimpleNamespace(
            attention_inputs=attn,
            input_ids=torch.arange(4, dtype=torch.int32),
        )
        captured = {}

        def fake_set_cp_info(*_args, **_kwargs):
            v4._cp_info = object()
            v4._cp_size = 2

        def fake_forward_layers(
            _v4, _kv_cache, _input_ids, positions, *_args, **_kwargs
        ):
            captured["positions"] = positions
            return torch.empty((4, 1))

        with patch.object(
            prefill_forward, "set_cp_info", fake_set_cp_info
        ), patch.object(
            prefill_forward, "_cp_prepare_fusion_supported", return_value=True
        ), patch.object(
            prefill_forward, "build_block_tables_batched", return_value={}
        ), patch.object(
            prefill_forward, "forward_layers", fake_forward_layers
        ), patch.object(
            prefill_forward, "PyModelOutputs", side_effect=lambda hidden: hidden
        ), patch.object(
            prefill_forward, "_build_positions_from_lengths"
        ) as build_positions:
            output = prefill_forward.forward_prefill(
                v4=v4,
                kv_cache=None,
                parallelism_config=SimpleNamespace(),
                inputs=inputs,
            )

        build_positions.assert_not_called()
        self.assertEqual(captured["positions"].numel(), 0)
        self.assertEqual(tuple(output.shape), (4, 1))

    def test_cp_context_setters_are_cached(self):
        class Sink:
            def __init__(self):
                self.values = []

            def set_cp_ctx(self, value):
                self.values.append(value)

        def make_transformer():
            attn = Sink()
            attn.compressor = Sink()
            attn.indexer = Sink()
            attn.indexer.compressor = Sink()
            return SimpleNamespace(layers=[SimpleNamespace(attn=attn)]), (
                attn,
                attn.compressor,
                attn.indexer,
                attn.indexer.compressor,
            )

        transformer, sinks = make_transformer()
        V4Transformer._propagate_cp_ctx(transformer, "first")
        setters = transformer._cp_ctx_setters
        V4Transformer._propagate_cp_ctx(transformer, "second")
        self.assertIs(setters, transformer._cp_ctx_setters)
        for sink in sinks:
            self.assertEqual(sink.values, ["first", "second"])

    def test_disable_record_function_ranges_is_scoped(self):
        calls = []

        def fake_record_function(name):
            calls.append(name)
            return nullcontext()

        with patch.object(_profiler, "_RANGES_ENABLED", True), patch.object(
            torch.profiler, "record_function", fake_record_function
        ):
            with _profiler.record_function_range("before"):
                pass
            with _profiler.disable_record_function_ranges():
                with _profiler.record_function_range("disabled"):
                    pass
                with _profiler.disable_record_function_ranges():
                    with _profiler.record_function_range("nested_disabled"):
                        pass
            with _profiler.record_function_range("after"):
                pass

        self.assertEqual(calls, ["before", "after"])

    def test_disabled_record_function_range_uses_reusable_noop_context(self):
        with patch.object(_profiler, "_RANGES_ENABLED", True):
            with _profiler.disable_record_function_ranges():
                ctx1 = _profiler.record_function_range("disabled_1")
                ctx2 = _profiler.record_function_range("disabled_2")

        self.assertIs(ctx1, _profiler._NOOP_RECORD_FUNCTION_RANGE)
        self.assertIs(ctx2, _profiler._NOOP_RECORD_FUNCTION_RANGE)
        with ctx1:
            pass

    def test_fast_path_is_default_on(self):
        class FakeBlock(Block):
            pass

        class FakeAttention(AttentionFP8):
            pass

        class FakeV4:
            fp8_kv_cache = True
            layers = [FakeBlock.__new__(FakeBlock)]

        object.__setattr__(
            FakeV4.layers[0], "attn", FakeAttention.__new__(FakeAttention)
        )
        object.__setattr__(FakeV4.layers[0], "engram", None)

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(prefill_forward._fwd_dbg, "enabled", lambda: False):
            self.assertTrue(
                prefill_forward._prefill_fast_path_enabled(
                    FakeV4(), prepare_hidden_fn=None
                )
            )

    def test_fast_path_layer_calls_are_cached_but_env_gated(self):
        v4 = _FakeV4()

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(prefill_forward._fwd_dbg, "enabled", lambda: False):
            layer_calls = prefill_forward._prefill_fast_path_layer_calls(v4)
            self.assertEqual(len(layer_calls), 2)
            self.assertIs(
                layer_calls, prefill_forward._prefill_fast_path_layer_calls(v4)
            )
            self.assertTrue(
                prefill_forward._prefill_fast_path_enabled(
                    v4, prepare_hidden_fn=None, layer_calls=layer_calls
                )
            )

        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_FAST_PATH": "0"},
            clear=True,
        ), patch.object(prefill_forward._rt, "ENABLED", False), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ):
            self.assertFalse(
                prefill_forward._prefill_fast_path_enabled(
                    v4, prepare_hidden_fn=None, layer_calls=layer_calls
                )
            )

    def test_fast_path_can_be_disabled(self):
        class FakeBlock(Block):
            pass

        class FakeAttention(AttentionFP8):
            pass

        class FakeV4:
            fp8_kv_cache = True
            layers = [FakeBlock.__new__(FakeBlock)]

        object.__setattr__(
            FakeV4.layers[0], "attn", FakeAttention.__new__(FakeAttention)
        )

        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_FAST_PATH": "0"},
            clear=True,
        ), patch.object(prefill_forward._rt, "ENABLED", False), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ):
            self.assertFalse(
                prefill_forward._prefill_fast_path_enabled(
                    FakeV4(), prepare_hidden_fn=None
                )
            )

    def test_fast_path_fails_closed_for_unsupported_contexts(self):
        class FakeBlock(Block):
            pass

        class FakeAttention(AttentionFP8):
            pass

        class FakeV4:
            fp8_kv_cache = True
            layers = [FakeBlock.__new__(FakeBlock)]

        object.__setattr__(
            FakeV4.layers[0], "attn", FakeAttention.__new__(FakeAttention)
        )

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(prefill_forward._fwd_dbg, "enabled", lambda: False):
            self.assertFalse(
                prefill_forward._prefill_fast_path_enabled(
                    FakeV4(), prepare_hidden_fn=lambda **_: None
                )
            )

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", True
        ), patch.object(prefill_forward._fwd_dbg, "enabled", lambda: False):
            self.assertFalse(
                prefill_forward._prefill_fast_path_enabled(
                    FakeV4(), prepare_hidden_fn=None
                )
            )

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(prefill_forward._fwd_dbg, "enabled", lambda: True):
            self.assertFalse(
                prefill_forward._prefill_fast_path_enabled(
                    FakeV4(), prepare_hidden_fn=None
                )
            )

    def test_fast_path_requires_supported_fp8_layers(self):
        class NoLayersV4:
            pass

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(prefill_forward._fwd_dbg, "enabled", lambda: False):
            self.assertFalse(
                prefill_forward._prefill_fast_path_enabled(
                    NoLayersV4(), prepare_hidden_fn=None
                )
            )

        class NonFp8V4:
            fp8_kv_cache = False
            layers = []

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(prefill_forward._fwd_dbg, "enabled", lambda: False):
            self.assertFalse(
                prefill_forward._prefill_fast_path_enabled(
                    NonFp8V4(), prepare_hidden_fn=None
                )
            )

        class FakeV4:
            fp8_kv_cache = True
            layers = [object()]

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(prefill_forward._fwd_dbg, "enabled", lambda: False):
            self.assertFalse(
                prefill_forward._prefill_fast_path_enabled(
                    FakeV4(), prepare_hidden_fn=None
                )
            )

    def test_forward_layers_dispatches_fast_path_by_default(self):
        v4 = _FakeV4()
        input_ids = torch.tensor([3, 4], dtype=torch.long)
        positions = torch.tensor([7, 8], dtype=torch.long)
        cu_seqlens = torch.tensor([0, 2], dtype=torch.long)
        block_tables = {0: torch.tensor([[1]], dtype=torch.int32)}
        attn_inputs = SimpleNamespace(
            input_lengths=torch.tensor([2], dtype=torch.int32),
            prefix_lengths=torch.tensor([7], dtype=torch.int32),
        )

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ), patch.object(
            prefill_forward, "build_and_propagate_prefill_meta_fp8"
        ) as build_meta, patch.object(
            prefill_forward, "clear_prefill_meta_shared_fp8"
        ) as clear_meta:
            out = prefill_forward.forward_layers(
                v4,
                kv_cache=None,
                input_ids=input_ids,
                positions=positions,
                cu_seqlens=cu_seqlens,
                block_tables_by_type=block_tables,
                attn_inputs=attn_inputs,
            )

        self.assertEqual(
            [call[0] for call in v4.calls], ["fast", "fast", "head_reduce"]
        )
        torch.testing.assert_close(v4.calls[0][4], positions)
        torch.testing.assert_close(v4.calls[0][5], cu_seqlens)
        self.assertIs(v4.calls[0][7], block_tables)
        torch.testing.assert_close(
            out,
            torch.tensor([[106.0, 106.5], [107.0, 107.5]]),
        )
        build_meta.assert_called_once()
        clear_meta.assert_called_once_with(v4)

    def test_fast_path_keeps_only_outer_layer_ranges(self):
        v4 = _FakeV4()
        input_ids = torch.tensor([3, 4], dtype=torch.long)
        positions = torch.tensor([7, 8], dtype=torch.long)
        cu_seqlens = torch.tensor([0, 2], dtype=torch.long)
        events = []
        nested_range_enabled = []

        for layer in v4.layers:
            original = layer.forward_prefill_fast

            def wrapped(*args, _original=original, **kwargs):
                nested_range_enabled.append(_profiler.record_function_ranges_enabled())
                return _original(*args, **kwargs)

            layer.forward_prefill_fast = wrapped

        @contextmanager
        def layer_range(layer_idx):
            events.append(("enter", layer_idx))
            try:
                yield
            finally:
                events.append(("exit", layer_idx))

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ), patch.object(
            prefill_forward, "build_and_propagate_prefill_meta_fp8"
        ), patch.object(
            prefill_forward, "clear_prefill_meta_shared_fp8"
        ), patch.object(
            _profiler, "make_layer_forward_range", return_value=layer_range
        ):
            prefill_forward.forward_layers(
                v4,
                kv_cache=None,
                input_ids=input_ids,
                positions=positions,
                cu_seqlens=cu_seqlens,
                block_tables_by_type=None,
            )

        self.assertEqual(
            events,
            [("enter", 0), ("exit", 0), ("enter", 1), ("exit", 1)],
        )
        self.assertEqual(nested_range_enabled, [False, False])

    def test_forward_layers_fast_path_preserves_varlen_batch_metadata(self):
        v4 = _FakeV4()
        input_ids = torch.tensor([3, 4, 5, 6], dtype=torch.long)
        positions = torch.tensor([5, 6, 100, 101], dtype=torch.long)
        cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.long)
        block_tables = {0: torch.tensor([[1], [2]], dtype=torch.int32)}
        attn_inputs = SimpleNamespace(
            input_lengths=torch.tensor([2, 2], dtype=torch.int32),
            prefix_lengths=torch.tensor([5, 100], dtype=torch.int32),
        )

        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ), patch.object(
            prefill_forward, "build_and_propagate_prefill_meta_fp8"
        ) as build_meta, patch.object(
            prefill_forward, "clear_prefill_meta_shared_fp8"
        ):
            out = prefill_forward.forward_layers(
                v4,
                kv_cache=None,
                input_ids=input_ids,
                positions=positions,
                cu_seqlens=cu_seqlens,
                block_tables_by_type=block_tables,
                attn_inputs=attn_inputs,
            )

        self.assertEqual(
            [call[0] for call in v4.calls], ["fast", "fast", "head_reduce"]
        )
        torch.testing.assert_close(v4.calls[0][4], positions)
        torch.testing.assert_close(v4.calls[0][5], cu_seqlens)
        kwargs = build_meta.call_args.kwargs
        self.assertEqual(kwargs["batch_size"], 2)
        torch.testing.assert_close(
            kwargs["sp_per_req"], torch.tensor([5, 100], dtype=torch.int64)
        )
        torch.testing.assert_close(
            kwargs["req_id_per_token"], torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            kwargs["input_lengths"], torch.tensor([2, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(
            kwargs["prefix_lengths"], torch.tensor([5, 100], dtype=torch.int32)
        )
        self.assertEqual(kwargs["max_seqlen_q"], 2)
        torch.testing.assert_close(
            out,
            torch.tensor(
                [[106.0, 106.5], [107.0, 107.5], [108.0, 108.5], [109.0, 109.5]]
            ),
        )

    def test_active_profiler_keeps_nested_ranges_on_fast_path(self):
        v4 = _FakeV4()
        observed = []
        for layer in v4.layers:
            original = layer.forward_prefill_fast

            def wrapped(*args, _original=original, **kwargs):
                observed.append(_profiler.record_function_ranges_enabled())
                return _original(*args, **kwargs)

            layer.forward_prefill_fast = wrapped
        with patch.dict(prefill_forward.os.environ, {}, clear=True), patch.object(
            prefill_forward._rt, "ENABLED", False
        ), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ), patch.object(
            _profiler, "_torch_profiler_enabled", return_value=True
        ), patch.object(
            prefill_forward, "build_and_propagate_prefill_meta_fp8"
        ), patch.object(
            prefill_forward, "clear_prefill_meta_shared_fp8"
        ):
            prefill_forward.forward_layers(
                v4,
                None,
                torch.tensor([3, 4]),
                torch.tensor([0, 1]),
                torch.tensor([0, 2]),
                None,
            )
        self.assertEqual(observed, [True, True])

    def test_forward_layers_uses_normal_layer_call_when_fast_path_disabled(self):
        v4 = _FakeV4()
        input_ids = torch.tensor([3, 4], dtype=torch.long)
        positions = torch.tensor([7, 8], dtype=torch.long)
        cu_seqlens = torch.tensor([0, 2], dtype=torch.long)

        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_FAST_PATH": "0"},
            clear=True,
        ), patch.object(prefill_forward._rt, "ENABLED", False), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ), patch.object(
            prefill_forward, "build_and_propagate_prefill_meta_fp8"
        ), patch.object(
            prefill_forward, "clear_prefill_meta_shared_fp8"
        ):
            out = prefill_forward.forward_layers(
                v4,
                kv_cache=None,
                input_ids=input_ids,
                positions=positions,
                cu_seqlens=cu_seqlens,
                block_tables_by_type=None,
            )

        self.assertEqual(
            [call[0] for call in v4.calls], ["normal", "normal", "head_reduce"]
        )
        torch.testing.assert_close(
            out,
            torch.tensor([[124.0, 124.5], [125.0, 125.5]]),
        )


def _normal_path_rms(x, weight, eps):
    value = x.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        * weight.float()
    ).to(x.dtype)


class _NormalPathNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.clone())
        return _normal_path_rms(x, self.weight, self.variance_epsilon)


class _NormalPathAttention(AttentionFP8):
    """CPU operator boundary; the real Block and delayed HC remain unmocked."""

    def __init__(self, *, supported=True, config=True):
        nn.Module.__init__(self)
        if config is not False:
            self.v41_config = {} if config is True else None
        self.supported = supported
        self.gate_calls = []
        self.fused_calls = []
        self.shared_calls = []
        self.ordinary_calls = []
        self.pair = (
            torch.tensor([17], dtype=torch.uint8),
            torch.tensor([23], dtype=torch.int32),
        )

    def can_fuse_prefill_attn_norm_input_quant(self, x, weight):
        self.gate_calls.append((x.clone(), weight))
        # This isolates Block dispatch using small CPU shapes. GPU shape/device
        # eligibility is tested separately by the norm/quant kernel tests.
        return self.supported and x.dtype == weight.dtype == torch.bfloat16

    def prefill_fused_attn_norm_input_quant(self, x, weight, eps):
        self.fused_calls.append((x.clone(), weight, eps))
        x.copy_(_normal_path_rms(x, weight, eps))
        return x, self.pair

    def forward_with_shared_input_quant(
        self, x, positions, shared_input_quant, **kwargs
    ):
        self.shared_calls.append((x.clone(), positions, shared_input_quant, kwargs))
        return (x.float() * 0.375 + 0.25).to(x.dtype)

    def forward(self, x, positions, **kwargs):
        self.ordinary_calls.append((x.clone(), positions, kwargs))
        return (x.float() * 0.375 + 0.25).to(x.dtype)


class _NormalPathEngram(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden, hashes, mask):
        self.calls.append((hidden.clone(), hashes, mask))
        lane_delta = hidden.new_tensor([0.25, -0.5, 0.75, 1.0]).view(1, 4, 1)
        return hidden + lane_delta * mask[:, None, None]


class _NormalPathFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, ids):
        self.inputs.append((x.clone(), ids, getattr(self, "_dbg_positions", None)))
        return (x.float() * 0.25 - 0.125).to(x.dtype)


class NormalBlockNormQuantTest(unittest.TestCase):
    @staticmethod
    def hc(pre_bias):
        base = torch.linspace(-0.7, 0.9, 24)
        base[:4] = torch.tensor(pre_bias)
        return DelayedHCUnit(
            torch.zeros(24, 12),
            base,
            torch.ones(3),
            dim=3,
            hc_mult=4,
            hc_sinkhorn_iters=4,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )

    def case(
        self,
        *,
        engram=False,
        previous=True,
        dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        supported=True,
        config=True,
    ):
        layer = Block.__new__(Block)
        nn.Module.__init__(layer)
        layer.layer_id = 14 if engram else 0
        layer.attn = _NormalPathAttention(supported=supported, config=config)
        layer.attn_hc = self.hc([-3.0, -1.0, 1.0, 3.0])
        layer.ffn_hc = self.hc([3.0, 1.0, -1.0, -3.0])
        layer.ffn_hc.set_previous(layer.attn_hc)
        layer.attn_norm = _NormalPathNorm(
            torch.tensor([0.5, 1.25, -1.75], dtype=weight_dtype), 3e-5
        )
        layer.ffn_norm = _NormalPathNorm(
            torch.tensor([1.5, -0.75, 0.25], dtype=weight_dtype), 7e-5
        )
        layer.ffn = _NormalPathFFN()
        layer.engram = _NormalPathEngram() if engram else None
        layer.engram_hashes = torch.arange(6).view(3, 2)
        layer.engram_token_mask = torch.tensor([True, False, True])
        # Only optional GPU mega-mHC fusion is stubbed; pre/post are real HC.
        layer._try_mega_mhc = Mock(return_value=None)
        layer._sync_after_first_cp_prefill_attention = Mock()
        hidden = (torch.arange(36).reshape(3, 4, 3).float() / 11 - 1.0).to(dtype)
        predecessor = self.hc([-2.0, 0.0, 2.0, 4.0]) if previous else None
        if predecessor is not None:
            predecessor.pre(hidden * 0.5)
            layer.attn_hc.set_previous(predecessor)
        return SimpleNamespace(
            layer=layer,
            hidden=hidden,
            previous=predecessor,
            ids=torch.tensor([11, 13, 17]),
            positions=torch.tensor([7, 8, 9]),
            cu=torch.tensor([0, 3]),
            cache=object(),
            tables={0: object()},
        )

    def exercise(self, case, *, fused, debug=False):
        from rtp_llm.models_py.modules.dsv4 import _record_tensor

        c, layer = case, case.layer
        original = c.hidden.clone()
        gamma = layer.attn_norm.weight.clone()
        injected = original.clone()
        if layer.engram is not None:
            delta = original.new_tensor([0.25, -0.5, 0.75, 1.0]).view(1, 4, 1)
            injected += delta * layer.engram_token_mask[:, None, None]
        expected_pre = (
            injected[:, 0].contiguous()
            if c.previous is None
            else (injected.float() * c.previous.pre_mix_out.unsqueeze(-1))
            .sum(-2)
            .to(injected.dtype)
        )
        expected_norm = _normal_path_rms(
            expected_pre, gamma, layer.attn_norm.variance_epsilon
        )
        with patch.object(
            _record_tensor, "should_record_layer", return_value=debug
        ), patch.object(
            _record_tensor, "_DBG_GLOBAL_POS", 8 if debug else -1
        ), patch.object(
            _record_tensor, "record_if_level"
        ) as record, patch.object(
            layer.attn_hc, "post", wraps=layer.attn_hc.post
        ) as attn_post, patch.object(
            layer.ffn_hc, "post", wraps=layer.ffn_hc.post
        ) as ffn_post:
            # nn.Module.__call__ executes the actual, complete Block.forward.
            out = layer(
                c.hidden,
                c.ids,
                c.positions,
                c.cu,
                kv_cache=c.cache,
                block_tables_by_type=c.tables,
            )
        self.assertEqual(len(layer.attn.fused_calls), int(fused))
        self.assertEqual(len(layer.attn.shared_calls), int(fused))
        self.assertEqual(len(layer.attn.ordinary_calls), int(not fused))
        self.assertEqual(len(layer.attn_norm.inputs), int(not fused))
        self.assertEqual(len(layer.ffn_norm.inputs), 1)
        self.assertTrue(
            torch.equal(c.hidden, original), "HC residual was overwritten by norm"
        )
        self.assertTrue(torch.equal(layer.attn_norm.weight, gamma))
        self.assertEqual(attn_post.call_count, 1)
        attn_value, attn_residual, post, comb = attn_post.call_args.args
        self.assertTrue(torch.equal(attn_residual, injected))
        expected_attn = (expected_norm.float() * 0.375 + 0.25).to(injected.dtype)
        self.assertTrue(torch.equal(attn_value, expected_attn))
        middle = (
            post.float() * expected_attn.float().unsqueeze(-2)
            + comb.float().transpose(-1, -2) @ injected.float()
        ).to(injected.dtype)
        # FFN reads the attention sublayer's pre_mix, not its newly computed own mix.
        expected_ffn_pre = (
            (middle.float() * layer.attn_hc.pre_mix_out.unsqueeze(-1))
            .sum(-2)
            .to(middle.dtype)
        )
        self.assertTrue(torch.equal(layer.ffn_norm.inputs[0], expected_ffn_pre))
        wrong_mix = (
            (middle.float() * layer.ffn_hc.pre_mix_out.unsqueeze(-1))
            .sum(-2)
            .to(middle.dtype)
        )
        self.assertFalse(torch.equal(expected_ffn_pre, wrong_mix))
        ff_norm = _normal_path_rms(
            expected_ffn_pre, layer.ffn_norm.weight, layer.ffn_norm.variance_epsilon
        )
        self.assertTrue(torch.equal(layer.ffn.inputs[0][0], ff_norm))
        self.assertIs(layer.ffn.inputs[0][1], c.ids)
        ff_value, ff_residual, ff_post, ff_comb = ffn_post.call_args.args
        self.assertTrue(torch.equal(ff_residual, middle))
        expected_value = (ff_norm.float() * 0.25 - 0.125).to(middle.dtype)
        self.assertTrue(torch.equal(ff_value, expected_value))
        expected = (
            ff_post.float() * expected_value.float().unsqueeze(-2)
            + ff_comb.float().transpose(-1, -2) @ middle.float()
        ).to(middle.dtype)
        self.assertTrue(torch.equal(out, expected))
        if fused:
            raw, weight, eps = layer.attn.fused_calls[0]
            self.assertTrue(torch.equal(raw, expected_pre))
            self.assertEqual(weight.data_ptr(), layer.attn_norm.weight.data_ptr())
            self.assertEqual(eps, 3e-5)
            norm, positions, pair, kwargs = layer.attn.shared_calls[0]
            self.assertIs(pair, layer.attn.pair)
            self.assertIs(positions, c.positions)
        else:
            self.assertTrue(torch.equal(layer.attn_norm.inputs[0], expected_pre))
            norm, positions, kwargs = layer.attn.ordinary_calls[0]
        self.assertTrue(torch.equal(norm, expected_norm))
        self.assertIs(kwargs["kv_cache"], c.cache)
        self.assertIs(kwargs["block_tables_by_type"], c.tables)
        if layer.engram is not None:
            self.assertEqual(len(layer.engram.calls), 1)
            self.assertIs(layer.engram.calls[0][1], layer.engram_hashes)
            self.assertIs(layer.engram.calls[0][2], layer.engram_token_mask)
        if debug:
            self.assertFalse(layer.attn.gate_calls)
            layer._try_mega_mhc.assert_not_called()
            self.assertIn(
                f"L{layer.layer_id:02d}_attn_in_pos8",
                [call.args[1] for call in record.call_args_list],
            )
            self.assertIsNone(layer.ffn._dbg_positions)
        else:
            record.assert_not_called()
            layer._try_mega_mhc.assert_called_once()
        layer._sync_after_first_cp_prefill_attention.assert_called_once()

    def test_normal_v41_fuses_with_engram_and_without_engram(self):
        for engram in (False, True):
            for previous in (False, True):
                with self.subTest(engram=engram, previous=previous):
                    self.exercise(
                        self.case(engram=engram, previous=previous), fused=True
                    )

    def test_v41_engram_preserves_normal_layer_and_fast_neighbors(self):
        from rtp_llm.models_py.modules.dsv4 import _record_tensor

        def make_chain():
            cases = [self.case(engram=i == 1, previous=i == 0) for i in range(3)]
            for i, c in enumerate(cases):
                c.layer.layer_id = i
                # Input-dependent, distinct mixes expose stale/cross-layer reads.
                for unit in (c.layer.attn_hc, c.layer.ffn_hc):
                    unit.fn.copy_(
                        torch.linspace(-0.03, 0.05, unit.fn.numel()).reshape_as(unit.fn)
                        * (i + 1)
                    )
                if i:
                    c.layer.attn_hc.set_previous(cases[i - 1].layer.ffn_hc)
            return cases

        normal, mixed = make_chain(), make_chain()
        model = SimpleNamespace(fp8_kv_cache=True, layers=[c.layer for c in mixed])
        calls = prefill_forward._prefill_fast_path_layer_calls(model)
        self.assertEqual(calls[0], mixed[0].layer.prefill_fast_callable())
        self.assertIs(calls[1], mixed[1].layer)
        self.assertEqual(calls[2], mixed[2].layer.prefill_fast_callable())
        self.assertIs(calls, prefill_forward._prefill_fast_path_layer_calls(model))

        inputs = normal[0]

        def run_chain(cases, layer_calls, normal_call_counts):
            hidden = inputs.hidden.clone()
            outputs = []
            for i, (c, call) in enumerate(zip(cases, layer_calls)):
                layer = c.layer
                residual = hidden.clone()
                if layer.engram is not None:
                    delta = residual.new_tensor([0.25, -0.5, 0.75, 1.0]).view(1, 4, 1)
                    residual += delta * layer.engram_token_mask[:, None, None]
                previous = cases[i - 1].layer.ffn_hc if i else c.previous
                expected_pre = (
                    (residual.float() * previous.pre_mix_out.unsqueeze(-1))
                    .sum(-2)
                    .to(residual.dtype)
                )
                before = hidden.clone()
                with patch.object(layer, "forward", wraps=layer.forward) as forward:
                    hidden = call(
                        hidden,
                        inputs.ids,
                        inputs.positions,
                        inputs.cu,
                        kv_cache=inputs.cache,
                        block_tables_by_type=inputs.tables,
                    )
                self.assertEqual(forward.call_count, normal_call_counts[i])
                self.assertEqual(len(layer.attn.fused_calls), 1)
                self.assertTrue(torch.equal(layer.attn.fused_calls[0][0], expected_pre))
                wrong_pre = (
                    (residual.float() * layer.attn_hc.pre_mix_out.unsqueeze(-1))
                    .sum(-2)
                    .to(residual.dtype)
                )
                self.assertFalse(torch.equal(expected_pre, wrong_pre))
                (
                    attn_value,
                    attn_residual,
                    post,
                    comb,
                ) = layer._try_mega_mhc.call_args.args
                self.assertTrue(torch.equal(attn_residual, residual))
                middle = (
                    post.float() * attn_value.float().unsqueeze(-2)
                    + comb.float().transpose(-1, -2) @ residual.float()
                ).to(residual.dtype)
                expected_ffn_pre = (
                    (middle.float() * layer.attn_hc.pre_mix_out.unsqueeze(-1))
                    .sum(-2)
                    .to(middle.dtype)
                )
                self.assertEqual(len(layer.ffn_norm.inputs), 1)
                self.assertTrue(torch.equal(layer.ffn_norm.inputs[0], expected_ffn_pre))
                self.assertEqual(len(layer.ffn.inputs), 1)
                self.assertIs(layer.ffn.inputs[0][1], inputs.ids)
                self.assertEqual(len(layer.attn.shared_calls), 1)
                _, positions, pair, kwargs = layer.attn.shared_calls[0]
                self.assertIs(positions, inputs.positions)
                self.assertIs(pair, layer.attn.pair)
                self.assertIs(kwargs["kv_cache"], inputs.cache)
                self.assertIs(kwargs["block_tables_by_type"], inputs.tables)
                if layer.engram is not None:
                    self.assertEqual(len(layer.engram.calls), 1)
                    seen, hashes, mask = layer.engram.calls[0]
                    self.assertTrue(torch.equal(seen, before))
                    self.assertIs(hashes, layer.engram_hashes)
                    self.assertIs(mask, layer.engram_token_mask)
                layer._sync_after_first_cp_prefill_attention.assert_called_once()
                outputs.append(hidden.clone())
            return outputs

        with patch.object(_record_tensor, "should_record_layer", return_value=False):
            expected = run_chain(normal, [c.layer for c in normal], [1, 1, 1])
            actual = run_chain(mixed, calls, [0, 1, 0])
        for i, (reference, result) in enumerate(zip(expected, actual)):
            with self.subTest(layer=i):
                self.assertTrue(torch.equal(reference, result))
                for name in ("attn_hc", "ffn_hc"):
                    self.assertTrue(
                        torch.equal(
                            getattr(normal[i].layer, name).pre_mix_out,
                            getattr(mixed[i].layer, name).pre_mix_out,
                        )
                    )

    def test_non_v41_engram_still_disables_fast_stack(self):
        plain, engram = self.case(), self.case(engram=True, config=None)
        model = SimpleNamespace(fp8_kv_cache=True, layers=[plain.layer, engram.layer])
        self.assertIsNone(prefill_forward._prefill_fast_path_layer_calls(model))

    def test_v4_and_missing_v41_config_preserve_ordinary_norm(self):
        for config in (None, False):
            with self.subTest(config=config):
                c = self.case(config=config)
                self.exercise(c, fused=False)
                self.assertFalse(c.layer.attn.gate_calls)

    def test_small_gate_rejection_and_float_inputs_or_gamma_keep_ordinary_norm(self):
        for kwargs in (
            {"supported": False},
            {"dtype": torch.float32},
            {"weight_dtype": torch.float32},
        ):
            with self.subTest(kwargs=kwargs):
                c = self.case(**kwargs)
                self.exercise(c, fused=False)
                self.assertEqual(len(c.layer.attn.gate_calls), 1)

    def test_debug_preserves_norm_recording_and_delayed_hc(self):
        self.exercise(self.case(engram=True), fused=False, debug=True)


if __name__ == "__main__":
    unittest.main()
