import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.prefill import forward as prefill_forward


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
        self.norm = lambda h: h + 100

    def _propagate_cp_ctx(self, cp_ctx):
        self.cp_ctx = cp_ctx

    def embed(self, input_ids):
        return torch.stack((input_ids.float(), input_ids.float() + 0.5), dim=-1)

    def _hc_head_reduce(self, h):
        self.calls.append(("head_reduce", h.clone()))
        return h.squeeze(-2)


class PrefillFastPathTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
