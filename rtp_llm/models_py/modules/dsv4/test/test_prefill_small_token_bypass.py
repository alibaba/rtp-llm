import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4.prefill import forward as prefill_forward


class PrefillSmallTokenBypassTest(unittest.TestCase):
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

    def test_globally_disabled_record_function_range_uses_reusable_noop_context(self):
        calls = []

        def fake_record_function(name):
            calls.append(name)
            return nullcontext()

        with patch.object(_profiler, "_RANGES_ENABLED", False), patch.object(
            torch.profiler, "record_function", fake_record_function
        ):
            ctx = _profiler.record_function_range("globally_disabled")
            with ctx:
                pass

        self.assertIs(ctx, _profiler._NOOP_RECORD_FUNCTION_RANGE)
        self.assertEqual(calls, [])

    def test_bypass_token_cap_uses_global_cp_length(self):
        class FakeV4:
            layers = []

        input_ids = torch.zeros(8192, dtype=torch.long)
        cp_ctx = SimpleNamespace(seq_len_full=65536)
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS": "1",
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS_MAX_TOKENS": "8192",
            },
        ), patch.object(prefill_forward._rt, "ENABLED", False), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ):
            enabled = prefill_forward._small_token_bypass_enabled(
                FakeV4(),
                input_ids,
                prepare_hidden_fn=None,
                cp_ctx=cp_ctx,
            )

        self.assertFalse(enabled)

    def test_bypass_malformed_token_cap_fails_closed(self):
        class FakeV4:
            layers = []

        input_ids = torch.zeros(512, dtype=torch.long)
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS": "1",
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS_MAX_TOKENS": "bad",
            },
        ), patch.object(prefill_forward._rt, "ENABLED", False), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ):
            enabled = prefill_forward._small_token_bypass_enabled(
                FakeV4(),
                input_ids,
                prepare_hidden_fn=None,
                cp_ctx=None,
            )

        self.assertFalse(enabled)

    def test_env_int_fail_closed(self):
        with patch.dict(prefill_forward.os.environ, {}, clear=True):
            self.assertEqual(prefill_forward._env_int_fail_closed("X", 7), 7)
        with patch.dict(prefill_forward.os.environ, {"X": "11"}):
            self.assertEqual(prefill_forward._env_int_fail_closed("X", 7), 11)
        with patch.dict(prefill_forward.os.environ, {"X": "bad"}):
            self.assertIsNone(prefill_forward._env_int_fail_closed("X", 7))

    def test_request_token_count_uses_cp_global_length(self):
        input_ids = torch.zeros(512, dtype=torch.long)
        cp_ctx = SimpleNamespace(seq_len_full=4096)

        self.assertEqual(
            prefill_forward._prefill_request_token_count(input_ids, cp_ctx),
            4096,
        )
        self.assertEqual(
            prefill_forward._prefill_request_token_count(input_ids, None),
            512,
        )

    def test_static_eager_graph_diag_gate_accepts_fully_bound_state(self):
        v4 = SimpleNamespace(
            fp8_kv_cache=True,
            _last_prefill_graph_state=SimpleNamespace(valid=True),
            _last_prefill_graph_bind_static_error=None,
            _last_prefill_graph_bind_static_meta_error=None,
            _last_prefill_graph_bound_capture_surface=SimpleNamespace(
                static_bound=True
            ),
        )
        decision = SimpleNamespace(enabled=True)
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN": "1",
                "DSV4_PREFILL_GRAPH_BIND_STATIC_INPUTS": "1",
                "DSV4_PREFILL_GRAPH_BIND_STATIC_META": "1",
            },
            clear=True,
        ):
            self.assertTrue(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4,
                    decision,
                    kv_cache=None,
                    static_state_updated_this_forward=True,
                    graph_static_bind_allowed=True,
                    graph_replay_requested=False,
                    use_small_token_bypass=True,
                    write_cache_store_impl=None,
                    rt_on=False,
                )
            )

    def test_static_eager_graph_diag_gate_fails_closed(self):
        v4 = SimpleNamespace(
            fp8_kv_cache=True,
            _last_prefill_graph_state=SimpleNamespace(valid=True),
            _last_prefill_graph_bind_static_error=None,
            _last_prefill_graph_bind_static_meta_error=None,
            _last_prefill_graph_bound_capture_surface=SimpleNamespace(
                static_bound=True
            ),
        )
        decision = SimpleNamespace(enabled=True)
        common = dict(
            kv_cache=None,
            static_state_updated_this_forward=True,
            graph_static_bind_allowed=True,
            graph_replay_requested=False,
            use_small_token_bypass=True,
            write_cache_store_impl=None,
            rt_on=False,
        )
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN": "1",
                "DSV4_PREFILL_GRAPH_BIND_STATIC_INPUTS": "1",
            },
            clear=True,
        ):
            self.assertFalse(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4, decision, **common
                )
            )
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN": "1",
                "DSV4_PREFILL_GRAPH_BIND_STATIC_INPUTS": "1",
                "DSV4_PREFILL_GRAPH_BIND_STATIC_META": "1",
            },
            clear=True,
        ):
            self.assertFalse(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4,
                    decision,
                    **{**common, "graph_replay_requested": True},
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4,
                    decision,
                    **{**common, "static_state_updated_this_forward": False},
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4,
                    decision,
                    **{**common, "kv_cache": object()},
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4,
                    decision,
                    **{**common, "write_cache_store_impl": object()},
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4,
                    decision,
                    **{**common, "rt_on": True},
                )
            )
            with patch.dict(
                prefill_forward.os.environ,
                {
                    "DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN": "1",
                    "DSV4_PREFILL_GRAPH_BIND_STATIC_INPUTS": "1",
                    "DSV4_PREFILL_GRAPH_BIND_STATIC_META": "1",
                    "DSV4_PREFILL_GRAPH_STATIC_EAGER_ALLOW_LIVE_KV": "1",
                },
                clear=True,
            ):
                self.assertTrue(
                    prefill_forward._prefill_graph_static_eager_run_allowed(
                        v4,
                        decision,
                        **{**common, "kv_cache": object()},
                    )
                )
            v4._last_prefill_graph_bound_capture_surface = None
            self.assertFalse(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4, decision, **common
                )
            )
            v4._last_prefill_graph_bound_capture_surface = SimpleNamespace(
                static_bound=False
            )
            self.assertFalse(
                prefill_forward._prefill_graph_static_eager_run_allowed(
                    v4, decision, **common
                )
            )

    def test_copy_shadow_graph_diag_gate_is_fail_closed(self):
        v4 = SimpleNamespace(
            _last_prefill_graph_state=SimpleNamespace(valid=True),
            _last_prefill_graph_bind_static_error=None,
            _last_prefill_graph_bind_static_meta_error=None,
            _last_prefill_graph_bound_capture_surface=SimpleNamespace(
                static_bound=True
            ),
        )
        decision = SimpleNamespace(enabled=True)
        common = dict(
            static_args_bound_this_forward=True,
            static_state_updated_this_forward=True,
            use_small_token_bypass=True,
            write_cache_store_impl=None,
            rt_on=False,
        )
        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_GRAPH_COPY_SHADOW": "1"},
            clear=True,
        ):
            self.assertTrue(
                prefill_forward._prefill_graph_copy_shadow_allowed(
                    v4, decision, **common
                )
            )
            for override in (
                {"static_args_bound_this_forward": False},
                {"static_state_updated_this_forward": False},
                {"use_small_token_bypass": False},
                {"write_cache_store_impl": object()},
                {"rt_on": True},
            ):
                self.assertFalse(
                    prefill_forward._prefill_graph_copy_shadow_allowed(
                        v4, decision, **{**common, **override}
                    )
                )
            v4._last_prefill_graph_bound_capture_surface = None
            self.assertFalse(
                prefill_forward._prefill_graph_copy_shadow_allowed(
                    v4, decision, **common
                )
            )

    def test_bound_capture_surface_is_required_for_copy_shadow(self):
        with patch.dict(prefill_forward.os.environ, {}, clear=True):
            self.assertFalse(prefill_forward._prefill_graph_needs_bound_capture_surface())
        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_GRAPH_COPY_SHADOW": "1"},
            clear=True,
        ):
            self.assertTrue(prefill_forward._prefill_graph_needs_bound_capture_surface())
        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN": "1"},
            clear=True,
        ):
            self.assertTrue(prefill_forward._prefill_graph_needs_bound_capture_surface())
        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_GRAPH_CAPTURE_SURFACE_LOG": "1"},
            clear=True,
        ):
            self.assertTrue(prefill_forward._prefill_graph_needs_bound_capture_surface())

    def test_copy_shadow_result_state_is_not_stale(self):
        v4 = SimpleNamespace(
            _last_prefill_graph_copy_shadow_stats={"old": True},
            _last_prefill_graph_copy_shadow_error="old",
        )

        prefill_forward._clear_prefill_graph_copy_shadow_result(v4)
        self.assertIsNone(v4._last_prefill_graph_copy_shadow_stats)
        self.assertIsNone(v4._last_prefill_graph_copy_shadow_error)

        prefill_forward._set_prefill_graph_copy_shadow_stats(
            v4,
            mode="replay",
            exact=True,
            max_abs=0.0,
            mean_abs=0.0,
        )
        self.assertEqual(
            v4._last_prefill_graph_copy_shadow_stats,
            {"mode": "replay", "exact": True, "max_abs": 0.0, "mean_abs": 0.0},
        )
        self.assertIsNone(v4._last_prefill_graph_copy_shadow_error)

        prefill_forward._set_prefill_graph_copy_shadow_error(v4, "boom")
        self.assertIsNone(v4._last_prefill_graph_copy_shadow_stats)
        self.assertEqual(v4._last_prefill_graph_copy_shadow_error, "boom")

    def test_attn_body_shadow_gate_is_explicit_and_layer_scoped(self):
        from rtp_llm.models_py.modules.dsv4.prefill_graph import PrefillGraphKey

        key = PrefillGraphKey(token_bucket=3, batch_bucket=1, cp_size=1)
        v4 = SimpleNamespace(
            _last_prefill_graph_state=SimpleNamespace(
                valid=True,
                key=key,
                cuda_graph=None,
            ),
            _last_prefill_graph_bind_static_error=None,
            _last_prefill_graph_bind_static_meta_error=None,
        )
        decision = SimpleNamespace(enabled=True, key=key)
        common = dict(
            loop_kv_cache=object(),
            static_state_updated_this_forward=True,
            graph_replay_requested=False,
            write_cache_store_impl=None,
            rt_on=False,
        )
        with patch.dict(prefill_forward.os.environ, {}, clear=True):
            self.assertIsNone(prefill_forward._prefill_graph_attn_body_shadow_layer())
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW": "1",
                "DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW_LAYER": "2",
            },
            clear=True,
        ):
            self.assertEqual(prefill_forward._prefill_graph_attn_body_shadow_layer(), 2)
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4, decision, layer_idx=2, **common
                )
            )
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW": "1",
                "DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW_LAYER": "2",
                "DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW_ALLOW_GRAPH_KV": "1",
            },
            clear=True,
        ):
            self.assertTrue(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4, decision, layer_idx=2, **common
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4, decision, layer_idx=3, **common
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4,
                    decision,
                    layer_idx=2,
                    **{**common, "static_state_updated_this_forward": False},
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4,
                    decision,
                    layer_idx=2,
                    **{**common, "graph_replay_requested": True},
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4,
                    decision,
                    layer_idx=2,
                    **{**common, "write_cache_store_impl": object()},
                )
            )
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4,
                    decision,
                    layer_idx=2,
                    **{**common, "rt_on": True},
                )
            )
            decision.key = PrefillGraphKey(
                token_bucket=3, batch_bucket=1, cp_size=1, prefix_bucket=1
            )
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4, decision, layer_idx=2, **common
                )
            )
            decision.key = PrefillGraphKey(token_bucket=4, batch_bucket=1, cp_size=1)
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4, decision, layer_idx=2, **common
                )
            )
            decision.key = key
            v4._last_prefill_graph_state.cuda_graph = object()
            self.assertFalse(
                prefill_forward._prefill_graph_attn_body_shadow_allowed(
                    v4, decision, layer_idx=2, **common
                )
            )

    def test_attn_body_shadow_result_state_is_not_stale(self):
        v4 = SimpleNamespace(
            _last_prefill_graph_attn_body_shadow_stats={"old": True},
            _last_prefill_graph_attn_body_shadow_error="old",
        )

        prefill_forward._clear_prefill_graph_attn_body_shadow_result(v4)
        self.assertIsNone(v4._last_prefill_graph_attn_body_shadow_stats)
        self.assertIsNone(v4._last_prefill_graph_attn_body_shadow_error)

        prefill_forward._set_prefill_graph_attn_body_shadow_stats(
            v4,
            layer_idx=2,
            exact=True,
            max_abs=0.0,
            mean_abs=0.0,
        )
        self.assertEqual(
            v4._last_prefill_graph_attn_body_shadow_stats,
            {"layer_idx": 2, "exact": True, "max_abs": 0.0, "mean_abs": 0.0},
        )
        self.assertIsNone(v4._last_prefill_graph_attn_body_shadow_error)

        prefill_forward._set_prefill_graph_attn_body_shadow_error(v4, "boom")
        self.assertIsNone(v4._last_prefill_graph_attn_body_shadow_stats)
        self.assertEqual(v4._last_prefill_graph_attn_body_shadow_error, "boom")

    def test_attn_body_shadow_clear_if_enabled(self):
        v4 = SimpleNamespace(
            _last_prefill_graph_attn_body_shadow_stats={"old": True},
            _last_prefill_graph_attn_body_shadow_error="old",
        )

        with patch.dict(prefill_forward.os.environ, {}, clear=True):
            self.assertFalse(
                prefill_forward._clear_prefill_graph_attn_body_shadow_result_if_enabled(
                    v4
                )
            )
            self.assertEqual(v4._last_prefill_graph_attn_body_shadow_stats, {"old": True})
            self.assertEqual(v4._last_prefill_graph_attn_body_shadow_error, "old")
        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW": "1"},
            clear=True,
        ):
            self.assertTrue(
                prefill_forward._clear_prefill_graph_attn_body_shadow_result_if_enabled(
                    v4
                )
            )
            self.assertIsNone(v4._last_prefill_graph_attn_body_shadow_stats)
            self.assertIsNone(v4._last_prefill_graph_attn_body_shadow_error)

    def test_attn_body_shadow_compare_sync_error_propagates(self):
        v4 = SimpleNamespace()
        shadow = torch.ones(1)
        eager = torch.ones(1)

        with patch.object(
            prefill_forward.torch.cuda,
            "synchronize",
            side_effect=RuntimeError("live eager failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "live eager failed"):
                prefill_forward._compare_prefill_graph_attn_body_shadow_tensors(
                    v4,
                    layer_idx=0,
                    shadow_out=shadow,
                    eager_out=eager,
                )
        self.assertFalse(hasattr(v4, "_last_prefill_graph_attn_body_shadow_stats"))

    def test_release_prefill_graph_shadow_kv_clears_state_owned_kv(self):
        state = SimpleNamespace(
            graph_kv_cache=object(),
            graph_kv_block_cap=64,
            _graph_kv_signature=("sig",),
        )

        prefill_forward._release_prefill_graph_shadow_kv(state)

        self.assertIsNone(state.graph_kv_cache)
        self.assertEqual(state.graph_kv_block_cap, 0)
        self.assertIsNone(state._graph_kv_signature)

    def test_graph_prefix_facts_use_cp_prefix_lengths_max(self):
        cp_ctx = SimpleNamespace(
            prefix_length=0,
            prefix_lengths=torch.tensor([0, 17], dtype=torch.int32),
        )

        (
            prefix_length,
            max_prefix_length,
            prefix_unknown,
        ) = prefill_forward._graph_prefix_facts(cp_ctx, attn_inputs=None)

        self.assertEqual(prefix_length, 17)
        self.assertEqual(max_prefix_length, 17)
        self.assertFalse(prefix_unknown)

    def test_graph_prefix_facts_uses_cp_scalar_for_single_cuda_prefix_length(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for CUDA prefix length gate")
        cp_ctx = SimpleNamespace(
            prefix_length=0,
            prefix_lengths=torch.tensor([0, 17], device="cuda", dtype=torch.int32),
        )

        (
            prefix_length,
            max_prefix_length,
            prefix_unknown,
        ) = prefill_forward._graph_prefix_facts(cp_ctx, attn_inputs=None, batch_size=1)

        self.assertEqual(prefix_length, 0)
        self.assertEqual(max_prefix_length, 0)
        self.assertFalse(prefix_unknown)

    def test_graph_prefix_facts_fail_closed_for_multi_request_cuda_prefix_lengths(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for CUDA prefix length gate")
        cp_ctx = SimpleNamespace(
            prefix_length=0,
            prefix_lengths=torch.tensor([0, 17], device="cuda", dtype=torch.int32),
        )

        (
            prefix_length,
            max_prefix_length,
            prefix_unknown,
        ) = prefill_forward._graph_prefix_facts(cp_ctx, attn_inputs=None, batch_size=2)

        self.assertEqual(prefix_length, 0)
        self.assertEqual(max_prefix_length, 0)
        self.assertTrue(prefix_unknown)

    def test_island_bridge_shape_count_is_kind_specific(self):
        from rtp_llm.models_py.modules.dsv4.prefill_island_graph import (
            PrefillIslandGraphManager,
        )

        manager = PrefillIslandGraphManager()
        current_layer = object()
        next_layer = object()
        common = SimpleNamespace(freqs_cis=torch.empty(1, 2, dtype=torch.float32))

        ffn_out = torch.empty(1, 2, dtype=torch.bfloat16)
        residual = torch.empty(1, 4, 2, dtype=torch.bfloat16)
        post = torch.empty(1, 4, 1, dtype=torch.float32)
        comb = torch.empty(1, 4, 4, dtype=torch.float32)

        plain_key = manager._bridge_key(
            current_layer, next_layer, ffn_out, residual, post, comb
        )
        qkv_key = manager._bridge_key(
            current_layer,
            next_layer,
            ffn_out,
            residual,
            post,
            comb,
            common=common,
            include_qkv=True,
        )
        qkv_q_key = manager._bridge_key(
            current_layer,
            next_layer,
            ffn_out,
            residual,
            post,
            comb,
            common=common,
            include_qkv=True,
            include_q=True,
        )
        manager._bridge_states[plain_key] = object()
        manager._bridge_states[qkv_key] = object()
        manager._bridge_states[qkv_q_key] = object()
        manager._bridge_states[
            (
                id(current_layer),
                id(current_layer),
                "attn_post_ffn_pre",
                manager._tensor_sig(ffn_out),
                manager._tensor_sig(residual),
                manager._tensor_sig(post),
                manager._tensor_sig(comb),
            )
        ] = object()

        self.assertEqual(
            manager._bridge_count_for_pair(
                current_layer, next_layer, "ffn_post_attn_pre"
            ),
            1,
        )
        self.assertEqual(
            manager._bridge_count_for_pair(
                current_layer, next_layer, "ffn_post_attn_pre_qkv"
            ),
            1,
        )
        self.assertEqual(
            manager._bridge_count_for_pair(
                current_layer, next_layer, "ffn_post_attn_pre_qkv_q"
            ),
            1,
        )
        self.assertEqual(
            manager._bridge_count_for_pair(
                current_layer, current_layer, "attn_post_ffn_pre"
            ),
            1,
        )
        self.assertEqual(
            manager._bridge_count_for_pair(current_layer, next_layer, "other"),
            0,
        )

    def test_attn_ffn_bridge_limits_default_to_512_tokens(self):
        from rtp_llm.models_py.modules.dsv4.prefill_island_graph import (
            PrefillIslandGraphManager,
        )

        manager = PrefillIslandGraphManager()
        small = (
            torch.empty(512, 8, dtype=torch.bfloat16),
            torch.empty(512, 4, 8, dtype=torch.bfloat16),
            torch.empty(512, 4, 1, dtype=torch.float32),
            torch.empty(512, 4, 4, dtype=torch.float32),
        )
        too_many_tokens = (
            torch.empty(513, 8, dtype=torch.bfloat16),
            torch.empty(513, 4, 8, dtype=torch.bfloat16),
            torch.empty(513, 4, 1, dtype=torch.float32),
            torch.empty(513, 4, 4, dtype=torch.float32),
        )

        with patch.dict(prefill_forward.os.environ, {}, clear=True):
            self.assertTrue(manager._attn_ffn_bridge_within_limits(small))
            self.assertFalse(
                manager._attn_ffn_bridge_within_limits(too_many_tokens)
            )

    def test_attn_ffn_bridge_limits_respect_memory_budget(self):
        from rtp_llm.models_py.modules.dsv4.prefill_island_graph import (
            PrefillIslandGraphManager,
        )

        manager = PrefillIslandGraphManager()
        inputs = (
            torch.empty(4, 8, dtype=torch.bfloat16),
            torch.empty(4, 4, 8, dtype=torch.bfloat16),
            torch.empty(4, 4, 1, dtype=torch.float32),
            torch.empty(4, 4, 4, dtype=torch.float32),
        )
        estimated = manager._attn_ffn_bridge_estimated_bytes(inputs)
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_ISLAND_ATTN_FFN_BRIDGE_MAX_ESTIMATED_BYTES": str(
                    estimated - 1
                )
            },
        ):
            self.assertFalse(manager._attn_ffn_bridge_within_limits(inputs))
        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_ISLAND_ATTN_FFN_BRIDGE_MAX_ESTIMATED_BYTES": str(
                    estimated
                )
            },
        ):
            self.assertTrue(manager._attn_ffn_bridge_within_limits(inputs))

    def test_attn_ffn_bridge_malformed_limits_fail_closed(self):
        from rtp_llm.models_py.modules.dsv4.prefill_island_graph import (
            PrefillIslandGraphManager,
        )

        manager = PrefillIslandGraphManager()
        inputs = (
            torch.empty(4, 8, dtype=torch.bfloat16),
            torch.empty(4, 4, 8, dtype=torch.bfloat16),
            torch.empty(4, 4, 1, dtype=torch.float32),
            torch.empty(4, 4, 4, dtype=torch.float32),
        )

        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_ISLAND_ATTN_FFN_BRIDGE_MAX_TOKENS": "bad"},
        ):
            self.assertFalse(manager._attn_ffn_bridge_within_limits(inputs))

        with patch.dict(
            prefill_forward.os.environ,
            {"DSV4_PREFILL_ISLAND_ATTN_FFN_BRIDGE_MAX_ESTIMATED_BYTES": "bad"},
        ):
            self.assertFalse(manager._attn_ffn_bridge_within_limits(inputs))

    def test_attn_ffn_bridge_returns_none_before_cp_sync_barrier(self):
        from rtp_llm.models_py.modules.dsv4.prefill_island_graph import (
            PrefillIslandGraphManager,
        )

        manager = PrefillIslandGraphManager()
        fake_cuda = SimpleNamespace(is_cuda=True)
        layer = SimpleNamespace(_cp_sync_after_attn_done=False)
        with patch.object(torch.cuda, "is_available", lambda: True), patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: False
        ):
            out = manager.run_attn_post_ffn_pre_bridge(
                layer=layer,
                attn_out=fake_cuda,
                residual=fake_cuda,
                post=fake_cuda,
                comb=fake_cuda,
            )

        self.assertIsNone(out)

    def test_disable_bridge_prefix_releases_cached_states(self):
        from rtp_llm.models_py.modules.dsv4.prefill_island_graph import (
            PrefillIslandGraphManager,
        )

        manager = PrefillIslandGraphManager()
        prefix = (1, 2, "attn_post_ffn_pre")
        other_prefix = (1, 3, "attn_post_ffn_pre")
        manager._bridge_states[prefix + ("shape_a",)] = object()
        manager._bridge_states[prefix + ("shape_b",)] = object()
        manager._bridge_states[other_prefix + ("shape_c",)] = object()
        manager._bridge_disabled.add(prefix + ("shape_disabled",))

        manager._disable_bridge_prefix(prefix)

        self.assertIn(prefix, manager._bridge_disabled_prefixes)
        self.assertNotIn(prefix + ("shape_a",), manager._bridge_states)
        self.assertNotIn(prefix + ("shape_b",), manager._bridge_states)
        self.assertNotIn(prefix + ("shape_disabled",), manager._bridge_disabled)
        self.assertIn(other_prefix + ("shape_c",), manager._bridge_states)

    def test_forward_layers_uses_fast_layer_under_token_cap(self):
        calls = []

        from rtp_llm.models_py.modules.dsv4.block import Block

        class FakeEmbed:
            def __call__(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 2)

        class FakeLayer(Block):
            def __init__(self):
                pass

            def __call__(self, h, *args, **kwargs):
                calls.append("slow_layer")
                return h + 1

            def forward_prefill_fast(self, h, *args, **kwargs):
                calls.append("fast_layer")
                return h + 2

        class FakeHead:
            def _head_impl(self, h):
                calls.append("fast_head")
                return h[:, 0, :]

        class FakeV4:
            fp8_kv_cache = False
            hc_mult = 2
            embed = FakeEmbed()
            layers = [FakeLayer()]
            head_hc = FakeHead()
            _mtp_hidden_buffer = None
            _mtp_last_hidden_buffer = None

            def _propagate_cp_ctx(self, cp_ctx):
                self.cp_ctx = cp_ctx

            def _hc_head_reduce(self, h):
                calls.append("slow_head")
                return h[:, 0, :]

            def norm(self, h):
                return h

        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        positions = torch.arange(3, dtype=torch.long)
        cu_seqlens = torch.tensor([0, 3], dtype=torch.long)

        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS": "1",
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS_MAX_TOKENS": "8",
            },
        ), patch.object(prefill_forward._rt, "ENABLED", False), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ):
            out = prefill_forward.forward_layers(
                FakeV4(),
                None,
                input_ids,
                positions,
                cu_seqlens,
                None,
            )

        self.assertEqual(calls, ["fast_layer", "fast_head"])
        expected = torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
        self.assertTrue(torch.equal(out, expected))

    def test_static_eager_gate_failure_restores_live_layer_loop_args(self):
        calls = []

        from rtp_llm.models_py.modules.dsv4.block import Block
        from rtp_llm.models_py.modules.dsv4.prefill_graph import (
            PrefillGraphKey,
            StaticPrefillGraphState,
            with_static_state_invariants,
        )

        class FakeEmbed:
            def __call__(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 2)

        class FakeLayer(Block):
            def __init__(self):
                pass

            def forward_prefill_fast(self, h, *args, **kwargs):
                calls.append(("layer_first", float(h.flatten()[0].item())))
                return h + 2

        class FakeHead:
            def _head_impl(self, h):
                calls.append(("head_first", float(h.flatten()[0].item())))
                return h[:, 0, :]

        class FakeV4:
            fp8_kv_cache = False
            hc_mult = 2
            embed = FakeEmbed()
            layers = [FakeLayer()]
            head_hc = FakeHead()
            _mtp_hidden_buffer = None
            _mtp_last_hidden_buffer = None
            _cp_size = 1

            def _propagate_cp_ctx(self, cp_ctx):
                self.cp_ctx = cp_ctx

            def _hc_head_reduce(self, h):
                raise AssertionError("small-token head should use fast head")

            def norm(self, h):
                return h

        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        positions = torch.arange(3, dtype=torch.long)
        cu_seqlens = torch.tensor([0, 3], dtype=torch.long)
        stale_hidden = torch.full((3, 2, 2), 99.0, dtype=torch.float32)
        key = with_static_state_invariants(
            PrefillGraphKey(token_bucket=3, batch_bucket=1, cp_size=1),
            local_token_bucket=3,
            hidden_shape_tail=(2, 2),
            hidden_dtype=torch.float32,
            block_cap=1,
            block_table_keys=(),
        )
        state = StaticPrefillGraphState(
            key=key,
            device="cpu",
            hidden_shape_tail=(2, 2),
            hidden_dtype=torch.float32,
            block_cap=1,
            block_table_keys=(),
        )
        state.update(
            input_ids=input_ids,
            hidden=stale_hidden,
            position_ids=positions,
            req_id_per_token=torch.tensor([0, 0, 0], dtype=torch.int32),
            cu_seqlens=cu_seqlens,
            input_lengths=torch.tensor([3], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            block_tables_by_type={},
            seq_len_full=3,
            prefix_length=0,
            meta_by_ratio=None,
        )
        v4 = FakeV4()
        v4._last_prefill_graph_state = state

        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS": "1",
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS_MAX_TOKENS": "8",
                "DSV4_PREFILL_GRAPH_MANAGER": "1",
                "DSV4_PREFILL_GRAPH_ALLOW_STATIC_EAGER": "1",
                "DSV4_PREFILL_GRAPH_BIND_STATIC_INPUTS": "1",
                "DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN": "1",
                "DSV4_PREFILL_GRAPH_BUCKETS": "3",
                "DSV4_PREFILL_GRAPH_BATCH_BUCKETS": "1",
                "DSV4_PREFILL_GRAPH_CP_SIZE": "1",
            },
            clear=True,
        ), patch.object(prefill_forward._rt, "ENABLED", False), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ):
            out = prefill_forward.forward_layers(
                v4,
                None,
                input_ids,
                positions,
                cu_seqlens,
                None,
            )

        self.assertEqual(calls, [("layer_first", 1.0), ("head_first", 3.0)])
        expected = torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
        self.assertTrue(torch.equal(out, expected))

    def test_block_fast_path_uses_hc_prefill_fast_pre(self):
        from rtp_llm.models_py.modules.dsv4 import block as block_module
        from rtp_llm.models_py.modules.dsv4.block import Block

        calls = []

        class FakeHC:
            def __init__(self, name):
                self.name = name

            def _pre_impl(self, x, dbg_tag=None):
                calls.append(f"{self.name}_slow_pre")
                return x + 100, "post", "comb"

            def _pre_impl_prefill_fast(self, x):
                calls.append(f"{self.name}_fast_pre")
                return x + 1, "post", "comb"

            def _post_impl(self, x, residual, post, comb):
                calls.append(f"{self.name}_slow_post")
                return x + residual + 100

            def _post_impl_prefill_fast(self, x, residual, post, comb):
                calls.append(f"{self.name}_fast_post")
                return x + residual

        class FakeAttention:
            _cp_ctx = None

            def __call__(self, x, positions, kv_cache=None, block_tables_by_type=None):
                calls.append("attn")
                return x + 2

        class FakeFFN:
            _strategy = SimpleNamespace(name="none")

            def __call__(self, x, input_ids):
                calls.append("ffn")
                return x + input_ids.to(x.dtype).unsqueeze(-1)

        class FakeBlock(Block):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.attn = FakeAttention()
                self.ffn = FakeFFN()
                self.attn_hc = FakeHC("attn")
                self.ffn_hc = FakeHC("ffn")
                self.attn_norm = lambda x: x + 3
                self.ffn_norm = lambda x: x + 5
                self._cp_sync_after_attn_done = False

        x = torch.ones(2, 1)
        input_ids = torch.tensor([10, 20], dtype=torch.long)
        positions = torch.arange(2, dtype=torch.long)
        cu_seqlens = torch.tensor([0, 2], dtype=torch.long)

        with patch.object(block_module, "AttentionFP8", FakeAttention):
            out = FakeBlock().forward_prefill_fast(x, input_ids, positions, cu_seqlens)

        self.assertEqual(
            calls,
            [
                "attn_fast_pre",
                "attn",
                "attn_fast_post",
                "ffn_fast_pre",
                "ffn",
                "ffn_fast_post",
            ],
        )
        expected = torch.tensor([[32.0], [42.0]])
        self.assertTrue(torch.equal(out, expected))

    def test_block_fast_path_prefers_attention_prefill_fast_entry(self):
        from rtp_llm.models_py.modules.dsv4 import block as block_module
        from rtp_llm.models_py.modules.dsv4.block import Block

        calls = []

        class FakeHC:
            def _pre_impl_prefill_fast(self, x):
                return x, "post", "comb"

            def _post_impl_prefill_fast(self, x, residual, post, comb):
                return x

        class FakeAttention:
            _cp_ctx = None

            def __call__(self, x, positions, kv_cache=None, block_tables_by_type=None):
                calls.append("attn_slow")
                return x + 100

            def forward_prefill_fast(
                self, x, positions, kv_cache=None, block_tables_by_type=None
            ):
                calls.append("attn_fast")
                return x + 1

        class FakeFFN:
            _strategy = SimpleNamespace(name="none")

            def __call__(self, x, input_ids):
                calls.append("ffn")
                return x

        class FakeBlock(Block):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.attn = FakeAttention()
                self.ffn = FakeFFN()
                self.attn_hc = FakeHC()
                self.ffn_hc = FakeHC()
                self.attn_norm = lambda x: x
                self.ffn_norm = lambda x: x
                self._cp_sync_after_attn_done = False

        x = torch.ones(2, 1)
        input_ids = torch.tensor([10, 20], dtype=torch.long)
        positions = torch.arange(2, dtype=torch.long)
        cu_seqlens = torch.tensor([0, 2], dtype=torch.long)

        with patch.object(block_module, "AttentionFP8", FakeAttention):
            out = FakeBlock().forward_prefill_fast(x, input_ids, positions, cu_seqlens)

        self.assertEqual(calls, ["attn_fast", "ffn"])
        self.assertTrue(torch.equal(out, torch.full((2, 1), 2.0)))

    def test_block_fast_path_requires_input_ids_for_fp8_attention(self):
        from rtp_llm.models_py.modules.dsv4 import block as block_module
        from rtp_llm.models_py.modules.dsv4.block import Block

        class FakeAttention:
            pass

        class FakeBlock(Block):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.attn = FakeAttention()

        with patch.object(block_module, "AttentionFP8", FakeAttention):
            with self.assertRaisesRegex(RuntimeError, "requires input_ids"):
                FakeBlock().forward_prefill_fast(
                    torch.ones(2, 1),
                    None,
                    torch.arange(2, dtype=torch.long),
                    torch.tensor([0, 2], dtype=torch.long),
                )

    def test_block_fast_path_falls_back_for_non_fp8_attention(self):
        from rtp_llm.models_py.modules.dsv4.block import Block

        calls = []

        class FakeBlock(Block):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.attn = object()

            def forward(
                self,
                x,
                input_ids,
                positions,
                cu_seqlens,
                kv_cache=None,
                block_tables_by_type=None,
            ):
                calls.append((input_ids, positions, cu_seqlens, kv_cache, block_tables_by_type))
                return x + 7

        x = torch.ones(2, 1)
        input_ids = torch.tensor([1, 2], dtype=torch.long)
        positions = torch.arange(2, dtype=torch.long)
        cu_seqlens = torch.tensor([0, 2], dtype=torch.long)
        out = FakeBlock().forward_prefill_fast(
            x,
            input_ids,
            positions,
            cu_seqlens,
            kv_cache="kv",
            block_tables_by_type="bt",
        )

        self.assertTrue(torch.equal(out, x + 7))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], input_ids)
        self.assertIs(calls[0][1], positions)
        self.assertIs(calls[0][2], cu_seqlens)
        self.assertEqual(calls[0][3:], ("kv", "bt"))

    def test_attention_prefill_fast_restores_inplace_flag(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        attn = AttentionFP8.__new__(AttentionFP8)
        attn._prefill_fast_qkv_inplace = False
        calls = []

        def fake_forward(x, positions, kv_cache=None, block_tables_by_type=None):
            calls.append(attn._prefill_fast_qkv_inplace)
            return x + 1

        attn.forward = fake_forward
        out = AttentionFP8.forward_prefill_fast(
            attn,
            torch.ones(2, 1),
            torch.arange(2, dtype=torch.long),
        )

        self.assertTrue(torch.equal(out, torch.full((2, 1), 2.0)))
        self.assertEqual(calls, [True])
        self.assertFalse(attn._prefill_fast_qkv_inplace)

    def test_attention_prefill_fast_restores_inplace_flag_on_exception(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        attn = AttentionFP8.__new__(AttentionFP8)
        attn._prefill_fast_qkv_inplace = True

        def fake_forward(x, positions, kv_cache=None, block_tables_by_type=None):
            self.assertTrue(attn._prefill_fast_qkv_inplace)
            raise RuntimeError("boom")

        attn.forward = fake_forward
        with self.assertRaisesRegex(RuntimeError, "boom"):
            AttentionFP8.forward_prefill_fast(
                attn,
                torch.ones(2, 1),
                torch.arange(2, dtype=torch.long),
            )

        self.assertTrue(attn._prefill_fast_qkv_inplace)

    def test_attention_rmsnorm_weighted_out_validation(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        attn = AttentionFP8.__new__(AttentionFP8)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        weight = torch.ones(4, dtype=torch.bfloat16)

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            AttentionFP8._rmsnorm_weighted(
                attn,
                x,
                weight,
                out=torch.empty(2, 5, dtype=torch.bfloat16),
            )
        with self.assertRaisesRegex(ValueError, "dtype/device"):
            AttentionFP8._rmsnorm_weighted(
                attn,
                x,
                weight,
                out=torch.empty_like(x, dtype=torch.float32),
            )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            AttentionFP8._rmsnorm_weighted(
                attn,
                x,
                weight,
                out=torch.empty(4, 2, dtype=torch.bfloat16).t(),
            )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.version.hip is not None,
        "NVIDIA CUDA required",
    )
    def test_prefill_fast_norm_reuses_cuda_rmsnorm_input(self):
        from rtp_llm.models_py.modules import RMSNorm
        from rtp_llm.models_py.modules.dsv4.block import _prefill_fast_norm

        torch.manual_seed(7)
        for tokens, dim in ((16, 128), (512, 7168)):
            weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16).abs()
            norm = RMSNorm(weight, eps=1e-6)
            self.assertEqual(
                norm.__class__.__module__, "rtp_llm.models_py.modules.base.cuda.norm"
            )
            x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
            ref = norm(x)
            x_inplace = x.clone()
            out = _prefill_fast_norm(norm, x_inplace)
            torch.cuda.synchronize()

            self.assertEqual(out.data_ptr(), x_inplace.data_ptr())
            self.assertTrue(torch.allclose(out.float(), ref.float(), rtol=0, atol=0))

    def test_forward_layers_uses_default_layer_over_token_cap(self):
        calls = []

        from rtp_llm.models_py.modules.dsv4.block import Block

        class FakeEmbed:
            def __call__(self, input_ids):
                return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 2)

        class FakeLayer(Block):
            def __init__(self):
                pass

            def __call__(self, h, *args, **kwargs):
                calls.append("slow_layer")
                return h + 1

            def forward_prefill_fast(self, h, *args, **kwargs):
                calls.append("fast_layer")
                return h + 2

        class FakeHead:
            def _head_impl(self, h):
                calls.append("fast_head")
                return h[:, 0, :]

        class FakeV4:
            fp8_kv_cache = False
            hc_mult = 2
            embed = FakeEmbed()
            layers = [FakeLayer()]
            head_hc = FakeHead()
            _mtp_hidden_buffer = None
            _mtp_last_hidden_buffer = None

            def _propagate_cp_ctx(self, cp_ctx):
                self.cp_ctx = cp_ctx

            def _hc_head_reduce(self, h):
                calls.append("slow_head")
                return h[:, 0, :]

            def norm(self, h):
                return h

        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        positions = torch.arange(3, dtype=torch.long)
        cu_seqlens = torch.tensor([0, 3], dtype=torch.long)

        with patch.dict(
            prefill_forward.os.environ,
            {
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS": "1",
                "DSV4_PREFILL_SMALL_TOKEN_BYPASS_MAX_TOKENS": "2",
            },
        ), patch.object(prefill_forward._rt, "ENABLED", False), patch.object(
            prefill_forward._fwd_dbg, "enabled", lambda: False
        ):
            out = prefill_forward.forward_layers(
                FakeV4(),
                None,
                input_ids,
                positions,
                cu_seqlens,
                None,
            )

        self.assertEqual(calls, ["slow_layer", "slow_head"])
        expected = torch.tensor([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()
