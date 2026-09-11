"""CPU call-order checks for CMP's explicit TRT prepared-input handoff.

Execute the production CMP/TRT methods with GPU launches replaced by spies.
These tests check dependencies and ownership, not CUDA scheduling or numerics.
"""

import gc
import unittest
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import trtllm_sparse_integration_test as integration


class CmpPrepareTest(integration.CpuOnlyTest):
    def test_token_retains_resident_mapping_until_consumed_and_released(self):
        op, launches, query, cache, topk = (
            integration.ForwardCallContractTest.make_forward_state(self)
        )
        physical = torch.zeros_like(topk)
        ref = weakref.ref(physical)
        token = op.prepare(query, cache, topk, physical_indices=physical)
        self.assertIs(token.inputs[3], physical)
        launches.reset_mock()
        del physical
        gc.collect()
        self.assertIsNotNone(ref())
        op.forward_prepared(token)
        del token
        gc.collect()
        self.assertIsNone(ref())

    def test_token_retains_inputs_until_release_without_op_retention(self):
        for rows in (0, 1):
            for view_inputs in (False, True):
                with self.subTest(rows=rows, view_inputs=view_inputs):
                    op, launches, query, cache, topk = (
                        integration.ForwardCallContractTest.make_forward_state(
                            self, tokens=rows
                        )
                    )
                    if view_inputs:
                        # The token must retain the supplied owners even when
                        # prepare makes additional dtype/shape views internally.
                        cache = cache.view(torch.float8_e4m3fn).unsqueeze(2)
                        topk = topk.unsqueeze(1)
                    refs = tuple(weakref.ref(value) for value in (query, cache, topk))
                    token = op.prepare(query, cache, topk)
                    self.assertIs(token.inputs[0], query)
                    self.assertIs(token.inputs[1], cache)
                    self.assertIs(token.inputs[2], topk)
                    # The converter spy would otherwise retain its arguments
                    # independently and hide a missing token reference.
                    launches.reset_mock()
                    del query, cache, topk
                    gc.collect()
                    self.assertTrue(all(ref() is not None for ref in refs))
                    op.forward_prepared(token)
                    launches.reset_mock()
                    gc.collect()
                    self.assertTrue(all(ref() is not None for ref in refs))
                    del token
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in refs))
                    # Keep the runner itself alive: it must not own the input
                    # tensors after the per-call handle is released.
                    self.assertEqual(op._capacity, rows)

    def make_state(self, *, indexer, long_context=False):
        calls = []
        stream_name = ["caller"]

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self):
                calls.append((stream_name[0], self.name + ".record"))

            def wait(self):
                calls.append((stream_name[0], self.name + ".wait"))

        @contextmanager
        def stream(name):
            previous = stream_name[0]
            stream_name[0] = name
            try:
                yield
            finally:
                stream_name[0] = previous

        namespace = {
            "__name__": "_cpu_cmp_prepare",
            "torch": torch,
            "W": SimpleNamespace(mla_kc="kc"),
            "_page_block_table": lambda inputs: inputs.kv_cache_block_id_device,
        }
        integration.compile_nodes(
            integration.MODULES / "hybrid/glm5_cmp.py", namespace, {"Glm5Cmp"}
        )
        cmp = object.__new__(namespace["Glm5Cmp"])
        cmp.layer_idx = 0
        cmp._events = SimpleNamespace(
            **{
                name: Event(name)
                for name in (
                    "caller_to_main",
                    "side_streams_complete",
                    "norm_to_indexer_k",
                    "qkv_to_indexer_q",
                    "indexer_q_to_score",
                    "q_path_complete",
                    "indexer_complete",
                )
            }
        )
        cmp._side_streams = lambda device: ("main", "index", "indexer_q")
        cmp._packed_head_gate_weight = object()
        cmp._qkv_projection = cmp._q_b_proj = (None, None)
        cmp._indexer_k_projection = (None, None)
        cmp._indexer_q_projection = (torch.empty((4096, 1)), None)
        norm = SimpleNamespace(
            weight=torch.ones(1), beta=torch.zeros(1), variance_epsilon=1e-5
        )
        cmp.input_layernorm = norm
        hidden = torch.zeros((1, 6144), dtype=torch.bfloat16)
        query = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
        topk = torch.zeros((1, 2048), dtype=torch.int32)

        def stage(name, result):
            def run(*args, **kwargs):
                calls.append((stream_name[0], name))
                event = kwargs.get("notify_event")
                if event is not None:
                    event.record()
                return kwargs.get("out", result)

            return run

        cmp.self_attn = SimpleNamespace(
            has_indexer=indexer,
            q_a_layernorm=norm,
            kv_a_layernorm=norm,
            indexer=SimpleNamespace(
                k_norm=norm,
                indexer_op=SimpleNamespace(_get_topk_paged=stage("topk", topk)),
            ),
        )
        op, launches, _, cache, _ = (
            integration.ForwardCallContractTest.make_forward_state(self, tokens=1)
        )
        op.num_heads = 64
        op.output = torch.empty((1, 64, 512), dtype=torch.bfloat16)
        launches.convert.side_effect = stage("convert", None)
        launches.decode.side_effect = stage("decode", None)
        launches.mask.side_effect = stage("mask", None)
        # Reuse the production op's validation/generation code. Spy on prepare
        # only to locate it relative to the real prologue's existing waits.
        prepare = op.prepare
        op.prepare = Mock(wraps=prepare)
        cmp.ops = SimpleNamespace(
            allocate_outputs=lambda _: (hidden, hidden, hidden, None),
            add_norm_quant=stage("norm", (hidden, hidden, hidden, None)),
            qkv_a_proj=stage("qkv", hidden),
            qkv_rmsnorm_quant_rope_cached=stage("kv_write", (hidden, None, cache)),
            q_b_proj=stage("q_b", (query, query)),
            absorbed_q_nope_bmm=stage("absorbed_q", None),
            indexer_q_proj=stage("indexer_q", hidden),
            indexer_q_rope_quant=stage("indexer_q_rope", (hidden, hidden)),
            indexer_k_cache=stage("indexer_k_write", None),
        )
        table = torch.ones((1, 8192 if long_context else 2), dtype=torch.int32)
        implementation = SimpleNamespace(
            fmha_impl=op,
            fmha_params=SimpleNamespace(
                slot_mapping=torch.zeros(1, dtype=torch.int64),
                positions_d=torch.zeros(1, dtype=torch.int32),
            ),
            weights=[{"kc": None}],
            attn_inputs=SimpleNamespace(kv_cache_block_id_device=table),
            _cos_sin_cache=None,
        )
        layer_cache = SimpleNamespace(
            kv_cache_base=cache,
            kv_scale_base=torch.zeros((3, 64, 1, 132), dtype=torch.uint8),
        )
        return SimpleNamespace(
            cmp=cmp,
            op=op,
            implementation=implementation,
            layer_cache=layer_cache,
            hidden=hidden,
            query=query,
            topk=topk,
            calls=calls,
            launches=launches,
            stream=stream,
        )

    def prologue(self, state, *, reuse=False):
        with patch.object(torch.cuda, "stream", side_effect=state.stream):
            return state.cmp.mla_prologue(
                state.hidden,
                state.hidden,
                state.implementation,
                state.layer_cache,
                state.topk,
                reuse_topk_indices=reuse,
            )

    def test_normal_prepare_follows_q_and_topk_before_existing_join(self):
        for long_context in (False, True):
            with self.subTest(long_context=long_context):
                state = self.make_state(indexer=True, long_context=long_context)
                _, prepared, topk = self.prologue(state)
                self.assertNotIsInstance(prepared, torch.Tensor)
                self.assertIs(topk, state.topk)
                self.assertEqual(state.op.prepare.call_count, 1)
                convert = state.calls.index(("main", "convert"))
                for before in (
                    ("main", "kv_write"),
                    ("main", "absorbed_q"),
                    ("index", "topk"),
                    ("main", "indexer_complete.wait"),
                ):
                    self.assertLess(state.calls.index(before), convert)
                for after in (
                    ("main", "side_streams_complete.record"),
                    ("caller", "side_streams_complete.wait"),
                ):
                    self.assertGreater(state.calls.index(after), convert)
                state.launches.decode.assert_not_called()
                result = state.cmp.sparse_mla(
                    prepared,
                    topk,
                    state.implementation,
                    state.layer_cache,
                )
                self.assertIs(result, state.op.output)
                self.assertEqual(
                    [call[0] for call in state.launches.mock_calls],
                    ["convert", "decode", "mask"],
                )
                self.assertEqual(
                    state.calls[-2:], [("caller", "decode"), ("caller", "mask")]
                )

    def test_reused_topk_prepares_fresh_on_each_call_after_current_kv_and_q(self):
        for has_indexer in (False, True):
            with self.subTest(has_indexer=has_indexer):
                state = self.make_state(indexer=has_indexer)
                tokens = []
                for step in range(3):
                    state.query.fill_(step)
                    state.topk.fill_(step)
                    state.layer_cache.kv_cache_base.fill_(step)
                    state.calls.clear()
                    _, prepared, topk = self.prologue(state, reuse=True)
                    tokens.append(prepared)
                    self.assertIs(topk, state.topk)
                    self.assertEqual(
                        state.calls,
                        [
                            ("caller", name)
                            for name in (
                                "norm",
                                "qkv",
                                "kv_write",
                                "q_b",
                                "absorbed_q",
                                "convert",
                            )
                        ],
                    )
                    state.cmp.sparse_mla(
                        prepared, topk, state.implementation, state.layer_cache
                    )
                self.assertEqual(len({id(token) for token in tokens}), 3)
                self.assertEqual(state.op.prepare.call_count, 3)
                self.assertEqual(state.launches.convert.call_count, 3)
                self.assertEqual(state.launches.decode.call_count, 3)
                with self.assertRaises(RuntimeError):
                    state.cmp.sparse_mla(
                        tokens[0],
                        state.topk,
                        state.implementation,
                        state.layer_cache,
                    )
                self.assertEqual(state.launches.decode.call_count, 3)

    def test_flash_and_missing_backend_keep_tensor_and_do_not_prepare(self):
        state = self.make_state(indexer=False)
        for op in (None, SimpleNamespace(prepare=Mock(), backend_name="flashmla")):
            with self.subTest(op=op):
                state.implementation.fmha_impl = op
                _, query, _ = self.prologue(state, reuse=True)
                self.assertIs(query, state.query)
                if op is not None:
                    op.prepare.assert_not_called()
        state.launches.convert.assert_not_called()

    def test_pinned_flash_path_keeps_write_and_physical_indices_contract(self):
        state = self.make_state(indexer=False)
        flash = SimpleNamespace(
            backend_name="flashmla",
            expects_paged_kv=True,
            prepare=Mock(),
            forward=Mock(),
        )
        working = SimpleNamespace(
            write=Mock(),
            backing=[torch.empty(0, dtype=torch.uint8)],
            resident=[state.layer_cache.kv_cache_base],
            physical_indices=state.topk,
        )
        state.implementation.fmha_impl = flash
        state.implementation.pinned_mla_groups = {0: (working, 0)}
        state.implementation.prefetch_kv = Mock()
        _, query, topk = self.prologue(state, reuse=True)
        self.assertIs(query, state.query)
        flash.prepare.assert_not_called()
        state.implementation.prefetch_kv.assert_called_once_with(0, state.topk)
        working.write.assert_called_once()
        state.cmp.sparse_mla(query, topk, state.implementation, state.layer_cache)
        flash.forward.assert_called_once_with(
            query,
            working.resident[0],
            topk,
            layer_id=0,
            physical_indices=working.physical_indices,
        )

    def test_tensor_direct_call_still_uses_forward(self):
        state = self.make_state(indexer=False)
        result = state.cmp.sparse_mla(
            state.query,
            state.topk,
            state.implementation,
            state.layer_cache,
        )
        self.assertIs(result, state.op.output)
        self.assertEqual(
            [call[0] for call in state.launches.mock_calls],
            ["convert", "decode", "mask"],
        )


if __name__ == "__main__":
    unittest.main()
