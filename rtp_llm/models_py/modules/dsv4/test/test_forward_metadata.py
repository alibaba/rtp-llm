"""Forward cache lifetime, workspace budgeting and geometry isolation."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

_SPEC = importlib.util.spec_from_file_location(
    "forward_metadata_under_test", Path(__file__).parents[1] / "forward_metadata.py"
)
meta = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(meta)


class ForwardMetadataTest(unittest.TestCase):
    def setUp(self):
        self.inputs = SimpleNamespace(attention_inputs=SimpleNamespace(is_prefill=True))
        self.model = SimpleNamespace(config=SimpleNamespace(model_type="glm5_3_flash"))
        self.patch_object(meta, "_SCORE_WORKSPACE_MB", 32)
        self.query = self.start_patch(
            mock.patch("torch.cuda.mem_get_info", return_value=(1 << 30, 2 << 30))
        )
        self.start_patch(mock.patch("torch.cuda.memory_reserved", return_value=0))
        self.start_patch(mock.patch("torch.cuda.memory_allocated", return_value=0))
        self.start_patch(mock.patch("torch.cuda.current_device", return_value=0))

    def start_patch(self, patch):
        value = patch.start()
        self.addCleanup(patch.stop)
        return value

    def patch_object(self, obj, name, value):
        return self.start_patch(mock.patch.object(obj, name, value))

    def test_workspace_reused_once_per_forward(self):

        @meta.scoped_forward_metadata
        def forward(_, inputs):
            self.assertIsNotNone(meta.metadata_cache())
            for _ in range(12):
                self.assertEqual(meta.score_workspace_budget("cuda:0"), 32 << 20)

        forward(self.model, self.inputs)
        self.assertEqual(self.query.call_count, 1)
        forward(self.model, self.inputs)
        self.assertEqual(self.query.call_count, 2)

    def test_new_scope_for_reused_inputs_and_exception(self):
        scopes = []

        @meta.scoped_forward_metadata
        def forward(_, inputs, fail=False):
            cache = meta.metadata_cache()
            self.assertNotIn("value", cache)
            scopes.append(cache)
            cache["value"] = object()
            if fail:
                raise ValueError("test failure")

        forward(self.model, self.inputs)
        with self.assertRaisesRegex(ValueError, "test failure"):
            forward(self.model, self.inputs, fail=True)
        forward(self.model, self.inputs)
        self.assertEqual(len({id(cache) for cache in scopes}), 3)
        self.assertIsNone(meta.metadata_cache())

    def test_nested_forward_restores_outer_scope(self):
        @meta.scoped_forward_metadata
        def inner(_, inputs):
            self.assertNotIn("outer", meta.metadata_cache())
            meta.metadata_cache()["inner"] = True

        @meta.scoped_forward_metadata
        def outer(_, inputs):
            cache = meta.metadata_cache()
            cache["outer"] = True
            inner(self.model, inputs)
            self.assertIs(meta.metadata_cache(), cache)
            self.assertTrue(cache["outer"])
            self.assertNotIn("inner", cache)
            meta.score_workspace_budget("cuda:0")

        outer(self.model, self.inputs)
        self.assertEqual(self.query.call_count, 2)

    def test_decode_does_not_query_memory(self):
        self.inputs.attention_inputs.is_prefill = False

        @meta.scoped_forward_metadata
        def forward(_, inputs):
            self.assertIsNone(meta.metadata_cache())

        forward(self.model, self.inputs)
        self.query.assert_not_called()

    def test_other_models_do_not_query_memory(self):
        self.model.config.model_type = "kimi_linear"

        @meta.scoped_forward_metadata
        def forward(_, inputs):
            self.assertIsNone(meta.metadata_cache())

        forward(self.model, self.inputs)
        self.query.assert_not_called()

    def test_target_verify_does_not_query_memory(self):
        self.inputs.attention_inputs.is_target_verify = True

        @meta.scoped_forward_metadata
        def forward(_, inputs):
            self.assertIsNone(meta.metadata_cache())

        forward(self.model, self.inputs)
        self.query.assert_not_called()

    def test_glm53_mtp_has_its_own_scope(self):
        self.model.config.model_type = "glm_5_mtp"
        self.model.config.is_glm53_mtp = True

        @meta.scoped_forward_metadata
        def forward(_, inputs):
            self.assertIsNotNone(meta.metadata_cache())
            meta.score_workspace_budget("cuda:0")

        forward(self.model, self.inputs)
        self.assertEqual(self.query.call_count, 1)
        self.assertIsNone(meta.metadata_cache())

    def test_budget_caps_at_available_and_reusable_memory(self):
        self.query.return_value = (16 << 20, 1 << 30)
        with mock.patch(
            "torch.cuda.memory_reserved", return_value=24 << 20
        ), mock.patch("torch.cuda.memory_allocated", return_value=8 << 20):
            self.assertEqual(meta.score_workspace_budget("cuda:0"), 16 << 20)

    def test_cp_geometry_is_part_of_cache_identity(self):
        padding = torch.ones(32, dtype=torch.int32)
        restore = torch.arange(32)

        def info(lengths):
            return SimpleNamespace(
                prefill_qkv_padding_mask=padding,
                prefill_qkv_restore_indice=restore,
                prefill_actual_input_lengths_cpu=torch.tensor(lengths),
                prefill_cp_chunk_lengths=torch.tensor([2, 2]),
            )

        @meta.cached_cp_context
        def build(cp_info, *args):
            return tuple(cp_info.prefill_actual_input_lengths_cpu.tolist())

        a, b = info([15, 16]), info([16, 15])
        token = meta._FORWARD_METADATA.set({})
        try:
            self.assertEqual(build(a, 8, 0, 4, "cpu"), (15, 16))
            self.assertEqual(build(b, 8, 0, 4, "cpu"), (16, 15))
        finally:
            meta._FORWARD_METADATA.reset(token)

    def test_indexer_reuses_geometry_but_separates_physical_tables(self):
        ctx = object()
        state_table = torch.tensor([[1, 2]], dtype=torch.int32)
        kv_table = torch.tensor([[3, 4]], dtype=torch.int32)
        calls = []

        @meta.cached_indexer_prefill
        def prepare(layer, **kwargs):
            result = object()
            calls.append(result)
            return result

        def layer():
            return SimpleNamespace(
                _cp_ctx=ctx,
                compressor=SimpleNamespace(kpool_mode=True),
                compress_ratio=4,
                _state_block_table=state_table,
                _kv_pool_view=torch.empty(4, 132),
                _state_eb=128,
                _kv_eb=132,
                _state_tokens_per_block=128,
                _kv_tokens_per_block=32,
                _kv_owner_tokens_per_block=4,
            )

        first, second = layer(), layer()
        token = meta._FORWARD_METADATA.set({})
        try:
            a = prepare(first, kv_block_table=kv_table, device=torch.device("cpu"))
            self.assertIs(
                a, prepare(second, kv_block_table=kv_table, device=torch.device("cpu"))
            )
            b = prepare(
                second, kv_block_table=kv_table.clone(), device=torch.device("cpu")
            )
            self.assertIsNot(a, b)
            second._state_block_table = state_table.clone()
            c = prepare(second, kv_block_table=kv_table, device=torch.device("cpu"))
            self.assertIsNot(a, c)
            second._kv_owner_tokens_per_block = 8
            d = prepare(second, kv_block_table=kv_table, device=torch.device("cpu"))
            self.assertIsNot(c, d)
            self.assertEqual(len(calls), 4)
        finally:
            meta._FORWARD_METADATA.reset(token)


if __name__ == "__main__":
    unittest.main()
