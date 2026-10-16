"""Forward cache lifetime, workspace budgeting and geometry isolation."""

import importlib.util
import random
import sys
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

_KDA_SPEC = importlib.util.spec_from_file_location(
    "kda_state_metadata_under_test",
    Path(__file__).parents[3] / "triton_kernels/kimi_kda/chunk_delta_h.py",
)
kda_meta = importlib.util.module_from_spec(_KDA_SPEC)
_KDA_SPEC.loader.exec_module(kda_meta)


class ForwardMetadataTest(unittest.TestCase):
    def setUp(self):
        self.inputs = SimpleNamespace(attention_inputs=SimpleNamespace(is_prefill=True))
        self.model = SimpleNamespace(config=SimpleNamespace(model_type="glm5_3_flash"))
        self.patch_object(meta, "_SCORE_WORKSPACE_MB", 32)
        self.query = self.start_patch(
            mock.patch("torch.cuda.mem_get_info", return_value=(1 << 30, 2 << 30))
        )
        self.allocator = self.start_patch(
            mock.patch("torch.cuda.memory_stats_as_nested_dict", return_value={})
        )
        self.start_patch(mock.patch("torch.cuda.current_device", return_value=0))

    def start_patch(self, patch):
        value = patch.start()
        self.addCleanup(patch.stop)
        return value

    def patch_object(self, obj, name, value):
        return self.start_patch(mock.patch.object(obj, name, value))

    def test_prefix_summary_once_and_next_forward_refresh(self):
        prefix = torch.tensor([8192, 0], dtype=torch.int32)
        transfers = []
        self.start_patch(
            mock.patch.object(torch.Tensor, "cpu", lambda t: transfers.append(t) or t)
        )
        for expected in ((True, False), (True, True)):
            token = meta._FORWARD_METADATA.set({})
            try:
                for _ in range(11):
                    self.assertEqual(
                        meta.cached_prefill_prefix_summary(prefix), expected
                    )
            finally:
                meta._FORWARD_METADATA.reset(token)
            prefix[1] = 128
        self.assertEqual(len(transfers), 2)

    def test_logical_indexer_lengths_reuse_and_match_reference(self):
        cu = torch.tensor([0, 419, 837, 1349], dtype=torch.int32)
        lengths = torch.diff(cu)
        prefix = torch.tensor([8192, 4096, 0], dtype=torch.int32)
        token = meta._FORWARD_METADATA.set({})
        try:
            meta.metadata_cache()[("prefill_cu_host", meta.tensor_key(cu))] = tuple(
                cu.tolist()
            )
            a = meta.cached_indexer_host_lengths(cu, prefix, lengths, 4, "cpu")
            b = meta.cached_indexer_host_lengths(cu, prefix, lengths, 4, "cpu")
            self.assertIs(a, b)
            expected = (prefix.long() + lengths.long()) // 4
            torch.testing.assert_close(a[0].long(), expected, rtol=0, atol=0)
            torch.testing.assert_close(
                a[1].long(),
                torch.cat((torch.zeros(1, dtype=torch.int64), expected.cumsum(0))),
                rtol=0,
                atol=0,
            )
            self.assertEqual(a[2], int(expected.sum()))
            self.assertEqual(a[3], 8611)
        finally:
            meta._FORWARD_METADATA.reset(token)

    def test_indexer_bounds_exact_reuse_and_scope_refresh(self):
        req = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)
        positions = torch.tensor([8192, 8193, 128, 129, 130, 0], dtype=torch.int32)
        counts = torch.tensor([2048, 32, 0], dtype=torch.int32)
        cu = torch.tensor([0, 2048, 2080, 2080], dtype=torch.int32)
        with mock.patch.dict("os.environ", {"GLM53_INDEXER_SHARED_BOUNDS": "1"}):
            for _ in range(2):
                expected_ks = cu.long()[req.long()].int()
                expected_ke = (
                    expected_ks.long()
                    + torch.minimum(
                        (positions.long() + 1) // 4, counts.long()[req.long()]
                    )
                ).int()
                token = meta._FORWARD_METADATA.set({})
                try:
                    a = meta.indexer_varlen_bounds(req, positions, counts, cu, 4, "cpu")
                    for _ in range(11):
                        self.assertIs(
                            a,
                            meta.indexer_varlen_bounds(
                                req, positions, counts, cu, 4, "cpu"
                            ),
                        )
                    self.assertTrue(torch.equal(a[1], expected_ks))
                    self.assertTrue(torch.equal(a[2], expected_ke))
                    self.assertTrue(torch.equal(a[3], expected_ks))
                    different_ratio = meta.indexer_varlen_bounds(
                        req, positions, counts, cu, 8, "cpu"
                    )
                    self.assertIsNot(a, different_ratio)
                    changed = positions.clone() + 16
                    b = meta.indexer_varlen_bounds(req, changed, counts, cu, 4, "cpu")
                    self.assertIsNot(a, b)
                    # The cache owns the source to prevent pointer recycling.
                    self.assertTrue(
                        any(
                            value[0][1] is positions
                            for value in meta.metadata_cache().values()
                        )
                    )
                finally:
                    meta._FORWARD_METADATA.reset(token)
                counts.add_(8)
                cu[1:] += torch.tensor([8, 16, 24], dtype=torch.int32)
                positions.add_(4)
            a = meta.indexer_varlen_bounds(req, positions, counts, cu, 4, "cpu")
            b = meta.indexer_varlen_bounds(req, positions, counts, cu, 4, "cpu")
            self.assertIsNot(a, b)

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

    def test_kda_boundaries_shared_and_refreshed_for_reused_tensor(self):
        self.start_patch(
            mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.modules.dsv4.forward_metadata": meta,
                },
            )
        )
        self.start_patch(
            mock.patch.dict("os.environ", {"RTP_KDA_SHARED_HOST_METADATA": "1"})
        )
        boundaries = torch.tensor([0, 32, 64], dtype=torch.int32)
        transfers = []
        self.start_patch(
            mock.patch.object(
                torch.Tensor, "cpu", lambda tensor: transfers.append(tensor) or tensor
            )
        )

        @meta.scoped_forward_metadata
        def forward(_, inputs, expected):
            for _ in range(34):
                self.assertEqual(
                    kda_meta._sequence_boundaries(boundaries, 64), expected
                )

        forward(self.model, self.inputs, [0, 32, 64])
        self.assertEqual(len(transfers), 1)
        original_ptr = boundaries.data_ptr()
        boundaries[1] = 16
        self.assertEqual(boundaries.data_ptr(), original_ptr)
        forward(self.model, self.inputs, [0, 16, 64])
        self.assertEqual(len(transfers), 2)

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
        self.allocator.return_value = {
            "reserved_bytes": {"all": {"current": 24 << 20}},
            "allocated_bytes": {"all": {"current": 8 << 20}},
        }
        self.assertEqual(meta.score_workspace_budget("cuda:0"), 16 << 20)
        self.allocator.assert_called_once_with("cuda:0")

    def test_saturated_budget_needs_no_cuda_query(self):
        self.allocator.return_value = {
            "reserved_bytes": {"all": {"current": 80 << 20}},
            "allocated_bytes": {"all": {"current": 16 << 20}},
        }
        self.assertEqual(meta.score_workspace_budget("cuda:0"), 32 << 20)
        self.query.assert_not_called()
        # Falling below the saturation point must still query current free
        # memory; no previous forward's budget or free-byte estimate is reused.
        self.query.return_value = (0, 1 << 30)
        self.allocator.return_value = {
            "reserved_bytes": {"all": {"current": 63 << 20}},
        }
        self.assertEqual(meta.score_workspace_budget("cuda:0"), (63 << 20) // 2)
        self.query.assert_called_once()

    def test_demand_budget_expires_and_falls_back_to_current_memory(self):
        self.patch_object(meta, "_SCORE_WORKSPACE_DEMAND", True)
        self.allocator.return_value = {
            "reserved_bytes": {"all": {"current": 24 << 20}},
        }

        @meta.scoped_forward_metadata
        def forward(_, inputs):
            if self.allocator.return_value:
                self.assertTrue(meta.score_workspace_fits("cuda:0", 12 << 20))
                self.assertFalse(meta.score_workspace_fits("cuda:0", (12 << 20) + 1))
                self.query.assert_not_called()
            else:
                self.assertFalse(meta.score_workspace_fits("cuda:0", 1))
                self.assertEqual(meta.score_workspace_budget("cuda:0"), 8 << 20)
                self.assertTrue(meta.score_workspace_fits("cuda:0", 8 << 20))
                self.query.assert_called_once()

        forward(self.model, self.inputs)
        self.assertFalse(meta.score_workspace_fits("cuda:0", 1))
        self.allocator.return_value = {}
        self.query.return_value = (16 << 20, 1 << 30)
        forward(self.model, self.inputs)

    def test_allocator_snapshot_matches_scalar_statistics(self):
        # Keep the real public scalar functions, replacing only their native
        # statistics source. Include missing keys and over-allocation, whose
        # reusable contribution must be clamped to zero.
        for reserved, allocated in ((0, 0), (11, 7), (7, 11), (1 << 40, 123)):
            self.allocator.return_value = {
                "reserved_bytes": {"all": {"current": reserved}},
                "allocated_bytes": {"all": {"current": allocated}},
                "irrelevant": {"small_pool": {"peak": 999}},
            }
            # cuda re-exports the function; the scalar wrappers resolve it
            # through the defining memory module's globals.
            with mock.patch.object(
                torch.cuda.memory, "memory_stats_as_nested_dict", self.allocator
            ):
                expected = max(
                    0,
                    torch.cuda.memory_reserved("cuda:0")
                    - torch.cuda.memory_allocated("cuda:0"),
                )
            self.allocator.reset_mock()
            self.assertEqual(meta._reusable_allocator_bytes("cuda:0"), expected)
            self.allocator.assert_called_once_with("cuda:0")
        self.allocator.return_value = {}
        self.assertEqual(meta._reusable_allocator_bytes("cuda:0"), 0)
        self.query.assert_not_called()

    def test_demand_bound_preserves_unchunked_row_geometry(self):
        rng = random.Random(193905)
        for _ in range(10000):
            rows = rng.choice((1, 81, 127, 128, 129, 255, 256, 257, 9889, 32768))
            row_bytes = rng.randrange(1, 200000)
            required = ((rows + 127) // 128 * 128) * row_bytes
            cap = 32 << 20
            bound = min(cap, required + rng.choice((-1, 0, 1, 128 * row_bytes)))
            token = meta._FORWARD_METADATA.set(
                {("indexer_score_budget_lower", torch.device("cuda:0")): bound}
            )
            try:
                if meta.score_workspace_fits("cuda:0", required):
                    free = rng.randrange(0, 1 << 32)
                    original_budget = min(cap, (free + 2 * bound) // 2)
                    original_rows = min(
                        rows, max(1, (original_budget // row_bytes) // 128 * 128)
                    )
                    self.assertEqual(original_rows, rows)
            finally:
                meta._FORWARD_METADATA.reset(token)
        self.query.assert_not_called()

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
