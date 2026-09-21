"""Host-only startup plan contracts; these tests never import a CUDA runtime."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_FP8 = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_v41_warmup_cpu_contract", _FP8 / "_v41_attention_jit_warmup.py"
)
warmup = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = warmup
_SPEC.loader.exec_module(warmup)


def attention(layer, ratio=2, producer=False):
    return SimpleNamespace(
        layer_id=layer,
        v41_config={},
        compress_ratio=ratio,
        n_heads=64,
        head_dim=512,
        window_size=128,
        index_topk=512,
        index_n_heads=32,
        index_head_dim=128,
        eps=1e-6,
        freqs_cis=SimpleNamespace(shape=(524288, 32)),
        is_kv_source=producer,
    )


class V41AttentionWarmupCPU(unittest.TestCase):
    def test_collects_live_ratios_and_deduplicates_layers(self):
        modules = [
            attention(0, 0),
            attention(2),
            attention(8, producer=True),
            attention(20, 1, True),
        ]
        v4 = SimpleNamespace(layers=[SimpleNamespace(attn=a) for a in modules])
        collected = warmup._collect_attentions(v4)
        self.assertEqual({key[0] for key in collected}, {0, 1, 2})
        self.assertEqual(len(collected), 3)
        self.assertIn(modules[2], collected.values())

    def test_model_and_norm_layouts_are_not_coalesced(self):
        a, b = attention(2), attention(8)
        b.freqs_cis = SimpleNamespace(shape=(1048576, 32))
        self.assertNotEqual(warmup._attention_key(a), warmup._attention_key(b))
        b.freqs_cis = a.freqs_cis
        b.global_norm = SimpleNamespace(dtype="bfloat16")
        self.assertNotEqual(warmup._attention_key(a), warmup._attention_key(b))

    def test_deferred_pool_collection_never_touches_attention(self):
        class Unreadable:
            def values(self):
                raise AssertionError("No pool is bound")

        self.assertEqual(warmup._collect_pool_layouts(Unreadable(), None, 4, True), {})

    def test_real_schema_read_uses_metadata_and_reconstructs_cp_swa_stride(self):
        # These objects intentionally have no tensor read/write/view interface.
        # Collecting warmup layouts must not inspect real cache contents.
        class Base:
            def __init__(self, columns):
                self.shape = (10000, columns)

            def element_size(self):
                return 1

        attn = attention(2, producer=True)
        u8, f32 = SimpleNamespace(itemsize=1), SimpleNamespace(itemsize=4)
        attn._pool_spec = {7: (u8, 528), 1: (u8, 288), 3: (u8, 68), 5: (f32, 1024)}
        sizes = {7: 19072, 1: 128 * 288, 3: 256 * 68, 5: 2 * 1024 * 4}
        cache = SimpleNamespace(
            seq_size_per_block=4096,
            get_layer_cache=lambda layer, region: SimpleNamespace(
                kv_cache_base=Base(sizes[region])
            ),
        )
        modules = {
            "rtp_llm.models_py.modules.dsv4.attn_type": SimpleNamespace(
                CSA_KV=1, HCA_KV=2, INDEXER_KV=3, CSA_STATE=5, SWA_KV=7
            ),
            "rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils": SimpleNamespace(
                require_pool_tokens_per_block=lambda cache, region: (
                    4096 if region in (5, 7) else 256
                )
            ),
            "rtp_llm.models_py.modules.dsv4.fp8.attention": SimpleNamespace(
                _ATTN_TYPE_ENUM_BY_INT={i: i for i in sizes}
            ),
        }
        with patch.dict(sys.modules, modules):
            layouts = warmup._collect_pool_layouts(
                {warmup._attention_key(attn): attn}, cache, 4, True
            )
        by_region = {layout.region: layout for layout, _ in layouts}
        self.assertEqual(set(by_region), {1, 3, 5, 7})
        self.assertEqual(by_region[7].stride_bytes, 19072 * 4)
        self.assertEqual(by_region[7].entries, (19072 * 4) // 528)
        self.assertEqual(by_region[1].entries, 128)
        self.assertEqual(by_region[3].entries, 256)
        self.assertEqual(by_region[5].entries, 2)
        self.assertEqual(by_region[1].owner_tokens_per_block, 4096)

    def test_required_wrapper_rejection_cannot_be_marked_warmed(self):
        for result in (False, None):
            with self.assertRaisesRegex(RuntimeError, "rejected reachable"):
                warmup._require_launch(result, "producer")
            self.assertIs(
                warmup._require_launch(result, "disabled", enabled=False), result
            )
        result = object()
        self.assertIs(warmup._require_launch(result, "producer"), result)

    def test_short_context_does_not_warm_unreachable_candidates(self):
        for length in (4096, 8192, 16384):
            self.assertEqual(warmup._candidate_widths(length, 8, 2048), ())

    def test_every_reachable_bitmap_tile_is_covered_at_512k(self):
        widths = warmup._candidate_widths(524288, 8, 2048)
        self.assertLess(len(widths), 20)
        self.assertEqual(widths[-1], 524288)
        tiles = {1 << (((n + 255) // 256) - 1).bit_length() for n in widths}
        self.assertEqual(tiles, {128, 256, 512, 1024, 2048})

    def test_indexer_warmup_covers_unbound_bounds_pointer_layouts(self):
        layouts = warmup._indexer_warmup_layouts()
        self.assertEqual(len(layouts), 12)
        self.assertEqual({rows for rows, _, _ in layouts}, {1, 2, 4})
        self.assertEqual({stride for _, stride, _ in layouts}, {1, 2})
        self.assertEqual({offset for _, _, offset in layouts}, {0, 1})
        # [3,M] unbind yields (row_ke, end) at byte offsets (4M, 8M).
        self.assertEqual(
            {(4 * rows % 16 == 0, 8 * rows % 16 == 0) for rows, _, _ in layouts},
            {(True, True), (False, True), (False, False)},
        )
        calls = []
        with patch.object(
            warmup,
            "_warm_indexer_layout",
            side_effect=lambda *args, **kw: calls.append(kw),
        ):
            warmup._warm_indexer(object(), 524288, "cuda")
        self.assertEqual(
            {(c["rows"], c["vector_stride"], c["vector_offset"]) for c in calls},
            set(layouts),
        )

    def test_slot_table_alignment_is_independent_of_position_alignment(self):
        self.assertEqual(
            set(warmup._slot_metadata_layouts()),
            {(0, 0), (0, 1), (1, 0), (1, 1)},
        )

    def test_swa_metadata_warmup_calls_all_production_helpers(self):
        source = ast.parse((_FP8 / "_v41_attention_jit_warmup.py").read_text())
        fn = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name == "_warm_swa_metadata"
        )
        calls = {
            n.func.attr
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        self.assertTrue(
            {
                "compute_window_topk_and_length_varlen",
                "compute_prefill_gather_lens",
                "compute_swa_slot_mapping",
                "compute_swa_slot_mapping_from_positions",
                "compute_swa_slot_in_flat_from_cu",
            }
            <= calls
        )

    def test_swa_metadata_uses_runtime_batch_and_real_page_schema(self):
        class Tensor:
            def __init__(self, values, dtype):
                self.values, self.dtype = values, dtype

            def __getitem__(self, item):
                return Tensor(self.values[item], self.dtype)

            def view(self, *shape):
                return self

            def to(self, dtype):
                return Tensor(self.values, dtype)

        def filled(shape, value, *, dtype, device):
            self.assertEqual(device, "private-dummy-device")
            count = shape if isinstance(shape, int) else shape[0]
            return Tensor([value] * count, dtype)

        fake_torch = SimpleNamespace(
            int32="int32",
            int64="int64",
            arange=lambda *args, dtype, device: Tensor(list(range(*args)), dtype),
            full=filled,
            ones=lambda shape, **kw: filled(shape, 1, **kw),
            zeros=lambda shape, **kw: filled(shape, 0, **kw),
        )
        ops = Mock()
        planner = Mock(return_value=(1, 3, 17))
        modules = {
            "torch": fake_torch,
            "rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup": SimpleNamespace(
                _swa_slot_batch_block_warmup_sizes=planner
            ),
            "rtp_llm.models_py.modules.dsv4.fp8": SimpleNamespace(_swa_ops_triton=ops),
        }
        layout = warmup.PoolLayout(7, 72, 72 * 528, 8192, 8192, 0)
        with patch.dict(sys.modules, modules):
            warmup._warm_swa_metadata(
                SimpleNamespace(window_size=63), layout, 17, "private-dummy-device"
            )
        planner.assert_called_once_with(17)
        mapping = ops.compute_swa_slot_mapping.call_args.kwargs
        self.assertEqual(mapping["pool_entries_per_block"], 72)
        self.assertEqual(mapping["ring_entries"], 72)
        self.assertEqual(mapping["tokens_per_block_for_block_table"], 8192)
        flat = ops.compute_swa_slot_in_flat_from_cu.call_args_list
        self.assertEqual(
            [call.kwargs["num_tokens"] for call in flat], [2, 2, 6, 6, 34, 34]
        )
        self.assertTrue(all(call.kwargs["window_size"] == 63 for call in flat))
        self.assertEqual({call.args[1].dtype for call in flat}, {"int32", "int64"})

    def test_ratio_one_compression_uses_none_previous_state(self):
        source = ast.parse((_FP8 / "_v41_attention_jit_warmup.py").read_text())
        fn = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name == "_warm_pool"
        )
        previous = [
            n.value
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "previous" for t in n.targets)
        ]
        self.assertEqual(len(previous), 1)
        self.assertIsInstance(previous[0], ast.IfExp)
        self.assertIsNone(ast.literal_eval(previous[0].orelse))

    def test_reuse_lengths_are_runtime_jit_arguments(self):
        # Compile-contract regression: arbitrary reuse must not key the K2048
        # sort, metadata bounds, or dense finite filtering on exact lengths.
        cases = {
            "_v41_sparse_prefill_indexer.py": {
                "_prepare_sparse_prefill_plan_kernel": {"KEY_COUNT"}
            },
            "_v41_prefill_metadata.py": {
                "_prefill_slots_kernel": {"ROWS", "REQUESTS", "COLS", "TABLE_STRIDE"},
                "_prefill_bounds_kernel": {"ROWS", "WIDTH"},
            },
            "_v41_prefill_topk.py": {
                "_prefill_topk_bounds_kernel": {"ROWS", "WIDTH"},
                "_prefill_topk_finite_kernel": {"STRIDE"},
            },
            "_v41_prefill_global.py": {
                "_prefill_compress_main_kernel": {"POOL_STRIDE", "BLOCKS"},
                "_prefill_state_store_kernel": {"STATE_ROWS"},
                "_prefill_index_store_kernel": {"POOL_STRIDE", "BLOCKS"},
            },
            "_v41_prefill_candidates.py": {
                "_prefill_candidate_pool_kernel": {"WIDTH", "STRIDE", "NBLOCKS"},
                "_prefill_candidate_store_bitmap_kernel": {"FLAG_WORDS", "FLAG_STRIDE"},
                "_prefill_candidate_mask_kernel": {"WIDTH", "STRIDE", "FLAG_STRIDE"},
                "_prefill_candidate_build_bitmap_kernel": {
                    "NBLOCKS",
                    "FLAG_WORDS",
                    "FLAG_STRIDE",
                },
            },
        }
        for filename, functions in cases.items():
            nodes = {
                node.name: node
                for node in ast.parse((_FP8 / filename).read_text()).body
                if isinstance(node, ast.FunctionDef)
            }
            for name, dynamic in functions.items():
                with self.subTest(kernel=name):
                    node = nodes[name]
                    arguments = {arg.arg: arg.annotation for arg in node.args.args}
                    for argument in dynamic:
                        self.assertIsNone(arguments[argument])
                    decorators = [
                        d for d in node.decorator_list if isinstance(d, ast.Call)
                    ]
                    skipped = set()
                    for decorator in decorators:
                        for keyword in decorator.keywords:
                            if keyword.arg == "do_not_specialize":
                                skipped.update(ast.literal_eval(keyword.value))
                    self.assertTrue(dynamic <= skipped)


if __name__ == "__main__":
    unittest.main()
