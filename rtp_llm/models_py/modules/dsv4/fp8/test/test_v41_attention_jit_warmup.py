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
    def test_joint_pool_layout_fallback_and_required_compilation(self):
        package = "rtp_llm.models_py.modules.dsv4.fp8"
        joint = SimpleNamespace(
            PoolLayout=SimpleNamespace,
            is_supported_layout=Mock(return_value=True),
            warmup=Mock(return_value=True),
        )
        modules = {
            "torch": SimpleNamespace(int32="int32", int64="int64"),
            package: SimpleNamespace(_v41_joint_pool=joint),
        }
        main = warmup.PoolLayout(1, 64, 64 * 288, 128, 512, 2)
        index = warmup.PoolLayout(2, 128, 128 * 68, 128, 512, 2)
        with patch.dict(sys.modules, modules):
            warmup._warm_joint_pool(main, index, 3, "private-device")
            self.assertEqual(joint.warmup.call_count, 4)
            self.assertEqual(
                {
                    (c.args[0].table_dtype, c.args[1].table_dtype)
                    for c in joint.warmup.call_args_list
                },
                {(a, b) for a in ("int32", "int64") for b in ("int32", "int64")},
            )
            for call in joint.warmup.call_args_list:
                self.assertEqual(call.args[0].entries_per_block, 64)
                self.assertEqual(call.args[1].entries_per_block, 128)
                self.assertEqual(call.args[0].owner_tokens_per_block, 512)
                self.assertEqual(call.args[0].ratio, 2)
                self.assertEqual(
                    call.kwargs, dict(cp_size=4, cp_rank=3, device="private-device")
                )
            joint.warmup.reset_mock()
            joint.is_supported_layout.return_value = False
            warmup._warm_joint_pool(main, index, 3, "private-device")
            joint.warmup.assert_not_called()
            joint.is_supported_layout.return_value = True
            joint.warmup.return_value = False
            with self.assertRaisesRegex(RuntimeError, "joint pool readback"):
                warmup._warm_joint_pool(main, index, 3, "private-device")

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

    def test_short_context_has_no_topk_width_buckets(self):
        for length in (4096, 8192, 16384):
            self.assertEqual(warmup._candidate_widths(length, 8, 2048), ())

    def test_candidate_warmup_covers_all_blocks_fit_boundary_and_visible_views(self):
        import torch

        # Execute the unchanged candidate-only section, with the preceding Q
        # preparation excluded. No GPU/JIT implementation is imported here.
        path = _FP8 / "_v41_attention_jit_warmup.py"
        tree = ast.parse(path.read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_warm_indexer_layout"
        )
        first = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "config" for t in n.targets)
        )
        last = next(
            i
            for i, n in enumerate(fn.body[first:], first)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == "width"
        )
        candidate_fn = ast.parse(
            "def candidate_only(attn, max_seq_len, device, rows, vector_stride, vector_offset, candidates):\n    pass\n"
        ).body[0]
        candidate_fn.body = fn.body[first : last + 1]
        scope = dict(vars(warmup), torch=torch)
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[candidate_fn], type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            scope,
        )
        attn = attention(20, ratio=1, producer=True)
        attn.v41_config = {"candidate_topk_blocks": 2048, "candidate_block_size": 8}

        def select(
            logits,
            visible,
            block,
            count,
            *,
            build_bitmap=True,
            mask_tail=False,
            token_indices=None,
            token_ends=None,
        ):
            if mask_tail:
                self.assertEqual(token_indices.shape, (len(logits), 512))
                self.assertEqual(token_indices.dtype, torch.int32)
                self.assertTrue(token_indices.is_contiguous())
                self.assertIs(token_ends, visible)
                self.assertTrue(token_ends.is_contiguous())
            ids = torch.zeros(
                (len(logits), min(count, (logits.shape[1] + block - 1) // block)),
                dtype=torch.int32,
            )
            return (
                ids,
                (
                    torch.zeros((len(logits), 1), dtype=torch.int32)
                    if build_bitmap
                    else None
                ),
            )

        candidates = SimpleNamespace(
            select_candidates=Mock(side_effect=select),
            is_supported=lambda *a: True,
            bitmap_is_bounded=lambda *a: True,
            build_flags=Mock(return_value=torch.zeros((2, 1), dtype=torch.int32)),
            mask_candidates=Mock(return_value=True),
        )
        with patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only fixture")
        ) as cuda_init:
            for length in (7, 8192, 16384, 16385, 32768):
                for stride in (1, 2):
                    for offset in (0, 1):
                        candidates.select_candidates.reset_mock()
                        candidates.build_flags.reset_mock()
                        candidates.mask_candidates.reset_mock()
                        scope["candidate_only"](
                            attn, length, "cpu", 4, stride, offset, candidates
                        )
                        all_calls = candidates.select_candidates.call_args_list
                        calls = [c for c in all_calls if not c.kwargs.get("mask_tail")]
                        fused = [c for c in all_calls if c.kwargs.get("mask_tail")]
                        direct = [
                            c for c in calls if c.kwargs.get("build_bitmap") is False
                        ]
                        bitmap = [
                            c for c in calls if c.kwargs.get("build_bitmap", True)
                        ]
                        widths = set(warmup._candidate_widths(length, 8, 2048)) | {
                            min(length, 16384)
                        }
                        self.assertEqual(
                            len(fused), 2 * len(widths) if stride == 1 else 0
                        )
                        if stride == 1:
                            self.assertEqual(
                                {
                                    (c.args[0].shape[1], c.kwargs["build_bitmap"])
                                    for c in fused
                                },
                                {
                                    (width, bitmap)
                                    for width in widths
                                    for bitmap in (False, True)
                                },
                            )
                        self.assertEqual({c.args[0].shape[1] for c in direct}, widths)
                        self.assertEqual(len(direct), 2 * len(widths))
                        self.assertEqual(len(bitmap), len(direct))
                        self.assertEqual(
                            {c.args[1].dtype for c in direct},
                            {torch.int32, torch.int64},
                        )
                        self.assertTrue(
                            any((c.args[0].shape[1] + 7) // 8 <= 2048 for c in direct)
                        )
                        self.assertEqual(
                            any(c.args[0].shape[1] > 16384 for c in direct),
                            length > 16384,
                        )
                        for call in direct:
                            logits, visible = call.args[:2]
                            self.assertEqual(logits.stride(0), logits.shape[1] + 256)
                            self.assertEqual(visible.stride(0), stride)
                            self.assertEqual(visible.storage_offset(), offset)
                        self.assertEqual(candidates.build_flags.call_count, len(direct))
                        self.assertEqual(
                            candidates.mask_candidates.call_count, 2 * len(direct)
                        )
            # These row-layout passes must not repeat the candidate bucket work.
            for rows in (1, 2):
                candidates.select_candidates.reset_mock()
                scope["candidate_only"](attn, 32768, "cpu", rows, 1, 0, candidates)
                candidates.select_candidates.assert_not_called()
            candidates.select_candidates.side_effect = None
            candidates.select_candidates.return_value = None
            with self.assertRaisesRegex(RuntimeError, "candidate selection"):
                scope["candidate_only"](attn, 16384, "cpu", 4, 1, 0, candidates)
            cuda_init.assert_not_called()

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

    def test_dense_and_sparse_warmup_use_required_metadata_without_toggle_helpers(self):
        import torch

        attn = attention(20, ratio=1, producer=True)
        attn.v41_config = {"candidate_topk_blocks": 2048, "candidate_block_size": 8}
        selected = torch.zeros((1, 512), dtype=torch.int32)
        meta = SimpleNamespace(
            try_score_bounds=Mock(return_value=(torch.zeros(1), torch.ones(1)))
        )
        plan = SimpleNamespace(end=torch.ones(1, dtype=torch.int32))
        sparse = SimpleNamespace(
            prepare_plan=Mock(return_value=plan),
            score=Mock(return_value=torch.zeros(1, 16392, dtype=torch.bfloat16)),
            remap=Mock(return_value=selected),
        )
        indexer = SimpleNamespace(
            is_supported=lambda *args: True,
            quantize_indexer_q=Mock(
                return_value=(torch.zeros(1, 32, 64), torch.ones(1, 32))
            ),
            PrefillIndexerKeys=lambda quant, scale: SimpleNamespace(
                quant=quant, scale=scale
            ),
            score_indexer_chunk=Mock(return_value=torch.zeros(1, 1024)),
        )
        package = SimpleNamespace(
            _v41_deepselect=SimpleNamespace(
                is_available=lambda device: True,
                try_select_sparse_tokens=Mock(return_value=selected),
            ),
            _v41_indexer_q_triton=SimpleNamespace(
                is_supported=lambda *args: True,
                try_fused_indexer_q=lambda *args: (
                    torch.zeros(1, 1, 32, 64),
                    torch.ones(1, 1, 32),
                    torch.ones(1, 1, 32),
                ),
            ),
            _v41_prefill_candidates=SimpleNamespace(),
            _v41_prefill_indexer=indexer,
            _v41_prefill_metadata=meta,
            _v41_prefill_topk=SimpleNamespace(
                is_supported=lambda *args: True,
                try_select_tokens=Mock(return_value=selected),
            ),
            _v41_sparse_prefill_indexer=sparse,
        )

        class Kernel:
            def __init__(self):
                self.launch = Mock()

            def __getitem__(self, grid):
                return self.launch

        grouped_bounds, mask_tail = Kernel(), Kernel()
        grouped_score = Mock(return_value=torch.zeros(1, 1024))
        package_name = "rtp_llm.models_py.modules.dsv4.fp8"
        modules = {
            package_name: package,
            package_name
            + "._indexer_score": SimpleNamespace(
                fp8_fp4_mqa_indexer_score=grouped_score
            ),
            package_name
            + "._v41_grouped_prefill_score": SimpleNamespace(
                _grouped_score_bounds_kernel=grouped_bounds,
                _mask_tail_kernel=mask_tail,
            ),
        }
        # Exercise SM100 branches with CPU tensors; fail on any real CUDA init.
        with patch.dict(sys.modules, modules), patch.object(
            warmup, "__package__", package_name
        ), patch.object(
            warmup,
            "__spec__",
            importlib.util.spec_from_loader(
                package_name + "._v41_attention_jit_warmup", loader=None
            ),
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 0)
        ) as capability, patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only fixture")
        ) as cuda_init:
            warmup._warm_indexer_layout(
                attn, 32768, "cpu", rows=1, vector_stride=1, vector_offset=0
            )
            meta.try_score_bounds.assert_called_once()
            capability.assert_called_once_with("cpu")
            grouped_bounds.launch.assert_called_once()
            grouped_score.assert_called_once()
            mask_tail.launch.assert_called_once()
            # Each visible dtype warms both plain and mixed-request lookup plans.
            self.assertEqual(sparse.prepare_plan.call_count, 8)
            self.assertEqual(
                {
                    call.kwargs.get("positions_ratio")
                    for call in sparse.prepare_plan.call_args_list
                },
                {None, 1, 2},
            )
            self.assertEqual(sparse.remap.call_count, 4)
            for i, dtype in enumerate((torch.int32, torch.int64)):
                plain, lookup, ratio1, ratio2 = sparse.prepare_plan.call_args_list[
                    4 * i : 4 * i + 4
                ]
                self.assertEqual(ratio1.kwargs["positions_ratio"], 1)
                self.assertEqual(ratio2.kwargs["positions_ratio"], 2)
                self.assertEqual(ratio1.args[1].dtype, dtype)
                self.assertEqual(ratio2.args[1].dtype, dtype)
                self.assertEqual(plain.args[1].dtype, dtype)
                self.assertEqual(plain.kwargs, {})
                self.assertEqual(lookup.kwargs["request_ids"].dtype, torch.int64)
                self.assertEqual(lookup.kwargs["request_ids"].tolist(), [1])
                counts = lookup.kwargs["request_key_counts"]
                self.assertEqual(counts.dtype, torch.int32)
                self.assertEqual(counts.tolist(), [16392, 16392])
                self.assertEqual(lookup.args[2], 16640 + 16392)
            # Supported warmup work must fail loudly if a helper stops launching.
            meta.try_score_bounds.return_value = None
            with self.assertRaisesRegex(RuntimeError, "score bounds"):
                warmup._warm_indexer_layout(
                    attn, 32768, "cpu", rows=1, vector_stride=1, vector_offset=0
                )
            cuda_init.assert_not_called()

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
