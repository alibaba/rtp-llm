"""CPU contracts for physical decoder SWA table/metadata isolation.

Load the small Python helpers directly; the CUDA translator is replaced only
at its kernel boundary with a scalar indexing oracle. No model/native import
or GPU allocation is required.
"""

import ast
import contextlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "rtp_llm.models_py.modules.dsv4"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NativeEnum:
    """Pybind-like enum with deliberately no equality to Python integers."""

    def __init__(self, value):
        self.value = value

    def __int__(self):
        return self.value


class Cache:
    def __init__(self, decoder_layers=(21, 22), count=23):
        self.group_region_names = [NativeEnum(7), NativeEnum(8)]
        self.rows = [[-1] * 9 for _ in range(count)]
        for layer, row in enumerate(self.rows):
            row[8 if layer in decoder_layers else 7] = (
                1 if layer in decoder_layers else 0
            )
        self.map_reads = 0

    @property
    def layer_region_to_group_id(self):
        self.map_reads += 1
        return self.rows


class Meta(NamedTuple):
    region: int
    common: object
    workspace: object = None


def read_oracle(
    reqs, tables, positions, entries_per_block, tokens_per_block_for_block_table
):
    out = torch.full_like(positions, -1)
    for token in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            pos = int(positions[token, j])
            if pos < 0:
                continue
            block = int(
                tables[int(reqs[token]), pos // tokens_per_block_for_block_table]
            )
            if block > 0:
                out[token, j] = block * entries_per_block + pos % entries_per_block
    return out


class BoundedReplayCacheTest(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        load(PREFIX + ".attn_type", "attn_type.py")
        self.utils = load(PREFIX + ".kv_cache_utils", "kv_cache_utils.py")
        load(
            PREFIX + ".fp8.decode.pool_slot_mapping", "fp8/decode/pool_slot_mapping.py"
        )
        translator = types.ModuleType(PREFIX + ".fp8.decode.paged_topk_translator")
        translator.translate_local_to_global_slots = read_oracle
        sys.modules[translator.__name__] = translator
        self.decode = load(
            "bounded_decode_metadata_under_test", "fp8/decode/decode_attn_metadata.py"
        )
        # Execute the actual binding helper without importing attention's CUDA ops.
        tree = ast.parse((ROOT / "fp8/attention.py").read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "bind_attn_cache"
        )
        attention = types.ModuleType(PREFIX + ".fp8.attention")
        attention.__dict__.update(
            contextmanager=contextlib.contextmanager, BIND_KEEP=object()
        )
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == PREFIX + ".attn_type"
        ]
        exec(
            compile(
                ast.Module(body=[*imports, function], type_ignores=[]),
                "bind_attn_cache",
                "exec",
            ),
            attention.__dict__,
        )
        self.bind = attention.bind_attn_cache
        sys.modules[attention.__name__] = attention
        profiler = types.ModuleType(PREFIX + "._profiler")
        profiler.record_function_range = lambda *a: contextlib.nullcontext()
        sys.modules[profiler.__name__] = profiler
        self.prefill = load(
            "bounded_prefill_metadata_under_test", "fp8/prefill_meta.py"
        )

    def tearDown(self):
        self.modules.stop()

    def test_native_enum_and_once_per_cache_resolution(self):
        cache = Cache()
        owner = SimpleNamespace()
        self.assertNotEqual(cache.group_region_names[1], 8)
        for _ in range(3):
            for layer in range(23):
                self.assertEqual(
                    self.utils.cached_swa_region(owner, cache, layer),
                    8 if layer >= 21 else 7,
                )
        self.assertEqual(cache.map_reads, 1)
        draft = Cache(decoder_layers=(0, 1, 2), count=3)
        self.assertEqual(self.utils.cached_swa_region(owner, draft, 0), 8)
        self.assertEqual(draft.map_reads, 1)
        legacy = SimpleNamespace(group_region_names=[NativeEnum(7)])
        self.assertEqual(self.utils.cached_swa_region(owner, legacy, 21), 7)

    def test_table_selection_and_restore_including_failure(self):
        cache = Cache()
        tables = {7: torch.tensor([[11]]), 8: torch.tensor([[29]])}
        original_encoder_table = tables[7]
        old_table = object()
        attn = SimpleNamespace(
            layer_id=21, _kv_cache=None, _block_tables_by_type=old_table, _cp_ctx=None
        )
        with self.bind(attn, cache, tables):
            self.assertEqual(attn._swa_cache_region, 8)
            self.assertIs(attn._block_tables_by_type[7], tables[8])
            self.assertIs(tables[7], original_encoder_table)
            other = Cache(decoder_layers=())
            with self.bind(attn, other, tables, cp_ctx="inner"):
                self.assertEqual(attn._swa_cache_region, 7)
                self.assertIs(attn._block_tables_by_type[7], original_encoder_table)
                self.assertEqual(attn._cp_ctx, "inner")
            self.assertEqual(attn._swa_cache_region, 8)
            self.assertIs(attn._block_tables_by_type[7], tables[8])
            self.assertIsNone(attn._cp_ctx)
        self.assertIs(attn._block_tables_by_type, old_table)
        self.assertIsNone(attn._kv_cache)
        with self.assertRaisesRegex(RuntimeError, "Missing physical SWA"):
            with self.bind(attn, cache, {7: tables[7]}):
                self.fail("missing region8 accepted")
        self.assertIs(attn._block_tables_by_type, old_table)
        self.assertIsNone(attn._kv_cache)

    def test_broadcast_splits_equal_ratio_across_physical_pools(self):
        calls = []
        layers = []
        shared = {}
        for layer, ratio in ((0, 0), (2, 2), (20, 1), (21, 1), (22, 1)):
            attn = SimpleNamespace(
                layer_id=layer,
                compress_ratio=ratio,
                _kv_cache=None,
                _block_tables_by_type=None,
                _cp_ctx=None,
                _shared_attention=shared,
            )

            def build(*args, a=attn, **kwargs):
                common = kwargs["reuse_common_meta"]
                if common is not None:
                    self.assertEqual(common.region, a._swa_cache_region)
                calls.append(
                    (
                        a.layer_id,
                        a._swa_cache_region,
                        int(a._block_tables_by_type[7][0, 0]),
                    )
                )
                return Meta(a._swa_cache_region, common)

            attn._build_shared_prefill_meta = build
            attn._ensure_freqs_cis_bound = lambda: None
            attn._set_prefill_meta_shared = lambda meta, a=attn: setattr(
                a, "meta", meta
            )
            layers.append(SimpleNamespace(attn=attn))
        model = SimpleNamespace(layers=layers)
        tables = {7: torch.tensor([[11]]), 8: torch.tensor([[29]])}
        self.prefill.build_and_propagate_prefill_meta_fp8(
            model, torch.empty((4, 2)), 0, Cache(), tables, workspace=None
        )
        self.assertEqual(calls, [(0, 7, 11), (2, 7, 11), (20, 7, 11), (21, 8, 29)])
        self.assertIs(layers[-1].attn.meta, layers[-2].attn.meta)
        self.assertIsNot(layers[-1].attn.meta, layers[-3].attn.meta)

    def test_decode_swa_distinct_buffers_and_boundary_updates(self):
        meta = self.decode.allocate_decode_metadata_fp8(
            max_batch_size=2,
            q_len=2,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[0],
            index_topk=16,
            device=torch.device("cpu"),
            paged_pool_specs={7: (136, 512, 4), 8: (136, 512, 4)},
        )
        child = meta.decoder_swa_metadata
        self.assertIsNotNone(child)
        self.assertIs(child.position_ids, meta.position_ids)
        self.assertIs(child.swa_abs_idx, meta.swa_abs_idx)
        self.assertIsNot(child.swa_global_slots, meta.swa_global_slots)
        ptrs = (
            child.swa_global_slots.data_ptr(),
            child.pool_write_slot_mappings[7].data_ptr(),
            child.pool_block_tables[7].data_ptr(),
        )
        for start, base in ((511, 40), (640, 60)):
            meta.pool_block_tables[7].fill_(3)
            for row in range(2):
                meta.pool_block_tables[8][row] = torch.arange(
                    base + 4 * row, base + 4 * row + 4
                )
            positions = [start, start + 1, start + 20, start + 21]
            meta.position_ids.copy_(torch.tensor(positions))
            for token, pos in enumerate(positions):
                meta.swa_abs_idx.reshape(4, 128)[token] = torch.arange(
                    pos - 127, pos + 1
                )
            self.decode.update_decoder_swa_metadata(
                meta, {7: 136, 8: 136}, batch_size=2
            )
            expected_writes = []
            for token, pos in enumerate(positions):
                req = token // 2
                expected_writes.append((base + 4 * req + pos // 512) * 136 + pos % 136)
            self.assertEqual(
                child.pool_write_slot_mappings[7].tolist(), expected_writes
            )
            self.assertEqual(child.swa_global_slots[:, -1].tolist(), expected_writes)
            self.assertTrue(torch.all(meta.swa_global_slots == -1))
            self.assertEqual(
                ptrs,
                (
                    child.swa_global_slots.data_ptr(),
                    child.pool_write_slot_mappings[7].data_ptr(),
                    child.pool_block_tables[7].data_ptr(),
                ),
            )

    def test_default_off_has_no_decoder_allocations(self):
        meta = self.decode.allocate_decode_metadata_fp8(
            max_batch_size=1,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[0],
            index_topk=16,
            device=torch.device("cpu"),
            paged_pool_specs={7: (136, 512, 4)},
        )
        self.assertIsNone(meta.decoder_swa_metadata)
        self.assertEqual(set(meta.pool_write_slot_mappings), {7})

    def test_dspark_uses_its_physical_swa_table(self):
        path = ROOT.parents[1] / "model_desc/deepseek_v4_dspark_model.py"
        tree = ast.parse(path.read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_swa_block_table"
        )
        scope = {
            "torch": torch,
            "Any": object,
            "cached_swa_region": self.utils.cached_swa_region,
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            scope,
        )
        model = SimpleNamespace(kv_cache=Cache(decoder_layers=(0, 1, 2), count=3))
        # The legacy encoder slot can be present but unallocated; selecting it
        # must never hide a valid region8 table from the draft model.
        tables = [torch.tensor([[-1, -1]]), torch.tensor([[31, 32]])]
        inputs = SimpleNamespace(kv_cache_kernel_block_id_device_by_group=tables)
        result = scope["_swa_block_table"](model, inputs, 1)
        self.assertEqual(result.tolist(), [[31, 32]])
        self.assertEqual(result.data_ptr(), tables[1].data_ptr())


if __name__ == "__main__":
    unittest.main()
