"""Source-only contract checks for the fixed sparse-attention geometry."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path


_ROOT = Path(__file__).parents[1]


class RichContractTest(unittest.TestCase):
    def test_scheduler_key_and_buckets(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = object
        fake_torch.dtype = object
        fake_torch.bfloat16 = object()
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False, is_current_stream_capturing=lambda: False
        )
        fake_flash = types.ModuleType("flash_mla")
        calls = []
        fake_flash.get_mla_metadata = lambda **kw: (calls.append(kw) or (object(), None))
        old_torch, old_flash = sys.modules.get("torch"), sys.modules.get("flash_mla")
        sys.modules.update(torch=fake_torch, flash_mla=fake_flash)
        try:
            path = _ROOT / "decode" / "decode_attn_metadata.py"
            spec = importlib.util.spec_from_file_location("rich_metadata_test", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            metadata = types.SimpleNamespace(sched_meta_cache={}, _sched_meta_capturing=False)
            widths = {None: None, module.RICH_CSA_BUCKET: 512, module.RICH_HCA_BUCKET: 8192}
            for bucket in module.RICH_SCHED_BUCKETS:
                module.get_or_build_sched_meta(
                    metadata, batch_size=1, q_len=1, num_heads=64,
                    topk=128, extra_attn_type=bucket,
                    extra_index_width=widths[bucket]
                )
            self.assertEqual(len(metadata.sched_meta_cache), 3)
            self.assertEqual(len(calls), 3)
            module.get_or_build_sched_meta(
                metadata, batch_size=1, q_len=1, num_heads=64,
                topk=128, extra_attn_type=module.RICH_CSA_BUCKET,
                extra_index_width=1024
            )
            self.assertEqual(len(metadata.sched_meta_cache), 4)
            with self.assertRaises(ValueError):
                module.get_or_build_sched_meta(
                    metadata, batch_size=1, q_len=1, num_heads=64,
                    topk=255, extra_attn_type=None
                )
            module.get_or_build_sched_meta(
                metadata, batch_size=1, q_len=1, num_heads=64,
                topk=512, extra_attn_type=None
            )
            module.get_or_build_sched_meta(
                metadata, batch_size=1, q_len=1, num_heads=64,
                topk=2048, extra_attn_type=None
            )
            self.assertEqual(len(metadata.sched_meta_cache), 6)
        finally:
            sys.modules.pop("rich_metadata_test", None)
            if old_torch is None: sys.modules.pop("torch", None)
            else: sys.modules["torch"] = old_torch
            if old_flash is None: sys.modules.pop("flash_mla", None)
            else: sys.modules["flash_mla"] = old_flash

    def test_striped_geometry_and_length_contract(self):
        source = (_ROOT / "decode" / "attention_kernels.py").read_text()
        self.assertIn("_validate_model1_cache_tensor", source)
        self.assertIn("length.dtype != torch.int32", source)
        self.assertIn("length.device != q.device", source)

    def test_flash_mla_wrapper_keyword_mapping(self):
        path = _ROOT / "decode" / "fp8_sparse_attn_decode_op.py"
        tree = ast.parse(path.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "flash_mla_with_kvcache"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [keyword.arg for keyword in calls[0].keywords],
            [
                "q",
                "k_cache",
                "block_table",
                "head_dim_v",
                "cache_seqlens",
                "tile_scheduler_metadata",
                "num_splits",
                "is_fp8_kvcache",
                "indices",
                "softmax_scale",
                "topk_length",
                "attn_sink",
                "extra_k_cache",
                "extra_indices_in_kvcache",
                "extra_topk_length",
            ],
        )


if __name__ == "__main__":
    unittest.main()
