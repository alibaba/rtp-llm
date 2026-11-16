"""Source-only contract checks for the fixed sparse-attention geometry."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path


_ROOT = Path(__file__).parents[5] / "models_py/modules/dsv4/fp8"
_CONTRACT = Path(__file__).parents[6] / "rtp_llm/models_py/modules/dsv4/ppu/attention/attention_contract.py"


class RichContractTest(unittest.TestCase):
    def test_dspark_block_index_width_is_admitted(self):
        """The SWA pool accepts the doubled DSpARK block width, nothing else."""

        class Fake:
            def __init__(self, shape, dtype, strides=None, contiguous=True):
                self.shape = tuple(shape)
                self.dtype = dtype
                self.device = "ppu"
                self.ndim = len(self.shape)
                self._strides = tuple(strides or [1] * self.ndim)
                self._contiguous = contiguous

            def stride(self, dim):
                return self._strides[dim]

            def is_contiguous(self):
                return self._contiguous

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = Fake
        fake_torch.uint8 = "uint8"
        fake_torch.bfloat16 = "bf16"
        fake_torch.int32 = "int32"
        fake_torch.float32 = "fp32"
        namespace = {
            "torch": fake_torch,
            "RICH_SWA_INDEX_WIDTH": 128,
            "RICH_SWA_PAGE": 128,
            "RICH_CSA_INDEX_WIDTH": 512,
            "RICH_HCA_INDEX_WIDTH": 64,
        }
        tree = ast.parse(_CONTRACT.read_text())
        wanted = (
            "_validate_rich_pool_page",
            "_validate_model1_cache_tensor",
            "_validate_rich_pool",
            "_validate_lengths",
            "validate_swa_pool",
        )
        nodes = [
            n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted)
            or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "RICH_SWA_MAX_INDEX_WIDTH")
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_CONTRACT), "exec"),
             namespace)
        validate = namespace["validate_swa_pool"]
        # 128 physical SWA entries per block, padded row stride as in the pool.
        pool = Fake((16, 128, 584), "uint8", strides=(74880, 584, 1))

        def call(width):
            validate(
                q=Fake((2, 5, 64, 512), "bf16"),
                swa_pool_3d=pool,
                swa_topk_3d=Fake((2, 5, width), "int32"),
                attn_sink=Fake((64,), "fp32"),
                topk_length=Fake((2,), "int32"),
            )

        call(128)
        call(256)
        with self.assertRaisesRegex(ValueError, "unsupported width 320"):
            call(320)

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
        legacy_name = "rtp_llm.models_py.utils.flash_mla_legacy"
        old_legacy = sys.modules.get(legacy_name)
        sys.modules.update(torch=fake_torch, flash_mla=fake_flash)
        sys.modules[legacy_name] = fake_flash
        try:
            path = _ROOT / "decode" / "decode_attn_metadata.py"
            spec = importlib.util.spec_from_file_location("rich_metadata_test", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            implementation = _CONTRACT.parent / "fp8/decode/decode_attn_metadata.py"
            implementation_tree = ast.parse(implementation.read_text())
            scheduler = next(n for n in implementation_tree.body
                             if isinstance(n, ast.FunctionDef) and n.name == "get_or_build_sched_meta")
            constants = [n for n in implementation_tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id.startswith("RICH_") for t in n.targets)]
            exec(compile(ast.Module(body=constants + [scheduler], type_ignores=[]), str(implementation), "exec"), module.__dict__)
            tree = ast.parse(_CONTRACT.read_text())
            key = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "scheduler_key")
            scope = dict(module.__dict__)
            exec(compile(ast.Module(body=[key], type_ignores=[]), str(_CONTRACT), "exec"), scope)
            provider = types.SimpleNamespace(scheduler_key=scope["scheduler_key"])
            metadata = types.SimpleNamespace(sched_meta_cache={}, _sched_meta_capturing=False)
            widths = {None: None, module.RICH_CSA_BUCKET: 512, module.RICH_HCA_BUCKET: 8192}
            for bucket in module.RICH_SCHED_BUCKETS:
                module.get_or_build_sched_meta(
                    metadata, platform_provider=provider, batch_size=1, q_len=1, num_heads=64,
                    topk=128, extra_attn_type=bucket,
                    extra_index_width=widths[bucket]
                )
            self.assertEqual(len(metadata.sched_meta_cache), 3)
            self.assertEqual(len(calls), 3)
            module.get_or_build_sched_meta(
                metadata, platform_provider=provider, batch_size=1, q_len=1, num_heads=64,
                topk=128, extra_attn_type=module.RICH_CSA_BUCKET,
                extra_index_width=1024
            )
            self.assertEqual(len(metadata.sched_meta_cache), 4)
            with self.assertRaises(ValueError):
                module.get_or_build_sched_meta(
                    metadata, platform_provider=provider, batch_size=1, q_len=1, num_heads=64,
                    topk=255, extra_attn_type=None
                )
            module.get_or_build_sched_meta(
                metadata, platform_provider=provider, batch_size=1, q_len=1, num_heads=64,
                topk=512, extra_attn_type=None
            )
            module.get_or_build_sched_meta(
                metadata, platform_provider=provider, batch_size=1, q_len=1, num_heads=64,
                topk=2048, extra_attn_type=None
            )
            self.assertEqual(len(metadata.sched_meta_cache), 6)
            metadata.sched_meta_cache.clear()
            module.get_or_build_sched_meta(
                metadata, batch_size=1, q_len=1, num_heads=64,
                topk=255, extra_attn_type="existing_cuda_bucket"
            )
            self.assertEqual(list(metadata.sched_meta_cache), [(1, "existing_cuda_bucket")])
        finally:
            sys.modules.pop("rich_metadata_test", None)
            if old_torch is None: sys.modules.pop("torch", None)
            else: sys.modules["torch"] = old_torch
            if old_flash is None: sys.modules.pop("flash_mla", None)
            else: sys.modules["flash_mla"] = old_flash
            if old_legacy is None: sys.modules.pop(legacy_name, None)
            else: sys.modules[legacy_name] = old_legacy

    def test_striped_geometry_and_length_contract(self):
        source = _CONTRACT.read_text()
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
