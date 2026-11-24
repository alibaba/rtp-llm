"""Static contract: the PPU DSpARK draft must not reach CUDA-only operators.

Importing the real modules needs the native runtime, so these assertions read
the sources instead.  They exist to catch the two regressions that silently
put PPU back on CUDA kernels: dropping a ``_dspark_*`` hook from the shared
attention body, and dropping the PPU branch from model construction.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[6]
PPU_DSPARK = ROOT / "rtp_llm/models_py/model_desc/deepseek_v4_ppu/dspark_model.py"
SHARED_DSPARK = ROOT / "rtp_llm/models_py/model_desc/deepseek_v4_dspark_model.py"
MODEL_ENTRY = ROOT / "rtp_llm/models/deepseek_v4.py"


def _parse(path):
    return ast.parse(Path(path).read_text())


def _class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls, name):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found on {cls.name}")


def _imports_from(tree, module_suffix):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith(
            module_suffix
        ):
            names.update(alias.name for alias in node.names)
    return names


class PpuDsparkDraftContractTest(unittest.TestCase):
    def test_ppu_draft_mixes_shared_dspark_with_ppu_buffers(self):
        cls = _class(_parse(PPU_DSPARK), "PpuDeepSeekV4DSparkModel")
        bases = [ast.unparse(base) for base in cls.bases]
        self.assertEqual(bases, ["DeepSeekV4DSparkModel", "PpuDeepSeekV4Model"])

    def test_ppu_draft_binds_every_decode_operator_to_ppu(self):
        tree = _parse(PPU_DSPARK)
        self.assertIn(
            "decode_compute_qkv",
            _imports_from(tree, "ppu.attention.fp8.decode.compute_qkv"),
        )
        self.assertIn(
            "decode_output_proj",
            _imports_from(tree, "ppu.attention.fp8.decode.output_proj"),
        )
        self.assertIn(
            "attn_fp8_swa_paged",
            _imports_from(tree, "ppu.attention.fp8.decode.attention_kernels"),
        )
        self.assertIn(
            "get_or_build_sched_meta",
            _imports_from(tree, "ppu.attention.fp8.decode.decode_attn_metadata"),
        )
        load = _method(
            _class(tree, "PpuDeepSeekV4DSparkModel"), "_load_extra_weights"
        )
        self.assertIn("platform_provider=self._platform_provider", ast.unparse(load))

    def test_ppu_draft_declares_device_only_graph_metadata(self):
        cls = _class(_parse(PPU_DSPARK), "PpuDeepSeekV4DSparkModel")
        prepare = ast.unparse(_method(cls, "prepare_fmha_impl"))
        self.assertIn("uses_device_only_cuda_graph_metadata = True", prepare)

    def test_shared_attention_body_only_talks_to_hooks(self):
        cls = _class(_parse(SHARED_DSPARK), "DeepSeekV4DSparkModel")
        body = ast.unparse(_method(cls, "_forward_dspark_attention"))
        for hook in (
            "self._dspark_compute_qkv",
            "self._dspark_sched_meta",
            "self._dspark_attend",
            "self._dspark_output_proj",
        ):
            self.assertIn(hook, body)
        # Both platforms share one KV writer; it stays inline instead of
        # becoming an unused seam.
        self.assertIn("decode_write_swa_fp8(", body)
        for cuda_only in (
            "decode_compute_qkv(",
            "decode_output_proj(",
            "_get_fp8_decode_op",
        ):
            self.assertNotIn(cuda_only, body)

    def test_embedding_dispatch_preserves_cuda_and_ppu_contracts(self):
        query = object()
        v4 = SimpleNamespace(
            embed=Mock(return_value="local"),
            _embed=Mock(return_value="replicated"),
        )
        for source, cls_name, expected in (
            (SHARED_DSPARK, "DeepSeekV4DSparkModel", "local"),
            (PPU_DSPARK, "PpuDeepSeekV4DSparkModel", "replicated"),
        ):
            node = _method(_class(_parse(source), cls_name), "_dspark_embed")
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            scope = {}
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(source), "exec"), scope)
            actual = scope["_dspark_embed"](SimpleNamespace(v4=v4), query)
            self.assertEqual(actual, expected)
        v4.embed.assert_called_once_with(query)
        v4._embed.assert_called_once_with(query)
        cls = _class(_parse(SHARED_DSPARK), "DeepSeekV4DSparkModel")
        body = ast.unparse(_method(cls, "_forward_layers"))
        self.assertIn("self._dspark_embed(query_ids)", body)

    def test_ppu_capability_enforces_draft_and_target_depth_bounds(self):
        path = ROOT / "rtp_llm/models_py/modules/dsv4/ppu/manifest.py"
        scope = {}
        # The manifest is import-light; stub only the native enum and constructor.
        exec(compile(path.read_text(), str(path), "exec"), scope)
        cache_type = SimpleNamespace(FP8=object())
        imports = {
            "rtp_llm.ops": SimpleNamespace(KvCacheDataType=cache_type),
            "rtp_llm.models_py.modules.dsv4.ppu.ppu_decode_provider": SimpleNamespace(
                PpuDecodeProvider=Mock()
            ),
        }
        parallel = SimpleNamespace(
            role_type=SimpleNamespace(name="DECODE"),
            pp_size=1,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
            world_size=8,
            tp_size=1,
            dp_size=8,
            ep_size=8,
        )
        moe = SimpleNamespace(
            use_deepep_moe=True,
            use_deepep_low_latency=True,
            use_deepep_internode=False,
            use_all_gather=False,
        )
        # Explicit expected bounds keep disabled target-only distinct from draft.
        cases = (
            ("deepseek_v4_dspark", 3, [], 1, 5),
            ("deepseek_v4", 43, [40, 41, 42], 0, 5),
            ("deepseek_v4_mtp", 1, [], 0, 3),
        )
        for model_type, layers, ids, minimum, maximum in cases:
            config = SimpleNamespace(
                model_type=model_type,
                hidden_size=4096,
                num_layers=layers,
                dspark_target_layer_ids=ids,
                lora_infos=None,
                eplb_config=SimpleNamespace(enable_eplb=lambda: False),
                attn_config=SimpleNamespace(
                    layer_compress_ratios=[0] * layers,
                    kv_cache_dtype=cache_type.FP8,
                    kernel_tokens_per_block=256,
                    tokens_per_block=256,
                ),
            )
            with patch.dict(sys.modules, imports), patch.dict(
                scope["os"].environ, {}, clear=True
            ):
                for gamma in (-1, 0, 1, 3, 4, 5, 6, 8):
                    config.gen_num_per_cycle = gamma
                    with self.subTest(model_type=model_type, gamma=gamma):
                        if minimum <= gamma <= maximum:
                            scope["build_provider"](config, parallel, moe, 64)
                        else:
                            with self.assertRaisesRegex(ValueError, "speculative depth"):
                                scope["build_provider"](config, parallel, moe, 64)

    def test_model_entry_selects_ppu_draft_on_ppu_flash(self):
        cls = _class(_parse(MODEL_ENTRY), "DeepSeekV4DSpark")
        body = ast.unparse(_method(cls, "_create_python_model"))
        self.assertIn("_is_ppu_flash(self.model_config)", body)
        self.assertIn("deepseek_v4_ppu.dspark_model", body)
        self.assertIn("PpuDeepSeekV4DSparkModel", body)
        self.assertIn("deepseek_v4_dspark_model", body)


if __name__ == "__main__":
    unittest.main()
