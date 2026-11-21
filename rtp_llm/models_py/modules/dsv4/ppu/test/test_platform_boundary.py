"""Model selection and descriptor isolation without native runtime imports."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlatformBoundaryTest(unittest.TestCase):
    def test_scheduler_capacity_is_local_to_ppu_flash_target_and_draft(self):
        root = Path(__file__).resolve().parents[6]
        # Execute the real small helpers without importing the native model runtime.
        tree = ast.parse((root / "rtp_llm/models/deepseek_v4.py").read_text())
        nodes = [
            n for n in tree.body
            if (isinstance(n, ast.FunctionDef)
                and n.name in ("_is_ppu_flash", "configure_ppu_scheduler"))
            or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "_PPU_FLASH_LAYERS")
        ]
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / "rtp_llm/models/deepseek_v4.py"), "exec"), namespace)
        for ppu in (False, True):
            for name, layers in (("deepseek_v4", 43), ("deepseek_v4_mtp", 1), ("deepseek_v4_dspark", 3), ("deepseek_v4_dspark", 1), ("qwen_3", 43), ("deepseek_v4", 61)):
                for width in (4096, 7168):
                    for chunk, expected in ((-1, 1024), (0, 1024), (128, 256), (512, 512)):
                        config = SimpleNamespace(model_type=name, hidden_size=width, num_layers=layers,
                            max_seq_len=1024, moe_prefill_max_tokens_per_rank=256)
                        scheduler = SimpleNamespace(prefill_chunk_size=chunk,
                            max_context_batch_size=4, max_batch_tokens_size=256)
                        with patch.dict(sys.modules, {
                            "rtp_llm.device.device_type": SimpleNamespace(is_ppu=lambda: ppu),
                        }):
                            namespace["configure_ppu_scheduler"](config, scheduler)
                        selected = ppu and width == 4096 and (
                            (name == "deepseek_v4" and layers == 43)
                            or (name == "deepseek_v4_mtp" and layers == 1)
                            or (name == "deepseek_v4_dspark" and layers == 3)
                        )
                        self.assertEqual(config.moe_prefill_max_tokens_per_rank, expected if selected else 256)
                        self.assertEqual(getattr(config, "_ppu_prefill_chunk_size", None), chunk if selected else None)

    def test_weight_rules_do_not_mutate_other_instances_or_class_tables(self):
        path = Path(__file__).resolve().parents[6] / "rtp_llm/models_py/modules/dsv4/ppu/resources.py"
        module = load_source("ppu_weight_resources_test", path)
        baseline = lambda *args: "baseline"
        custom = lambda *args: "ppu"

        class Leaf:
            name = "routed"
            gpt_style_tp_strategy = {"routed": baseline}

        ppu, cuda = Leaf(), Leaf()
        parent = SimpleNamespace(name="pair", sub_weights={"routed": ppu})
        info = SimpleNamespace(layer_weights=[[parent]], weights=[])
        # This test exercises descriptor isolation without loading native dependencies.
        with patch.dict(sys.modules, {
            "rtp_llm.model_loader.ffn_weight": SimpleNamespace(MoeAtomicWeight=type("MoeAtomicWeight", (), {})),
            "rtp_llm.models_py.modules.dsv4.ppu.moe.routed_weight": SimpleNamespace(PpuRoutedWeight=type("PpuRoutedWeight", (), {})),
        }), patch.object(module, "tp_split_strategies", return_value={"routed": custom}):
            module.configure_weight_splits(info)
        self.assertIs(ppu.gpt_style_tp_strategy["routed"], custom)
        self.assertIs(cuda.gpt_style_tp_strategy["routed"], baseline)
        self.assertIs(Leaf.gpt_style_tp_strategy["routed"], baseline)


if __name__ == "__main__":
    unittest.main()
