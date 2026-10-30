"""Model selection and descriptor isolation without native runtime imports."""

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
    def test_only_ppu_dsv4_selects_platform_models(self):
        root = Path(__file__).resolve().parents[2]
        factory = load_source("platform_model_dispatch_test", root / "model_dispatch.py")
        target, draft, original = object(), object(), object()
        classes = SimpleNamespace(PpuDeepSeekV4=target, PpuDeepSeekV4Mtp=draft)
        for ppu in (False, True):
            with patch.dict(sys.modules, {
                "rtp_llm.device.device_type": SimpleNamespace(is_ppu=lambda: ppu),
            }):
                # Resolve the relative import as it is used by the real package.
                factory.__package__ = "rtp_llm.platforms"
                with patch.dict(sys.modules, {"rtp_llm.platforms.ppu.models.dsv4.model": classes}):
                    for name, expected in (("deepseek_v4", target), ("deepseek_v4_mtp", draft)):
                        config = SimpleNamespace(hidden_size=4096, num_layers=1 if name.endswith("mtp") else 43)
                        self.assertIs(factory.resolve_model_class(name, original, config), expected if ppu else original)
                        self.assertIs(factory.resolve_model_class(name, original), original)
                        config.hidden_size = 7168
                        self.assertIs(factory.resolve_model_class(name, original, config), original)
                    for name in ("deepseek_v4_dspark", "qwen_3", "deepseek3"):
                        self.assertIs(factory.resolve_model_class(name, original), original)
                        self.assertIs(factory.resolve_model_class(name, original, object()), original)
                    if not ppu:
                        self.assertIs(factory.resolve_model_class("deepseek_v4", original, object()), original)

    def test_weight_rules_do_not_mutate_other_instances_or_class_tables(self):
        path = Path(__file__).resolve().parents[1] / "models/dsv4/resources.py"
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
            "rtp_llm.platforms.ppu.models.dsv4.routed_weight": SimpleNamespace(PpuRoutedWeight=type("PpuRoutedWeight", (), {})),
        }), patch.object(module, "tp_split_strategies", return_value={"routed": custom}):
            module.configure_weight_splits(info)
        self.assertIs(ppu.gpt_style_tp_strategy["routed"], custom)
        self.assertIs(cuda.gpt_style_tp_strategy["routed"], baseline)
        self.assertIs(Leaf.gpt_style_tp_strategy["routed"], baseline)


if __name__ == "__main__":
    unittest.main()
