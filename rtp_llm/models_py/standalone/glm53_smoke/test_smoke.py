import os
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.standalone.glm53_smoke.golden import compare, save_exclusive
from rtp_llm.models_py.standalone.glm53_smoke.model import remap_weights
from rtp_llm.models_py.standalone.glm53_smoke.launch import signal_verified


class SmokeContractTest(unittest.TestCase):
    def test_remap_real_moe_layers_and_preserve_global_names(self):
        names = []
        layers = []
        for _ in range(4):
            weights = [
                SimpleNamespace(name="model.layers.{i}.mlp.weight"),
                SimpleNamespace(name="model.embed_tokens.weight"),
            ]
            names.append(weights)
            component = SimpleNamespace(weights=weights)
            layers.append([SimpleNamespace(get_components=lambda c=component: [c])])
        remap_weights(SimpleNamespace(layer_weights=layers))
        for source, values in zip((4, 5, 6, 7), names):
            self.assertEqual(values[0].name, f"model.layers.{source}.mlp.weight")
            self.assertEqual(values[1].name, "model.embed_tokens.weight")
        with self.assertRaises(ValueError):
            remap_weights(SimpleNamespace(layer_weights=layers[:3]))

    def test_numeric_gate_rejects_perturbation_and_nan(self):
        ref = {"layer0.logits": torch.tensor([3.0, 4.0])}
        self.assertTrue(compare(ref, ref)["layer0.logits"]["passed"])
        value = compare({"layer0.logits": torch.tensor([3.0, 5.0])}, ref)[
            "layer0.logits"
        ]
        self.assertFalse(value["passed"])
        self.assertAlmostEqual(value["relative_l2"], 0.2)
        self.assertFalse(
            compare({"layer0.logits": torch.tensor([float("nan"), 4.0])}, ref)[
                "layer0.logits"
            ]["passed"]
        )
        with self.assertRaises(ValueError):
            compare({"layer0.logits": torch.ones(3)}, ref)

    def test_cleanup_never_signals_a_reused_pid(self):
        saved = {"pid": 123, "start_ticks": "10", "state": "S"}
        path = "rtp_llm.models_py.standalone.glm53_smoke.launch.process_identity"
        with patch(path, return_value={**saved, "start_ticks": "11"}), patch.object(
            os, "kill"
        ) as kill:
            signal_verified(saved, 15)
            kill.assert_not_called()
        with patch(path, return_value=saved), patch.object(os, "kill") as kill:
            signal_verified(saved, 15)
            kill.assert_called_once_with(123, 15)

    def test_golden_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.pt"
            tensors = {"layer0.logits": torch.ones(2)}
            save_exclusive(path, tensors, {"source_layers": [4, 5, 6, 7]})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_exclusive(path, {"bad": torch.zeros(2)}, {})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
