import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint, RuntimeTensorBinding


class FTInstallTest(unittest.TestCase):
    def test_custom_install_does_not_rematerialize_validated_layout(self):
        model = torch.nn.Linear(3, 2, bias=True)
        source = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        value = torch.tensor([-1, 3], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            save_file({"w": source, "b": value}, str(path))
            checkpoint = FTCheckpoint(path.parent, CheckpointTopology(), {"w": ("w", path), "b": ("b", path)})
            conversions, installations = [], []

            def convert(values):
                conversions.append("w")
                return values[0].t().contiguous()

            def install(values):
                installations.append("w")
                model.weight.copy_(values[0].t())

            weight = RuntimeTensorBinding(("w",), "weight", convert, install=install)
            wrong = RuntimeTensorBinding(("b",), "bias", lambda values: values[0].double())
            start = model.weight.detach().clone()
            with self.assertRaisesRegex(ValueError, "FT dtype mismatch"):
                checkpoint.load_parameters(model, (weight, wrong))
            self.assertEqual(installations, [])
            torch.testing.assert_close(model.weight.detach(), start)
            conversions.clear()
            bias = RuntimeTensorBinding(("b",), "bias", lambda values: values[0])
            checkpoint.load_parameters(model, (weight, bias))
            self.assertEqual(conversions, ["w"])
            self.assertEqual(installations, ["w"])
            torch.testing.assert_close(model.weight.detach(), source.t())
            torch.testing.assert_close(model.bias.detach(), value)


if __name__ == "__main__":
    unittest.main()
