import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class TorchPatchTest(unittest.TestCase):
    def load_patch(self, *, has_ue8m0):
        torch = types.ModuleType("torch")
        torch.__version__ = "2.11.0" if has_ue8m0 else "2.5.1"
        torch.Tensor = object
        torch.concat = Mock()
        torch.uint8 = object()
        if has_ue8m0:
            torch.float8_e8m0fnu = object()
        dist = types.ModuleType("torch.distributed")
        original = dist.broadcast = Mock(return_value=object())
        torch.distributed = dist
        with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "torch_patch.py"))
        return torch, dist, original

    def test_older_torch_import_preserves_broadcast(self):
        with patch("logging.warning") as warning:
            _, dist, original = self.load_patch(has_ue8m0=False)
        self.assertIs(dist.broadcast, original)
        warning.assert_not_called()

    def test_other_dtypes_preserve_tensor_and_arguments(self):
        _, dist, original = self.load_patch(has_ue8m0=True)
        tensor = Mock(dtype=object())
        group = object()
        actual = dist.broadcast(tensor, src=2, group=group, async_op=True)
        original.assert_called_once_with(tensor, src=2, group=group, async_op=True)
        tensor.view.assert_not_called()
        self.assertIs(actual, original.return_value)

    def test_ue8m0_uses_byte_view_and_preserves_async_result(self):
        torch, dist, original = self.load_patch(has_ue8m0=True)
        tensor = Mock(dtype=torch.float8_e8m0fnu)
        actual = dist.broadcast(tensor, 0, async_op=True)
        tensor.view.assert_called_once_with(torch.uint8)
        original.assert_called_once_with(tensor.view.return_value, 0, async_op=True)
        self.assertIs(actual, original.return_value)


if __name__ == "__main__":
    unittest.main()
