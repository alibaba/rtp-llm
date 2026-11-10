import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

path = Path(__file__).parents[1] / "cp_mtp_hidden_handoff.py"
spec = importlib.util.spec_from_file_location("cp_mtp_hidden_handoff", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class _Base:
    def initialize(self, resource):
        return True

    def get_mtp_target_hidden_states(self, rows):
        return self.fallback


class _Model(module.CpMtpHiddenHandoff, _Base):
    def __init__(self, cp=True):
        self.parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: cp)
        )
        self.weight = SimpleNamespace(global_weights={"embedding": torch.ones(2, 4)})
        self.config = SimpleNamespace(hidden_size=4)
        self.fallback = torch.zeros(1, 4)


def _resource(speculative=True, decode=False):
    return SimpleNamespace(is_speculative=speculative, is_decode_role=decode)


class CpMtpHiddenHandoffTest(unittest.TestCase):
    def test_declares_empty_handoff_then_retains_all_local_rows(self):
        model = _Model()
        self.assertTrue(model.initialize(_resource()))
        self.assertTrue(model.has_mtp_hidden_buffer())
        self.assertEqual(model.get_mtp_target_hidden_states(-1).shape, (0, 4))
        hidden = torch.arange(20).view(5, 4).float()
        model.capture_cp_mtp_hidden(hidden)
        self.assertIs(model.get_mtp_target_hidden_states(-1), hidden)
        torch.testing.assert_close(model.get_mtp_target_hidden_states(3), hidden[:3])
        with self.assertRaisesRegex(RuntimeError, "local rows"):
            model.get_mtp_target_hidden_states(6)

    def test_non_cp_non_speculative_and_decode_keep_base_contract(self):
        for model, resource in [
            (_Model(False), _resource()),
            (_Model(), _resource(False)),
            (_Model(), _resource(decode=True)),
        ]:
            model.initialize(resource)
            self.assertFalse(model.has_mtp_hidden_buffer())
            model.capture_cp_mtp_hidden(torch.ones(3, 4))
            self.assertIs(model.get_mtp_target_hidden_states(-1), model.fallback)

    def test_reinitialize_drops_stale_rows(self):
        model = _Model()
        model.initialize(_resource())
        model.capture_cp_mtp_hidden(torch.ones(5, 4))
        model.initialize(_resource())
        self.assertEqual(model.get_mtp_target_hidden_states(-1).shape[0], 0)


if __name__ == "__main__":
    unittest.main()
