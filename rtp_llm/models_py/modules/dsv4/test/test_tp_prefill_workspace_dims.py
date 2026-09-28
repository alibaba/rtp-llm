"""CPU-only contract tests for TP-local DeepSeek-V4 prefill-Q sizing."""

import importlib.util
from pathlib import Path
import unittest

_PATH = Path(__file__).resolve().parents[1] / "prefill_workspace.py"
_SPEC = importlib.util.spec_from_file_location("dsv4_prefill_workspace_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
tp_local_prefill_q_dim = _MODULE.tp_local_prefill_q_dim


class TpPrefillWorkspaceDimsTest(unittest.TestCase):
    def test_prefill_q_dim_is_tp_local(self):
        self.assertEqual(tp_local_prefill_q_dim(128, 256, 1), 32768)
        self.assertEqual(tp_local_prefill_q_dim(128, 256, 8), 4096)

    def test_prefill_q_dim_rejects_invalid_head_partition(self):
        with self.assertRaisesRegex(AssertionError, "not divisible"):
            tp_local_prefill_q_dim(10, 256, 8)


if __name__ == "__main__":
    unittest.main()
