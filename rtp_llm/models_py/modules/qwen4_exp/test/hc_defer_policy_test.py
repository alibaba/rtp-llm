"""CPU-only tests for the experimental HC selection policy."""

import importlib.util
import pathlib
import unittest


_path = pathlib.Path(__file__).resolve().parents[3] / "model_desc/qwen4_exp_runtime/hc_defer.py"
_spec = importlib.util.spec_from_file_location("hc_defer_policy", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class HcDeferPolicyTest(unittest.TestCase):
    def test_sm90_is_opt_in_and_graph_compatible(self):
        args = dict(cuda=True, hip=False, shape=(16, 10240), bf16=True, sm=(9, 0), graph=True)
        self.assertFalse(_module.should_defer_hc_residual(enabled=False, **args))
        self.assertTrue(_module.should_defer_hc_residual(enabled=True, **args))
        for shape in ((0, 10240), (33, 10240), (16, 10239), (16, 4, 2560)):
            self.assertFalse(_module.should_defer_hc_residual(enabled=True, **(args | {"shape": shape})))
        for change in ({"cuda": False}, {"hip": True}, {"bf16": False}, {"sm": (8, 9)}):
            self.assertFalse(_module.should_defer_hc_residual(enabled=True, **(args | change)))

    def test_existing_sm120_policy_unchanged(self):
        args = dict(enabled=False, cuda=True, hip=False, shape=(4096, 10240), bf16=True, sm=(12, 0), graph=False)
        self.assertTrue(_module.should_defer_hc_residual(**args))
        self.assertFalse(_module.should_defer_hc_residual(**(args | {"graph": True})))
        self.assertFalse(_module.should_defer_hc_residual(**(args | {"shape": (4092, 10240)})))


if __name__ == "__main__":
    unittest.main()
