"""CPU-only tests for the experimental HC selection policy."""

import importlib.util
import pathlib
import unittest


_path = pathlib.Path(__file__).resolve().parents[3] / "model_desc/qwen4_exp_runtime/hc_defer.py"
_spec = importlib.util.spec_from_file_location("hc_defer_policy", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class HcDeferPolicyTest(unittest.TestCase):
    def test_sm90_is_default_and_graph_compatible(self):
        args = dict(cuda=True, hip=False, shape=(16, 10240), bf16=True, sm=(9, 0), graph=True)
        for rows in (1, 16, 32, 33, 2155, 12288):
            self.assertTrue(_module.should_defer_hc_residual(**(args | {"shape": (rows, 10240)})))
        for shape in ((0, 10240), (12289, 10240), (16, 10239), (16, 4, 2560)):
            self.assertFalse(_module.should_defer_hc_residual(**(args | {"shape": shape})))
        for change in ({"cuda": False}, {"hip": True}, {"bf16": False}, {"sm": (8, 9)}):
            self.assertFalse(_module.should_defer_hc_residual(**(args | change)))

    def test_existing_sm120_policy_unchanged(self):
        args = dict(cuda=True, hip=False, shape=(4096, 10240), bf16=True, sm=(12, 0), graph=False)
        self.assertTrue(_module.should_defer_hc_residual(**args))
        self.assertFalse(_module.should_defer_hc_residual(**(args | {"graph": True})))
        self.assertFalse(_module.should_defer_hc_residual(**(args | {"shape": (4092, 10240)})))

    def test_sm90_long_prefill_extends_the_eager_limit(self):
        args = dict(cuda=True, hip=False, bf16=True, sm=(9, 0))
        for rows in (12289, 32768):
            with self.subTest(rows=rows):
                shape = (rows, 10240)
                self.assertTrue(
                    _module.should_defer_hc_residual(**args, shape=shape, graph=False)
                )
                self.assertFalse(
                    _module.should_defer_hc_residual(**args, shape=shape, graph=True)
                )
        self.assertFalse(
            _module.should_defer_hc_residual(
                **args, shape=(32769, 10240), graph=False
            )
        )


if __name__ == "__main__":
    unittest.main()
