"""CPU regression entry point; does not run accelerator target methods."""

import types
import unittest

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_loader_test import (
    fp4_linear_test,
    fp4_moe_test,
    fp4_read_spec_test,
    nvfp4_backend_test,
    nvfp4_cache_test,
)
from rtp_llm.models_py.quant_methods.test import test_mxfp4_quark


class AcceleratorFixtureConstructionTest(unittest.TestCase):
    def test_real_typed_configs_construct_without_allocating_gpu(self):
        from rtp_llm.models_py.new_loader_test.fp4_accelerator_checkpoint_test import (
            _AcceleratorRoot,
        )

        for recipe in ("dense", "offline", "online", "quark"):
            with self.subTest(recipe=recipe):
                model = _AcceleratorRoot(
                    types.SimpleNamespace(recipe=recipe), NewLoaderConfig(device="cpu")
                )
                self.assertTrue(all(t.device.type == "cpu" for t in model.parameters()))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite([tests])
    for module in (
        fp4_linear_test,
        fp4_moe_test,
        fp4_read_spec_test,
        nvfp4_backend_test,
        nvfp4_cache_test,
        test_mxfp4_quark,
    ):
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


if __name__ == "__main__":
    unittest.main()
