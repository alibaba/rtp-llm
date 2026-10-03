"""CPU-only regression for the DSpark kernel's wheel/runfiles dependency.

Do not import the module here: the CPU Bazel lock intentionally has no Triton
package.  Finding the module is sufficient to catch the packaging regression;
the CUDA wheel smoke imports it and exercises its public entry point.
"""

import importlib.util
import unittest


class DSparkWheelImportTest(unittest.TestCase):
    def test_root_triton_kernel_is_packaged(self):
        for module in ("dspark_swa", "dspark_gemma_rope"):
            with self.subTest(module=module):
                spec = importlib.util.find_spec(
                    f"rtp_llm.models_py.triton_kernels.{module}"
                )
                self.assertIsNotNone(spec)


if __name__ == "__main__":
    unittest.main()
