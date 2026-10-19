import sys
import unittest

from rtp_llm.models_py.distributed.deepep_wrapper import DeepepWrapperConfig


class DeepepWrapperImportTest(unittest.TestCase):
    def test_import_does_not_load_newloader_quant_runtime(self):
        self.assertIsNotNone(DeepepWrapperConfig)
        self.assertNotIn("rtp_llm.models_py.quant_methods.base", sys.modules)


if __name__ == "__main__":
    unittest.main()
