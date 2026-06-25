import importlib
import unittest


class FrontendPackageImportTest(unittest.TestCase):
    def test_frontend_package_imports_without_async_model_dep(self):
        importlib.import_module("rtp_llm.frontend.frontend_app")


if __name__ == "__main__":
    unittest.main()
