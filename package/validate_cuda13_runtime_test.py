import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_cuda13_runtime.py")
SPEC = importlib.util.spec_from_file_location("validate_cuda13_runtime", MODULE_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class EnqueueProviderTest(unittest.TestCase):
    def test_skips_an_inspected_state(self):
        provider = Path("/runtime/libdependency.so")
        inherited_rpath = (Path("/runtime/first"),)
        pending = []

        VALIDATOR.enqueue_provider(
            pending, {(provider, inherited_rpath)}, provider, inherited_rpath
        )

        self.assertEqual(pending, [])

    def test_revisits_provider_with_different_inherited_rpath(self):
        provider = Path("/runtime/libdependency.so")
        first_rpath = (Path("/runtime/first"),)
        second_rpath = (Path("/runtime/second"),)
        pending = []

        VALIDATOR.enqueue_provider(pending, {(provider, first_rpath)}, provider, second_rpath)

        self.assertEqual(pending, [(provider, second_rpath)])


if __name__ == "__main__":
    unittest.main()
