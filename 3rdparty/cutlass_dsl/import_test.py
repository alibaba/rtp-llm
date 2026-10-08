import importlib.util
import sys
import unittest
from importlib import metadata
from pathlib import Path


class CutlassBuildInstallTest(unittest.TestCase):
    def test_complete_install_is_importable_before_rtp_initialization(self):
        import cutlass
        import cutlass.cute

        self.assertIn("cutlass_site", Path(cutlass.__file__).parts)
        self.assertEqual(cutlass.__version__, "4.7.1")
        self.assertTrue(callable(cutlass.cute.arch.map_dsmem_ptr))
        repo = Path(__file__).parents[2]
        helper = repo / "rtp_llm/models_py/utils/cutlass.py"
        spec = importlib.util.spec_from_file_location("cutlass_helper", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = cutlass.__file__
        module.setup_cutlass_import_path()
        # Match the backend's parent-directory alias without duplicating payloads.
        sys.path.append(str(repo / "rtp_llm" / ".."))
        module.setup_cutlass_import_path()
        self.assertEqual(cutlass.__file__, before)
        self.assertEqual(metadata.version("nvidia-cutlass-dsl"), "4.7.1")


if __name__ == "__main__":
    unittest.main()
