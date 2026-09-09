import importlib.util
import os
import unittest
from pathlib import Path

from rtp_llm.config.py_config_modules import MMRdmaConfig


def _load_exporter_module():
    runfiles_root = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ["TEST_WORKSPACE"]
    extension_path = (
        runfiles_root / workspace / "rtp_llm/libs/libmm_rdma_exporter.so"
    )
    if not extension_path.is_file():
        raise FileNotFoundError(f"RDMA exporter shared library not found: {extension_path}")

    spec = importlib.util.spec_from_file_location("libmm_rdma_exporter", extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create module spec for {extension_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MMRdmaExporterLibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_exporter_module()

    def test_real_library_loads_and_extracts_python_config(self):
        self.assertTrue(hasattr(self.module, "MMRdmaExporter"))
        self.assertIsInstance(self.module.MMRdmaExporter.available(), bool)

        config = MMRdmaConfig()
        config.bind_ip = "127.0.0.1"
        config.port = 12345
        config.connect_timeout_ms = 321
        config.read_timeout_ms = 4321
        config.qp_count = 0
        config.slot_gc_timeout_ms = 5432
        config.max_slot_bytes = 4 * 1024 * 1024
        config.max_receipt_bytes = 8 * 1024 * 1024 * 1024

        # qp_count=0 disables the selected provider before hardware initialization,
        # while construction still extracts every Python field into C++ RdmaConfig.
        exporter = self.module.MMRdmaExporter(config)
        self.assertFalse(exporter.enabled())

    def test_missing_config_field_is_rejected(self):
        config = MMRdmaConfig()
        del config.max_slot_bytes

        with self.assertRaises(AttributeError):
            self.module.MMRdmaExporter(config)

    def test_wrong_config_field_type_is_rejected(self):
        config = MMRdmaConfig()
        config.qp_count = "two"

        with self.assertRaises((TypeError, RuntimeError)):
            self.module.MMRdmaExporter(config)


if __name__ == "__main__":
    unittest.main()
