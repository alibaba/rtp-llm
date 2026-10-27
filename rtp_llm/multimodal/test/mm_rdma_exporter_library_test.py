import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from rtp_llm.config.py_config_modules import MMRdmaConfig

RDMA_FIRST_IMPORT = r"""
import importlib.util
import json
import os
from pathlib import Path
import sys

extension = Path(sys.argv[1])
owner = extension.with_name("librtp_execution_stage_recorder.so")

def mapped_libraries(prefix):
    found = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and Path(fields[5]).name.startswith(prefix):
            found.add((os.path.realpath(fields[5]), fields[3], fields[4]))
    return sorted(found)

def compute_modules():
    return sorted(
        name for name in sys.modules
        if name == "librtp_compute_ops" or name.endswith(".librtp_compute_ops")
    )

# Register only the configuration bindings used by the production ViT path.
# Do not import compute ops, construct an exporter, create QPs, or run a model.
from rtp_llm.config.py_config_modules import MMRdmaConfig

before = {
    "compute_modules": compute_modules(),
    "compute_mappings": mapped_libraries("librtp_compute_ops.so"),
    "config_registered": "libth_transformer_config" in sys.modules,
    "config_mappings": mapped_libraries("libth_transformer_config.so"),
}
if before["compute_modules"] or before["compute_mappings"]:
    raise AssertionError("RDMA-first prerequisite violated: " + json.dumps(before))
if not before["config_registered"] or not before["config_mappings"]:
    raise AssertionError("configuration bindings were not registered: " + json.dumps(before))
if not owner.is_file():
    raise AssertionError("colocated recorder owner is missing: " + str(owner))

spec = importlib.util.spec_from_file_location("libmm_rdma_exporter", extension)
if spec is None or spec.loader is None:
    raise ImportError("cannot create exporter module spec: " + str(extension))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not hasattr(module, "MMRdmaExporter"):
    raise AssertionError("real exporter extension has no MMRdmaExporter binding")

owners = mapped_libraries(owner.name)
after = {
    "compute_modules": compute_modules(),
    "compute_mappings": mapped_libraries("librtp_compute_ops.so"),
    "owner_mappings": owners,
}
if after["compute_modules"] or after["compute_mappings"]:
    raise AssertionError("exporter import loaded the compute DSO: " + json.dumps(after))
if len(owners) != 1 or not os.path.samefile(owners[0][0], owner):
    raise AssertionError("expected one colocated recorder owner: " + json.dumps(after))
print("RTP_RDMA_FIRST_IMPORT_RESULT=" + json.dumps({
    "pid": os.getpid(),
    "trace_setting": os.environ.get("RTP_LLM_GPU_STAGE_TRACE"),
    "exporter": str(extension),
    "owner": str(owner),
    "before": before,
    "after": after,
    "passed": True,
}), flush=True)
"""


def _load_exporter_module():
    runfiles_root = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ["TEST_WORKSPACE"]
    extension_path = runfiles_root / workspace / "rtp_llm/libs/libmm_rdma_exporter.so"
    if not extension_path.is_file():
        raise FileNotFoundError(
            f"RDMA exporter shared library not found: {extension_path}"
        )

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

    def test_rdma_first_import_in_fresh_processes(self):
        extension = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "rtp_llm/libs/libmm_rdma_exporter.so"
        )
        self.assertTrue(extension.is_file(), str(extension))
        for setting in ("0", "1"):
            with self.subTest(trace_setting=setting):
                environment = os.environ.copy()
                environment["RTP_LLM_GPU_STAGE_TRACE"] = setting
                environment["PYTHONPATH"] = os.pathsep.join(
                    str(Path(path).absolute()) for path in sys.path if path
                )
                result = subprocess.run(
                    [sys.executable, "-c", RDMA_FIRST_IMPORT, str(extension)],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                reports = [
                    line.removeprefix("RTP_RDMA_FIRST_IMPORT_RESULT=")
                    for line in result.stdout.splitlines()
                    if line.startswith("RTP_RDMA_FIRST_IMPORT_RESULT=")
                ]
                self.assertEqual(len(reports), 1, result.stdout + result.stderr)
                report = json.loads(reports[0])
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["trace_setting"], setting)
                self.assertNotEqual(report["pid"], os.getpid())
                self.assertEqual(report["before"]["compute_modules"], [])
                self.assertEqual(report["before"]["compute_mappings"], [])
                self.assertEqual(report["after"]["compute_modules"], [])
                self.assertEqual(report["after"]["compute_mappings"], [])
                self.assertEqual(len(report["after"]["owner_mappings"]), 1)
                print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
