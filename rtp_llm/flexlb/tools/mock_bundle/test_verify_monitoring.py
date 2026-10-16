import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from verify_monitoring import REQUIRED, verify


class MonitoringArtifactTest(unittest.TestCase):
    def test_flat_and_nested_monitoring_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.jar"
            for nested in (False, True):
                with zipfile.ZipFile(path, "w") as jar:
                    if nested:
                        dependency = io.BytesIO()
                        with zipfile.ZipFile(dependency, "w") as library:
                            for name in REQUIRED:
                                library.writestr(name, b"class")
                        jar.writestr("BOOT-INF/lib/kmonitor.jar", dependency.getvalue())
                    else:
                        for name in REQUIRED:
                            jar.writestr(name, b"class")
                verify(path)

    def test_opensource_master_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master.jar"
            with zipfile.ZipFile(path, "w") as jar:
                jar.writestr("BOOT-INF/classes/application.yml", b"flexlb:")
            with self.assertRaisesRegex(ValueError, "missing KMonitor"):
                verify(path)
