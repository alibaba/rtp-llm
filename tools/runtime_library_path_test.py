"""Exercise the startup helper in Bash and the Linux dynamic loader."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HELPER = Path(
    os.environ.get(
        "RUNTIME_LIBRARY_PATH_HELPER",
        Path(__file__).with_name("runtime_library_path.sh"),
    )
)


class RuntimeLibraryPathTest(unittest.TestCase):
    def run_child(self, directory, inherited, defaults, code):
        profile = Path(directory) / "image library path"
        if defaults is not None:
            profile.write_text(defaults)
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        if inherited is not None:
            env["LD_LIBRARY_PATH"] = inherited
        return subprocess.run(
            [
                "bash",
                "-eu",
                "-c",
                'source "$1"; restore_image_library_path "$2"; exec "$3" -c "$4"',
                "--",
                str(HELPER),
                str(profile),
                sys.executable,
                code,
            ],
            env=env,
            text=True,
            capture_output=True,
        )

    def test_path_order_and_optional_defaults(self):
        for inherited, defaults, expected in (
            (
                "/deployment:/platform",
                "/image/lib\n",
                "/deployment:/platform:/image/lib",
            ),
            (None, "/image/lib\n", "/image/lib"),
            ("", "/image/lib\n", "/image/lib"),
            ("/deployment", "", "/deployment"),
            ("/deployment", None, "/deployment"),
            (None, None, None),
        ):
            with self.subTest(inherited=inherited, defaults=defaults):
                with tempfile.TemporaryDirectory() as directory:
                    result = self.run_child(
                        directory,
                        inherited,
                        defaults,
                        'import os,json; print(json.dumps(os.getenv("LD_LIBRARY_PATH")))',
                    )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux loader contract")
    def test_child_can_load_library_after_restoring_image_path(self):
        compiler = shutil.which(os.environ.get("CC", "cc"))
        self.assertIsNotNone(compiler, "image compiler is required for the loader test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "probe.c"
            source.write_text("int runtime_library_path_value(void) { return 42; }\n")
            library = root / "librtp_runtime_path_probe.so"
            subprocess.run(
                [compiler, "-shared", "-fPIC", str(source), "-o", str(library)],
                check=True,
                capture_output=True,
            )
            code = (
                "import ctypes; "
                "lib=ctypes.CDLL('librtp_runtime_path_probe.so'); "
                "assert lib.runtime_library_path_value() == 42"
            )
            missing = self.run_child(directory, "/deployment-only", None, code)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn(library.name, missing.stderr)
            restored = self.run_child(directory, "/deployment-only", directory, code)
            self.assertEqual(restored.returncode, 0, restored.stderr)


if __name__ == "__main__":
    unittest.main()
