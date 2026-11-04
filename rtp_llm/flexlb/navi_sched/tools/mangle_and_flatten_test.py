import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile


class MangleAndFlattenTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        self.source = self.root / "probe.cpp"
        self.source.write_text(
            "namespace navi_sched {\n"
            "struct SymbolProbe { int value; };\n"
            "__attribute__((noinline)) int retainedProbe(SymbolProbe *p) {\n"
            "    return p->value + 1;\n"
            "}\n}\n"
            'extern "C" int entry(int value) {\n'
            "    navi_sched::SymbolProbe probe{value};\n"
            "    return navi_sched::retainedProbe(&probe);\n}\n"
        )
        self.library = self.root / "_navi_sched_core.so"
        subprocess.run(
            ["c++", "-g", "-O2", "-fno-omit-frame-pointer", "-fPIC",
             "-fvisibility=hidden", "-shared", str(self.source), "-o", str(self.library)],
            check=True,
        )

    def package(self):
        input_path = self.root / "input.whl"
        output_path = self.root / "output.whl"
        with zipfile.ZipFile(input_path, "w") as wheel:
            wheel.write(self.library, "navi_sched/_navi_sched_core.so")
            wheel.writestr("navi_sched/__init__.py", "")
            wheel.writestr("navi_sched_raw-0.1.0.dist-info/METADATA", "Name: navi_sched_raw\n")
            wheel.writestr("navi_sched_raw-0.1.0.dist-info/WHEEL", "Root-Is-Purelib: true\nTag: py3-none-any\n")
            wheel.writestr("navi_sched_raw-0.1.0.dist-info/RECORD", "")
        result = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).with_name("mangle_and_flatten.py")),
             "--input", str(input_path), "--output", str(output_path),
             "--dist-name", "navi_sched", "--wheel-tag", "cp310-cp310-linux_x86_64"],
            cwd=self.root, capture_output=True, text=True,
        )
        return result, output_path

    def test_packaged_library_preserves_perf_and_gdb_symbols(self):
        original = self.library.read_bytes()
        result, output = self.package()
        self.assertEqual(0, result.returncode, result.stderr)
        with zipfile.ZipFile(output) as wheel:
            packaged = wheel.read("navi_sched/_navi_sched_core.so")
        self.assertEqual(original, packaged)
        self.library.write_bytes(packaged)
        symbols = subprocess.check_output(["nm", "-C", str(self.library)], text=True)
        self.assertIn("navi_sched::retainedProbe(navi_sched::SymbolProbe*)", symbols)
        debugger = subprocess.check_output(
            ["gdb", "-nx", "-q", "-batch", str(self.library),
             "-ex", "info line navi_sched::retainedProbe", "-ex", "ptype navi_sched::SymbolProbe"],
            text=True, stderr=subprocess.STDOUT,
        )
        self.assertIn('of "' + str(self.source) + '"', debugger)
        self.assertIn("int value;", debugger)

    def test_rejects_library_without_debug_information(self):
        subprocess.run(["strip", "--strip-debug", str(self.library)], check=True)
        result, _ = self.package()
        self.assertNotEqual(0, result.returncode)
        self.assertIn(".debug_info", result.stderr)

    def test_rejects_library_without_function_symbols(self):
        subprocess.run(["strip", "--strip-unneeded", str(self.library)], check=True)
        result, _ = self.package()
        self.assertNotEqual(0, result.returncode)
        self.assertIn(".symtab", result.stderr)


if __name__ == "__main__":
    unittest.main()
