import subprocess
import tempfile
import unittest
from pathlib import Path


FLEXLB_DIR = Path(__file__).resolve().parents[3]
SETENV = FLEXLB_DIR / "APP-META/docker-config/environment/common/bin/setenv.sh"


class JvmMemoryDefaultsTest(unittest.TestCase):
    def run_function(self, script, *args):
        return subprocess.run(
            ["bash", "-c", 'SETENV_SETTED=true; source "$1"; shift; ' + script,
             "memory-test", str(SETENV), *map(str, args)],
            check=True, capture_output=True, text=True,
        )

    def test_small_container_budget_leaves_native_memory_headroom(self):
        for memory_mb in (1024, 2048, 2049, 4096, 8192, 12288, 16384):
            with self.subTest(memory_mb=memory_mb):
                result = self.run_function(
                    'configure_default_jvm_memory "$1"; '
                    'printf "%s %s %s %s %s" "$DEFAULT_JVM_XMS" "$DEFAULT_JVM_XMX" '
                    '"$maxDirectMemory" "$maxMetaspace" "$reservedCodeCache"', memory_mb,
                )
                sizes = [int(value[:-1]) for value in result.stdout.split()]
                self.assertEqual(sizes[0], sizes[1])
                self.assertLessEqual(sum(sizes[1:]), memory_mb * 13 // 16)

    def test_larger_container_profiles_are_preserved(self):
        for memory_mb, heap, direct in (
            (19456, "12g", "1g"), (24576, "12g", "1g"),
            (32768, "18g", "1g"), (65536, "32g", "2g"),
        ):
            with self.subTest(memory_mb=memory_mb):
                result = self.run_function(
                    'configure_default_jvm_memory "$1"; '
                    'printf "%s %s" "$DEFAULT_JVM_XMX" "$maxDirectMemory"', memory_mb,
                )
                self.assertEqual(f"{heap} {direct}", result.stdout)

    def test_cgroup_limits_and_unlimited_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            meminfo = root / "meminfo"
            meminfo.write_text("MemTotal:       67108864 kB\n")
            v2 = root / "memory.max"
            v1 = root / "memory.limit_in_bytes"
            for v2_value, v1_value, expected in (
                (str(4 * 1024**3), "9223372036854771712", "4096"),
                ("max", str(8 * 1024**3), "8192"),
                ("max", "9223372036854771712", "65536"),
                ("invalid", "invalid", "1024"),
            ):
                with self.subTest(v2=v2_value, v1=v1_value):
                    v2.write_text(v2_value)
                    v1.write_text(v1_value)
                    result = self.run_function('available_memory_mb "$1" "$2" "$3"',
                                               meminfo, v2, v1)
                    self.assertEqual(expected, result.stdout.strip())
            meminfo.write_text("invalid\n")
            result = self.run_function('available_memory_mb "$1" "$2" "$3"',
                                       meminfo, v2, v1)
            self.assertEqual("1024", result.stdout.strip())
            self.assertIn("conservative", result.stderr)


if __name__ == "__main__":
    unittest.main()
