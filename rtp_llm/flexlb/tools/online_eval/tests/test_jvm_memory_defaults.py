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
        for memory_mb in (1024, 2048, 2049, 4096, 8192, 12288, 16383, 16384):
            with self.subTest(memory_mb=memory_mb):
                result = self.run_function(
                    'configure_default_jvm_memory "$1"; '
                    'printf "%s %s %s %s %s" "$DEFAULT_JVM_XMS" "$DEFAULT_JVM_XMX" '
                    '"$maxDirectMemory" "$maxMetaspace" "$reservedCodeCache"', memory_mb,
                )
                sizes = [int(value[:-1]) * (1024 if value.endswith("g") else 1)
                         for value in result.stdout.split()]
                self.assertEqual(sizes[0], sizes[1])
                self.assertLessEqual(sum(sizes[1:]), memory_mb * 13 // 16)

    def test_larger_container_profiles_are_preserved(self):
        for memory_mb, heap, direct in (
            (16383, "10239m", "1023m"), (16384, "10240m", "2g"),
            (19456, "12g", "2g"), (24576, "12g", "2g"),
            (32768, "18g", "2g"), (65536, "32g", "2g"),
        ):
            with self.subTest(memory_mb=memory_mb):
                result = self.run_function(
                    'configure_default_jvm_memory "$1"; '
                    'printf "%s %s" "$DEFAULT_JVM_XMX" "$maxDirectMemory"', memory_mb,
                )
                self.assertEqual(f"{heap} {direct}", result.stdout)

    def test_large_profile_boundaries_fit_all_jvm_pools_and_native_headroom(self):
        for memory_mb in (16384, 16385, 17408, 19456, 24576, 24577, 32768,
                          32769, 33792, 40959, 40960, 65536):
            with self.subTest(memory_mb=memory_mb):
                result = self.run_function(
                    'configure_default_jvm_memory "$1"; '
                    'validate_jvm_memory_budget "$1" "$DEFAULT_JVM_XMS" "$DEFAULT_JVM_XMX"; '
                    'for size in "$DEFAULT_JVM_XMX" "$maxDirectMemory" "$maxMetaspace" '
                    '"$reservedCodeCache"; do jvm_memory_mb "$size"; done; '
                    'echo "$NATIVE_MEMORY_HEADROOM_MB"', memory_mb,
                )
                sizes = [int(value) for value in result.stdout.split()]
                self.assertEqual(2048, sizes[1])
                self.assertEqual(memory_mb // 8, sizes[-1])
                self.assertLessEqual(sum(sizes), memory_mb)

    def test_heap_overrides_use_the_same_budget(self):
        for memory_mb, initial_heap, maximum_heap in (
            (2048, "512m", "1g"), (8192, "6g", "6g"),
            (16384, "11g", "11g"), (65536, "32g", "40g"),
        ):
            with self.subTest(memory_mb=memory_mb, maximum_heap=maximum_heap):
                self.run_function(
                    'configure_default_jvm_memory "$1"; '
                    'validate_jvm_memory_budget "$1" "$2" "$3"',
                    memory_mb, initial_heap, maximum_heap,
                )

    def test_invalid_or_overcommitted_heap_is_rejected_before_launch(self):
        for memory_mb, initial_heap, maximum_heap, error in (
            (8192, "8g", "8g", "exceeds container limit 8192MB"),
            (32769, "32g", "32g", "native_headroom=4096MB"),
            (8192, "6g", "4g", "initial heap 6g exceeds maximum heap 4g"),
            (8192, "invalid", "6g", "invalid JVM initial heap size"),
            (8192, "6g", "0", "invalid JVM maximum heap size"),
        ):
            with self.subTest(memory_mb=memory_mb, maximum_heap=maximum_heap):
                with self.assertRaises(subprocess.CalledProcessError) as raised:
                    self.run_function(
                        'configure_default_jvm_memory "$1"; '
                        'validate_jvm_memory_budget "$1" "$2" "$3"',
                        memory_mb, initial_heap, maximum_heap,
                    )
                self.assertIn(error, raised.exception.stderr)

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
