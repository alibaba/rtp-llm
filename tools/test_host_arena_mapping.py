"""Compile the real mmap helper with non-CUDA warning flags and exercise its bounds.

This tests conditional compilation and CPU mappings, not CUDA registration.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = r"""
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <cerrno>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <sys/mman.h>
#include <unistd.h>
#define RTP_LLM_LOG_INFO(...) ((void)0)
#define RTP_LLM_LOG_WARNING(...) ((void)0)
"""
MAIN = r"""
int main() {
 for (const char* flag : {"0", "1"}) {
  setenv("RTP_LLM_HOST_BLOCK_POOL_HUGE_PAGE", flag, 1);
  for (size_t n : {size_t(1),size_t(4095),size_t(4096),size_t(2097151),size_t(2097152),size_t(2097153)}) {
   void* ptr = mmapHostArena(n); assert(ptr != MAP_FAILED);
   if (*flag == '1') assert(reinterpret_cast<uintptr_t>(ptr) % 2097152 == 0);
   auto* bytes = static_cast<unsigned char*>(ptr);
   assert(bytes[0] == 0 && bytes[n-1] == 0);
   bytes[0] = 3; bytes[n-1] = 7; assert(bytes[n-1] == 7);
   assert(munmap(ptr, n) == 0);
  }
 }
 for (const char* flag : {"", "2", "bad"}) {
  setenv("RTP_LLM_HOST_BLOCK_POOL_HUGE_PAGE", flag, 1);
  bool caught = false;
  try { mmapHostArena(4096); } catch (const std::invalid_argument&) { caught = true; }
  assert(caught);
 }
 setenv("RTP_LLM_HOST_BLOCK_POOL_HUGE_PAGE", "1", 1);
 for (size_t n : {size_t(0),std::numeric_limits<size_t>::max()}) {
  bool caught = false;
  try { mmapHostArena(n); } catch (const std::invalid_argument&) { caught = true; }
  assert(caught);
 }
 unsetenv("RTP_LLM_HOST_BLOCK_POOL_HUGE_PAGE");
 void* ptr = mmapHostArena(4096); assert(ptr != MAP_FAILED); assert(munmap(ptr,4096) == 0);
}
"""


class HostArenaMappingTests(unittest.TestCase):
    def test_build_and_mapping(self):
        source = (ROOT / "rtp_llm/cpp/cache/BlockPool.cc").read_text()
        comment = source.index("// Opt-in PMD alignment")
        # Include the surrounding real preprocessor guard, rather than adding
        # a guard in the test that would hide an unguarded production helper.
        start = source.rfind("};", 0, comment) + 2
        end = source.index("torch::Tensor allocateRegisteredCpuTensor", comment)
        helper = source[start:end]
        with tempfile.TemporaryDirectory() as tmp:
            for cuda in (0, 1):
                with self.subTest(using_cuda=cuda):
                    cc = Path(tmp) / f"mapping{cuda}.cc"
                    exe = Path(tmp) / f"mapping{cuda}"
                    cc.write_text(
                        HEADERS
                        + "\nnamespace {\n"
                        + helper
                        + "\n}\n"
                        + (MAIN if cuda else "int main() { return 0; }\n")
                    )
                    subprocess.run(
                        [
                            "g++",
                            "-std=c++17",
                            "-O2",
                            "-Wall",
                            "-Wextra",
                            "-Werror",
                            "-Werror=unused-function",
                            f"-DUSING_CUDA={cuda}",
                            str(cc),
                            "-o",
                            str(exe),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
