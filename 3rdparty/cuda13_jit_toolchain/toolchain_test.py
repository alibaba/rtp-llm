import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from rtp_llm.utils.jit_toolchain import configure_bazel_jit_toolchain


class JitToolchainTest(unittest.TestCase):
    def test_non_bazel_environment_is_unchanged(self):
        env = {"PATH": "/usr/bin", "CXX": "/custom/g++"}
        expected = dict(env)
        self.assertIsNone(configure_bazel_jit_toolchain(env))
        self.assertEqual(env, expected)

    def test_explicit_host_compiler_and_flags_are_preserved(self):
        env = dict(os.environ, CXX="/custom/g++",
                   NVCC_PREPEND_FLAGS="--compiler-bindir=/custom/cuda-g++ --use_fast_math")
        root = configure_bazel_jit_toolchain(env)
        self.assertIsNotNone(root)
        self.assertEqual(env["CXX"], "/custom/g++")
        self.assertEqual(env["NVCC_PREPEND_FLAGS"],
                         "--compiler-bindir=/custom/cuda-g++ --use_fast_math")
        first = dict(env)
        configure_bazel_jit_toolchain(env)
        self.assertEqual(env, first)

    def test_manifest_resolution(self):
        env = dict(os.environ)
        root = configure_bazel_jit_toolchain(env)
        self.assertIsNotNone(root)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "MANIFEST"
            manifest.write_text("cuda13_jit_gcc/toolchain/bin/g++ " + str(root / "bin/g++") + "\n")
            resolved = configure_bazel_jit_toolchain({"RUNFILES_MANIFEST_FILE": str(manifest)})
            self.assertEqual(resolved, root)

    def test_cpp20_and_nvcc_sm103a_from_runfiles(self):
        env = dict(os.environ)
        for key in ("CC", "CXX", "CUDAHOSTCXX", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
            env.pop(key, None)
        root = configure_bazel_jit_toolchain(env)
        self.assertIsNotNone(root)
        self.assertIn("external/cuda13_jit_gcc/", env["CXX"])
        version = subprocess.check_output([env["CXX"], "--version"], env=env, text=True)
        self.assertIn("12.3.0", version)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env["TMPDIR"] = str(tmp)
            host = tmp / "host.cc"
            host.write_text("#include <vector>\n"
                            "template<float V> constexpr float value() { return V; }\n"
                            "static_assert(value<1.5f>() == 1.5f);\n"
                            "int main() { return std::vector<int>{1,2}.size() != 2; }\n")
            subprocess.run([env["CXX"], "-std=c++20", str(host), "-o", str(tmp / "host")],
                           env=env, cwd=tmp, check=True, timeout=60)
            subprocess.run([str(tmp / "host")], env=env, cwd=tmp, check=True, timeout=10)
            cuda = tmp / "kernel.cu"
            cuda.write_text("template<float V> __global__ void set(float* x) { x[0] = V; }\n"
                            "template __global__ void set<1.5f>(float*);\n")
            nvcc = Path(env.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
            subprocess.run([str(nvcc), "-std=c++20", "-arch=sm_103a", "--cubin",
                            str(cuda), "-o", str(tmp / "kernel.cubin")],
                           env=env, cwd=tmp, check=True, timeout=120)
            self.assertGreater((tmp / "kernel.cubin").stat().st_size, 0)
        print("Bazel GCC:", env["CXX"], flush=True)


if __name__ == "__main__":
    unittest.main()
