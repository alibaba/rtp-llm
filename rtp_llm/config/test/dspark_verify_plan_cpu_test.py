"""Compile and execute the actual non-CUDA planner body against CPU LibTorch."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class DSparkVerifyPlanCpuTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "requires host C++ compiler")
    def test_actual_fallback_matches_stable_reference(self):
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths

        source = (ROOT / "rtp_llm/models_py/bindings/core/ExecOps.cc").read_text()
        start = source.index(
            "std::pair<torch::Tensor, torch::Tensor>\nexecDSparkVerifyPlan("
        )
        end = source.index("\n#endif\n}", start) + len("\n#endif\n}")
        body = source[start:end]
        program = (
            r"""
#include <torch/types.h>
#include <ATen/Functions.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#define USING_CUDA 0
#define RTP_LLM_CHECK_WITH_INFO(condition, ...) do { if (!(condition)) throw std::invalid_argument("invalid input"); } while (0)
"""
            + body
            + r"""
int main() {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    float nan = std::numeric_limits<float>::quiet_NaN();
    float inf = std::numeric_limits<float>::infinity();
    for (auto values : std::vector<std::vector<float>>{
            {1, 1, 1, 1}, {0, 0, 0, 0}, {.9f, .5f, .8f, .7f},
            {nan, .9f, .8f, .7f}, {inf, .9f, -inf, .7f}, {-1, 2, .8f, .7f}}) {
        auto confidence = torch::tensor(values, opts).reshape({2, 2});
        std::vector<float> survival(4);
        for (int request = 0; request < 2; ++request) {
            float p = 1;
            for (int position = 0; position < 2; ++position) {
                float c = values[request * 2 + position];
                p *= std::isfinite(c) ? std::min(1.f, std::max(0.f, c)) : 0.f;
                survival[request * 2 + position] = p;
            }
        }
        std::vector<int> order{0, 1, 2, 3};
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (survival[a] != survival[b]) return survival[a] > survival[b];
            if (a % 2 != b % 2) return a % 2 < b % 2;
            return a / 2 < b / 2;
        });
        for (int budget = 0; budget <= 4; ++budget) {
            auto result = execDSparkVerifyPlan(confidence, budget);
            std::vector<int32_t> lengths{1, 1}, mapping;
            for (int i = 0; i < budget; ++i) ++lengths[order[i] / 2];
            for (int request = 0; request < 2; ++request)
                for (int row = 0; row < lengths[request]; ++row)
                    mapping.push_back(request * 3 + row);
            assert(result.first.equal(torch::tensor(lengths, torch::kInt32)));
            assert(result.second.equal(torch::tensor(mapping, torch::kInt32)));
        }
    }
    // Empty DP rank has neither candidates nor compact rows.
    auto empty = execDSparkVerifyPlan(torch::empty({0, 7}, opts), 0);
    assert(empty.first.numel() == 0 && empty.second.numel() == 0);
}
"""
        )
        with tempfile.TemporaryDirectory(prefix="dspark_cpu_plan_") as temp:
            binary = str(Path(temp) / "planner_test")
            cmd = [
                "g++",
                "-std=c++17",
                "-O0",
                "-g0",
                "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
            ]
            cmd += ["-I" + path for path in include_paths()]
            cmd += ["-x", "c++", "-", "-o", binary]
            for path in library_paths():
                cmd += ["-L" + path, "-Wl,-rpath," + path]
            cmd += ["-ltorch", "-ltorch_cpu", "-lc10"]
            subprocess.run(
                cmd, input=program, text=True, check=True, capture_output=True
            )
            subprocess.run([binary], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
