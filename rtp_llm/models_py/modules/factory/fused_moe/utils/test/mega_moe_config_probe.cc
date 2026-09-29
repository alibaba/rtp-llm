#include <iostream>
#include "csrc/utils/exception.hpp"
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe_fp8.cuh>
#include "csrc/jit_kernels/heuristics/mega_moe.hpp"
#include "csrc/jit_kernels/heuristics/mega_moe_fp8.hpp"
// Standalone probe: include the pinned backend headers, do not copy heuristics.
int main(int argc, char** argv) {
    if (argc != 7)
        return 2;
    int ranks = std::stoi(argv[1]), experts = std::stoi(argv[2]), topk = std::stoi(argv[3]);
    int hidden = std::stoi(argv[4]), inter = std::stoi(argv[5]), cap = std::stoi(argv[6]);
    for (int t = 0; t <= cap; ++t) {
        auto a = deep_gemm::get_mega_moe_config(
            ranks, experts, experts / ranks, 34560, t, topk, hidden, inter, 34560, 36864, deep_gemm::MmaKind::MXFP8FP4);
        auto b = deep_gemm::get_mega_moe_fp8_config(
            ranks, experts, experts / ranks, 34560, t, topk, hidden, inter, 34560, 36864);
        std::cout << t << " " << a << " " << b << " " << (t * topk <= 32768) << "\n";
    }
}
