#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkCli.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkRunner.h"

namespace rtp_llm::benchmark {
namespace {
cudaDeviceProp selectCudaDevice(int cuda_device) {
    const cudaError_t set_status = cudaSetDevice(cuda_device);
    if (set_status != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice(" + std::to_string(cuda_device)
                                 + ") failed: " + cudaGetErrorString(set_status));
    }
    cudaDeviceProp    prop{};
    const cudaError_t prop_status = cudaGetDeviceProperties(&prop, cuda_device);
    if (prop_status != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties(" + std::to_string(cuda_device)
                                 + ") failed: " + cudaGetErrorString(prop_status));
    }
    return prop;
}

}  // namespace
}  // namespace rtp_llm::benchmark

int main(int argc, char** argv) {
    using namespace rtp_llm::benchmark;
    return runBenchmarkCli(argc, argv, [](const BenchmarkCommand& command) {
        const auto& common = command.common;
        const auto  prop   = selectCudaDevice(common.cuda_device);
        std::cout << "Using GPU: " << prop.name << " (device " << common.cuda_device << ")\n";
        if (command.subcommand == "tree") {
            return TreeBenchmarkRunner(command.profile,
                                       command.tree,
                                       common.seed,
                                       common.repetition_id,
                                       common.cuda_device,
                                       common.max_device_memory_fraction,
                                       common.output_json_path)
                .run();
        }
        return TransferBenchmarkRunner(command.profile, command.transfer, common.seed, common.output_json_path).run();
    });
}
