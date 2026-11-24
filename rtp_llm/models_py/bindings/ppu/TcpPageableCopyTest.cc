#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstring>
#include <future>
#include <limits>
#include <string>

namespace rtp_llm {
namespace {
using transfer::tcp::CopyTask;
using transfer::tcp::CudaCopyUtil;
using transfer::TransferErrorCode;

TEST(PpuTcpCopyTest, RoundTripPageablePayloadsAndHostBlocks) {
    constexpr size_t bytes = 2048;
    std::string input(bytes + 2, 'i'), output(bytes + 2, 'o');
    for (size_t i = 0; i < bytes; ++i) {
        input[i + 1] = static_cast<char>(i % 251);
    }
    auto device = torch::zeros({bytes + 2}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto* gpu = static_cast<char*>(device.data_ptr()) + 1;
    char cpu_source = 'x', cpu_result = '-';
    CudaCopyUtil copy;
    std::vector<CopyTask> send = {{input.data() + 1, bytes, gpu, device.get_device(), true}};
    std::vector<CopyTask> recv = {{gpu, bytes, output.data() + 1, device.get_device(), true},
                                  {&cpu_source, 1, &cpu_result, -1, false}};
    ASSERT_TRUE(copy.batchCopyToDevice(send));
    ASSERT_TRUE(copy.batchCopyToHost(recv));
    EXPECT_EQ(output.substr(1, bytes), input.substr(1, bytes));
    EXPECT_EQ(output.front(), 'o');
    EXPECT_EQ(output.back(), 'o');
    EXPECT_EQ(cpu_result, cpu_source);
}

TEST(PpuTcpCopyTest, RpcWorkersRoundTripConcurrentInteriorBlocksWithoutPooledDma) {
    constexpr int workers = 4, bytes = 2048;
    auto pool = torch::zeros({workers, bytes + 2}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<std::future<bool>> transfers;
    for (int i = 0; i < workers; ++i) {
        transfers.emplace_back(std::async(std::launch::async, [&, i] {
            // The old implementation fails here even if its host staging is pinned.
            struct ResetHook {
                ~ResetHook() { no_block_copy_detail::cuda_result_for_test = nullptr; }
            } reset;
            no_block_copy_detail::cuda_result_for_test = [](const char* phase, int result) {
                return std::strcmp(phase, "generic_submit") == 0 ? static_cast<int>(cudaErrorInvalidValue) : result;
            };
            CudaCopyUtil copy;
            std::string input(bytes, static_cast<char>(i + 1)), output(bytes + 2, '-');
            char* block = static_cast<char*>(pool.data_ptr()) + i * (bytes + 2) + 1;
            std::vector<CopyTask> upload{{input.data(), bytes, block, pool.get_device(), true}};
            std::vector<CopyTask> download{{block, bytes, output.data() + 1, pool.get_device(), true}};
            return copy.batchCopyToDevice(upload) && copy.batchCopyToHost(download)
                   && output == "-" + input + "-";
        }));
    }
    for (auto& transfer : transfers) {
        EXPECT_TRUE(transfer.get());
    }
    auto result = pool.cpu();
    for (int i = 0; i < workers; ++i) {
        EXPECT_EQ(result[i][0].item<int>(), 0);
        EXPECT_EQ(result[i][bytes + 1].item<int>(), 0);
    }
}

TEST(PpuTcpCopyTest, RejectsUnavailableDeviceWithoutPublishingPageablePayload) {
    auto gpu = torch::ones({16}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::string payload(16, 'x');
    int unavailable = 0;
    ASSERT_EQ(cudaGetDeviceCount(&unavailable), cudaSuccess);
    std::vector<CopyTask> send = {{gpu.data_ptr(), payload.size(), payload.data(), unavailable, true}};
    std::vector<CopyTask> recv = {{payload.data(), payload.size(), static_cast<char*>(gpu.data_ptr()), unavailable, true}};
    CudaCopyUtil copy;
    EXPECT_FALSE(copy.batchCopyToHost(send));
    EXPECT_EQ(payload, std::string(16, 'x'));
    EXPECT_EQ(copy.batchCopyToDevice(recv, std::numeric_limits<int64_t>::max()), TransferErrorCode::COPY_FAILED);
}
}  // namespace
}  // namespace rtp_llm
