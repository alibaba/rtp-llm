// Copyright (c) RTP-LLM

#include <array>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gtest/gtest.h"

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm::test {
namespace {

struct CopyScratch {
    StagedMemoryCopyScratch value;
    ~CopyScratch() {
        releaseStagedMemoryCopyScratch(value);
    }
};

class StagedMemoryCopyTest: public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        int device_count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
        if (device_count == 0) {
            GTEST_SKIP() << "CUDA device required";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    void checkRoundTrip(const std::vector<size_t>& sizes, int repetitions) {
        std::vector<size_t> offsets;
        size_t              total_bytes = 0;
        for (const auto size : sizes) {
            offsets.push_back(total_bytes);
            total_bytes += size + 256;
        }
        auto  host_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(GetParam());
        auto  source       = torch::empty({static_cast<int64_t>(total_bytes)}, host_options);
        auto  result       = torch::empty_like(source, host_options);
        auto  device       = torch::empty({static_cast<int64_t>(total_bytes)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0));
        auto* source_ptr   = source.data_ptr<uint8_t>();
        auto* result_ptr   = result.data_ptr<uint8_t>();
        auto* device_ptr   = device.data_ptr<uint8_t>();
        std::memset(source_ptr, 0xa7, total_bytes);
        ASSERT_EQ(cudaMemset(device_ptr, 0xa7, total_bytes), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        StagedMemoryCopyParams h2d;
        StagedMemoryCopyParams d2h;
        h2d.direction    = StagedMemoryCopyDirection::H2D;
        d2h.direction    = StagedMemoryCopyDirection::D2H;
        h2d.device_index = 0;
        d2h.device_index = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            const size_t compact_offset = h2d.host_bytes;
            h2d.tiles.push_back({device_ptr + offsets[i], compact_offset, sizes[i]});
            d2h.tiles.push_back({device_ptr + offsets[i], compact_offset, sizes[i]});
            h2d.host_segments.push_back({source_ptr + offsets[i], compact_offset, sizes[i]});
            d2h.host_segments.push_back({result_ptr + offsets[i], compact_offset, sizes[i]});
            h2d.host_bytes += sizes[i];
            d2h.host_bytes += sizes[i];
        }
        CopyScratch scratch;

        for (int iteration = 0; iteration < repetitions; ++iteration) {
            SCOPED_TRACE(iteration);
            for (size_t i = 0; i < sizes.size(); ++i) {
                std::memset(source_ptr + offsets[i], (i + iteration) % 251, sizes[i]);
            }
            ASSERT_TRUE(execStagedMemoryCopy(h2d, &scratch.value));
            if (iteration == 0) {
                cudaDeviceProp properties{};
                ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
                if (properties.pageableMemoryAccess && properties.pageableMemoryAccessUsesHostPageTables
                    && properties.canUseHostPointerForRegisteredMem) {
                    EXPECT_EQ(scratch.value.host_staging, nullptr);
                }
            }
            // Inspect the whole allocation, including gaps that must not be overwritten.
            ASSERT_EQ(cudaMemcpy(result_ptr, device_ptr, total_bytes, cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(std::memcmp(source_ptr, result_ptr, total_bytes), 0);
            std::memset(result_ptr, 0xa7, total_bytes);
            ASSERT_TRUE(execStagedMemoryCopy(d2h, &scratch.value));
            ASSERT_EQ(std::memcmp(source_ptr, result_ptr, total_bytes), 0);
        }
    }
};

TEST_P(StagedMemoryCopyTest, MixedSizeCacheTiles) {
    // One cache page mixes SWA, compressed KV, indexer and state regions.
    constexpr std::array<size_t, 54> pattern = {
        18048, 18048, 18432, 8704,  8192,  18048, 18048, 18048, 18048, 18048, 18048, 18432, 8704,  8192,
        18048, 18048, 18048, 18048, 18048, 18048, 18432, 8704,  8192,  18048, 18048, 18048, 18048, 18048,
        18048, 36864, 8704,  18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048,
        18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048, 18048};
    std::vector<size_t> sizes(1024);
    for (size_t i = 0; i < sizes.size(); ++i) {
        sizes[i] = pattern[i % pattern.size()];
    }
    checkRoundTrip(sizes, 100);
}

TEST_P(StagedMemoryCopyTest, ManyDistinctCopySizes) {
    std::vector<size_t> sizes(1024);
    for (size_t i = 0; i < sizes.size(); ++i) {
        sizes[i] = 16 * ((i * 97) % 4096 + 1);
    }
    checkRoundTrip(sizes, 100);
}

TEST_P(StagedMemoryCopyTest, EqualSizeCacheTiles) {
    checkRoundTrip(std::vector<size_t>(256, 18048), 10);
}

TEST_P(StagedMemoryCopyTest, UnalignedTilesAndByteTails) {
    checkRoundTrip({1, 7, 8, 15, 17, 31, 65, 257}, 20);
}

TEST_P(StagedMemoryCopyTest, TileSpanningHostSegmentsUsesCpuPacking) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(GetParam());
    auto first   = torch::full({16}, 13, options);
    auto second  = torch::full({32}, 27, options);
    auto device  = torch::empty({48}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0));
    StagedMemoryCopyParams params;
    params.device_index  = 0;
    params.host_bytes    = 48;
    params.tiles         = {{device.data_ptr(), 0, 48}};
    params.host_segments = {{second.data_ptr(), 16, 32}, {first.data_ptr(), 0, 16}};
    CopyScratch scratch;
    ASSERT_TRUE(execStagedMemoryCopy(params, &scratch.value));
    EXPECT_NE(scratch.value.host_staging, nullptr);
    std::array<uint8_t, 48> result{};
    ASSERT_EQ(cudaMemcpy(result.data(), device.data_ptr(), result.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(result[i], i < 16 ? 13 : 27);
    }
}

TEST_P(StagedMemoryCopyTest, EmptyAndInvalidParams) {
    StagedMemoryCopyParams params;
    EXPECT_TRUE(execStagedMemoryCopy(params));
    params.tiles.push_back({nullptr, 0, 0});
    EXPECT_FALSE(execStagedMemoryCopy(params));
    params.device_index = 0;
    EXPECT_FALSE(execStagedMemoryCopy(params));
}

INSTANTIATE_TEST_SUITE_P(HostMemory, StagedMemoryCopyTest, ::testing::Bool());

}  // namespace
}  // namespace rtp_llm::test
