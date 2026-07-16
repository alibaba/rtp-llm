#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {
namespace {

constexpr size_t kKvBytesPerLayer    = 41984;
constexpr size_t kScaleBytesPerLayer = 8448;
constexpr size_t kLayerNum           = 79;
constexpr size_t kDeviceGapBytes     = 256;
constexpr size_t kBlockBytes         = kLayerNum * (kKvBytesPerLayer + kScaleBytesPerLayer);

struct TileLayout {
    size_t host_offset   = 0;
    size_t device_offset = 0;
    size_t bytes         = 0;
};

struct BenchmarkResult {
    double mean_us = 0.0;
    double p50_us  = 0.0;
    double p90_us  = 0.0;
};

int envInt(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    return std::max(1, std::atoi(value));
}

double percentile(std::vector<double> values, double ratio) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1, static_cast<size_t>(ratio * static_cast<double>(values.size())));
    return values[index];
}

class NoBlockCopyBenchmarkTest: public ::testing::Test {
protected:
    void allocate(size_t block_count) {
        layouts_.clear();
        size_t host_offset   = 0;
        size_t device_offset = 0;
        for (size_t block = 0; block < block_count; ++block) {
            for (size_t layer = 0; layer < kLayerNum; ++layer) {
                for (const size_t bytes : {kKvBytesPerLayer, kScaleBytesPerLayer}) {
                    layouts_.push_back(TileLayout{host_offset, device_offset, bytes});
                    host_offset += bytes;
                    device_offset += bytes + kDeviceGapBytes;
                }
            }
        }
        host_bytes_   = host_offset;
        device_bytes_ = device_offset;
        ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void**>(&host_), host_bytes_, cudaHostAllocPortable), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_), device_bytes_), cudaSuccess);
        ASSERT_EQ(cudaMemset(device_, 0, device_bytes_), cudaSuccess);
    }

    void release() {
        if (device_ != nullptr) {
            EXPECT_EQ(cudaFree(device_), cudaSuccess);
            device_ = nullptr;
        }
        if (host_ != nullptr) {
            EXPECT_EQ(cudaFreeHost(host_), cudaSuccess);
            host_ = nullptr;
        }
        layouts_.clear();
        host_bytes_ = device_bytes_ = 0;
    }

    void TearDown() override {
        release();
    }

    void fillHostPattern(uint8_t seed) {
        for (size_t tile_idx = 0; tile_idx < layouts_.size(); ++tile_idx) {
            const auto& tile = layouts_[tile_idx];
            for (size_t i = 0; i < tile.bytes; ++i) {
                host_[tile.host_offset + i] = static_cast<uint8_t>(seed + tile_idx * 17 + i);
            }
        }
    }

    void fillDevicePattern(uint8_t seed) {
        std::vector<uint8_t> tile_buffer;
        for (size_t tile_idx = 0; tile_idx < layouts_.size(); ++tile_idx) {
            const auto& tile = layouts_[tile_idx];
            tile_buffer.resize(tile.bytes);
            for (size_t i = 0; i < tile.bytes; ++i) {
                tile_buffer[i] = static_cast<uint8_t>(seed + tile_idx * 17 + i);
            }
            ASSERT_EQ(cudaMemcpy(device_ + tile.device_offset, tile_buffer.data(), tile.bytes, cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
    }

    void verifyHostPattern(uint8_t seed) {
        for (size_t tile_idx = 0; tile_idx < layouts_.size(); ++tile_idx) {
            const auto& tile = layouts_[tile_idx];
            for (size_t i = 0; i < tile.bytes; ++i) {
                ASSERT_EQ(host_[tile.host_offset + i], static_cast<uint8_t>(seed + tile_idx * 17 + i))
                    << "tile=" << tile_idx << " offset=" << i;
            }
        }
    }

    void verifyDevicePattern(uint8_t seed) {
        std::vector<uint8_t> tile_buffer;
        for (size_t tile_idx = 0; tile_idx < layouts_.size(); ++tile_idx) {
            const auto& tile = layouts_[tile_idx];
            tile_buffer.resize(tile.bytes);
            ASSERT_EQ(cudaMemcpy(tile_buffer.data(), device_ + tile.device_offset, tile.bytes, cudaMemcpyDeviceToHost),
                      cudaSuccess);
            for (size_t i = 0; i < tile.bytes; ++i) {
                ASSERT_EQ(tile_buffer[i], static_cast<uint8_t>(seed + tile_idx * 17 + i))
                    << "tile=" << tile_idx << " offset=" << i;
            }
        }
    }

    MultiCopyParams legacyParams(bool h2d) const {
        MultiCopyParams params;
        params.multi_dst.reserve(layouts_.size());
        params.multi_src.reserve(layouts_.size());
        const auto host_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        const auto gpu_options  = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);
        for (const auto& tile : layouts_) {
            auto host_tensor =
                torch::from_blob(host_ + tile.host_offset, {static_cast<int64_t>(tile.bytes)}, host_options);
            auto gpu_tensor =
                torch::from_blob(device_ + tile.device_offset, {static_cast<int64_t>(tile.bytes)}, gpu_options);
            params.multi_dst.push_back(h2d ? gpu_tensor : host_tensor);
            params.multi_src.push_back(h2d ? host_tensor : gpu_tensor);
        }
        return params;
    }

    BatchedMemoryCopyParams batchParams(bool h2d) const {
        BatchedMemoryCopyParams params;
        params.device_index = 0;
        params.tiles.reserve(layouts_.size());
        for (const auto& tile : layouts_) {
            void* dst =
                h2d ? static_cast<void*>(device_ + tile.device_offset) : static_cast<void*>(host_ + tile.host_offset);
            const void* src = h2d ? static_cast<const void*>(host_ + tile.host_offset) :
                                    static_cast<const void*>(device_ + tile.device_offset);
            params.tiles.push_back(BatchedMemoryCopyTile{dst, src, tile.bytes});
        }
        return params;
    }

    StagedMemoryCopyParams stagedDirectH2DParams(size_t block_count, int sm_copy_block_num = 0) const {
        StagedMemoryCopyParams params;
        params.host_bytes                  = host_bytes_;
        params.direct_pinned_host_segments = true;
        params.device_index                = 0;
        params.sm_copy_block_num           = sm_copy_block_num;
        params.direction                   = StagedMemoryCopyDirection::H2D;
        params.host_segments.reserve(block_count);
        for (size_t block = 0; block < block_count; ++block) {
            const size_t block_offset = block * kBlockBytes;
            params.host_segments.push_back(
                StagedMemoryCopyHostSegment{host_ + block_offset, block_offset, kBlockBytes});
        }
        params.tiles.reserve(layouts_.size());
        for (const auto& tile : layouts_) {
            params.tiles.push_back(StagedMemoryCopyTile{device_ + tile.device_offset, tile.host_offset, tile.bytes});
        }
        return params;
    }

    template<typename Fn>
    BenchmarkResult measure(Fn&& fn) {
        const int warmup = envInt("RTP_LLM_COPY_BENCH_WARMUP", 5);
        const int iters  = envInt("RTP_LLM_COPY_BENCH_ITERS", 20);
        for (int i = 0; i < warmup; ++i) {
            fn();
        }
        std::vector<double> samples;
        samples.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            fn();
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        }
        const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        return BenchmarkResult{sum / samples.size(), percentile(samples, 0.50), percentile(samples, 0.90)};
    }

    void printResult(const char* mode, const char* direction, size_t blocks, const BenchmarkResult& result) const {
        const double gbps = static_cast<double>(host_bytes_) / result.mean_us / 1.0e3;
        std::cout << std::fixed << std::setprecision(3) << "COPY_BENCH mode=" << mode << " direction=" << direction
                  << " blocks=" << blocks << " tiles=" << layouts_.size() << " bytes=" << host_bytes_
                  << " mean_us=" << result.mean_us << " p50_us=" << result.p50_us << " p90_us=" << result.p90_us
                  << " effective_gbps=" << gbps << std::endl;
    }

    std::vector<TileLayout> layouts_;
    uint8_t*                host_         = nullptr;
    uint8_t*                device_       = nullptr;
    size_t                  host_bytes_   = 0;
    size_t                  device_bytes_ = 0;
};

TEST_F(NoBlockCopyBenchmarkTest, CorrectnessAndObservedLatency) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    const std::vector<size_t> block_counts = {1, 8, 32, 51, 96};
    for (const size_t blocks : block_counts) {
        allocate(blocks);
        StagedMemoryCopyScratch scratch;
        SCOPED_TRACE(::testing::Message() << "blocks=" << blocks);
        for (const bool h2d : {true, false}) {
            const uint8_t seed   = h2d ? 0x31 : 0x73;
            auto          legacy = legacyParams(h2d);
            auto          batch  = batchParams(h2d);
            auto          staged = stagedDirectH2DParams(blocks);
            if (h2d) {
                fillHostPattern(seed);
                ASSERT_EQ(cudaMemset(device_, 0, device_bytes_), cudaSuccess);
                execNoBlockCopy(legacy);
                ASSERT_NO_FATAL_FAILURE(verifyDevicePattern(seed));
                ASSERT_EQ(cudaMemset(device_, 0, device_bytes_), cudaSuccess);
                ASSERT_TRUE(execBatchedMemoryCopy(batch));
                ASSERT_NO_FATAL_FAILURE(verifyDevicePattern(seed));
                ASSERT_EQ(cudaMemset(device_, 0, device_bytes_), cudaSuccess);
                ASSERT_TRUE(execStagedMemoryCopy(staged, &scratch));
                ASSERT_NO_FATAL_FAILURE(verifyDevicePattern(seed));
            } else {
                fillDevicePattern(seed);
                std::memset(host_, 0, host_bytes_);
                execNoBlockCopy(legacy);
                ASSERT_NO_FATAL_FAILURE(verifyHostPattern(seed));
                std::memset(host_, 0, host_bytes_);
                ASSERT_TRUE(execBatchedMemoryCopy(batch));
                ASSERT_NO_FATAL_FAILURE(verifyHostPattern(seed));
            }

            const auto legacy_result = measure([&]() { execNoBlockCopy(legacy); });
            printResult("legacy", h2d ? "H2D" : "D2H", blocks, legacy_result);
            if (h2d) {
                const auto batch_result = measure([&]() { ASSERT_TRUE(execBatchedMemoryCopy(batch)); });
                printResult("tile_batch", "H2D", blocks, batch_result);
                const auto staged_result = measure([&]() { ASSERT_TRUE(execStagedMemoryCopy(staged, &scratch)); });
                printResult("block_batch_scatter", "H2D", blocks, staged_result);
                if (blocks == 51) {
                    for (const int sm_blocks : {8, 16, 32, 64, 128}) {
                        auto limited_staged = stagedDirectH2DParams(blocks, sm_blocks);
                        ASSERT_EQ(cudaMemset(device_, 0, device_bytes_), cudaSuccess);
                        ASSERT_TRUE(execStagedMemoryCopy(limited_staged, &scratch));
                        ASSERT_NO_FATAL_FAILURE(verifyDevicePattern(seed));
                        const auto limited_result =
                            measure([&]() { ASSERT_TRUE(execStagedMemoryCopy(limited_staged, &scratch)); });
                        const std::string mode = "block_batch_scatter_sm_blocks_" + std::to_string(sm_blocks);
                        printResult(mode.c_str(), "H2D", blocks, limited_result);
                    }
                }
            } else {
                const auto batch_result = measure([&]() { ASSERT_TRUE(execBatchedMemoryCopy(batch)); });
                printResult("batch", "D2H", blocks, batch_result);
            }
        }
        releaseStagedMemoryCopyScratch(scratch);
        release();
    }
}

}  // namespace
}  // namespace rtp_llm
