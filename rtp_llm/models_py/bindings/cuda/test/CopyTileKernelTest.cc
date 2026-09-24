#include "rtp_llm/models_py/bindings/common/kernels/CopyTileKernel.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
struct Allocation {
    void* data = nullptr;
    explicit Allocation(size_t bytes) {
        checkCuda(cudaMalloc(&data, bytes));
    }
    ~Allocation() {
        cudaFree(data);
    }
    Allocation(const Allocation&)               = delete;
    Allocation&    operator=(const Allocation&) = delete;
    unsigned char* bytes() const {
        return static_cast<unsigned char*>(data);
    }
};
struct Stream {
    cudaStream_t value{};
    Stream() {
        checkCuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    ~Stream() {
        cudaStreamDestroy(value);
    }
};
class CopyTileKernelTest: public ::testing::TestWithParam<int> {};

// Each tile has an independent payload oracle and guard bytes on both sides.
// Vary the alignment of each endpoint independently, including mismatches,
// and straddle vector/warp-loop boundaries. A round trip alone would not find
// matching gather and scatter bugs, so check each direction against CPU bytes.
TEST_P(CopyTileKernelTest, MixedAlignmentRaggedTilesAndGuards) {
    constexpr size_t             count = 137, slot = 32768, extent = count * slot;
    constexpr unsigned char      guard = 0xd3;
    const std::array<size_t, 16> widths{0, 1, 2, 3, 7, 15, 16, 17, 31, 63, 511, 513, 2049, 8193, 16385, 19008};
    const std::array<std::pair<size_t, size_t>, 9> shifts{
        {{0, 0}, {8, 8}, {4, 4}, {2, 2}, {1, 1}, {0, 1}, {1, 0}, {8, 4}, {4, 2}}};
    Allocation                 device(extent), staging(extent), descriptors(count * sizeof(CopyTile));
    Stream                     stream;
    std::vector<unsigned char> deviceExpected(extent, guard), stagingExpected(extent, guard), observed(extent);
    std::vector<CopyTile>      tiles;
    for (size_t i = 0; i < count; ++i) {
        const auto   shift         = shifts[i % shifts.size()];
        const size_t deviceOffset  = i * slot + 32 + shift.first;
        const size_t stagingOffset = i * slot + 48 + shift.second;
        const size_t width         = widths[i % widths.size()];
        tiles.push_back({device.bytes() + deviceOffset, stagingOffset, width});
        for (size_t j = 0; j < width; ++j) {
            const auto byte                    = static_cast<unsigned char>((i * 73 + j * 37 + (j >> 8)) & 255);
            deviceExpected[deviceOffset + j]   = byte;
            stagingExpected[stagingOffset + j] = byte;
        }
    }
    checkCuda(cudaMemcpyAsync(
        descriptors.data, tiles.data(), count * sizeof(CopyTile), cudaMemcpyHostToDevice, stream.value));
    checkCuda(cudaMemcpyAsync(device.data, deviceExpected.data(), extent, cudaMemcpyHostToDevice, stream.value));
    checkCuda(cudaMemsetAsync(staging.data, guard, extent, stream.value));
    ASSERT_EQ(
        cudaSuccess,
        launchCopyTiles(
            static_cast<const CopyTile*>(descriptors.data), count, staging.data, false, stream.value, GetParam()));
    checkCuda(cudaMemcpyAsync(observed.data(), staging.data, extent, cudaMemcpyDeviceToHost, stream.value));
    checkCuda(cudaStreamSynchronize(stream.value));
    EXPECT_EQ(observed, stagingExpected) << "gather payload or staging guards";
    checkCuda(cudaMemcpyAsync(observed.data(), device.data, extent, cudaMemcpyDeviceToHost, stream.value));
    checkCuda(cudaStreamSynchronize(stream.value));
    EXPECT_EQ(observed, deviceExpected) << "gather changed source bytes";

    checkCuda(cudaMemcpyAsync(staging.data, stagingExpected.data(), extent, cudaMemcpyHostToDevice, stream.value));
    checkCuda(cudaMemsetAsync(device.data, guard, extent, stream.value));
    ASSERT_EQ(cudaSuccess,
              launchCopyTiles(
                  static_cast<const CopyTile*>(descriptors.data), count, staging.data, true, stream.value, GetParam()));
    checkCuda(cudaMemcpyAsync(observed.data(), device.data, extent, cudaMemcpyDeviceToHost, stream.value));
    checkCuda(cudaStreamSynchronize(stream.value));
    EXPECT_EQ(observed, deviceExpected) << "scatter payload or device guards";
    checkCuda(cudaMemcpyAsync(observed.data(), staging.data, extent, cudaMemcpyDeviceToHost, stream.value));
    checkCuda(cudaStreamSynchronize(stream.value));
    EXPECT_EQ(observed, stagingExpected) << "scatter changed staging bytes";
}

TEST(CopyTileKernelLaunchTest, EmptyBatchAndInvalidArguments) {
    Stream stream;
    EXPECT_EQ(cudaSuccess, launchCopyTiles(nullptr, 0, nullptr, false, stream.value));
    EXPECT_EQ(cudaSuccess, launchCopyTiles(nullptr, 0, nullptr, true, stream.value));
    Allocation  staging(64), descriptors(sizeof(CopyTile));
    const auto* tiles = static_cast<const CopyTile*>(descriptors.data);
    for (int threads : {0, 1, 31, 33, 1025})
        EXPECT_EQ(cudaErrorInvalidValue, launchCopyTiles(tiles, 1, staging.data, false, stream.value, threads));
    EXPECT_EQ(cudaErrorInvalidValue, launchCopyTiles(nullptr, 1, staging.data, false, stream.value));
    EXPECT_EQ(cudaErrorInvalidValue, launchCopyTiles(tiles, 1, nullptr, true, stream.value));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream.value));
}

INSTANTIATE_TEST_SUITE_P(ThreadCandidates, CopyTileKernelTest, ::testing::Values(32, 64, 128, 256, 512, 1024));
}  // namespace
}  // namespace rtp_llm
